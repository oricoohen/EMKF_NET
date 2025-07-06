"""
This file contains the class Pipeline_ERTS,
which is used to train and test RTSNet in both linear and non-linear cases.
"""

import torch
import torch.nn as nn
import time
import random
from Plot import Plot_extended as Plot
from RTSNet.PsmoothNN import PsmoothNN  # Ensure the PsmoothNN class is correctly imported



class Pipeline_ERTS:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.SysModel = ssModel #the dinamic system model contains the F, Q, H, R, T, T_test

    def setModel(self, model):
        self.model = model # the RTSNet model contains the parameters of the RTSNet

    def setTrainingParams(self, args, alpha=0.5, b=0.5):
        self.N_steps = args.n_steps  # Number of Training Steps
        self.N_B = args.n_batch  # Number of Samples in Batch
        self.learningRate = args.lr  # Learning Rate
        self.weightDecay = args.wd  # L2 Weight Regularization - Weight Decay
        self.alpha = alpha  # Composition loss factor
        self.b = b  # Weight factor between main loss and P-smooth loss
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',factor=0.9, patience=20)

        # ori add p smoothed
        # Save args for later
        self.args = args
        # Initialize PsmoothNN
        self.PsmoothNN = PsmoothNN(self.SysModel.m, self.args)
        # Optimizer for PsmoothNN
        self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,
                                                    weight_decay=self.weightDecay)
        # P-smooth loss weight
        self.p_smooth_weight = 0.1  # Default weight for p-smooth loss

    def P_smooth_Train(self,SysModel, cv_input, cv_target, train_input, train_target, path_results,path_rtsnet='best-model.pt', generate_f=None,
                 randomInit=False, cv_init=None, train_init=None):

        '''train P-smooth network with RTSNet fixed. dont change the RTSNet'''

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.model = torch.load(path_results + path_rtsnet, weights_only=False)  # Load the best RTSNet model
        self.model.eval() # Freeze RTSNet if needed, so it doesn't change

        self.PsmoothNN.train()  # Set P-smooth network to train mode

        # Preallocate arrays for logging training performance

        self.MSE_train_linear_epoch = torch.empty([self.N_steps])
        self.MSE_train_dB_epoch = torch.empty([self.N_steps])
        self.MSE_cv_idx_opt = 0
        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps])
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps])
        ##############
        ### Epochs ###
        ##############


        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            # Zero gradients for both optimizers
            self.PsmoothNN_optimizer.zero_grad()

            Batch_Psmooth_LOSS_sum = 0

            for j in range(0, self.N_B):
                n_e = random.randint(0, self.N_E - 1)
                if generate_f != None:  ####if we train with different f
                    index = n_e // 10
                    SysModel.F = SysModel.F_train[index]
                    self.model.update_F(SysModel.F)
                y_training = train_input[n_e]
                SysModel.T = y_training.size()[-1]

                x_out_training_forward = torch.empty(SysModel.m, SysModel.T)
                x_out_training = torch.empty(SysModel.m, SysModel.T)

                if (randomInit):
                    self.model.InitSequence(train_init[n_e], SysModel.T)
                else:
                    self.model.InitSequence(SysModel.m1x_0, SysModel.T)
                ########add changes to compute P and S
                #####compute P ori
                self.model.sigma_list = []  # is added in every step_KGain_est(self, y) [1, 1, m²]
                self.model.smoother_gain_list = []  # is added in every step_RTSGain_est(self, filter_x_nexttime, smoother_x_tplus2)
                for t in range(0, SysModel.T):
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                    P_forward = self.model.h_Sigma.clone().detach()  # [1, 1, m²]
                    self.model.sigma_list.append(P_forward)  # [1, 1, m²]
                # -------- RTSNet Backward Pass: Compute smoother gains and smoothed states --------
                # Start backward smoothing: initialize using last forward estimate
                x_out_training[:, SysModel.T - 1] = x_out_training_forward[:,SysModel.T - 1]  # backward smoothing starts from x_T|T
                self.model.InitBackward(x_out_training[:, SysModel.T - 1])
                x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],
                                                               x_out_training_forward[:, SysModel.T - 1], None)
                self.model.smoother_gain_list.append(self.model.SGain.clone().detach())  # ori save the T-1 sgain
                for t in range(SysModel.T - 3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t],
                                                      x_out_training_forward[:, t + 1], x_out_training[:, t + 2])
                    self.model.smoother_gain_list.append(self.model.SGain.clone().detach())  # Save detached copy ori shape[m, m]
                # ---- Handle initial smoothed P at time T ----
                P_smoothed_seq = torch.empty(SysModel.m, SysModel.m, SysModel.T)
                dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)  # shape: [1, 1, m²] input to PsmoothNN
                sigma_T = self.model.sigma_list[-1] # shape: [1, 1, m²] input to PsmoothNN
                ####compute the P(T)
                P_flat = self.PsmoothNN(sigma_T, dummy_sgain).view(-1)# shape: [1, 1, m²] to [m²]
                P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(SysModel.m,SysModel.m))# shape: [m, m]
                P_smoothed_seq[:, :, SysModel.T - 1] = P_matrix  # shape: [m, m]
                # ---- Loop over t = 1 to T for learning P_smooth ----
                for t in range(SysModel.T - 2, -1, -1):  # Loop from T-2 down to 0
                    sigma_t = self.model.sigma_list[t].view(1, 1, -1)  # sigma_t: shape [1, 1, m²]
                    # Compute the proper index for smoother_gain_list
                    index = (SysModel.T - 2) - t
                    sgain_t = self.model.smoother_gain_list[index].reshape(1, 1, -1)  # Now sgain_t: [1, 1, m²]
                    # Forward pass through PsmoothNN
                    P_flat = self.PsmoothNN(sigma_t, sgain_t)  # [1, 1, m²]
                    # Enforce PSD properties
                    P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(-1).view(SysModel.m,SysModel.m))  # [m, m]
                    # Save result in the sequence tensor
                    P_smoothed_seq[:, :, t] = P_matrix  # [ m, m]


                # Compute P-smooth loss using PsmoothNN's compute_loss method
                # Detach x_out_training to prevent gradient flow to RTSNet
                psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq, train_target[n_e], x_out_training.detach())

                # Accumulate losses
                Batch_Psmooth_LOSS_sum += psmooth_loss

            # Average losses for this batch
            Batch_Psmooth_LOSS_mean = Batch_Psmooth_LOSS_sum / self.N_B
            MSE_train_psmooth_batch = Batch_Psmooth_LOSS_mean


            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_psmooth_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])



            # Then train PsmoothNN
            Batch_Psmooth_LOSS_mean.backward()
            self.PsmoothNN_optimizer.step()


            ##################
            ### Optimizing ###
            ##################

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.PsmoothNN.eval()  # Set PsmoothNN to eval mode
            with torch.no_grad():
                MSE_cv_psmooth_batch = torch.empty([self.N_CV])

                for j in range(0, self.N_CV):
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size()[-1]

                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test)

                    if generate_f != None:  ####if we valid with different f
                        index = j // 10
                        SysModel.F = SysModel.F_valid[index]
                        self.model.update_F(SysModel.F)


                    if(randomInit):
                        if(cv_init==None):
                            self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
                        else:
                            self.model.InitSequence(cv_init[j], SysModel.T_test)
                    else:
                        self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)

                    # Forward pass and compute P-smooth
                    # Initialize lists to store intermediate values
                    self.model.sigma_list = []  # List of [1, 1, m²] tensors for each time step
                    self.model.smoother_gain_list = []  # List of [m, m] tensors for each time step

                    # Forward pass through RTSNet
                    for t in range(0, SysModel.T_test):
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                        P_cv_forward = self.model.h_Sigma.clone().detach()
                        self.model.sigma_list.append(P_cv_forward)  # [1, 1, m²]
                    # Initialize backward pass
                    x_out_cv[:, SysModel.T_test-1] = x_out_cv_forward[:, SysModel.T_test-1]  # [m]
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test-1])
                    # First backward step
                    x_out_cv[:, SysModel.T_test-2] = self.model(None, x_out_cv_forward[:, SysModel.T_test-2], x_out_cv_forward[:, SysModel.T_test-1],None)  # [m]
                    self.model.smoother_gain_list.append(self.model.SGain.clone().detach())  # [m, m]
                    # Remaining backward steps
                    for t in range(SysModel.T_test-3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t+1],x_out_cv[:, t+2])  # [m]
                        self.model.smoother_gain_list.append(self.model.SGain.clone().detach())  # [m, m]

                    # Initialize P-smooth sequence tensor
                    P_smoothed_seq = torch.empty(SysModel.m, SysModel.m, SysModel.T_test)  # [m, m, T_test]
                    dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)  # shape: [1, 1, m²] input to PsmoothNN
                    sigma_T = self.model.sigma_list[-1]  # shape: [1, 1, m²] input to PsmoothNN
                    # Handle initial P-smooth at time T_test
                    P_flat = self.PsmoothNN(sigma_T, dummy_sgain).view(-1)  # shape: [1, 1, m²] to [m²]
                    P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(SysModel.m, SysModel.m))  # shape: [m, m]
                    P_smoothed_seq[:, :, SysModel.T_test - 1] = P_matrix  # shape: [m, m]

                    # Compute P-smooth for remaining time steps
                    for t in range(SysModel.T_test - 2, -1, -1):
                        sigma_t = self.model.sigma_list[t].view(1, 1, -1)  # [1, 1, m²]
                        index = (SysModel.T_test - 2) - t
                        sgain_t = self.model.smoother_gain_list[index].reshape(1, 1, -1)  # [1, 1, m²]
                        P_flat = self.PsmoothNN(sigma_t, sgain_t)  # [1, 1, m²]
                        P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(-1).view(SysModel.m,SysModel.m))  # [m, m]
                        P_smoothed_seq[:, :, t] = P_matrix  # [m, m]

                    # Compute P-smooth validation loss
                    MSE_cv_psmooth_batch[j] = self.PsmoothNN.compute_loss(P_smoothed_seq, cv_target[j], x_out_cv).item()  # Scalar

                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_psmooth_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti

                    torch.save(self.PsmoothNN, path_results + 'best-psmooth.pt')

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE P_smoothe Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE P_smoothe Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        return  [self.MSE_train_dB_epoch[ti],self.MSE_cv_dB_epoch[ti]]



    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, generate_f=True,
                CompositionLoss=False, randomInit=False, cv_init=None, train_init=None):

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        MSE_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps])
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps])

        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_steps])
        self.MSE_train_dB_epoch = torch.empty([self.N_steps])

        # Training Mode
        self.model.train()

        # Init Hidden State
        self.model.init_hidden()
        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        nan_streak = 0

        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            # Zero gradients for both optimizers
            self.optimizer.zero_grad()

            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):
                n_e = random.randint(0, self.N_E - 1)
                if generate_f != None:  ####if we train with different f
                    index = n_e // 10
                    SysModel.F = SysModel.F_train[index]
                    self.model.update_F(SysModel.F)
                    # Debug check
                    # print(f"[DEBUG] Sample {j}:")
                    # print("F matrix:\n", SysModel.F)
                    # print("f(x) output for [1.0, 1.0]:", SysModel.f(torch.tensor([1.0, 1.0])))
                y_training = train_input[n_e]
                SysModel.T = y_training.size()[-1]

                x_out_training_forward = torch.empty(SysModel.m, SysModel.T)
                x_out_training = torch.empty(SysModel.m, SysModel.T)

                if (randomInit):
                    self.model.InitSequence(train_init[n_e], SysModel.T)
                else:
                    self.model.InitSequence(SysModel.m1x_0, SysModel.T)

                for t in range(0, SysModel.T):
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                x_out_training[:, SysModel.T - 1] = x_out_training_forward[:,SysModel.T - 1]  # backward smoothing starts from x_T|T
                self.model.InitBackward(x_out_training[:, SysModel.T-1])
                x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],x_out_training_forward[:, SysModel.T - 1], None)
                for t in range(SysModel.T - 3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t],x_out_training_forward[:, t + 1], x_out_training[:, t + 2])

                # Compute losses separately
                if (CompositionLoss):
                    y_hat = torch.empty([SysModel.n, SysModel.T])
                    for t in range(SysModel.T):
                        y_hat[:, t] = SysModel.h(x_out_training[:, t])
                    rtsnet_loss = self.alpha * self.loss_fn(x_out_training, train_target[n_e]) + (1 - self.alpha) * self.loss_fn(y_hat, train_input[n_e])
                else:
                    rtsnet_loss = self.loss_fn(x_out_training, train_target[n_e])


                # Accumulate losses
                Batch_Optimizing_LOSS_sum += rtsnet_loss

                MSE_train_linear_batch[j] = rtsnet_loss.item()

            # Average losses for this batch
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            # Train RTSNet first
            Batch_Optimizing_LOSS_mean.backward()
            # 1) check every gradient tensor ori 2 blocks
            bad_grad = False
            for p in self.model.parameters():
                if p.grad is None:  # this param wasn’t used this pass
                    continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    bad_grad = True
                    break

            if bad_grad:  # → skip this batch
                print("NaN/Inf gradients → batch skipped")
                nan_streak += 1
                if nan_streak >= 3:  # three bad batches in a row
                    print("Stopping training (3 consecutive bad batches).")
                    return  # leave NNTrain early
                self.model.zero_grad(set_to_none=True)  # throw away bad grads
                continue  # start next epoch iteration


                # Calling the step function on an Optimizer makes an update to its
                # parameters
                nan_streak = 0


            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)#ori
            self.optimizer.step()


            # Average for logging
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            with torch.no_grad():
                MSE_cv_linear_batch = torch.empty([self.N_CV])

                for j in range(0, self.N_CV):
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size()[-1]

                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test)

                    if generate_f != None:  ####if we valid with different f
                        index = j // 10
                        SysModel.F = SysModel.F_valid[index]
                        self.model.update_F(SysModel.F)

                    if (randomInit):
                        if (cv_init == None):
                            self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
                        else:
                            self.model.InitSequence(cv_init[j], SysModel.T_test)
                    else:
                        self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)


                    # Forward pass through RTSNet
                    for t in range(0, SysModel.T_test):
                        # x_out_cv_forward: [m] - Forward state estimates
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                    # Initialize backward pass
                    x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]  # [m]
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])
                    # First backward step
                    x_out_cv[:, SysModel.T_test - 2] = self.model(None, x_out_cv_forward[:, SysModel.T_test - 2],x_out_cv_forward[:, SysModel.T_test - 1], None)  # [m]
                    # Remaining backward steps
                    for t in range(SysModel.T_test - 3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t + 1],x_out_cv[:, t + 2])  # [m]


                    MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j]).item()  # Scalar

                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.model, path_results + 'best-model.pt')



            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :",
                  self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")



        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def compute_cross_covariances(self,F, H, Ks, Ps, SGains):
        """
        Args:
            F: [m, m] state transition matrix
            H: [n, m] measurement matrix
            Ks: the last Kalman gain [m, n]
            Ps: list of filtered covariances [m, m] x [T]
            SGains: list of smoother gains [T-1] x [m, m]

        Returns:
            V: list of cross-covariance matrices V_t_tminus1, each [m, m]
        """
        T = Ps.shape[2]
        V = [None for _ in range(T)]#list in length T

        # Equation for V_T,T-1
        I = torch.eye(H.shape[1])
        V[T - 1] = (I - Ks @ H) @ F @ Ps[:, :, T - 2]
        #make_covariance_matrix
        V[T - 1] = self.PsmoothNN.enforce_covariance_properties(V[T - 1])
        # Backward recursion
        for t in range(T - 2,-1,-1): #from T-2 to 0
            Pt = Ps[:, :, t]
            St = SGains[(T - 2) - t]
            Stm1_T = SGains[(T - 1) - t]
            V[t] = Pt @ Stm1_T.T + St @ (V[t + 1] - F @ Pt) @ Stm1_T.T
            # make_covariance_matrix
            V[t] = self.PsmoothNN.enforce_covariance_properties(V[t])
        return V


    def NNTest(self, SysModel, test_input, test_target, path_results, generate_f=None, MaskOnState=False,
               randomInit=False, test_init=None, load_model=False, load_model_path=None,load_p_smoothe_model_path=None):

        print("Testing RTSNet...")
        self.N_T = len(test_input)


        self.MSE_test_linear_arr = torch.empty([self.N_T])
        self.MSE_test_psmooth_arr = torch.empty([self.N_T])

        if MaskOnState:
            mask = torch.tensor([True, False, False])
            if SysModel.m == 2:
                mask = torch.tensor([True, False])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Load models
        if load_model:
            self.model = torch.load(load_model_path,weights_only=False)
            self.PsmoothNN = torch.load(load_p_smoothe_model_path)
        else:
            self.model = torch.load(path_results + 'best-model.pt',weights_only=False)
            self.PsmoothNN = torch.load(path_results + 'best-psmooth.pt',weights_only=False)

        self.model.eval()
        self.PsmoothNN.eval()

        torch.no_grad()

        x_out_list = []
        P_smooth_list = []
        V_list = []
        start = time.time()
        self.model.K_T_list = []

        for j in range(0, self.N_T):
            y_mdl_tst = test_input[j]
            SysModel.T_test = y_mdl_tst.size()[-1]

            x_out_test_forward_1 = torch.empty(SysModel.m, SysModel.T_test)
            x_out_test = torch.empty(SysModel.m, SysModel.T_test)

            if (randomInit):
                self.model.InitSequence(test_init[j], SysModel.T_test)
            else:
                self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)

            if generate_f != None:  ####if we valid with different f
                index = j // 10
                SysModel.F = SysModel.F_test[index]
                self.model.update_F(SysModel.F)
            # Forward pass and compute P-smooth
            self.model.sigma_list = []
            self.model.smoother_gain_list = []
            for t in range(0, SysModel.T_test):
                x_out_test_forward_1[:, t] = self.model(y_mdl_tst[:, t], None, None, None)
                P_test_forward = self.model.h_Sigma.clone().detach()
                self.model.sigma_list.append(P_test_forward)  # [1, 1, m²]
                if t == SysModel.T_test - 1:
                    K = self.model.KGain.clone().detach()
                    self.model.K_T_list.append(K)  # [m, n]
            x_out_test[:, SysModel.T_test - 1] = x_out_test_forward_1[:, SysModel.T_test - 1]
            self.model.InitBackward(x_out_test[:, SysModel.T_test - 1])
            x_out_test[:, SysModel.T_test - 2] = self.model(None, x_out_test_forward_1[:, SysModel.T_test - 2],
                                                            x_out_test_forward_1[:, SysModel.T_test - 1], None)
            self.model.smoother_gain_list.append(self.model.SGain.clone().detach())

            for t in range(SysModel.T_test - 3, -1, -1):#T-3 to 0
                x_out_test[:, t] = self.model(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t + 1],
                                              x_out_test[:, t + 2])
                self.model.smoother_gain_list.append(self.model.SGain.clone().detach())##there are T-1 s gain

            # Compute P-smooth predictions
            P_smoothed_seq = torch.empty(SysModel.m, SysModel.m, SysModel.T_test)
            dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)  # shape: [1, 1, m²] input to PsmoothNN
            sigma_T = self.model.sigma_list[-1]  # shape: [1, 1, m²] input to PsmoothNN
            # Handle initial P-smooth at time T_test
            P_flat = self.PsmoothNN(sigma_T, dummy_sgain).view(-1)  # shape: [1, 1, m²] to [m²]
            P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(SysModel.m, SysModel.m))  # shape: [m, m]
            P_smoothed_seq[:, :, SysModel.T_test - 1] = P_matrix  # shape: [m, m]

            for t in range(SysModel.T_test - 2, -1, -1):
                sigma_t = self.model.sigma_list[t].view(1, 1, -1)
                index = (SysModel.T_test - 2) - t
                sgain_t = self.model.smoother_gain_list[index].reshape(1, 1, -1)
                P_flat = self.PsmoothNN(sigma_t, sgain_t)
                P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(-1).view(SysModel.m,SysModel.m))
                P_smoothed_seq[:, :, t] = P_matrix

            #compute s(0) for later use, by the # S_t = P_t * F.T * (P_t+1)^-1
            s_0 = self.SysModel.m2x_0@SysModel.F@torch.inverse(P_smoothed_seq[:, :, 0].view(SysModel.m,SysModel.m))
            self.model.smoother_gain_list.append(s_0.clone().detach())  # [m, m]


            if (MaskOnState):
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test[mask], test_target[j][mask]).item()
            else:
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()

            # Compute P-smooth loss
            self.MSE_test_psmooth_arr[j] = self.PsmoothNN.compute_loss(P_smoothed_seq, test_target[j],
                                                                       x_out_test).item()

            x_out_list.append(x_out_test)
            P_smooth_list.append(P_smoothed_seq)

            #######compute V############
            V =  self.compute_cross_covariances(self.SysModel.F_test[j//10], self.SysModel.H, K, P_smoothed_seq, self.model.smoother_gain_list)
            V_list.append(V)


        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        self.MSE_test_psmooth_avg = torch.mean(self.MSE_test_psmooth_arr)
        self.MSE_test_psmooth_dB_avg = 10 * torch.log10(self.MSE_test_psmooth_avg)
        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.MSE_test_psmooth_std = torch.std(self.MSE_test_psmooth_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg
        self.test_std_dB = 10 * torch.log10(self.MSE_test_psmooth_std + self.MSE_test_psmooth_avg) - self.MSE_test_psmooth_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        str = self.modelName + "-" + "P-smooth MSE Test:"
        print(str, self.MSE_test_psmooth_dB_avg, "[dB]")
        str = self.modelName + "-" + "P-smooth STD Test:"
        print(str, self.MSE_test_psmooth_std, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_list, t, P_smooth_list, V_list, self.model.K_T_list,
                self.MSE_test_psmooth_dB_avg, self.MSE_test_psmooth_std]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_steps, self.N_B, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)

    def PlotTrain_RTS(self, MSE_KF_linear_arr, MSE_KF_dB_avg, MSE_RTS_linear_arr, MSE_RTS_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_E, self.N_steps, self.N_B, MSE_KF_dB_avg, MSE_RTS_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, MSE_RTS_linear_arr, self.MSE_test_linear_arr)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)