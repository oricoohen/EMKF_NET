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

    def setModel(self, model,args):
        self.args = args
        self.model = model # the RTSNet model contains the parameters of the RTSNet
        # Initialize PsmoothNN
        self.PsmoothNN = PsmoothNN(self.SysModel.m, self.args)
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
        # Optimizer for PsmoothNN
        self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,
                                                    weight_decay=self.weightDecay)
        # P-smooth loss weight
        self.p_smooth_weight = 0.1  # Default weight for p-smooth loss

    def P_smooth_Train(self,SysModel, cv_input, cv_target, train_input, train_target, path_results,path_rtsnet=None,load_psmooth_path = None, generate_f=True):

        '''train P-smooth network with RTSNet fixed. dont change the RTSNet'''

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.model = torch.load(path_rtsnet, weights_only=False)  # Load the best RTSNet model
        self.model.eval() # Freeze RTSNet if needed, so it doesn't change
        if load_psmooth_path != None:
            self.PsmoothNN = torch.load(load_psmooth_path, weights_only=False)
            # Re-link the optimizer to the parameters of the newly loaded model
            self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,
                                                        weight_decay=self.weightDecay)
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
            self.PsmoothNN.train()
            Batch_Psmooth_LOSS_sum = 0

            for j in range(0, self.N_B):

                n_e = random.randint(0, self.N_E - 1)
                if generate_f != None:  ####if we train with different f
                    index = n_e // 10
                    SysModel.F = SysModel.F_train[index]
                    self.model.update_F(SysModel.F)
                    # self.PsmoothNN.update_F(SysModel.F)
                y_training = train_input[n_e]
                SysModel.T = y_training.size()[-1]

                self.model.InitSequence(SysModel.m1x_0, SysModel.T)

                x_out_training_forward = torch.empty(SysModel.m, SysModel.T)
                x_out_training = torch.empty(SysModel.m, SysModel.T)
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
                self.PsmoothNN.start = 0
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
                        # self.PsmoothNN.update_F(SysModel.F)



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
                    self.PsmoothNN.start = 0
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
                    torch.save(self.PsmoothNN, path_results)


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



    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target,path_results, load_model_path=None,generate_f=True,
                CompositionLoss=False):

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        MSE_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps])
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps])

        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_steps])
        self.MSE_train_dB_epoch = torch.empty([self.N_steps])


        if load_model_path is not None:
            print("loading model_and keep training them")
            self.model = torch.load(load_model_path, weights_only=False)
            # Re-link the optimizer to the parameters of the newly loaded model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
                                              weight_decay=self.weightDecay)

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
            self.model.train()
            self.optimizer.zero_grad()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):

                self.model.init_hidden()
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



                    self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)


                    # Forward pass through RTSN et
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

                    torch.save(self.model, path_results)


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

    def compute_cross_covariances(self, F, H, Ks, Ps, SGains):
        """
        Computes lag-one cross-covariances and returns them as a single tensor.

        Returns:
            V_tensor (torch.Tensor): A single tensor of shape [m, m, T]
        """
        # Get dimensions from input tensors
        m = Ps.shape[0]
        T = Ps.shape[2]

        # 1. Create a single empty tensor with the target shape [m, m, T]
        V_tensor = torch.empty(m, m, T, device=F.device)

        # Note: Ks is the Kalman Gain for the last time step, T-1.
        # Ps is the sequence of filtered covariances P_t|t
        # SGains is the sequence of Smoother Gains S_t for t = T-2 down to 0.

        # --- Calculation for V_{T-1, T-2 | T} ---
        # The last element of V_tensor will actually be V_{T-1}
        I = torch.eye(m, device=F.device)
        # P_{T-2|T-2} is at index T-2
        # V_{T-1} uses the filtered covariance from T-2, not T-1
        V_T_minus_1 = (I - Ks @ H) @ F @ Ps[:, :, T - 2]

        # 2. Assign the result to the last time-slice of the tensor
        V_tensor[:, :, T - 1] = self.PsmoothNN.enforce_covariance_properties(V_T_minus_1)

        # --- Backward recursion for t = T-2 down to 0 ---
        for t in range(T - 2, -1, -1):
            # Get values for time t
            Pt = Ps[:, :, t]
            # Smoother gain S_t has been stored in reverse order
            # For t=T-2, we need the first element of SGains (index 0)
            # For t=T-3, we need the second element (index 1), and so on.
            index = (T - 2) - t
            St = SGains[index]
            St_minus1 = SGains[index +1]
            # Get V_{t+1, t | T} from the tensor we are filling
            V_t_plus_1 = V_tensor[:, :, t + 1]

            # The cross-covariance update equation
            V_t = Pt @ St_minus1.T + St @ (V_t_plus_1 - F @ Pt) @ St_minus1.T

            # 3. Assign the result to the correct slice [:, :, t]
            V_tensor[:, :, t] = self.PsmoothNN.enforce_covariance_properties(V_t)

        # 4. Return the single tensor
        return V_tensor


    def NNTest_no_p(self, SysModel, test_input, test_target, load_model_path, generate_f=True):


        print("Testing RTSNet...")
        self.N_T = len(test_input)


        self.MSE_test_linear_arr = torch.empty([self.N_T])


             # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Load models
        self.model = torch.load(load_model_path,weights_only=False)


        self.model.eval()

        torch.no_grad()

        x_out_list = []

        start = time.time()
        self.model.K_T_list = []

        for j in range(0, self.N_T):
            y_mdl_tst = test_input[j]
            SysModel.T_test = y_mdl_tst.size()[-1]
            x_out_test_forward_1 = torch.empty(SysModel.m, SysModel.T_test)
            x_out_test = torch.empty(SysModel.m, SysModel.T_test)

            self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
            self.model.init_hidden()

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




            self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()


            x_out_list.append(x_out_test)





        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)


        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg


        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, torch.stack(x_out_list), t, self.model.K_T_list,]
    def NNTest(self, SysModel, test_input, test_target, load_model_path,load_p_smoothe_model_path=None, generate_f=True):


        print("Testing RTSNet...")
        self.N_T = len(test_input)


        self.MSE_test_linear_arr = torch.empty([self.N_T])
        self.MSE_test_psmooth_arr = torch.empty([self.N_T])

             # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Load models
        if load_p_smoothe_model_path is not None:
            self.PsmoothNN = torch.load(load_p_smoothe_model_path,weights_only=False)
        else:
            self.PsmoothNN = torch.load('RTSNet/full_info/best-model.pt', weights_only=False)
        self.model = torch.load(load_model_path,weights_only=False)


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

            self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
            self.model.init_hidden()

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
            self.PsmoothNN.start = 0
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
            P_1_0_pred = SysModel.F @ self.SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
            s_0 = self.SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
            self.model.smoother_gain_list.append(s_0.clone().detach())  # [m, m]


            self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()

            # Compute P-smooth loss
            self.MSE_test_psmooth_arr[j] = self.PsmoothNN.compute_loss(P_smoothed_seq, test_target[j],
                                                                       x_out_test).item()

            x_out_list.append(x_out_test)
            P_smooth_list.append(P_smoothed_seq)

            #######compute V############
            V =  self.compute_cross_covariances(self.SysModel.F_test[j//10], self.SysModel.H, K, P_smoothed_seq, self.model.smoother_gain_list)
            V_list.append(V)#[seq](tensor(m,m,T))


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

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, torch.stack(x_out_list), t, torch.stack(P_smooth_list), V_list, self.model.K_T_list,
                self.MSE_test_psmooth_dB_avg, self.MSE_test_psmooth_std]

    def NNTest_HybridP(self, SysModel, test_input, test_target, load_model_path):
        """
        This function tests the RTSNet, but computes the smoothed covariance P
        analytically using the Kalman Gain (K) and Smoother Gain (S) produced by the RTSNet.
        This replaces the PsmoothNN.
        """
        print("Testing Hybrid Smoother (RTSNet states/gains, Analytical P)...")
        self.N_T = len(test_input)

        # Load the trained RTSNet model
        self.model = torch.load(load_model_path, weights_only=False)
        self.model.eval()

        # Initialize a tensor to store the MSE for each sequence >>>
        self.MSE_test_linear_arr = torch.empty([self.N_T])
        loss_fn = nn.MSELoss(reduction='mean')

        # To store the final results for all sequences
        x_out_list = []
        P_smooth_list_analytical = []
        V_list = []
        with torch.no_grad():
            for j in range(0, self.N_T):
                SysModel.T_test = test_input[j].size()[-1]
                m = SysModel.m

                # Get the correct F for this sequence
                # This uses the j // 10 logic. If you change to the fundamental
                # per-sequence F list, this line becomes: SysModel.F = SysModel.F_test[j]
                index = j // 10
                SysModel.F = SysModel.F_test[index]
                self.model.update_F(SysModel.F)

                # --- INITIALIZATION ---
                self.model.init_hidden()
                self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)

                # Lists to store the results from our analytical filter
                P_filtered_seq = torch.empty(m, m, SysModel.T_test)
                P_predicted_seq = torch.empty(m, m, SysModel.T_test)
                # Initialize P for the filter using the prior
                P_filt_prev = SysModel.m2x_0
                # <<<Initialize the tensor to store the forward sequence here >>>
                x_out_test_forward_1 = torch.empty(m, SysModel.T_test)
                K_t = None
                y_mdl_tst = test_input[j]
                # --- FORWARD PASS: HYBRID KALMAN FILTER ---
                for t in range(0, SysModel.T_test):
                    # 1. ANALYTICAL PREDICTION STEP for covariance
                    P_pred = SysModel.F @ P_filt_prev @ SysModel.F.T + SysModel.Q

                    # <<<Capture the output of the forward pass into our tensor >>>
                    x_out_test_forward_1[:, t] = self.model(y_mdl_tst[:, t], None, None, None)

                    K_t = self.model.KGain.clone()

                    # 3. ANALYTICAL UPDATE STEP using the Kalman Gain from the network
                    I = torch.eye(m)
                    # Using the numerically stable Joseph form for the covariance update
                    P_filt = (I - K_t @ SysModel.H) @ P_pred @ (I - K_t @ SysModel.H).T + K_t @ SysModel.R @ K_t.T

                    # 4. Save results and update for next step
                    P_predicted_seq[:, :, t] = P_pred
                    P_filtered_seq[:, :, t] = P_filt
                    P_filt_prev = P_filt

                # --- BACKWARD PASS: HYBRID RTS SMOOTHER ---
                # Run the RTSNet backward pass once to get all state estimates and smoother gains
                x_out_test = torch.empty(m, SysModel.T_test)
                self.model.smoother_gain_list = []  # Clear the list before populating

                x_out_test[:, -1] = x_out_test_forward_1[:, -1]
                self.model.InitBackward(x_out_test[:, -1])
                # Special first backward step for t = T-2
                x_out_test[:, SysModel.T_test - 2] = self.model(None, x_out_test_forward_1[:, SysModel.T_test - 2],
                                                                x_out_test_forward_1[:, SysModel.T_test - 1], None)
                self.model.smoother_gain_list.append(self.model.SGain.clone().detach())
                for t in range(SysModel.T_test - 3, -1, -1):  #### T-3 all the way to 0 includes [T-3,0]
                    x_out_test[:, t] = self.model(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t + 1],
                                                  x_out_test[:, t + 2])
                    self.model.smoother_gain_list.append(self.model.SGain.clone())
                #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
                P_1_0_pred = SysModel.F @ self.SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                s_0 = self.SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(
                    P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
                self.model.smoother_gain_list.append(s_0.clone().detach())  # [m, m]

                # Now, run the analytical RTS backward pass for the covariance
                P_smoothed_seq = torch.empty(m, m, SysModel.T_test)
                P_smoothed_seq[:, :, -1] = P_filtered_seq[:, :, -1]  # P_T|T is the last filtered P

                for t in range(SysModel.T_test - 2, -1, -1):
                    # Get necessary matrices for this time step
                    P_filt_t = P_filtered_seq[:, :, t]
                    P_pred_t_plus_1 = P_predicted_seq[:, :, t + 1]
                    P_smooth_t_plus_1 = P_smoothed_seq[:, :, t + 1]

                    # Get the Smoother Gain from the network for this time step
                    reverse_time = SysModel.T_test - 1 - t
                    S_t = self.model.smoother_gain_list[reverse_time]  ####S[T-1] = S(0), S[0] = S(T-1)

                    # ANALYTICAL RTS UPDATE using the smoother gain from the network
                    P_smooth_t = P_filt_t + S_t @ (P_smooth_t_plus_1 - P_pred_t_plus_1) @ S_t.T
                    P_smoothed_seq[:, :, t] = P_smooth_t

                # Save the final results for this sequence
                x_out_list.append(x_out_test)
                P_smooth_list_analytical.append(P_smoothed_seq)

                # <<<Calculate and store the MSE for the j-th sequence >>>
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()
                #######compute V############
                V = self.compute_cross_covariances(self.SysModel.F, self.SysModel.H, K_t, P_smoothed_seq,
                                                   self.model.smoother_gain_list)
                V_list.append(V)  # [seq](tensor(m,m,T))

        # <<< Average the MSEs over all sequences and print the result >>>
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        print(f"Hybrid RTSNet - MSE Test: {self.MSE_test_dB_avg:.4f} [dB]")

        # Return the full tensors of results
        return torch.stack(x_out_list), torch.stack(P_smooth_list_analytical), V_list

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

    def Train_Joint(self, SysModel, cv_input, cv_target, train_input, train_target, path_results_rtsnet,path_results_psmooth,load_rtsnet = None,load_psmooth = None,
                    generate_f=True,beta=0.7):
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        # Logging arrays
        self.MSE_train_rts_dB_epoch = torch.empty([self.N_steps])
        self.MSE_train_psmooth_dB_epoch = torch.empty([self.N_steps])
        self.MSE_cv_rts_dB_epoch = torch.empty([self.N_steps])
        self.MSE_cv_psmooth_dB_epoch = torch.empty([self.N_steps])
        self.MSE_train_total_dB_epoch = torch.empty([self.N_steps])
        self.MSE_cv_total_dB_epoch = torch.empty([self.N_steps])

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        if load_psmooth != None:
            self.model = torch.load(load_rtsnet, weights_only=False)
            self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
                                                        weight_decay=self.weightDecay)
            self.PsmoothNN = torch.load(load_psmooth, weights_only=False)
            # Re-link the optimizer to the parameters of the newly loaded model
            self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,weight_decay=self.weightDecay)


        for ti in range(0, self.N_steps):

            # Set both models to train mode
            self.model.train()
            self.PsmoothNN.train()

            # Zero gradients for both optimizers
            self.optimizer.zero_grad()
            self.PsmoothNN_optimizer.zero_grad()

            Batch_RTS_LOSS_sum = 0
            Batch_Psmooth_LOSS_sum = 0
            Batch_Total_LOSS_sum = 0

            for j in range(0, self.N_B):
                n_e = random.randint(0, self.N_E - 1)
                if generate_f:
                    index = n_e // 10
                    SysModel.F = SysModel.F_train[index]
                    self.model.update_F(SysModel.F)

                y_training = train_input[n_e]
                SysModel.T = y_training.size()[-1]

                # Run RTSNet forward and backward pass to get smoothed states and intermediate values
                x_out_training_forward = torch.empty(SysModel.m, SysModel.T)
                x_out_training = torch.empty(SysModel.m, SysModel.T)

                self.model.init_hidden()
                self.model.InitSequence(SysModel.m1x_0, SysModel.T)

                sigma_list = []
                smoother_gain_list = []

                for t in range(SysModel.T):
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                    sigma_list.append(self.model.h_Sigma.clone())  # We need to keep the graph attached

                x_out_training[:, SysModel.T - 1] = x_out_training_forward[:, SysModel.T - 1]
                K_T_1 = self.model.KGain.clone()#compute the K(T)
                self.model.InitBackward(x_out_training[:, SysModel.T - 1])
                x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],
                                                               x_out_training_forward[:, SysModel.T - 1], None)
                smoother_gain_list.append(self.model.SGain.clone())
                for t in range(SysModel.T - 3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t], x_out_training_forward[:, t + 1],
                                                      x_out_training[:, t + 2])
                    smoother_gain_list.append(self.model.SGain.clone())

                # Run PsmoothNN using the stateless method
                P_smoothed_seq = torch.empty(SysModel.m, SysModel.m, SysModel.T)
                dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)

                sigma_T = sigma_list[-1]
                # sigma_T_processed = self.PsmoothNN.FC8(sigma_T.view(1, -1)).view(1, 1, -1)
                # in_Psmooth_T = torch.cat((sigma_T_processed, dummy_sgain), dim=2)
                # h_current = in_Psmooth_T[:, :, :self.PsmoothNN.d_hidden_Psmooth].clone()
                self.PsmoothNN.start = 0
                P_flat = self.PsmoothNN(sigma_T, dummy_sgain).view(-1)  # shape: [1, 1, m²] to [m²]
                P_smoothed_seq[:, :, SysModel.T - 1] = self.PsmoothNN.enforce_covariance_properties(
                    P_flat.view(SysModel.m, SysModel.m))

                for t in range(SysModel.T - 2, -1, -1):
                    sigma_t = sigma_list[t]
                    index = (SysModel.T - 2) - t
                    sgain_t = smoother_gain_list[index]
                    P_flat = self.PsmoothNN(sigma_t, sgain_t) # [1, 1, m²] and [1, 1, d_hidden_Psmooth]
                    P_smoothed_seq[:, :, t] = self.PsmoothNN.enforce_covariance_properties(
                        P_flat.view(-1).view(SysModel.m, SysModel.m))

                # Calculate the two separate losses
                rtsnet_loss = self.loss_fn(x_out_training, train_target[n_e])
                psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq, train_target[n_e], x_out_training)

                # Combine them into a total loss
                beta_change = beta/(ti/5+1)
                total_loss = beta_change*rtsnet_loss + (1-beta_change)* psmooth_loss

                # Accumulate for logging
                Batch_RTS_LOSS_sum += rtsnet_loss
                Batch_Psmooth_LOSS_sum += psmooth_loss
                Batch_Total_LOSS_sum += total_loss

            # Average losses for the batch
            Total_LOSS_mean = Batch_Total_LOSS_sum / self.N_B
            RTSNET_LOSS_mean = Batch_RTS_LOSS_sum / self.N_B
            Psmooth_LOSS_mean = Batch_Psmooth_LOSS_sum / self.N_B
            # Backward pass on the combined loss
            Total_LOSS_mean.backward()

            # Clip gradients and step both optimizers
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.PsmoothNN.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.PsmoothNN_optimizer.step()

            # Log training losses
            self.MSE_train_rts_dB_epoch[ti] = 10 * torch.log10(Batch_RTS_LOSS_sum / self.N_B)
            self.MSE_train_psmooth_dB_epoch[ti] = 10 * torch.log10(Batch_Psmooth_LOSS_sum / self.N_B)
            self.MSE_train_total_dB_epoch[ti] = 10 * torch.log10(Batch_Total_LOSS_sum / self.N_B)
            # Validation#####################################################
            self.model.eval()
            self.PsmoothNN.eval()
            with ((torch.no_grad())):
                CV_RTS_LOSS_sum = 0
                CV_Psmooth_LOSS_sum = 0
                CV_Total_LOSS_sum = 0
                for j in range(self.N_CV):
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size()[-1]

                    if generate_f:
                        index = j // 10
                        SysModel.F = SysModel.F_valid[index]
                        self.model.update_F(SysModel.F)

                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test)
                    self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)

                    sigma_list_cv, smoother_gain_list_cv = [], []
                    for t in range(SysModel.T_test):
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                        sigma_list_cv.append(self.model.h_Sigma)

                    x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])
                    x_out_cv[:, SysModel.T_test - 2] = self.model(None, x_out_cv_forward[:, SysModel.T_test - 2],
                                                                  x_out_cv_forward[:, SysModel.T_test - 1], None)
                    smoother_gain_list_cv.append(self.model.SGain.clone())
                    for t in range(SysModel.T_test - 3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t + 1],
                                                    x_out_cv[:, t + 2])
                        smoother_gain_list_cv.append(self.model.SGain.clone())

                    P_smoothed_seq_cv = torch.empty(SysModel.m, SysModel.m, SysModel.T_test)
                    dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)  # shape: [1, 1, m²] input to PsmoothNN
                    sigma_T_cv = sigma_list_cv[-1]
                    self.PsmoothNN.start = 0

                    P_flat_cv = self.PsmoothNN(sigma_T_cv, dummy_sgain)  # shape: [1, 1, m²] to [m²]
                    P_smoothed_seq_cv[:, :, SysModel.T_test - 1] = self.PsmoothNN.enforce_covariance_properties(
                        P_flat_cv.view(-1).view(SysModel.m, SysModel.m))

                    for t in range(SysModel.T_test - 2, -1, -1):
                        sigma_t_cv = sigma_list_cv[t]
                        index = (SysModel.T_test - 2) - t
                        sgain_t_cv = smoother_gain_list_cv[index]
                        P_flat_cv = self.PsmoothNN(sigma_t_cv, sgain_t_cv)
                        P_smoothed_seq_cv[:, :, t] = self.PsmoothNN.enforce_covariance_properties(
                            P_flat_cv.view(-1).view(SysModel.m, SysModel.m))

                    CV_RTS_LOSS_sum += self.loss_fn(x_out_cv, cv_target[j]).item()
                    CV_Psmooth_LOSS_sum += self.PsmoothNN.compute_loss(P_smoothed_seq_cv, cv_target[j], x_out_cv).item()
                    CV_Total_LOSS_sum += beta*self.loss_fn(x_out_cv, cv_target[j]).item() + (1 - beta
                                )* self.PsmoothNN.compute_loss(P_smoothed_seq_cv, cv_target[j], x_out_cv).item()

                self.MSE_cv_rts_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_RTS_LOSS_sum / self.N_CV))
                self.MSE_cv_psmooth_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_Psmooth_LOSS_sum / self.N_CV))
                self.MSE_cv_total_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_Total_LOSS_sum / self.N_CV))


                # Save best models based on the main RTSNet validation loss
                if self.MSE_cv_total_dB_epoch[ti] < self.MSE_cv_dB_opt:
                    self.MSE_cv_dB_opt = self.MSE_cv_total_dB_epoch[ti]
                    self.MSE_cv_rts_dB_opt = self.MSE_cv_rts_dB_epoch[ti]
                    self.MSE_cv_psmooth_dB_opt = self.MSE_cv_psmooth_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.model, path_results_rtsnet)
                    torch.save(self.PsmoothNN, path_results_psmooth)
                    print(f"**** Best Models Saved at Epoch {ti} with CV Loss {self.MSE_cv_dB_opt:.4f} dB ****")

            # --- Comprehensive Logging ---
            # Get current epoch's values for printing
            train_rts_loss = self.MSE_train_rts_dB_epoch[ti]
            train_psmooth_loss = self.MSE_train_psmooth_dB_epoch[ti]
            train_total_loss = self.MSE_train_total_dB_epoch[ti]

            cv_rts_loss = self.MSE_cv_rts_dB_epoch[ti]
            cv_psmooth_loss = self.MSE_cv_psmooth_dB_epoch[ti]
            cv_total_loss = self.MSE_cv_total_dB_epoch[ti]

            # Use an f-string for clean, aligned printing
            print(f"Epoch {ti:03d}/{self.N_steps - 1} | "
                  f"TRAIN: [RTS: {train_rts_loss:8.3f}, PSmooth: {train_psmooth_loss:8.3f}, Total: {train_total_loss:8.3f}] dB | "
                  f"CV: [RTS: {cv_rts_loss:8.3f}, PSmooth: {cv_psmooth_loss:8.3f}, Total: {cv_total_loss:8.3f}] dB | "
                  f"BEST Total: {self.MSE_cv_dB_opt:8.3f} dB (at epoch {self.MSE_cv_idx_opt})")

        # After all epochs are done, return the logged histories for plotting
        return [self.MSE_train_rts_dB_epoch, self.MSE_train_psmooth_dB_epoch,self.MSE_train_total_dB_epoch, self.MSE_cv_rts_dB_epoch,
                self.MSE_cv_psmooth_dB_epoch,self.MSE_cv_total_dB_epoch]



    def compute_gaussian_loss(self, P_estimated_seq, x_target_seq, x_estimated_seq):
        """
        Computes the Gaussian Log-Likelihood loss from Equation (21) of the paper.

        Args:
            P_estimated_seq (torch.Tensor): The sequence of predicted covariance matrices [m, m, T].
            x_target_seq (torch.Tensor): The ground truth state sequence [m, T].
            x_estimated_seq (torch.Tensor): The estimated state sequence [m, T].

        Returns:
            torch.Tensor: A single scalar value for the loss.
        """
        m, T = x_target_seq.shape
        total_loss = 0.0

        # A small identity matrix for numerical stability
        # This prevents taking the inverse or log-determinant of a singular matrix
        identity_matrix = torch.eye(m, device=P_estimated_seq.device) * 1e-5

        for t in range(T):
            # Get the tensors for the current time step t
            P_t = P_estimated_seq[:, :, t]
            x_true = x_target_seq[:, t]
            x_est = x_estimated_seq[:, t]

            # 1. Calculate the error vector
            e_t = (x_true - x_est).unsqueeze(1)  # Shape: [m, 1]

            # 2. Add the small identity matrix for stability before inverting
            P_t_stable = P_t + identity_matrix

            # 3. Calculate the two parts of the loss function
            mahalanobis_term = e_t.T @ torch.linalg.inv(P_t_stable) @ e_t
            log_det_term = torch.log(torch.det(P_t_stable))

            # 4. Add the loss for this time step to the total
            total_loss += mahalanobis_term + log_det_term

        # Return the average loss over all time steps
        return total_loss / T