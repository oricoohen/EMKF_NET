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
import torch.nn as nn
from emkf.main_emkf_func_AI import EMKF_F_Mstep



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

                self.model.init_hidden()
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
                #oprion 1
                psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq, train_target[n_e], x_out_training.detach())
                #option 2
                #psmooth_loss = self.compute_gaussian_loss1(P_smoothed_seq, train_target[n_e], x_out_training.detach())

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
            # right after Batch_Psmooth_LOSS_mean.backward()
            total_grad = 0.0
            # for p in self.PsmoothNN.parameters():
            #     if p.grad is not None:
            #         total_grad += p.grad.norm().item()
            # print(f"Epoch {ti:03d} – gradient L2-norm on PsmoothNN = {total_grad:.4e}")
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

                    self.model.init_hidden()
                    self.model.InitSequence(SysModel.m1x_0, SysModel.T)

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
                    #option 1
                    MSE_cv_psmooth_batch[j] = self.PsmoothNN.compute_loss(P_smoothed_seq, cv_target[j], x_out_cv)  # Scalar
                    #option 2
                    #MSE_cv_psmooth_batch[j] = self.compute_gaussian_loss1(P_smoothed_seq, cv_target[j], x_out_cv)
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

                # Init Hidden State
                self.model.InitSequence(SysModel.m1x_0, SysModel.T)
                self.model.init_hidden()



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
                    self.model.init_hidden()


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

    def NNTrain_with_F(self, SysModel, cv_input, cv_target, train_input, train_target,path_results, load_model_path=None,generate_f=True,beta = 0.5):

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        MSE_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps])
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps])

        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_steps])
        self.MSE_train_dB_epoch = torch.empty([self.N_steps])

        m = SysModel.m

        F_train_copy = [F.clone() for F in SysModel.F_train]
        F_valid_copy = [F.clone() for F in SysModel.F_valid]

        if load_model_path is not None:
            print("loading model_and keep training them")
            self.model = torch.load(load_model_path, weights_only=False)
            # Re-link the optimizer to the parameters of the newly loaded model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
                                              weight_decay=self.weightDecay)

        # Training Mode
        self.model.train()


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
            F_loss_batch = 0

            for j in range(0, self.N_B):


                n_e = random.randint(0, self.N_E - 1)
                if generate_f != None:  ####if we train with different f
                    index = n_e // 10
                    SysModel.F = SysModel.F_train[index]
                    self.model.update_F(SysModel.F)

                y_training = train_input[n_e]
                SysModel.T = y_training.size()[-1]

                V_list = []
                x_out_training_forward = torch.empty(SysModel.m, SysModel.T)
                x_out_training = torch.empty(SysModel.m, SysModel.T)

                # Init Hidden State
                self.model.InitSequence(SysModel.m1x_0, SysModel.T)
                self.model.init_hidden()

                # Lists to store the results from our analytical filter
                P_filtered_seq = torch.empty(m, m, SysModel.T)
                P_predicted_seq = torch.empty(m, m, SysModel.T)
                # Initialize P for the filter using the prior
                P_filt_prev = SysModel.m2x_0

                for t in range(0, SysModel.T):
                    # 1. ANALYTICAL PREDICTION STEP for covariance
                    P_pred = SysModel.F @ P_filt_prev @ SysModel.F.T + SysModel.Q
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                    K_t = self.model.KGain.clone()
                    # 3. ANALYTICAL UPDATE STEP using the Kalman Gain from the network
                    I = torch.eye(m)
                    # Using the numerically stable Joseph form for the covariance update
                    P_filt = (I - K_t @ SysModel.H) @ P_pred @ (I - K_t @ SysModel.H).T + K_t @ SysModel.R @ K_t.T
                    # 4. Save results and update for next step
                    P_predicted_seq[:, :, t] = P_pred
                    P_filtered_seq[:, :, t] = P_filt
                    P_filt_prev = P_filt

                self.model.smoother_gain_list = []  # Clear the list before populating
                x_out_training[:, SysModel.T - 1] = x_out_training_forward[:,SysModel.T - 1]  # backward smoothing starts from x_T|T
                self.model.InitBackward(x_out_training[:, SysModel.T-1])
                x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],x_out_training_forward[:, SysModel.T - 1], None)
                self.model.smoother_gain_list.append(self.model.SGain.clone())
                for t in range(SysModel.T - 3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t],x_out_training_forward[:, t + 1], x_out_training[:, t + 2])
                    self.model.smoother_gain_list.append(self.model.SGain.clone())
                #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
                P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.linalg.inv(P_1_0_pred)  ######COMPUTE S_0
                self.model.smoother_gain_list.append(s_0.clone())  # [m, m]

                P_smoothed_seq = torch.empty(m, m, SysModel.T)
                P_smoothed_seq[:, :, -1] = P_filtered_seq[:, :, -1]  # P_T|T is the last filtered P
                for t in range(SysModel.T - 2, -1, -1):
                    # Get necessary matrices for this time step
                    P_filt_t = P_filtered_seq[:, :, t]
                    P_pred_t_plus_1 = P_predicted_seq[:, :, t + 1]
                    P_smooth_t_plus_1 = P_smoothed_seq[:, :, t + 1]

                    # Get the Smoother Gain from the network for this time step
                    reverse_time = SysModel.T - 1 - t
                    S_t = self.model.smoother_gain_list[reverse_time]  ####S[T-1] = S(0), S[0] = S(T-1)

                    # ANALYTICAL RTS UPDATE using the smoother gain from the network
                    P_smooth_t = P_filt_t + S_t @ (P_smooth_t_plus_1 - P_pred_t_plus_1) @ S_t.T
                    P_smoothed_seq[:, :, t] = P_smooth_t

                V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_filtered_seq,
                                                   self.model.smoother_gain_list)
                V_list.append(V)  # [seq](tensor(m,m,T))

                ###################### 2) run M‑step on **this sequence** (batch of size 1)
                X_s = x_out_training.unsqueeze(0)  # → [1, m, T]
                P_smooth_s = P_smoothed_seq.unsqueeze(0)  # → [1, m, m, T]
                V_s = V.unsqueeze(0)  # → [1, m, m, T]
                m_state = SysModel.F.shape[0]

                F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, V_s, m_state)
                F_est = F_est[0]
                index_1 = n_e // 10
                ###########CHANGE F AS TRAINING##########
                F_train_copy[index_1] = F_est
                #####################################
                F_TRUE = SysModel.F_train_TRUE[index_1]
                eps_f = F_est - F_TRUE
                eps = torch.linalg.norm(eps_f, ord='fro')

                F_loss_batch += eps
                rtsnet_loss = self.loss_fn(x_out_training, train_target[n_e])
                # Accumulate losses
                Batch_Optimizing_LOSS_sum += rtsnet_loss
                MSE_train_linear_batch[j] = rtsnet_loss.item()

            # Average losses for this batch
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            F_loss_mean = F_loss_batch/ self.N_B
            loss_total = beta*Batch_Optimizing_LOSS_mean + (1-beta)*F_loss_mean
            loss_total_training_db = 10 * torch.log10(loss_total)

            # Train RTSNet first
            loss_total.backward()
            # 1) check every gradient tensor ori 2 blocks
            # bad_grad = False
            # for p in self.model.parameters():
            #     if p.grad is None:  # this param wasn’t used this pass
            #         continue
            #     if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            #         bad_grad = True
            #         break
            #
            # if bad_grad:  # → skip this batch
            #     print("NaN/Inf gradients → batch skipped")
            #     nan_streak += 1
            #     if nan_streak >= 3:  # three bad batches in a row
            #         print("Stopping training (3 consecutive bad batches).")
            #         return  # leave NNTrain early
            #     self.model.zero_grad(set_to_none=True)  # throw away bad grads
            #     continue  # start next epoch iteration
            # ── DEBUG F‑LOSS GRADIENT CHECK ──
            # print("=== DEBUG: gradient norms after F_loss.backward() ===")
            # no_grad = True
            # for name, param in self.model.named_parameters():
            #     if param.grad is None:
            #         print(f"{name:30s} grad is None")
            #     else:
            #         gnorm = param.grad.norm().item()
            #         print(f"{name:30s} grad norm = {gnorm:.6e}")
            #         if gnorm > 0:
            #             no_grad = False
            # print(">>> Any nonzero grads? ", not no_grad)
            # print(V.requires_grad)  # should be True
            # print(V.grad_fn)  # should NOT be None
            # print(type(V.grad_fn))  # e.g. <class 'StackBackward0'>




            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)#ori
            self.optimizer.step()


            # Average for logging
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])


            ##################
            ### Optimizing ###
            ##################

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            with torch.no_grad():
                F_loss_batch_cv = 0.0
                MSE_cv_linear_batch = torch.empty([self.N_CV])

                for j in range(0, self.N_CV):
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size()[-1]

                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test)

                    P_filtered_seq = torch.empty(m, m, SysModel.T_test)
                    P_predicted_seq = torch.empty(m, m, SysModel.T_test)


                    if generate_f != None:  ####if we valid with different f
                        index = j // 10
                        SysModel.F = SysModel.F_valid[index]
                        self.model.update_F(SysModel.F)

                    self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
                    self.model.init_hidden()

                    P_filt_prev = SysModel.m2x_0
                    # Forward pass through RTSN et
                    for t in range(0, SysModel.T_test):
                        P_pred = SysModel.F @ P_filt_prev @ SysModel.F.T + SysModel.Q
                        # x_out_cv_forward: [m] - Forward state estimates
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                        K_t = self.model.KGain.clone()
                        I = torch.eye(m, device=P_pred.device)
                        P_filt = (I - K_t @ SysModel.H) @ P_pred @ (I - K_t @ SysModel.H).T + K_t @ SysModel.R @ K_t.T

                        P_predicted_seq[:, :, t] = P_pred
                        P_filtered_seq[:, :, t] = P_filt
                        P_filt_prev = P_filt

                    # ---------- BACKWARD (RTS for x, analytic P_smooth using S from net) ----------
                    self.model.smoother_gain_list = []
                    # Initialize backward pass
                    x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]  # [m]
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])
                    # First backward step
                    x_out_cv[:, SysModel.T_test - 2] = self.model(None, x_out_cv_forward[:, SysModel.T_test - 2],x_out_cv_forward[:, SysModel.T_test - 1], None)  # [m]
                    self.model.smoother_gain_list.append(self.model.SGain.clone())
                    # Remaining backward steps
                    for t in range(SysModel.T_test - 3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t + 1],x_out_cv[:, t + 2])  # [m]
                        self.model.smoother_gain_list.append(self.model.SGain.clone())

                    # s_0
                    P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                    s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.linalg.inv(P_1_0_pred)
                    self.model.smoother_gain_list.append(s_0.clone())

                    ###################em steps
                    # analytic RTS for P
                    P_smoothed_seq = torch.empty(m, m, SysModel.T_test)
                    P_smoothed_seq[:, :, -1] = P_filtered_seq[:, :, -1]
                    for t in range(SysModel.T_test - 2, -1, -1):
                        P_filt_t = P_filtered_seq[:, :, t]
                        P_pred_t_plus_1 = P_predicted_seq[:, :, t + 1]
                        P_smooth_t_plus_1 = P_smoothed_seq[:, :, t + 1]
                        reverse_time = SysModel.T_test - 1 - t
                        S_t = self.model.smoother_gain_list[reverse_time]
                        P_smooth_t = P_filt_t + S_t @ (P_smooth_t_plus_1 - P_pred_t_plus_1) @ S_t.T
                        P_smoothed_seq[:, :, t] = P_smooth_t

                    # V_t tensor
                    V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_filtered_seq,
                                                       self.model.smoother_gain_list)
                    # ---------- M-step ----------
                    X_s = x_out_cv.unsqueeze(0)  # [1, m, T]
                    P_smooth_s = P_smoothed_seq.unsqueeze(0)  # [1, m, m, T]
                    V_s = V.unsqueeze(0)  # [1, m, m, T]

                    F_est_cv = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, V_s,m)[0]
                    # write into copy
                    index = j // 10
                    F_valid_copy[index] = F_est_cv.detach()

                    # F-loss for logging
                    F_TRUE_cv = SysModel.F_valid_TRUE[index]
                    F_loss_batch_cv += torch.linalg.norm(F_est_cv - F_TRUE_cv, ord='fro')

                    # state MSE
                    MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j]).item()
                    #######################################################

                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                F_loss_batch_cv_av = F_loss_batch_cv/self.N_CV
                mse_cv_total = self.MSE_cv_linear_epoch[ti].item()*0.8 + 0.2*F_loss_batch_cv_av
                mse_cv_total_loss_db = 10 * torch.log10(mse_cv_total)

                # if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):this is whtiout F
                #     self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                #     self.MSE_cv_idx_opt = ti
                #
                #     torch.save(self.model, path_results)

                if (mse_cv_total_loss_db < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = mse_cv_total_loss_db
                    self.MSE_cv_idx_opt = ti

                    torch.save(self.model, path_results)



            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE rts Training :", self.MSE_train_dB_epoch[ti], "[dB]","MSE F LOSS Training :", F_loss_mean, "MSE F LOSS TOTAL :",loss_total_training_db,"[db]",
                  "MSE rts Validation :",self.MSE_cv_dB_epoch[ti],"[dB]","MSE total Validation :",mse_cv_total_loss_db,"[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")



        return F_train_copy, F_valid_copy







    def compute_cross_covariances(self, F, H, Ks, Ps, SGains):
        """
        Computes lag-one cross-covariances and returns them as a single tensor.

        Returns:
            V_tensor (torch.Tensor): A single tensor of shape [m, m, T]
            F [m,m]
            ks [m,n]
            Ps [m,m,T]

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
        V_tensor[:, :, T - 1] = V_T_minus_1

        # --- Backward recursion for t = T-2 down to 0 ---
        for t in range(T - 2, -1, -1):
            # Get values for time t
            Pt = Ps[:, :, t]
            # Smoother gain S_t has been stored in reverse order
            # For t=T-2, we need the first element of SGains (index 0)
            # For t=T-3, we need the second element (index 1), and so on.
            # print('size ,p,v,k',Ps.size, len(SGains),SGains[0],Ks.size)
            index = (T - 2) - t
            St = SGains[index]
            St_minus1 = SGains[index +1]
            # Get V_{t+1, t | T} from the tensor we are filling
            V_t_plus_1 = V_tensor[:, :, t + 1]

            # The cross-covariance update equation
            V_t = Pt @ St_minus1.T + St @ (V_t_plus_1 - F @ Pt) @ St_minus1.T

            # 3. Assign the result to the correct slice [:, :, t]
            V_tensor[:, :, t] = V_t

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
        print('ORICHECK DEHIL FFFFFFFFFFFFFFFF',SysModel.F_test[0] )
        for j in range(0, self.N_T):
            y_mdl_tst = test_input[j]
            SysModel.T_test = y_mdl_tst.size()[-1]
            x_out_test_forward_1 = torch.empty(SysModel.m, SysModel.T_test)
            x_out_test = torch.empty(SysModel.m, SysModel.T_test)

            self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
            self.model.init_hidden()

            if generate_f == False:  ####if we valid with different f
                SysModel.F = SysModel.F_test[j]
                self.model.update_F(SysModel.F)
            else:
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
            P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
            s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
            self.model.smoother_gain_list.append(s_0.clone().detach())  # [m, m]


            self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()

            # Compute P-smooth loss
            #option 1
            self.MSE_test_psmooth_arr[j] = self.PsmoothNN.compute_loss(P_smoothed_seq, test_target[j], x_out_test).item()
            #option 2
            #self.MSE_test_psmooth_arr[j] = self.compute_gaussian_loss1(P_smoothed_seq, test_target[j], x_out_test).item()


            x_out_list.append(x_out_test)
            P_smooth_list.append(P_smoothed_seq)

            #######compute V############
            V =  self.compute_cross_covariances(SysModel.F_test[j//10], SysModel.H, K, P_smoothed_seq, self.model.smoother_gain_list)
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
                self.model.smoother_gain_list.append(self.model.SGain.clone())
                for t in range(SysModel.T_test - 3, -1, -1):  #### T-3 all the way to 0 includes [T-3,0]
                    x_out_test[:, t] = self.model(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t + 1],
                                                  x_out_test[:, t + 2])
                    self.model.smoother_gain_list.append(self.model.SGain.clone())
                #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
                P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(
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
                V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_filtered_seq,
                                                   self.model.smoother_gain_list)
                V_list.append(V)  # [seq](tensor(m,m,T))

        # <<< Average the MSEs over all sequences and print the result >>>
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        print(f"Hybrid RTSNet - MSE Test: {self.MSE_test_dB_avg:.4f} [dB]")


        #
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
                    generate_f=True,beta=0.0):
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

        if load_rtsnet is not None:
            self.model = torch.load(load_rtsnet, weights_only=False)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,weight_decay=self.weightDecay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,weight_decay=self.weightDecay)
        if load_psmooth != None:
            self.PsmoothNN = torch.load(load_psmooth, weights_only=False)
            # Re-link the optimizer to the parameters of the newly loaded model
            self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,weight_decay=self.weightDecay)
        else:
            self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,
                                                        weight_decay=self.weightDecay)


        eps_f = SysModel.F_train[2]- SysModel.F_train_TRUE[2]
        eps = torch.linalg.norm(eps_f, ord='fro')
        print('initial diviation is' , eps,'the first',SysModel.F_train_TRUE[2], 'the second', SysModel.F_train[2] )

        F_train_copy = [F.clone() for F in SysModel.F_train]
        F_valid_copy = [F.clone() for F in SysModel.F_valid]

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
            F_loss_batch = 0


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
                    K_t = self.model.KGain.clone()### save the last one

                x_out_training[:, SysModel.T - 1] = x_out_training_forward[:, SysModel.T - 1]
                self.model.InitBackward(x_out_training[:, SysModel.T - 1])
                x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],
                                                               x_out_training_forward[:, SysModel.T - 1], None)
                smoother_gain_list.append(self.model.SGain.clone())
                for t in range(SysModel.T - 3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t], x_out_training_forward[:, t + 1],
                                                      x_out_training[:, t + 2])
                    smoother_gain_list.append(self.model.SGain.clone())
                #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
                P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
                smoother_gain_list.append(s_0.clone())  # [m, m]

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
                ###################compute the M step for F#########################
                p_tilde_tensor = torch.empty(SysModel.m,SysModel.m, SysModel.T)
                for i,p_1 in enumerate(sigma_list):
                    p_1 = p_1.view(4, 4).mean(dim=1)  # shape: (4,)
                    p_tilde_tensor[:,:,i] =   self.PsmoothNN.enforce_covariance_properties(p_1.view(SysModel.m, SysModel.m), eps=1e-6)  # tensor  (n×n×T)
                V = self.compute_cross_covariances(SysModel.F,SysModel.H, K_t,P_smoothed_seq, smoother_gain_list)
                # 2) run M‑step on **this sequence** (batch of size 1)
                X_s = x_out_training.unsqueeze(0)  # → [1, m, T]
                P_smooth_s = P_smoothed_seq.unsqueeze(0)  # → [1, m, m, T]
                V_s = V.unsqueeze(0)  # → [1, m, m, T]
                n_state = SysModel.F.shape[0]

                F_est = EMKF_F_Mstep(SysModel,X_s, P_smooth_s, V_s, n_state)
                F_est = F_est[0]
                index = n_e // 10
                F_TRUE = SysModel.F_train_TRUE[index]
                if j == 0:
                    print('true',F_TRUE, 'F_false', F_est)

                eps_f = F_est -F_TRUE

                eps = torch.linalg.norm(eps_f, ord='fro')
                F_loss_batch += eps
                # Calculate the two separate losses
                rtsnet_loss = self.loss_fn(x_out_training, train_target[n_e])
                #option_1
                psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq, train_target[n_e], x_out_training)
                #option 2
                #psmooth_loss = self.compute_gaussian_loss1(P_smoothed_seq, train_target[n_e], x_out_training)
                # Combine them into a total loss
                # beta_change = beta/(ti/5+1)
                beta_change =0.8
                total_loss = beta_change*rtsnet_loss + (1-beta_change)* psmooth_loss
                # Accumulate for logging
                Batch_RTS_LOSS_sum += rtsnet_loss
                Batch_Psmooth_LOSS_sum += psmooth_loss
                Batch_Total_LOSS_sum += total_loss

            # Average losses for the batch
            old_weights = [p.clone() for p in self.PsmoothNN.parameters()]
            Total_LOSS_mean = Batch_Total_LOSS_sum / self.N_B
            RTSNET_LOSS_mean = Batch_RTS_LOSS_sum / self.N_B
            Psmooth_LOSS_mean = Batch_Psmooth_LOSS_sum / self.N_B
            F_loss_mean = F_loss_batch/self.N_B
            Total_LOSS_mean = Total_LOSS_mean*0.6 +F_loss_mean*0.4
            print('F_loss is:', F_loss_mean)
            # Backward pass on the combined loss
            Total_LOSS_mean.backward(retain_graph=True)
            # Clip gradients and step both optimizers
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # bach_to_p = Psmooth_LOSS_mean
            # bach_to_p.backward()
            torch.nn.utils.clip_grad_norm_(self.PsmoothNN.parameters(), max_norm=1.0)
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
                F_loss_batch_cv = 0
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
                        K_t = self.model.KGain.clone()  ### save the last one

                    x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])
                    x_out_cv[:, SysModel.T_test - 2] = self.model(None, x_out_cv_forward[:, SysModel.T_test - 2],
                                                                  x_out_cv_forward[:, SysModel.T_test - 1], None)
                    smoother_gain_list_cv.append(self.model.SGain.clone())
                    for t in range(SysModel.T_test - 3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t + 1],
                                                    x_out_cv[:, t + 2])
                        smoother_gain_list_cv.append(self.model.SGain.clone())
                    #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
                    P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                    s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(
                        P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
                    smoother_gain_list_cv.append(s_0.clone())  # [m, m]

                    P_smoothed_seq_cv = torch.empty(SysModel.m, SysModel.m, SysModel.T_test)
                    dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)  # shape: [1, 1, m²] input to PsmoothNN
                    sigma_T_cv = sigma_list_cv[-1]
                    self.PsmoothNN.start = 0

                    P_flat_cv = self.PsmoothNN(sigma_T_cv, dummy_sgain)  # shape: [1, 1, m²] to [m²]
                    P_smoothed_seq_cv[:, :, SysModel.T_test - 1] = self.PsmoothNN.enforce_covariance_properties(P_flat_cv.view(-1).view(SysModel.m, SysModel.m))

                    for t in range(SysModel.T_test - 2, -1, -1):
                        sigma_t_cv = sigma_list_cv[t]
                        index = (SysModel.T_test - 2) - t
                        sgain_t_cv = smoother_gain_list_cv[index]
                        P_flat_cv = self.PsmoothNN(sigma_t_cv, sgain_t_cv)
                        P_smoothed_seq_cv[:, :, t] = self.PsmoothNN.enforce_covariance_properties(
                            P_flat_cv.view(-1).view(SysModel.m, SysModel.m))

                    ###################compute the M step for F#########################
                    p_tilde_tensor = torch.empty(SysModel.m, SysModel.m, SysModel.T)
                    for i, p_1 in enumerate(sigma_list_cv):
                        p_1 = p_1.view(4, 4).mean(dim=1)  # shape: (4,)
                        p_tilde_tensor[:, :, i] = self.PsmoothNN.enforce_covariance_properties(
                            p_1.view(SysModel.m, SysModel.m), eps=1e-6)  # tensor  (n×n×T)
                    V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_smoothed_seq_cv,
                                                       smoother_gain_list_cv)
                    # 2) run M‑step on **this sequence** (batch of size 1)
                    X_s = x_out_cv.unsqueeze(0)  # → [1, m, T]
                    P_smooth_s = P_smoothed_seq_cv.unsqueeze(0)  # → [1, m, m, T]
                    V_s = V.unsqueeze(0)  # → [1, m, m, T]
                    n_state = SysModel.F.shape[0]

                    F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, V_s, n_state)
                    F_est = F_est[0]
                    index = j // 10
                    F_TRUE = SysModel.F_valid_TRUE[index]
                    eps_f = F_est - F_TRUE

                    eps = torch.linalg.norm(eps_f, ord='fro')
                    F_loss_batch_cv += eps

                    # option_1
                    psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq_cv, cv_target[j], x_out_cv).item()
                    # option 2
                    #psmooth_loss = self.compute_gaussian_loss1(P_smoothed_seq_cv, cv_target[j], x_out_cv).item()


                    CV_RTS_LOSS_sum += self.loss_fn(x_out_cv, cv_target[j]).item()
                    CV_Psmooth_LOSS_sum += psmooth_loss

                    CV_Total_LOSS_sum += beta_change*self.loss_fn(x_out_cv, cv_target[j]).item() + (1 - beta_change)* psmooth_loss

                f_loss = F_loss_batch_cv/self.N_CV
                CV_Total_LOSS_sum =  0.6*CV_Total_LOSS_sum / self.N_CV + 0.4*f_loss
                self.MSE_cv_rts_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_RTS_LOSS_sum / self.N_CV))
                self.MSE_cv_psmooth_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_Psmooth_LOSS_sum / self.N_CV))
                self.MSE_cv_total_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_Total_LOSS_sum))
                print('cv f loss is:', f_loss)


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
        return



    def compute_gaussian_loss1(self,Sigma,x_target,x_est):
        err = x_target - x_est  # [m, T]
        m, T = err.shape
        eps = 1e-6
        eye_m = torch.eye(m, device=err.device, dtype=err.dtype) * eps

        total = 0.0
        const = 0.5 * m * torch.log(torch.tensor(2 * torch.pi, device=err.device))

        for t in range(T):
            Σ = Sigma[:, :, t] + eye_m  # jitter for PD
            δ = err[:, t].unsqueeze(1)  # [m,1]

            Σ_inv = torch.inverse(Σ)  # explicit inverse
            maha = (δ.transpose(0, 1) @ Σ_inv @ δ).squeeze()

            sign, logdet = torch.slogdet(Σ)  # numerically stable log-det
            if sign <= 0:
                # in case numerical issues lead to non-PD
                logdet = torch.log(torch.det(Σ) + eps)

            total += maha + logdet + 2 * const  # note we’ll divide by 2T below

        return total / (2 * T)

    def compute_gaussian_loss(self,P_seq,  # [m, m, T]  – predicted covariances
                              x_target_seq,  # [m, T]      – ground–truth states
                              x_est_seq):  # [m, T]      – RTSNet state output
        """
        Negative log-likelihood  (up to the additive constant ½·m·log(2π))
        ℓ_t = (x_t − μ_t)ᵀ P_t^{-1} (x_t − μ_t) + log |P_t|
        averaged over the T time steps.
        """
        m, T = x_target_seq.shape
        eps = 1e-5  # keeps P positive-definite numerically
        total = 0.0

        for t in range(T):
            P_t = P_seq[:, :, t] + torch.eye(m, device=P_seq.device) * eps
            δ = (x_target_seq[:, t] - x_est_seq[:, t]).unsqueeze(1)  # [m,1]

            # ---------- Mahalanobis term without explicit inverse ----------
            # L is lower-triangular s.t. P_t = L Lᵀ  (Cholesky factorisation)
            L = torch.linalg.cholesky(P_t)
            # Solve L Lᵀ α = δ   → α = P_t^{-1} δ
            α = torch.cholesky_solve(δ, L)  # same size as δ
            mahal = δ.T @ α  # scalar

            # ---------- log-det term (log |P_t|) -----------
            # log |P_t| = 2·Σ log diag(L)
            log_det = 2.0 * torch.sum(torch.log(torch.diag(L)))

            total += mahal + log_det

        return total / T/2  # average over the sequence

    # def Train_Joint(self, SysModel, cv_input, cv_target, train_input, train_target, path_results_rtsnet,
    #                 path_results_psmooth, load_rtsnet=None, load_psmooth=None,
    #                 generate_f=True, beta=0.0):
    #     self.N_E = len(train_input)
    #     self.N_CV = len(cv_input)
    #
    #     # Logging arrays
    #     self.MSE_train_rts_dB_epoch = torch.empty([self.N_steps])
    #     self.MSE_train_psmooth_dB_epoch = torch.empty([self.N_steps])
    #     self.MSE_cv_rts_dB_epoch = torch.empty([self.N_steps])
    #     self.MSE_cv_psmooth_dB_epoch = torch.empty([self.N_steps])
    #     self.MSE_train_total_dB_epoch = torch.empty([self.N_steps])
    #     self.MSE_cv_total_dB_epoch = torch.empty([self.N_steps])
    #
    #     self.MSE_cv_dB_opt = 1000
    #     self.MSE_cv_idx_opt = 0
    #
    #     if load_rtsnet is not None:
    #         self.model = torch.load(load_rtsnet, weights_only=False)
    #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
    #                                           weight_decay=self.weightDecay)
    #     else:
    #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
    #                                           weight_decay=self.weightDecay)
    #     if load_psmooth != None:
    #         self.PsmoothNN = torch.load(load_psmooth, weights_only=False)
    #         # Re-link the optimizer to the parameters of the newly loaded model
    #         self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,
    #                                                     weight_decay=self.weightDecay)
    #     else:
    #         self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,
    #                                                     weight_decay=self.weightDecay)
    #
    #     eps_f = SysModel.F_train[2] - SysModel.F_train_TRUE[2]
    #     eps = torch.linalg.norm(eps_f, ord='fro')
    #     print('initial diviation is', eps, 'the first', SysModel.F_train_TRUE[2], 'the second', SysModel.F_train[2])
    #
    #     F_train_copy = [F.clone() for F in SysModel.F_train]
    #     F_valid_copy = [F.clone() for F in SysModel.F_valid]
    #
    #     for ti in range(0, self.N_steps):
    #
    #         # Set both models to train mode
    #         self.model.train()
    #         self.PsmoothNN.train()
    #
    #         # Zero gradients for both optimizers
    #         self.optimizer.zero_grad()
    #         self.PsmoothNN_optimizer.zero_grad()
    #
    #         Batch_RTS_LOSS_sum = 0
    #         Batch_Psmooth_LOSS_sum = 0
    #         Batch_Total_LOSS_sum = 0
    #         F_loss_batch = []
    #
    #         for j in range(0, self.N_B):
    #             n_e = random.randint(0, self.N_E - 1)
    #             if generate_f:
    #                 index = n_e // 10
    #                 SysModel.F = SysModel.F_train[index]
    #                 self.model.update_F(SysModel.F)
    #
    #             y_training = train_input[n_e]
    #             SysModel.T = y_training.size()[-1]
    #
    #             # Run RTSNet forward and backward pass to get smoothed states and intermediate values
    #             x_out_training_forward = torch.empty(SysModel.m, SysModel.T)
    #             x_out_training = torch.empty(SysModel.m, SysModel.T)
    #
    #             self.model.init_hidden()
    #             self.model.InitSequence(SysModel.m1x_0, SysModel.T)
    #
    #             sigma_list = []
    #             smoother_gain_list = []
    #
    #             for t in range(SysModel.T):
    #                 x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
    #                 sigma_list.append(self.model.h_Sigma.clone())  # We need to keep the graph attached
    #                 K_t = self.model.KGain.clone()  ### save the last one
    #
    #             x_out_training[:, SysModel.T - 1] = x_out_training_forward[:, SysModel.T - 1]
    #             self.model.InitBackward(x_out_training[:, SysModel.T - 1])
    #             x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],
    #                                                            x_out_training_forward[:, SysModel.T - 1], None)
    #             smoother_gain_list.append(self.model.SGain.clone())
    #             for t in range(SysModel.T - 3, -1, -1):
    #                 x_out_training[:, t] = self.model(None, x_out_training_forward[:, t],
    #                                                   x_out_training_forward[:, t + 1],
    #                                                   x_out_training[:, t + 2])
    #                 smoother_gain_list.append(self.model.SGain.clone())
    #             #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
    #             P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
    #             s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(
    #                 P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
    #             smoother_gain_list.append(s_0.clone())  # [m, m]
    #
    #             # Run PsmoothNN using the stateless method
    #             P_smoothed_seq = torch.empty(SysModel.m, SysModel.m, SysModel.T)
    #             dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)
    #
    #             sigma_T = sigma_list[-1]
    #             # sigma_T_processed = self.PsmoothNN.FC8(sigma_T.view(1, -1)).view(1, 1, -1)
    #             # in_Psmooth_T = torch.cat((sigma_T_processed, dummy_sgain), dim=2)
    #             # h_current = in_Psmooth_T[:, :, :self.PsmoothNN.d_hidden_Psmooth].clone()
    #             self.PsmoothNN.start = 0
    #             P_flat = self.PsmoothNN(sigma_T, dummy_sgain).view(-1)  # shape: [1, 1, m²] to [m²]
    #             P_smoothed_seq[:, :, SysModel.T - 1] = self.PsmoothNN.enforce_covariance_properties(
    #                 P_flat.view(SysModel.m, SysModel.m))
    #
    #             for t in range(SysModel.T - 2, -1, -1):
    #                 sigma_t = sigma_list[t]
    #                 index = (SysModel.T - 2) - t
    #                 sgain_t = smoother_gain_list[index]
    #                 P_flat = self.PsmoothNN(sigma_t, sgain_t)  # [1, 1, m²] and [1, 1, d_hidden_Psmooth]
    #                 P_smoothed_seq[:, :, t] = self.PsmoothNN.enforce_covariance_properties(
    #                     P_flat.view(-1).view(SysModel.m, SysModel.m))
    #             ###################compute the M step for F#########################
    #             p_tilde_tensor = torch.empty(SysModel.m, SysModel.m, SysModel.T)
    #             for i, p_1 in enumerate(sigma_list):
    #                 p_1 = p_1.view(4, 4).mean(dim=1)  # shape: (4,)
    #                 p_tilde_tensor[:, :, i] = self.PsmoothNN.enforce_covariance_properties(
    #                     p_1.view(SysModel.m, SysModel.m), eps=1e-6)  # tensor  (n×n×T)
    #             V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, p_tilde_tensor, smoother_gain_list)
    #             # 2) run M‑step on **this sequence** (batch of size 1)
    #             X_s = x_out_training.unsqueeze(0)  # → [1, m, T]
    #             P_smooth_s = P_smoothed_seq.unsqueeze(0)  # → [1, m, m, T]
    #             V_s = V.unsqueeze(0)  # → [1, m, m, T]
    #             n_state = SysModel.F.shape[0]
    #
    #             F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, V_s, n_state)
    #             F_est = F_est[0]
    #             index = n_e // 10
    #             F_TRUE = SysModel.F_train_TRUE[index]
    #             eps_f = F_est - F_TRUE
    #
    #             eps = torch.linalg.norm(eps_f, ord='fro')
    #             F_loss_batch.append(eps)
    #             # Calculate the two separate losses
    #             rtsnet_loss = self.loss_fn(x_out_training, train_target[n_e])
    #             # option_1
    #             psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq, train_target[n_e], x_out_training)
    #             # option 2
    #             # psmooth_loss = self.compute_gaussian_loss1(P_smoothed_seq, train_target[n_e], x_out_training)
    #             # Combine them into a total loss
    #             # beta_change = beta/(ti/5+1)
    #             beta_change = 0.6
    #             total_loss = beta_change * rtsnet_loss + (1 - beta_change) * psmooth_loss
    #             # Accumulate for logging
    #             Batch_RTS_LOSS_sum += rtsnet_loss
    #             Batch_Psmooth_LOSS_sum += psmooth_loss
    #             Batch_Total_LOSS_sum += total_loss
    #
    #         # Average losses for the batch
    #         old_weights = [p.clone() for p in self.PsmoothNN.parameters()]
    #         Total_LOSS_mean = Batch_Total_LOSS_sum / self.N_B
    #         RTSNET_LOSS_mean = Batch_RTS_LOSS_sum / self.N_B
    #         Psmooth_LOSS_mean = Batch_Psmooth_LOSS_sum / self.N_B
    #         F_loss_mean = torch.stack(F_loss_batch).mean()
    #         Total_LOSS_mean = Total_LOSS_mean * 1 + F_loss_mean * 0
    #         print('F_loss is:', F_loss_mean)
    #         # Backward pass on the combined loss
    #         Total_LOSS_mean.backward(retain_graph=True)
    #         # Clip gradients and step both optimizers
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #         self.optimizer.step()
    #         # bach_to_p = Psmooth_LOSS_mean
    #         # bach_to_p.backward()
    #         torch.nn.utils.clip_grad_norm_(self.PsmoothNN.parameters(), max_norm=1.0)
    #         self.PsmoothNN_optimizer.step()
    #
    #         # Log training losses
    #         self.MSE_train_rts_dB_epoch[ti] = 10 * torch.log10(Batch_RTS_LOSS_sum / self.N_B)
    #         self.MSE_train_psmooth_dB_epoch[ti] = 10 * torch.log10(Batch_Psmooth_LOSS_sum / self.N_B)
    #         self.MSE_train_total_dB_epoch[ti] = 10 * torch.log10(Batch_Total_LOSS_sum / self.N_B)
    #         # Validation#####################################################
    #         self.model.eval()
    #         self.PsmoothNN.eval()
    #         with ((torch.no_grad())):
    #             CV_RTS_LOSS_sum = 0
    #             CV_Psmooth_LOSS_sum = 0
    #             CV_Total_LOSS_sum = 0
    #             F_loss_batch_cv = []
    #             for j in range(self.N_CV):
    #                 y_cv = cv_input[j]
    #                 SysModel.T_test = y_cv.size()[-1]
    #
    #                 if generate_f:
    #                     index = j // 10
    #                     SysModel.F = SysModel.F_valid[index]
    #                     self.model.update_F(SysModel.F)
    #
    #                 x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test)
    #                 x_out_cv = torch.empty(SysModel.m, SysModel.T_test)
    #                 self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
    #
    #                 sigma_list_cv, smoother_gain_list_cv = [], []
    #                 for t in range(SysModel.T_test):
    #                     x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
    #                     sigma_list_cv.append(self.model.h_Sigma)
    #                     K_t = self.model.KGain.clone()  ### save the last one
    #
    #                 x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]
    #                 self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])
    #                 x_out_cv[:, SysModel.T_test - 2] = self.model(None, x_out_cv_forward[:, SysModel.T_test - 2],
    #                                                               x_out_cv_forward[:, SysModel.T_test - 1], None)
    #                 smoother_gain_list_cv.append(self.model.SGain.clone())
    #                 for t in range(SysModel.T_test - 3, -1, -1):
    #                     x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t + 1],
    #                                                 x_out_cv[:, t + 2])
    #                     smoother_gain_list_cv.append(self.model.SGain.clone())
    #                 #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
    #                 P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
    #                 s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(
    #                     P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
    #                 smoother_gain_list_cv.append(s_0.clone())  # [m, m]
    #
    #                 P_smoothed_seq_cv = torch.empty(SysModel.m, SysModel.m, SysModel.T_test)
    #                 dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)  # shape: [1, 1, m²] input to PsmoothNN
    #                 sigma_T_cv = sigma_list_cv[-1]
    #                 self.PsmoothNN.start = 0
    #
    #                 P_flat_cv = self.PsmoothNN(sigma_T_cv, dummy_sgain)  # shape: [1, 1, m²] to [m²]
    #                 P_smoothed_seq_cv[:, :, SysModel.T_test - 1] = self.PsmoothNN.enforce_covariance_properties(
    #                     P_flat_cv.view(-1).view(SysModel.m, SysModel.m))
    #
    #                 for t in range(SysModel.T_test - 2, -1, -1):
    #                     sigma_t_cv = sigma_list_cv[t]
    #                     index = (SysModel.T_test - 2) - t
    #                     sgain_t_cv = smoother_gain_list_cv[index]
    #                     P_flat_cv = self.PsmoothNN(sigma_t_cv, sgain_t_cv)
    #                     P_smoothed_seq_cv[:, :, t] = self.PsmoothNN.enforce_covariance_properties(
    #                         P_flat_cv.view(-1).view(SysModel.m, SysModel.m))
    #
    #                 ###################compute the M step for F#########################
    #                 p_tilde_tensor = torch.empty(SysModel.m, SysModel.m, SysModel.T)
    #                 for i, p_1 in enumerate(sigma_list_cv):
    #                     p_1 = p_1.view(4, 4).mean(dim=1)  # shape: (4,)
    #                     p_tilde_tensor[:, :, i] = self.PsmoothNN.enforce_covariance_properties(
    #                         p_1.view(SysModel.m, SysModel.m), eps=1e-6)  # tensor  (n×n×T)
    #                 V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, p_tilde_tensor,
    #                                                    smoother_gain_list_cv)
    #                 # 2) run M‑step on **this sequence** (batch of size 1)
    #                 X_s = x_out_cv.unsqueeze(0)  # → [1, m, T]
    #                 P_smooth_s = P_smoothed_seq_cv.unsqueeze(0)  # → [1, m, m, T]
    #                 V_s = V.unsqueeze(0)  # → [1, m, m, T]
    #                 n_state = SysModel.F.shape[0]
    #
    #                 F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, V_s, n_state)
    #                 F_est = F_est[0]
    #                 index = j // 10
    #                 F_TRUE = SysModel.F_valid_TRUE[index]
    #                 eps_f = F_est - F_TRUE
    #
    #                 eps = torch.linalg.norm(eps_f, ord='fro')
    #                 F_loss_batch_cv.append(eps)
    #
    #                 # option_1
    #                 psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq_cv, cv_target[j], x_out_cv).item()
    #                 # option 2
    #                 # psmooth_loss = self.compute_gaussian_loss1(P_smoothed_seq_cv, cv_target[j], x_out_cv).item()
    #
    #                 CV_RTS_LOSS_sum += self.loss_fn(x_out_cv, cv_target[j]).item()
    #                 CV_Psmooth_LOSS_sum += psmooth_loss
    #
    #                 CV_Total_LOSS_sum += beta_change * self.loss_fn(x_out_cv, cv_target[j]).item() + (
    #                             1 - beta_change) * psmooth_loss
    #
    #             f_loss = torch.stack(F_loss_batch_cv).mean()
    #             CV_Total_LOSS_sum = 0.5 * CV_Total_LOSS_sum / self.N_CV + 0 * f_loss
    #             self.MSE_cv_rts_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_RTS_LOSS_sum / self.N_CV))
    #             self.MSE_cv_psmooth_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_Psmooth_LOSS_sum / self.N_CV))
    #             self.MSE_cv_total_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_Total_LOSS_sum))
    #             print('cv f loss is:', f_loss)
    #
    #             # Save best models based on the main RTSNet validation loss
    #             if self.MSE_cv_total_dB_epoch[ti] < self.MSE_cv_dB_opt:
    #                 self.MSE_cv_dB_opt = self.MSE_cv_total_dB_epoch[ti]
    #                 self.MSE_cv_rts_dB_opt = self.MSE_cv_rts_dB_epoch[ti]
    #                 self.MSE_cv_psmooth_dB_opt = self.MSE_cv_psmooth_dB_epoch[ti]
    #                 self.MSE_cv_idx_opt = ti
    #                 torch.save(self.model, path_results_rtsnet)
    #                 torch.save(self.PsmoothNN, path_results_psmooth)
    #                 print(f"**** Best Models Saved at Epoch {ti} with CV Loss {self.MSE_cv_dB_opt:.4f} dB ****")
    #
    #         # --- Comprehensive Logging ---
    #         # Get current epoch's values for printing
    #         train_rts_loss = self.MSE_train_rts_dB_epoch[ti]
    #         train_psmooth_loss = self.MSE_train_psmooth_dB_epoch[ti]
    #         train_total_loss = self.MSE_train_total_dB_epoch[ti]
    #
    #         cv_rts_loss = self.MSE_cv_rts_dB_epoch[ti]
    #         cv_psmooth_loss = self.MSE_cv_psmooth_dB_epoch[ti]
    #         cv_total_loss = self.MSE_cv_total_dB_epoch[ti]
    #
    #         # Use an f-string for clean, aligned printing
    #         print(f"Epoch {ti:03d}/{self.N_steps - 1} | "
    #               f"TRAIN: [RTS: {train_rts_loss:8.3f}, PSmooth: {train_psmooth_loss:8.3f}, Total: {train_total_loss:8.3f}] dB | "
    #               f"CV: [RTS: {cv_rts_loss:8.3f}, PSmooth: {cv_psmooth_loss:8.3f}, Total: {cv_total_loss:8.3f}] dB | "
    #               f"BEST Total: {self.MSE_cv_dB_opt:8.3f} dB (at epoch {self.MSE_cv_idx_opt})")
    #
    #     # After all epochs are done, return the logged histories for plotting
    #     return

    def _run_rtsnet_sequence(self, SysModel, y_seq, model_index):
        """
        Run RTSNet forward and backward pass for a single sequence.
        """
        T = y_seq.size()[-1]
        m = SysModel.m

        # Initialize
        x_out_forward = torch.empty(m, T)
        x_out_smoothed = torch.empty(m, T)

        self.rtsnet_models[model_index].init_hidden()
        self.rtsnet_models[model_index].InitSequence(SysModel.m1x_0, T)

        sigma_list = []
        smoother_gain_list = []

        # Forward pass
        for t in range(T):
            x_out_forward[:, t] = self.rtsnet_models[model_index](y_seq[:, t], None, None, None)
            sigma_list.append(self.rtsnet_models[model_index].h_Sigma.clone())
            if t == T-1:
                K_t = self.rtsnet_models[model_index].KGain.clone()
        # Backward pass
        x_out_smoothed[:, T - 1] = x_out_forward[:, T - 1]


        self.rtsnet_models[model_index].InitBackward(x_out_smoothed[:, T - 1])
        x_out_smoothed[:, T - 2] = self.rtsnet_models[model_index](None, x_out_forward[:, T - 2],
                                                                   x_out_forward[:, T - 1], None)
        smoother_gain_list.append(self.rtsnet_models[model_index].SGain.clone())

        for t in range(T - 3, -1, -1):
            x_out_smoothed[:, t] = self.rtsnet_models[model_index](None, x_out_forward[:, t],
                                                                   x_out_forward[:, t + 1],
                                                                   x_out_smoothed[:, t + 2])
            smoother_gain_list.append(self.rtsnet_models[model_index].SGain.clone())

        # Run PsmoothNet
        P_smoothed_seq = torch.empty(m, m, T)
        dummy_sgain = torch.zeros(1, 1, m * m)

        # Final time step
        sigma_T = sigma_list[-1]
        self.psmooth_models[model_index].start = 0
        P_flat = self.psmooth_models[model_index](sigma_T, dummy_sgain).view(-1)
        P_smoothed_seq[:, :, T - 1] = self.psmooth_models[model_index].enforce_covariance_properties(
            P_flat.view(m, m))

        # Backward in time
        for t in range(T - 2, -1, -1):
            sigma_t = sigma_list[t]
            sgain_index = (T - 2) - t
            sgain_t = smoother_gain_list[sgain_index].reshape(1, 1, -1)
            P_flat = self.psmooth_models[model_index](sigma_t, sgain_t)
            P_smoothed_seq[:, :, t] = self.psmooth_models[model_index].enforce_covariance_properties(
                P_flat.view(-1).view(m, m))

        # Compute S_0
        P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
        s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(P_1_0_pred.view(m, m))
        smoother_gain_list.append(s_0.clone())

        # Extract filtered covariances
        P_filtered_seq = torch.empty(m, m, T)
        # for i, sigma in enumerate(sigma_list):
        #     sigma_processed = sigma.view(4, 4).mean(dim=1)
        #     P_filtered_seq[:, :, i] = self.psmooth_models[model_index].enforce_covariance_properties(
        #         sigma_processed.view(SysModel.m, SysModel.m), eps=1e-6)

        return x_out_forward, x_out_smoothed, P_smoothed_seq, P_filtered_seq, smoother_gain_list, K_t

    def _run_sequential_emkf_epoch(self, SysModel, input_data, target_data, emkf_iterations, is_training):
        """
        FIXED VERSION - Run one epoch with BATCH → SEQUENCE → EM order and F propagation.
        """
        N_data = len(input_data)

        # FIX 1: Correct batch calculation
        if is_training:
            N_batches = self.N_B  # Training: 400//10 = 40 batches
        else:
            N_batches = self.N_B  # Validation: ceil(100/10) = 10 batches

        # Initialize F matrices
        if is_training:
            F_current = [f.clone().detach() for f in SysModel.F_train]  # FIX: Detach initial F
            F_true = SysModel.F_train_TRUE
        else:
            F_current = [f.clone().detach() for f in SysModel.F_valid]  # FIX: Detach initial F
            F_true = SysModel.F_valid_TRUE

        # Accumulate all iteration losses across all batches
        all_iter_losses = [[] for _ in range(emkf_iterations)]
        all_iter_f_losses = [[] for _ in range(emkf_iterations)]

        # BATCH LOOP (outermost)
        for batch_idx in range(N_batches):
            # Get batch indices
            if is_training:
                batch_indices = [random.randint(0, N_data - 1) for _ in range(self.N_B)]
                for i in range(emkf_iterations):
                    self.rtsnet_optimizers[i].zero_grad()
                    self.psmooth_optimizers[i].zero_grad()
            else:
                start_idx = batch_idx * self.N_B
                end_idx = min(start_idx + self.N_B, N_data)
                batch_indices = list(range(start_idx, end_idx))


            # Make copy of F for this batch (so F updates happen within batch)
            F_batch = [f.clone() for f in F_current]

            # Track batch losses for printing
            batch_iter_losses = [[] for _ in range(emkf_iterations)]
            batch_iter_f_losses = [[] for _ in range(emkf_iterations)]

            # SEQUENCE LOOP (middle) - Process each sequence in the batch
            for seq_idx in batch_indices:
                y_seq = input_data[seq_idx]
                target_seq = target_data[seq_idx]
                f_index = seq_idx // 10

                # Store losses for each EM iteration for this sequence
                seq_iter_losses = []
                seq_iter_f_losses = []

                # FIX 2: Start with stable F initialization
                F_seq = F_batch[f_index].clone().detach()  # Ensure no gradients from previous sequences

                # EM ITERATIONS LOOP (innermost) - Sequential F updates for this sequence
                for em_iter in range(emkf_iterations):
                    # Set model modes
                    if is_training:
                        self.rtsnet_models[em_iter].train()
                        self.psmooth_models[em_iter].train()

                    # # FIX 3: Ensure F has gradients for this iteration only
                    # if is_training:
                    #     F_seq = F_seq.detach().requires_grad_(True)
                    # else:
                    #     F_seq = F_seq.detach()

                    # Use current F (updated from previous EM iteration)
                    SysModel.F = F_seq
                    self.rtsnet_models[em_iter].update_F(SysModel.F)

                    # E-STEP: Run networks
                    x_out_forward, x_out_smoothed, P_smoothed_seq, P_filtered_seq, smoother_gain_list, K_t = \
                        self._run_rtsnet_sequence(SysModel, y_seq, em_iter)

                    # Compute losses
                    rts_loss = self.loss_fn(x_out_smoothed, target_seq)
                    psmooth_loss = self.psmooth_models[em_iter].compute_loss(P_smoothed_seq, target_seq,
                                                                             x_out_smoothed)
                    total_seq_loss = 0.9 * rts_loss + 0.1 * psmooth_loss
                    seq_iter_losses.append(total_seq_loss)

                    # Debug print for first sequence in first batch
                    if batch_idx == 0 and seq_idx == batch_indices[0]:
                        print(f'EM iter: {em_iter}, loss: {total_seq_loss:.4f}')
                        print(f'F_seq: {F_seq}')

                    # FIX 4: Stable M-STEP with regularization
                    # print(SysModel.F)
                    V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_smoothed_seq,smoother_gain_list)
                    X_s = x_out_smoothed.unsqueeze(0)
                    P_smooth_s = P_smoothed_seq.unsqueeze(0)
                    list_V_s = []
                    list_V_s.append(V)

                    F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, list_V_s, SysModel.m)[0]
                    F_seq = F_est
                    # F loss
                    F_true_seq = F_true[f_index]
                    f_loss = torch.linalg.norm(F_seq.detach() - F_true_seq, ord='fro')
                    seq_iter_f_losses.append(f_loss)

                # Add this sequence's losses to both global and batch accumulators
                for em_iter in range(emkf_iterations):
                    all_iter_losses[em_iter].append(seq_iter_losses[em_iter])
                    all_iter_f_losses[em_iter].append(seq_iter_f_losses[em_iter])
                    batch_iter_losses[em_iter].append(seq_iter_losses[em_iter])
                    batch_iter_f_losses[em_iter].append(seq_iter_f_losses[em_iter])

                # ADDED: Batch-level optimization
            if is_training:
                # Compute batch losses
                batch_losses = []
                for em_iter in range(emkf_iterations):
                    if batch_iter_losses[em_iter]:
                        batch_avg_loss = torch.stack(batch_iter_losses[em_iter]).mean()
                        batch_losses.append(batch_avg_loss)
                    else:
                        print('problemmmmmmmmmmmmmmmmmmmmmmmm')

                    # Compute weighted total batch loss
                iteration_weights = [0.2, 0.3, 0.5]
                total_batch_loss = sum(w * loss for w, loss in zip(iteration_weights, batch_losses))

                if torch.isfinite(total_batch_loss) and total_batch_loss < 1000.0:
                    total_batch_loss.backward()

                    for i in range(emkf_iterations):
                        torch.nn.utils.clip_grad_norm_(self.rtsnet_models[i].parameters(), max_norm=0.5)
                        torch.nn.utils.clip_grad_norm_(self.psmooth_models[i].parameters(), max_norm=0.5)
                        self.rtsnet_optimizers[i].step()
                        self.psmooth_optimizers[i].step()
                else:
                    print(
                        f"WARNING: Skipping backward pass for batch {batch_idx} due to invalid loss: {total_batch_loss}")

            # BATCH LOSS PRINTING (only every 5 batches to reduce spam)
            if batch_idx % 2 == 0 or batch_idx == N_batches - 1:
                mode_str = "TRAIN" if is_training else "VALID"
                print(f"  {mode_str} Batch {batch_idx + 1}/{N_batches}:")
                for em_iter in range(emkf_iterations):
                    if batch_iter_losses[em_iter]:  # Check if batch has losses
                        batch_avg_loss = torch.stack(batch_iter_losses[em_iter]).mean()
                        batch_avg_f_loss = torch.stack(batch_iter_f_losses[em_iter]).mean()
                        batch_loss_db = 10 * torch.log10(batch_avg_loss.detach())
                        print(f"    EM-Iter {em_iter}: Loss={batch_loss_db:.2f}dB, F-loss={batch_avg_f_loss:.4f}")
                    else:
                        print(f"    EM-Iter {em_iter}: No data processed")

            # Update global F with final F from this batch
            for f_idx, f_val in enumerate(F_batch):
                F_current[f_idx] = f_val.detach().clone()

        # Compute final averages across all sequences for each iteration
        final_iter_losses = []
        final_iter_f_losses = []

        for em_iter in range(emkf_iterations):
            if all_iter_losses[em_iter]:  # This check prevents torch.stack([]) error
                iter_avg_loss = torch.stack(all_iter_losses[em_iter]).mean()
                iter_avg_f_loss = torch.stack(all_iter_f_losses[em_iter]).mean()
            else:
                # Fallback values when no losses were collected
                iter_avg_loss = torch.tensor(1000.0)  # High loss indicates problem
                iter_avg_f_loss = torch.tensor(0.0)  # Zero F-loss as neutral value

            final_iter_losses.append(iter_avg_loss)
            final_iter_f_losses.append(iter_avg_f_loss)

        return final_iter_losses, final_iter_f_losses

    def Train_EndToEnd_EMKF(self, SysModel, cv_input, cv_target, train_input, train_target,
                                  rtsnet_model_paths, psmooth_model_paths, emkf_iterations=3,
                                  load_base_rtsnet=None, load_base_psmooth=None):
        """
        FIXED VERSION - Main training function for end-to-end EMKF training.
        """

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        # Initialize multiple models
        self.rtsnet_models = []
        self.psmooth_models = []

        for i in range(emkf_iterations):
            rtsnet_model = torch.load(load_base_rtsnet, weights_only=False)
            self.rtsnet_models.append(rtsnet_model)

            psmooth_model = torch.load(load_base_psmooth, weights_only=False)
            self.psmooth_models.append(psmooth_model)

        # Create separate optimizers with LOWER learning rate for stability
        self.rtsnet_optimizers = []
        self.psmooth_optimizers = []

        # FIX 7: Reduce learning rate for stability
        stable_lr = self.learningRate   # 10x smaller learning rate

        for i in range(emkf_iterations):
            self.rtsnet_optimizers.append(torch.optim.Adam(self.rtsnet_models[i].parameters(), lr=stable_lr,
                                                           weight_decay=self.weightDecay))
            self.psmooth_optimizers.append(torch.optim.Adam(self.psmooth_models[i].parameters(), lr=stable_lr,
                                                            weight_decay=self.weightDecay))

        # Logging arrays
        self.MSE_train_total_dB_epoch = torch.empty([self.N_steps])
        self.MSE_cv_total_dB_epoch = torch.empty([self.N_steps])
        self.F_loss_train_epoch = torch.empty([self.N_steps])
        self.F_loss_cv_epoch = torch.empty([self.N_steps])

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        print(f"Starting FIXED End-to-End EMKF Training with {emkf_iterations} EM iterations")
        print(f"Using reduced learning rate: {stable_lr}")

        # MAIN EPOCH LOOP
        for epoch in range(self.N_steps):

             # TRAINING
            train_losses, train_f_losses = self._run_sequential_emkf_epoch(SysModel, train_input, train_target,
                                                                           emkf_iterations, is_training=True)

            # FIX 8: Check for NaN/Inf losses
            valid_train_losses = []
            for loss in train_losses:
                if torch.isfinite(loss):
                    valid_train_losses.append(loss)
                else:
                    print(f"WARNING: Invalid loss detected: {loss}, skipping")
                    valid_train_losses.append(torch.tensor(1000.0))  # High penalty
            train_losses = valid_train_losses

            # Compute weighted total training loss
            iteration_weights = [0.1, 0.2, 0.7]
            total_train_loss = sum(w * loss for w, loss in zip(iteration_weights, train_losses))

            # # FIX 9: Gradient clipping BEFORE backprop
            # if torch.isfinite(total_train_loss) and total_train_loss < 1000.0:
            #     # BACKPROPAGATION
            #     total_train_loss.backward()
            #
            #     # OPTIMIZER STEP with more aggressive clipping
            #     for i in range(emkf_iterations):
            #         torch.nn.utils.clip_grad_norm_(self.rtsnet_models[i].parameters(), max_norm=0.5)  # Reduced from 1.0
            #         torch.nn.utils.clip_grad_norm_(self.psmooth_models[i].parameters(),
            #                                        max_norm=0.5)  # Reduced from 1.0
            #         self.rtsnet_optimizers[i].step()
            #         self.psmooth_optimizers[i].step()
            # else:
            #     print(f"WARNING: Skipping backward pass due to invalid loss: {total_train_loss}")

            # VALIDATION
            with torch.no_grad():
                for i in range(emkf_iterations):
                    self.rtsnet_models[i].eval()
                    self.psmooth_models[i].eval()

                cv_losses, cv_f_losses = self._run_sequential_emkf_epoch(SysModel, cv_input, cv_target, emkf_iterations,
                                                                         is_training=False)

            # Compute weighted averages
            total_cv_loss = sum(w * loss for w, loss in zip(iteration_weights, cv_losses))
            total_train_f_loss = sum(w * loss for w, loss in zip(iteration_weights, train_f_losses))
            total_cv_f_loss = sum(w * loss for w, loss in zip(iteration_weights, cv_f_losses))

            # LOGGING
            self.MSE_train_total_dB_epoch[epoch] = 10 * torch.log10(total_train_loss.detach())
            self.MSE_cv_total_dB_epoch[epoch] = 10 * torch.log10(total_cv_loss.detach())
            self.F_loss_train_epoch[epoch] = total_train_f_loss.detach()
            self.F_loss_cv_epoch[epoch] = total_cv_f_loss.detach()

            # SAVE BEST MODELS
            if self.MSE_cv_total_dB_epoch[epoch] < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = self.MSE_cv_total_dB_epoch[epoch]
                self.MSE_cv_idx_opt = epoch

                for i in range(emkf_iterations):
                    torch.save(self.rtsnet_models[i], rtsnet_model_paths[i])
                    torch.save(self.psmooth_models[i], psmooth_model_paths[i])

                print(f"**** Best Models Saved at Epoch {epoch} with CV Loss {self.MSE_cv_dB_opt:.4f} dB ****")

            # PROGRESS PRINTING (every epoch)
            print(f"Epoch {epoch:03d}/{self.N_steps - 1} | ")
            print(
                f"  TRAIN: [Total: {self.MSE_train_total_dB_epoch[epoch]:.3f}, F-loss: {self.F_loss_train_epoch[epoch]:.6f}]")
            print(
                f"  CV:    [Total: {self.MSE_cv_total_dB_epoch[epoch]:.3f}, F-loss: {self.F_loss_cv_epoch[epoch]:.6f}]")
            print(f"  BEST:  {self.MSE_cv_dB_opt:.3f} dB (epoch {self.MSE_cv_idx_opt})")

            # Log individual iteration losses
            for i, (train_loss, cv_loss) in enumerate(zip(train_losses, cv_losses)):
                print(
                    f"    Iter {i}: Train={10 * torch.log10(train_loss.detach()):.2f}dB, CV={10 * torch.log10(cv_loss.detach()):.2f}dB")

        return [self.MSE_train_total_dB_epoch, self.MSE_cv_total_dB_epoch, self.F_loss_train_epoch,
                self.F_loss_cv_epoch]

    def Test_Only_EMKF(self, SysModel, test_input, test_target,
                       load_base_rtsnet=None, load_base_psmooth=None, emkf_iterations=3):
        """
        Test-only version - No training, no optimization, just run EMKF on test data
        """

        # Initialize multiple models
        self.rtsnet_models = []
        self.psmooth_models = []

        for i in range(emkf_iterations):
            rtsnet_model = torch.load(load_base_rtsnet, weights_only=False)
            self.rtsnet_models.append(rtsnet_model)

            psmooth_model = torch.load(load_base_psmooth, weights_only=False)
            self.psmooth_models.append(psmooth_model)

        print(f"Starting Test-Only EMKF with {emkf_iterations} EM iterations")

        # Run test only
        test_losses, test_f_losses = self._run_test_simple(SysModel, test_input, test_target, emkf_iterations)

        # # Compute weighted total test loss
        # iteration_weights = [0.1, 0.2, 0.7]
        # total_test_loss = sum(w * loss for w, loss in zip(iteration_weights, test_losses))
        # total_test_f_loss = sum(w * loss for w, loss in zip(iteration_weights, test_f_losses))

        # # LOGGING
        # test_loss_db = 10 * torch.log10(total_test_loss.detach())
        #
        # print(f"TEST RESULTS:")
        # print(f"  Total Loss: {test_loss_db:.3f} dB")
        # print(f"  F-loss: {total_test_f_loss:.6f}")


        # Log individual iteration losses
        for i, test_loss in enumerate(test_losses):
            print(f"    Iter {i}: {10 * torch.log10(test_loss.detach()):.2f}dB")

        return test_losses, test_f_losses

    def _run_test_simple(self, SysModel, input_data, target_data, emkf_iterations):
        """
        Simple test - just loop through each sequence one by one
        """
        N_data = len(input_data)

        # Initialize F matrices
        F_current = [f.clone().detach() for f in SysModel.F_test]
        F_true = SysModel.F_test_TRUE

        # Accumulate losses
        all_iter_losses = [[] for _ in range(emkf_iterations)]
        all_iter_f_losses = [[] for _ in range(emkf_iterations)]

        # SIMPLE LOOP - one sequence at a time
        for seq_idx in range(N_data):
            y_seq = input_data[seq_idx]
            target_seq = target_data[seq_idx]
            f_index = seq_idx // 10

            # Store losses for each EM iteration for this sequence
            seq_iter_losses = []
            seq_iter_f_losses = []

            # Start with F
            F_seq = F_current[f_index].clone().detach()

            # EM ITERATIONS for this sequence
            for em_iter in range(emkf_iterations):
                # Set model modes - EVAL ONLY
                self.rtsnet_models[em_iter].eval()
                self.psmooth_models[em_iter].eval()

                # Use current F
                SysModel.F = F_seq
                self.rtsnet_models[em_iter].update_F(SysModel.F)

                # E-STEP: Run networks
                with torch.no_grad():
                    x_out_forward, x_out_smoothed, P_smoothed_seq, P_filtered_seq, smoother_gain_list, K_t = \
                        self._run_rtsnet_sequence(SysModel, y_seq, em_iter)

                    # Compute losses
                    rts_loss = self.loss_fn(x_out_smoothed, target_seq)
                    psmooth_loss = self.psmooth_models[em_iter].compute_loss(P_smoothed_seq, target_seq, x_out_smoothed)
                    total_seq_loss = 1 * rts_loss + 0 * psmooth_loss
                    seq_iter_losses.append(total_seq_loss)

                    # Debug print for first sequence
                    if seq_idx == 0:
                        print(f'EM iter: {em_iter}, loss: {total_seq_loss:.4f}')
                        print(f'F_seq: {F_seq}')

                    # M-STEP
                    V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_smoothed_seq, smoother_gain_list)
                    X_s = x_out_smoothed.unsqueeze(0)
                    P_smooth_s = P_smoothed_seq.unsqueeze(0)
                    list_V_s = []
                    list_V_s.append(V)
                    F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, list_V_s, SysModel.m)[0]
                    F_seq = F_est

                    # F loss
                    F_true_seq = F_true[f_index]
                    f_loss = torch.linalg.norm(F_seq.detach() - F_true_seq, ord='fro')
                    seq_iter_f_losses.append(f_loss)

            # Add this sequence's losses to accumulators
            for em_iter in range(emkf_iterations):
                all_iter_losses[em_iter].append(seq_iter_losses[em_iter])
                all_iter_f_losses[em_iter].append(seq_iter_f_losses[em_iter])

        # Compute final averages
        final_iter_losses = []
        final_iter_f_losses = []

        for em_iter in range(emkf_iterations):
            if all_iter_losses[em_iter]:
                iter_avg_loss = torch.stack(all_iter_losses[em_iter]).mean()
                iter_avg_f_loss = torch.stack(all_iter_f_losses[em_iter]).mean()
            else:
                print('wrongggggggggggggggggggggggggggggggggggggggg')

            final_iter_losses.append(iter_avg_loss)
            final_iter_f_losses.append(iter_avg_f_loss)

        return final_iter_losses, final_iter_f_losses

    def Test_Only_EMKF_2(self, SysModel, test_input, test_target,
                       load_base_rtsnet=None, load_base_psmooth=None, emkf_iterations=3,generate_F = True):
        """
        Test-only version - Match the working EMKF exactly
        """

        # Initialize multiple models
        self.rtsnet_models = []
        self.psmooth_models = []

        for i in range(emkf_iterations):
            rtsnet_model = torch.load(load_base_rtsnet, weights_only=False)
            self.rtsnet_models.append(rtsnet_model)

            psmooth_model = torch.load(load_base_psmooth, weights_only=False)
            self.psmooth_models.append(psmooth_model)

        print(f"Starting Test-Only EMKF with {emkf_iterations} EM iterations")

        F_matrices = []
        F_matrices.append(torch.stack(SysModel.F_test))

        for q in range(emkf_iterations):

            # E-STEP: Process ALL sequences using the SAME method as your working code
            if q > 0:
                [_, _, _, x_out_tensor, _, P_smooth_tensor, V_list, K_T_list, _, _] = self.NNTest_modified(
                    SysModel, test_input, test_target, self.rtsnet_models[q], self.psmooth_models[q], generate_f=generate_F)
            else:
                [_, _, _, x_out_tensor, _, P_smooth_tensor, V_list, K_T_list, _, _] = self.NNTest_modified(
                SysModel, test_input, test_target,self.rtsnet_models[q], self.psmooth_models[q],generate_f=True)

            # M-STEP: Use ALL data like your working code
            F_est = EMKF_F_Mstep(SysModel, x_out_tensor, P_smooth_tensor, V_list, SysModel.m)

            F_matrices.append(F_est)
            print('q_iter:', q, 'F_est:', F_est[0])

            # Update F for next iteration
            new_F_list = []
            for f_matrix in F_est:
                new_F_list.append(f_matrix)
            SysModel.F_test = new_F_list

        # Final test with last F
        [_, _, _, _, _, _, _, _, _, _] = self.NNTest_modified(SysModel, test_input, test_target,self.rtsnet_models[-1], self.psmooth_models[-1],generate_f=generate_F)

        return F_matrices

    def NNTest_modified(self, SysModel, test_input, test_target, rtsnet_model, psmooth_model, generate_f=True):
        """
        Modified NNTest to work with single models instead of self.model and self.PsmoothNN
        """
        print("Testing RTSNet...")
        N_T = len(test_input)

        MSE_test_linear_arr = torch.empty([N_T])
        MSE_test_psmooth_arr = torch.empty([N_T])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        rtsnet_model.eval()
        psmooth_model.eval()

        with torch.no_grad():
            x_out_list = []
            P_smooth_list = []
            V_list = []
            start = time.time()
            K_T_list = []

            print('ORICHECK DEHIL FFFFFFFFFFFFFFFF', SysModel.F_test[0])

            for j in range(0, N_T):
                y_mdl_tst = test_input[j]
                T_test = y_mdl_tst.size()[-1]
                x_out_test_forward_1 = torch.empty(SysModel.m, T_test)
                x_out_test = torch.empty(SysModel.m, T_test)

                rtsnet_model.InitSequence(SysModel.m1x_0, T_test)
                rtsnet_model.init_hidden()

                if generate_f == False:
                    SysModel.F = SysModel.F_test[j]
                    rtsnet_model.update_F(SysModel.F)
                else:
                    index = j // 10
                    SysModel.F = SysModel.F_test[index]
                    rtsnet_model.update_F(SysModel.F)

                # Forward pass and compute P-smooth
                sigma_list = []
                smoother_gain_list = []

                for t in range(0, T_test):
                    x_out_test_forward_1[:, t] = rtsnet_model(y_mdl_tst[:, t], None, None, None)
                    P_test_forward = rtsnet_model.h_Sigma.clone().detach()
                    sigma_list.append(P_test_forward)  # [1, 1, m²]
                    if t == T_test - 1:
                        K = rtsnet_model.KGain.clone().detach()
                        K_T_list.append(K)  # [m, n]

                x_out_test[:, T_test - 1] = x_out_test_forward_1[:, T_test - 1]
                rtsnet_model.InitBackward(x_out_test[:, T_test - 1])
                x_out_test[:, T_test - 2] = rtsnet_model(None, x_out_test_forward_1[:, T_test - 2],
                                                         x_out_test_forward_1[:, T_test - 1], None)
                smoother_gain_list.append(rtsnet_model.SGain.clone().detach())

                for t in range(T_test - 3, -1, -1):  # T-3 to 0
                    x_out_test[:, t] = rtsnet_model(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t + 1],
                                                    x_out_test[:, t + 2])
                    smoother_gain_list.append(rtsnet_model.SGain.clone().detach())

                # Compute P-smooth predictions
                P_smoothed_seq = torch.empty(SysModel.m, SysModel.m, T_test)
                dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)
                sigma_T = sigma_list[-1]
                psmooth_model.start = 0

                # Handle initial P-smooth at time T_test
                P_flat = psmooth_model(sigma_T, dummy_sgain).view(-1)
                P_matrix = psmooth_model.enforce_covariance_properties(P_flat.view(SysModel.m, SysModel.m))
                P_smoothed_seq[:, :, T_test - 1] = P_matrix

                for t in range(T_test - 2, -1, -1):
                    sigma_t = sigma_list[t].view(1, 1, -1)
                    index = (T_test - 2) - t
                    sgain_t = smoother_gain_list[index].reshape(1, 1, -1)
                    P_flat = psmooth_model(sigma_t, sgain_t)
                    P_matrix = psmooth_model.enforce_covariance_properties(P_flat.view(-1).view(SysModel.m, SysModel.m))
                    P_smoothed_seq[:, :, t] = P_matrix

                # Compute s(0)
                P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(P_1_0_pred.view(SysModel.m, SysModel.m))
                smoother_gain_list.append(s_0.clone().detach())

                MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()
                MSE_test_psmooth_arr[j] = psmooth_model.compute_loss(P_smoothed_seq, test_target[j], x_out_test).item()

                x_out_list.append(x_out_test)
                P_smooth_list.append(P_smoothed_seq)

                # Compute V
                V = self.compute_cross_covariances(SysModel.F_test[j // 10], SysModel.H, K, P_smoothed_seq,
                                                   smoother_gain_list)
                V_list.append(V)

            end = time.time()
            t = end - start

            # Average
            MSE_test_linear_avg = torch.mean(MSE_test_linear_arr)
            MSE_test_dB_avg = 10 * torch.log10(MSE_test_linear_avg)
            MSE_test_psmooth_avg = torch.mean(MSE_test_psmooth_arr)
            MSE_test_psmooth_dB_avg = 10 * torch.log10(MSE_test_psmooth_avg)

            # Standard deviation
            MSE_test_linear_std = torch.std(MSE_test_linear_arr, unbiased=True)
            MSE_test_psmooth_std = torch.std(MSE_test_psmooth_arr, unbiased=True)

            # Print results
            print("RTSNet-MSE Test:", MSE_test_dB_avg, "[dB]")
            print("RTSNet-STD Test:", 10 * torch.log10(MSE_test_linear_std + MSE_test_linear_avg) - MSE_test_dB_avg,
                  "[dB]")
            print("RTSNet-P-smooth MSE Test:", MSE_test_psmooth_dB_avg, "[dB]")
            print("RTSNet-P-smooth STD Test:", MSE_test_psmooth_std, "[dB]")
            print("Inference Time:", t)

            return [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, torch.stack(x_out_list), t,
                    torch.stack(P_smooth_list), V_list, K_T_list, MSE_test_psmooth_dB_avg, MSE_test_psmooth_std]