"""
This file contains the class Pipeline_KF, 
which is used to train and test KalmanNet in linear cases.
"""

import torch
import torch.nn as nn
import random
import time
from Plot import Plot


class Pipeline_KF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName 
        self.PipelineName = self.folderName + "pipeline_" + self.modelName 

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, args):
        self.N_steps = args.n_steps  # Number of Training Steps
        self.N_B = args.n_batch # Number of Samples in Batch
        self.learningRate = args.lr # Learning Rate
        self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)


    def NNTrain(self, n_Examples, train_input, train_target, n_CV, cv_input, cv_target, many_F = True):

        self.N_E = n_Examples
        self.N_CV = n_CV

        MSE_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps])
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps])

        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_steps])
        self.MSE_train_dB_epoch = torch.empty([self.N_steps])

        nan_streak = 0  # <-- consecutive bad batches


        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_steps):

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            for j in range(0, self.N_CV):

                if many_F:
                    index = j // 10
                    self.ssModel.F = self.ssModel.F_valid[index]
                    self.model.update_F(self.ssModel.F)

                y_cv = cv_input[j, :, :]
                self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T)

                x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T)
                for t in range(0, self.ssModel.T):
                    x_out_cv[:, t] = self.model(y_cv[:, t])

                # Compute Training Loss
                MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j, :, :]).item()

            # Average
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            valid_batches = 0  # only count good batches

            for j in range(0, self.N_B):
                n_e = random.randint(0, self.N_E - 1)

                if many_F:
                    index = n_e // 10
                    self.ssModel.F = self.ssModel.F_train[index]
                    self.model.update_F(self.ssModel.F)

                y_training = train_input[n_e, :, :]
                self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T)


                x_out_training = torch.empty(self.ssModel.m, self.ssModel.T)
                for t in range(0, self.ssModel.T):
                    x_out_training[:, t] = self.model(y_training[:, t])

                # Compute Training Loss
                LOSS = self.loss_fn(x_out_training, train_target[n_e, :, :])

                MSE_train_linear_batch[j] = LOSS.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
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
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

    def NNTest(self, n_Test, test_input, test_target,sysmodel = None, many_F = True):

        self.N_T = n_Test


        if sysmodel is not None:
            self.setssModel(sysmodel)

        self.MSE_test_linear_arr = torch.empty([self.N_T])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        self.model = torch.load(self.modelFileName, weights_only=False)

        self.model.eval()

        torch.no_grad()
        
        start = time.time()

        # --------- NEW: trackers for the worst sequence ----------ori
        worst_mse = -float("inf")  # start lower than any real MSE
        worst_idx = -1
        worst_F = None
        # ----------------------------------------------------------




        for j in range(0, self.N_T):

            if many_F:
                index = j // 10
                self.ssModel.F = self.ssModel.F_test[index]
                self.model.update_F(self.ssModel.F)

            y_mdl_tst = test_input[j, :, :]

            self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T)

            x_out_test = torch.empty(self.ssModel.m, self.ssModel.T)

            for t in range(0, self.ssModel.T):
                x_out_test[:, t] = self.model(y_mdl_tst[:, t])

            self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j, :, :]).item()

            # ------------ NEW: keep the worst ----------------ori
            if self.MSE_test_linear_arr[j] > worst_mse:
                worst_mse = self.MSE_test_linear_arr[j]
                worst_idx = j
                worst_F   = self.ssModel.F.clone()  # make a copy
            # -------------------------------------------------





        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg


        # Print MSE Cross Validation
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        # ------------ NEW: report the worst sequence ------------ori
        worst_mse_dB = 10 * torch.log10(torch.tensor(worst_mse))
        print(f"Worst sequence index : {worst_idx}")
        print(f"Worst MSE            : {worst_mse:.6e}  ({worst_mse_dB:.3f} dB)")
        print("Associated F matrix  :")
        print(worst_F)
        # ---------------------------------------------------------



        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)