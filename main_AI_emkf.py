for i in range(num_models):
    # Create RTSNet
    rtsnet = RTSNetNN()
    rtsnet.NNBuild(sys_model_base, args)
    self.rtsnet_models.append(rtsnet)

    # Create PsmoothNN
    psmooth = PsmoothNN(sys_model_base.m, args)
    self.psmooth_models.append(psmooth)

    # Create Optimizers
    optimizer_rts = torch.optim.Adam(rtsnet.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer_psmooth = torch.optim.Adam(psmooth.parameters(), lr=args.lr, weight_decay=args.wd)
    self.rtsnet_optimizers.append(optimizer_rts)
    self.psmooth_optimizers.append(optimizer_psmooth)
self.loss_fn = nn.MSELoss(reduction='mean')
self.p_smooth_weight = args.p_smooth_weight if hasattr(args, 'p_smooth_weight') else 0.1
self.N_steps = args.n_steps
self.N_B = args.n_batch


def Train_Bank(self, F_list, sys_model_base, data_loader_func):
    """
    Trains the entire bank of RTSNet/PsmoothNN models simultaneously.
    """
    print(f"Starting joint training for a bank of {self.num_models} models...")

    for i in range(self.num_models):
        self.rtsnet_models[i].train()
        self.psmooth_models[i].train()

    for ti in range(self.N_steps):
        # For each training step, train each model on data specific to its F
        for i in range(self.num_models):

            # Get data generated with F_i
            F_i = F_list[i]
            sys_model_i = SystemModel(F_i, sys_model_base.Q, sys_model_base.H, sys_model_base.R, sys_model_base.T,
                                      sys_model_base.T_test)
            train_input, train_target, _, _ = data_loader_func(sys_model_i)  # Assumes a function that provides data

            # Zero gradients for this specific model pair
            self.rtsnet_optimizers[i].zero_grad()
            self.psmooth_optimizers[i].zero_grad()

            # --- This part is similar to your Train_Joint logic, but for model `i` ---
            # Get a random batch
            n_e = torch.randint(0, len(train_input), (self.N_B,))
            y_batch = train_input[n_e]
            target_batch = train_target[n_e]

            total_loss_i = 0
            for j in range(self.N_B):
                # Run the forward pass for model i
                x_smooth, sigma_list, sg_list = self.run_rtsnet_pass(self.rtsnet_models[i], y_batch[j], sys_model_i)
                P_smooth = self.run_psmooth_pass(self.psmooth_models[i], sigma_list, sg_list, sys_model_i)

                # Calculate loss
                rts_loss = self.loss_fn(x_smooth, target_batch[j])
                psmooth_loss = self.psmooth_models[i].compute_loss(P_smooth, target_batch[j], x_smooth)
                total_loss_i += rts_loss + self.p_smooth_weight * psmooth_loss

            # Backward pass for model i
            avg_loss_i = total_loss_i / self.N_B
            avg_loss_i.backward()

            # Step optimizers for model i
            self.rtsnet_optimizers[i].step()
            self.psmooth_optimizers[i].step()

        if ti % 20 == 0:
            print(f"Bank Training Epoch {ti}/{self.N_steps} Complete.")


def AI_EMKF_MM(num_models, initial_F_list, sys_model_true, observation_data, args):
    """
    Master function for the Multiple Model AI EMKF.
    """

    # 1. Initialize the Bank of Models
    pipeline_mm = Pipeline_MM(
        Time=strTime,
        folderName=path_results,
        modelName_prefix="mm_expert",
        num_models=num_models,
        sys_model_base=sys_model_true,  # Base model for dimensions
        args=args
    )

    F_k_list = initial_F_list

    num_em_iterations = 10
    for k in range(num_em_iterations):
        print(f"\n{'=' * 50}\nAI-EMKF MM Iteration {k + 1}/{num_em_iterations}\n{'=' * 50}")

        # --- E-STEP (Part 1): Train the entire bank of experts ---
        # Each expert RTSNet[i] learns to smooth based on F_k_list[i]
        pipeline_mm.Train_Bank(F_k_list, sys_model_true, data_loader_func=lambda sm: DataLoader(DataGen(...)))

        # --- E-STEP (Part 2): Evaluate all experts on real data ---
        # Now, run the REAL observation data through ALL trained experts
        # to see which one performs best.

        all_x_smooth = []
        all_P_smooth = []
        all_V_smooth = []
        model_likelihoods = []  # To store how well each model explains the data

        for i in range(num_models):
            # Set models to eval mode
            pipeline_mm.rtsnet_models[i].eval()
            pipeline_mm.psmooth_models[i].eval()

            # Get smoothed results from expert `i` on the real data
            # You would need a test/inference function in your Pipeline_MM
            # For now, let's conceptualize it:
            x_s, P_s, V_s = pipeline_mm.run_inference_on_real_data(i, observation_data)

            all_x_smooth.append(x_s)
            all_P_smooth.append(P_s)
            all_V_smooth.append(V_s)

            # Calculate the likelihood of the data given this model's smoothing
            # This is a simplified version; a true likelihood is more complex
            likelihood = -pipeline_mm.loss_fn(x_s, real_target_data)
            model_likelihoods.append(likelihood)

        # --- M-STEP: Update the F matrices ---

        # Here, you have choices. Let's implement one:
        # "Winner-take-all": Find the best model and only update its F.

        best_model_idx = torch.argmax(torch.tensor(model_likelihoods)).item()
        print(f"M-Step: Best performing model is Expert #{best_model_idx}.")

        # Get the smoothed results from only the BEST model
        best_x_s = all_x_smooth[best_model_idx]
        best_P_s = all_P_smooth[best_model_idx]
        best_V_s = all_V_smooth[best_model_idx]

        # Use its results to calculate its new F
        F_new = EMKF_F_solo(
            F_k_list[best_model_idx],  # Start from its previous F
            sys_model_true.H, sys_model_true.Q, sys_model_true.R,
            observation_data, m1_0, m2_0,
            best_x_s, best_P_s, best_V_s,
            ...  # other args for emkf_solo
        )

        print(f"Updating F for model #{best_model_idx}.")
        F_k_list[best_model_idx] = F_new  # Update only the winner

    return F_k_list