import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from layers import RBFLayer

# This print statement helps you confirm the new file is loaded
print("\n*** SUCCESSFULLY LOADED CORRECTED 2ND-ORDER MODEL ***\n")

class RBF_PINNs(tf.keras.layers.Layer):
    """
    Physics-Informed Neural Network (PINN) using Radial Basis Functions (RBF)
    to estimate the epsilon parameter of the stationary Swift-Hohenberg equation
    by solving a coupled system of two 2nd-order PDEs.
    """
    def __init__(
        self,
        units,       # Number of RBF nodes
        x,           # 1D array of x-coordinates
        y,           # 1D array of y-coordinates
        min_lr,      # Minimum learning rate for cyclical LR scheduler
        max_lr,      # Maximum learning rate for cyclical LR scheduler
        u,           # 2D numpy array of the input pattern data (un-normalized)
        dx,          # Grid spacing
        tol,         # Loss tolerance to stop training
        threshold_ep,# Iteration count to switch from data-fitting to physics-fitting
        batchsize,   # Batch size for training
        sigma2=2,    # Parameter for the RBF layer (related to width of basis functions)
    ):
        super().__init__()
        # Store basic parameters
        self.lr = max_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.tol = tol
        self.units = units
        self.dx = dx
        self.batchsize = batchsize
        self.n = len(x) # Grid size (n x n)
        self.epochs = 0 # Training epoch counter
        
        # --- Physical Parameters ---
        # Epsilon: The parameter we want to learn. It's trainable. Initial guess is 0.5.
        self.epsilon = tf.Variable([0.5], dtype=tf.float32, trainable=True)
        # Delta and Gamma: Fixed parameters based on the problem setup. Not trainable.
        self.delta = tf.Variable([0.406], dtype=tf.float32, trainable=False)
        self.gamma = tf.Variable([0.196], dtype=tf.float32, trainable=False)
        
        # --- Auxiliary Trainable Variables (Optional - Not used in core PDE) ---
        # These seem related to potential future normalization/scaling attempts, kept for compatibility.
        self.u_mean_var = tf.Variable([0.9068], dtype=tf.float32, trainable=True)
        self.u_scale_var = tf.Variable([2.5], dtype=tf.float32, trainable=True)
        
        # --- Data Preparation ---
        # Convert the input pattern `u` (numpy array) into TensorFlow Tensors.
        # This is done *once* during initialization for efficiency.
        # The data is flattened into a single column vector (shape [n*n, 1]).
        self.u = tf.constant(u.flatten()[:, None], dtype=tf.float32)
        # `u_in` seems redundant here if it's identical to `u`, but kept for compatibility.
        self.u_in = tf.constant(u.flatten()[:, None], dtype=tf.float32)

        # --- Coordinate Grid Preparation ---
        self.max_val = np.max(x) # Max coordinate value, used by RBFLayer
        # Threshold: Number of iterations for the initial data-fitting phase (_get_losses1).
        self.threshold = threshold_ep
        
        # Create a meshgrid of physical coordinates (X, Y)
        X_np, Y_np = np.meshgrid(x, y)
        # Total number of points in the grid (n*n)
        self.tot_len = len(X_np.flatten())
        
        # Convert coordinate arrays into TensorFlow Tensors (flattened column vectors).
        # Done *once* for efficiency.
        self.X = tf.constant(X_np.flatten()[:, None], dtype=tf.float32)
        self.Y = tf.constant(Y_np.flatten()[:, None], dtype=tf.float32)
        # `X_in`, `Y_in` seem redundant, kept for compatibility.
        self.X_in = tf.constant(X_np.flatten()[:, None], dtype=tf.float32)
        self.Y_in = tf.constant(Y_np.flatten()[:, None], dtype=tf.float32)
        
        # Total number of points (redundant, using self.tot_len is sufficient)
        self.tot_len_in = self.tot_len
        
        # --- History Tracking ---
        # Lists to store values during training for later analysis/plotting.
        self.tot_loss = []
        self.epsilon_array = []
        self.delta_array = []   # Will be constant
        self.gamma_array = []   # Will be constant
        self.u_m_array = []     # History of u_mean_var
        self.u_s_array = []     # History of u_scale_var
        self.loss_array = []    # History of total loss
        self.loss_u_array = []    # History of data loss component
        self.loss_pde_u_array = [] # History of the *sum* of PDE loss components (L_pde1 + L_pde2)

        # --- Model and Optimizer Setup ---
        # Concatenate X and Y coordinates for model input (shape [n*n, 2]). Not strictly needed for training batches.
        self.Data = tf.concat([self.X, self.Y], 1)
        # Using the legacy Adam optimizer.
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.max_lr)
        
        # Define the Neural Network Architecture
        # This model takes (x, y) coordinates as input.
        # It uses an RBF layer followed by a Dense layer.
        # CRITICAL: It has 2 outputs: [u_pred, p_pred]
        # u_pred is the network's prediction for the pattern u.
        # p_pred is the network's prediction for the auxiliary variable p = nabla^2(u).
        self.model_up = tf.keras.Sequential(
            [
                RBFLayer(self.units, 1 / sigma2, self.max_val), # RBF layer
                tf.keras.layers.Dense(2, input_shape=(self.units,), use_bias=False), # Output layer with 2 neurons
            ]
        )
        # Mean Squared Error loss function instance.
        self.mse = tf.keras.losses.MeanSquaredError()
        # Iteration counter.
        self.iterations = 0

        # --- Dynamic Weights for Loss Terms ---
        # These weights help balance the contribution of different loss terms during training.
        # They are updated periodically based on the relative magnitudes of the losses.
        self.weight_pde_u = tf.Variable(1.0, dtype=tf.float32, trainable=False) # Weight for PDE loss 1 (p - nabla^2(u) = 0)
        self.weight_pde_u_1 = tf.Variable(1.0, dtype=tf.float32, trainable=False)# Weight for PDE loss 2 (the reformulated SH equation)

    
    @tf.function # Compiles this function into a static graph for speed.
    def _get_losses1(self, x, y, u):
        """
        Phase 1: Data-Fitting Only (Pre-training)
        - Runs for the first `self.threshold` iterations.
        - Trains the network (`self.model_up`) to match the input data `u`.
        - ONLY the data loss is calculated and used for gradients.
        - The physics (PDE residuals) and `epsilon` are IGNORED here.
        - Goal: Get a good initial guess for the pattern `u` before introducing physics.
        """
        # Start recording operations for gradient calculation.
        with tf.GradientTape(persistent=False) as tape1:
            # Get model output [u_pred, p_pred] for the input batch coordinates (x, y).
            model_output = self.model_up(tf.concat([x, y], 1), training=True)
            # We only care about u_pred in this phase. The second output (p_pred) is ignored.
            u_pred, _ = tf.split(model_output, 2, axis=1)
            
            # Calculate the Mean Squared Error between the network's prediction `u_pred`
            # and the true data `u` for this batch. Normalize by the mean square of the data.
            loss_u = self.mse(u_pred, u) / tf.reduce_mean(tf.square(u))
            
            # Identify the trainable variables of the model (weights and biases).
            trainables_u = self.model_up.trainable_variables
            # Calculate the gradients of the data loss (`loss_u`) with respect to the model's variables.
            grads_u = tape1.gradient(loss_u, trainables_u)
        # `tape1` is automatically released here.

        # Apply the calculated gradients to update the model's weights using the optimizer.
        self.optimizer.apply_gradients(zip(grads_u, trainables_u))
        # The total loss for this phase is just the data loss.
        loss = loss_u
        return loss
    
    # --- CORRECTED _get_losses2 FUNCTION ---
    @tf.function # Compile into a static graph.
    def _get_losses2(self, x, y, u):
        """
        Phase 2: Physics-Informed Training
        - Runs after `self.threshold` iterations.
        - Calculates both data loss and PDE residual losses.
        - Trains the network (`self.model_up`) AND the parameter `self.epsilon` simultaneously.
        - Solves the coupled 2nd-order system:
            1. PDE 1: p - nabla^2(u) = 0
            2. PDE 2: (epsilon*u - delta*u^2 - gamma*u^3) - (u + 2*p + nabla^2(p)) = 0
        """
        
        # Use a single persistent GradientTape. It needs to be persistent because we will
        # call tape.gradient multiple times to compute 2nd order derivatives.
        with tf.GradientTape(persistent=True) as tape:
            # Explicitly tell the tape to "watch" the input tensors `x` and `y`.
            # This is necessary because they are inputs to the function, not tf.Variables,
            # but we need to compute gradients with respect to them (spatial derivatives).
            tape.watch(x)
            tape.watch(y)
            
            # --- 0th Order: Model Prediction ---
            # Get the network's output [u_pred, p_pred] for the current batch.
            model_output = self.model_up(tf.concat([x, y], 1), training=True)
            u_pred, p_pred = tf.split(model_output, 2, axis=1)
            
            # --- 1st Order Spatial Derivatives ---
            # Calculate du/dx, du/dy, dp/dx, dp/dy using the tape.
            u_x = tape.gradient(u_pred, x)
            u_y = tape.gradient(u_pred, y)
            p_x = tape.gradient(p_pred, x)
            p_y = tape.gradient(p_pred, y)

            # Handle potential `None` gradients (can happen if inputs `x` or `y` are constant
            # within a batch, though unlikely with spatial data). Replace None with zeros.
            if u_x is None: u_x = tf.zeros_like(x)
            if u_y is None: u_y = tf.zeros_like(y)
            if p_x is None: p_x = tf.zeros_like(x)
            if p_y is None: p_y = tf.zeros_like(y)
            
            # --- 2nd Order Spatial Derivatives ---
            # Calculate d^2u/dx^2, d^2u/dy^2, d^2p/dx^2, d^2p/dy^2.
            # This works because the tape is persistent and recorded the 1st derivative ops.
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)
            p_xx = tape.gradient(p_x, x)
            p_yy = tape.gradient(p_y, y)

            # Handle potential `None` gradients for 2nd derivatives.
            if u_xx is None: u_xx = tf.zeros_like(x)
            if u_yy is None: u_yy = tf.zeros_like(y)
            if p_xx is None: p_xx = tf.zeros_like(x)
            if p_yy is None: p_yy = tf.zeros_like(y)
            
            # --- Calculate Laplacians ---
            # nabla^2(u) = d^2u/dx^2 + d^2u/dy^2
            Laplace_u = u_xx + u_yy
            # nabla^2(p) = d^2p/dx^2 + d^2p/dy^2
            Laplace_p = p_xx + p_yy

            # --- PDE Residual Calculation ---
            # These residuals represent how much the network's predictions currently
            # violate the physics equations. We want to drive these to zero.
            
            # Residual for PDE 1: p - nabla^2(u) = 0
            pde_residual_1 = p_pred - Laplace_u

            # Residual for PDE 2: (epsilon*u - delta*u^2 - ...) - (u + 2*p + nabla^2(p)) = 0
            # First, calculate the substituted operator term: u + 2*p + nabla^2(p)
            sh_operator_new = u_pred + 2 * p_pred + Laplace_p
            
            # Apply Curriculum Learning to PDE 2's nonlinear terms:
            # Gradually introduce the nonlinear terms based on the iteration number.
            # This can help stabilize training in the early stages of the physics phase.
            if self.iterations < 10000: # Example threshold, adjust as needed
                # Only linear term and operator
                pde_residual_2 = self.epsilon * u_pred - sh_operator_new
                curriculum_weight = 0.1 # Weight for the PDE loss term
            elif self.iterations < 20000: # Example threshold
                # Introduce quadratic term partially
                pde_residual_2 = (self.epsilon * u_pred -
                                  0.3 * self.delta * tf.square(u_pred) - sh_operator_new)
                curriculum_weight = 0.3
            elif self.iterations < 30000: # Example threshold
                # Introduce cubic term partially
                pde_residual_2 = (self.epsilon * u_pred -
                                  0.7 * self.delta * tf.square(u_pred) -
                                  0.3 * self.gamma * tf.pow(u_pred, 3) - sh_operator_new)
                curriculum_weight = 0.6
            else:
                # Include all terms fully
                pde_residual_2 = (self.epsilon * u_pred -
                                  self.delta * tf.square(u_pred) -
                                  self.gamma * tf.pow(u_pred, 3) - sh_operator_new)
                curriculum_weight = 1.0

            # --- Loss Calculation (Inside the Tape) ---
            # Data Loss: How well u_pred matches the data u.
            loss_u = self.mse(u_pred, u) / tf.reduce_mean(tf.square(u))
            # PDE Loss 1: Mean squared error of the residual for the first PDE.
            loss_pde_1 = tf.reduce_mean(tf.square(pde_residual_1))
            # PDE Loss 2: Mean squared error of the residual for the second PDE.
            loss_pde_2 = tf.reduce_mean(tf.square(pde_residual_2))
            
            # Epsilon Boundary Penalty: Penalize epsilon if it goes outside the desired range [0.2, 0.8].
            # This acts as a soft constraint.
            upper_penalty = tf.maximum(0.0, self.epsilon - 0.8)**2
            lower_penalty = tf.maximum(0.0, 0.2 - self.epsilon)**2
            # Sum the penalties. epsilon_bound_penalty will have shape [1].
            epsilon_bound_penalty = upper_penalty + lower_penalty
            
            # --- Total Loss ---
            # Weighted sum of all loss components.
            # Adjust the coefficients (0.1, 0.01) to tune the relative importance.
            # Dynamic weights (self.weight_pde_u, self.weight_pde_u_1) adapt during training.
            loss1 = (0.1 * loss_u +                             # Data loss contribution
                     self.weight_pde_u * loss_pde_1 +          # PDE1 loss contribution
                     curriculum_weight * self.weight_pde_u_1 * loss_pde_2 + # PDE2 loss contribution (with curriculum)
                     1.0 * epsilon_bound_penalty[0])           # Epsilon penalty (increased weight)
                                                               # Index [0] converts shape [1] tensor to scalar
            
            # Identify all variables that need gradients calculated for optimization.
            # This includes the neural network weights/biases AND the epsilon parameter.
            trainables1 = (self.model_up.trainable_variables + [self.epsilon])

        # --- Apply Gradients (Calculation happens *after* the 'with tape' block) ---
        # Calculate the gradients of the total loss (`loss1`) with respect to all trainable variables.
        grads1 = tape.gradient(loss1, trainables1)
        
        # --- Gradient Filtering and Application ---
        # Create pairs of (gradient, variable).
        # Filter out any pairs where the gradient is None (which shouldn't happen often with the checks above).
        filtered_grads_and_vars = [(g, v) for g, v in zip(grads1, trainables1) if g is not None]
        
        # Check if any valid gradients were computed.
        if not filtered_grads_and_vars:
             tf.print("WARNING: All gradients are None in _get_losses2. Model is not training physics.")
        else:
            # Apply the valid gradients to update the corresponding variables via the optimizer.
            self.optimizer.apply_gradients(filtered_grads_and_vars)
        
        # We are done with the persistent tape, release its resources.
        del tape
        
        # --- Epsilon Constraints and Reset Logic (After gradient update) ---
        # If epsilon gets too close to the boundaries (0.1 or 0.9), reset it towards the middle.
        # This helps prevent it from getting stuck at the edges.
        if self.iterations % 500 == 0:
            epsilon_near_lower = (self.epsilon <= 0.1)
            epsilon_near_upper = (self.epsilon >= 0.9)
            
            if epsilon_near_lower or epsilon_near_upper:
                # Calculate a new value with some random noise around 0.3 or 0.5
                if epsilon_near_lower:
                    new_val = 0.3 + 0.1 * tf.random.normal([1])
                else: # Near upper boundary
                    new_val = 0.5 + 0.1 * tf.random.normal([1])
                # Assign the new value, ensuring it stays within the hard [0.1, 0.9] limits.
                self.epsilon.assign([tf.maximum(0.1, tf.minimum(0.9, new_val[0]))])
        
        # Enforce hard constraints: Ensure epsilon always stays within [0.1, 0.9].
        # This clips the value if the optimizer tries to push it outside this range.
        self.epsilon.assign(tf.minimum(tf.maximum(self.epsilon, 0.1), 0.9))
        
        # Return the individual loss components for logging/analysis.
        return loss1, loss_u, loss_pde_1, loss_pde_2

    # --- CORRECTED update_dynamic_weights FUNCTION ---
    @tf.function # Compile into a static graph.
    def update_dynamic_weights(self):
        """
        Update the dynamic loss weights (`self.weight_pde_u`, `self.weight_pde_u_1`).
        - Runs periodically during the physics-fitting phase.
        - Calculates the current data loss and PDE losses on the *entire* dataset.
        - Sets the weights based on the relative magnitudes of these losses.
        - Goal: Automatically balance the loss terms so they contribute appropriately.
        """
        # Use the full dataset coordinates (X_in, Y_in) and data (u_in).
        X1 = self.X_in
        Y1 = self.Y_in
        u1 = self.u_in

        # Use a persistent tape to calculate necessary 2nd derivatives, similar to _get_losses2.
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X1) # Watch coordinates as they are not tf.Variables
            tape.watch(Y1)
            
            # --- 0th Order ---
            model_output = self.model_up(tf.concat([X1, Y1], 1)) # Use current model state
            u_pred, p_pred = tf.split(model_output, 2, axis=1)
            
            # --- 1st Order ---
            u_x = tape.gradient(u_pred, X1)
            u_y = tape.gradient(u_pred, Y1)
            p_x = tape.gradient(p_pred, X1)
            p_y = tape.gradient(p_pred, Y1)
            # Handle None gradients
            if u_x is None: u_x = tf.zeros_like(X1); tf.print("Warning: u_x is None in weight update")
            if u_y is None: u_y = tf.zeros_like(Y1); tf.print("Warning: u_y is None in weight update")
            if p_x is None: p_x = tf.zeros_like(X1); tf.print("Warning: p_x is None in weight update")
            if p_y is None: p_y = tf.zeros_like(Y1); tf.print("Warning: p_y is None in weight update")

            # --- 2nd Order ---
            u_xx = tape.gradient(u_x, X1)
            u_yy = tape.gradient(u_y, Y1)
            p_xx = tape.gradient(p_x, X1)
            p_yy = tape.gradient(p_y, Y1)
            # Handle None gradients
            if u_xx is None: u_xx = tf.zeros_like(X1); tf.print("Warning: u_xx is None in weight update")
            if u_yy is None: u_yy = tf.zeros_like(Y1); tf.print("Warning: u_yy is None in weight update")
            if p_xx is None: p_xx = tf.zeros_like(X1); tf.print("Warning: p_xx is None in weight update")
            if p_yy is None: p_yy = tf.zeros_like(Y1); tf.print("Warning: p_yy is None in weight update")

            # Laplacians
            Laplace_u = u_xx + u_yy
            Laplace_p = p_xx + p_yy

            # --- Loss calculation (on full dataset) ---
            # Data loss
            loss_u = self.mse(u_pred, u1) # Not normalized here, using raw MSE
            
            # PDE 1 Loss
            pde_residual_1 = p_pred - Laplace_u
            loss_pde_1 = tf.reduce_mean(tf.square(pde_residual_1))
            
            # PDE 2 Loss (using the full, non-curriculum version of the equation)
            sh_operator_new = u_pred + 2 * p_pred + Laplace_p
            pde_residual_2 = (self.epsilon * u_pred -      # Current epsilon value
                              self.delta * tf.square(u_pred) -
                              self.gamma * tf.pow(u_pred, 3) - sh_operator_new)
            loss_pde_2 = tf.reduce_mean(tf.square(pde_residual_2))
            
        # Release the tape after calculating derivatives and losses.
        del tape
        
        # --- Assign Weights ---
        # Calculate weights based on the ratio of PDE loss to (PDE loss + Data loss).
        # Add a small epsilon to prevent division by zero.
        epsilon_div_zero = 1e-8
        # Weight for PDE1 loss term
        self.weight_pde_u.assign(loss_pde_1 / (loss_pde_1 + loss_u + epsilon_div_zero))
        # Weight for PDE2 loss term
        self.weight_pde_u_1.assign(loss_pde_2 / (loss_pde_2 + loss_u + epsilon_div_zero))
        # Optional: Print updated weights for debugging
        # tf.print("Updated dynamic weights: w1=", self.weight_pde_u, " w2=", self.weight_pde_u_1)


    def train(self, max_iterations=1e8, step_size=None):
        """
        Main training loop. Handles both the data-fitting phase and the
        physics-informed phase.
        """
        # Initialize the Cyclical Learning Rate scheduler callback.
        from clr_callback import CyclicLR
        clr_cb = CyclicLR(
            model_optimizer=self, # Pass the instance itself, needs access to self.optimizer.lr
            base_lr=self.min_lr,
            max_lr=self.max_lr,
            step_size=step_size, # Number of iterations per half cycle
        )
        self.callbacks = clr_cb # Store the callback
        
        # Load previous model weights if resuming training (iterations > 1).
        if self.iterations > 1:
            try:
                # Ensure weights_model_up exists before trying to set them
                self.model_up.set_weights(self.weights_model_up)
                tf.print("Resumed training from previous state.")
            except AttributeError:
                tf.print("Warning: Could not load previous weights. Starting fresh.")

        # Flag used for updating dynamic weights only once per frequency check.
        first_pass = True
        # Initialize the LR scheduler.
        self.callbacks.on_train_begin()
        
        # Main training loop continues until max_iterations or error condition.
        while self.iterations > -1: # Loop indefinitely until break condition
            # -------------------------------------
            # PHASE 1: DATA-FITTING (Pre-training)
            # -------------------------------------
            if self.iterations < self.threshold:
                # Use the full dataset size.
                len_dat = self.tot_len
                # Create a TensorFlow dataset pipeline:
                # 1. `range(len_dat)`: Generate indices 0, 1, ..., N-1.
                # 2. `shuffle(len_dat)`: Shuffle the indices randomly each epoch.
                # 3. `batch(self.batchsize)`: Group indices into batches.
                datindexes = tf.data.Dataset.range(len_dat).shuffle(len_dat).batch(self.batchsize)
                
                # Iterate through the batches of indices for one epoch.
                for batch_indices in datindexes:
                    # Gather the corresponding coordinate and data points using the batch indices.
                    x_batch = tf.gather(self.X, batch_indices)
                    y_batch = tf.gather(self.Y, batch_indices)
                    u_batch = tf.gather(self.u, batch_indices)
                    
                    # Perform one training step using only the data loss.
                    loss = self._get_losses1(x_batch, y_batch, u_batch)
                    
                    # Store the current model weights (used for potential resuming).
                    self.weights_model_up = self.model_up.get_weights()
                    
                    # --- Logging (Data-fitting phase) ---
                    if self.iterations % 200 == 0: # Print status periodically
                        tf.print(
                            "It: %d, Epoch: %d, Data-Only loss: %e, Learning rate: %2e"
                            % (
                                self.iterations,
                                self.epochs,
                                loss,
                                self.optimizer.lr, # Access LR directly from optimizer
                            )
                        )
                    if self.iterations % 2000 == 0: # Plot prediction periodically
                        # Predict `u` on the full grid using the current model state.
                        model_output = self.model_up(tf.concat([self.X, self.Y], 1))
                        u_pred, _ = tf.split(model_output, 2, axis=1) # Get u_pred
                        # Reshape the flattened prediction back into a 2D grid.
                        u_pred_plot = np.reshape(u_pred.numpy(), (self.n, self.n)) # Use .numpy() to convert Tensor
                        
                        # Display the predicted pattern.
                        plt.imshow(u_pred_plot, cmap=plt.cm.RdBu)
                        plt.colorbar(label = 'Predicted Amplitude (u)')
                        plt.title(f'Prediction at Iteration {self.iterations}')
                        plt.show()
                    
                    # --- Loop Control (Data-fitting phase) ---
                    # Check if maximum iterations reached.
                    if self.iterations >= max_iterations: # Use >= for safety
                        self.iterations = -2 # Set flag to exit outer while loop
                        tf.print("Max iterations reached. Stopping training.")
                        break # Exit the inner for loop (batches)
                    
                    # Increment iteration counter *after* checks.
                    self.iterations += 1
                    
                    # Check if loss tolerance is met (optional early stopping).
                    # if loss < self.tol:
                    #     tf.print("Loss tolerance met. Stopping training early.")
                    #     self.iterations = -2 # Set flag to exit outer while loop
                    #     break # Exit the inner for loop

                    # Update the learning rate using the scheduler *after* each batch.
                    self.callbacks.on_batch_end(self.epochs) # Pass current epoch

                # If max_iterations was reached inside the batch loop, exit outer loop too.
                if self.iterations == -2:
                    break

            # ----------------------------------------
            # PHASE 2: PHYSICS-INFORMED TRAINING
            # ----------------------------------------
            else:
                # Use the full dataset size (same as data-fitting phase).
                len_dat = self.tot_len_in
                # Create the dataset pipeline for batches, similar to phase 1.
                datindexes = tf.data.Dataset.range(len_dat).shuffle(len_dat).batch(self.batchsize)

                # --- Update Dynamic Weights Periodically ---
                # Define how often to update the weights. Fewer updates might be more stable.
                if self.iterations < (self.threshold + 10000): # Example threshold
                    update_freq = 500
                else:
                    update_freq = 1000

                # Update weights only on the first pass into this phase or every `update_freq` iterations.
                if first_pass or self.iterations % update_freq == 0:
                    self.update_dynamic_weights() # Calls the @tf.function compiled version
                    first_pass = False # Reset flag after first update
                            
                # Iterate through batches for one epoch.
                for batch_indices in datindexes:
                    # Gather data for the current batch.
                    x_batch = tf.gather(self.X_in, batch_indices)
                    y_batch = tf.gather(self.Y_in, batch_indices)
                    u_batch = tf.gather(self.u_in, batch_indices)
                
                    # Perform one training step using the combined data and physics losses (_get_losses2).
                    # This updates both the model weights and the epsilon parameter.
                    loss, loss_u, loss_pde_1, loss_pde_2 = self._get_losses2(x_batch, y_batch, u_batch)
                    
                    # --- Record History (Physics phase) ---
                    if self.iterations % 50 == 0: # Record history less frequently
                        self.tot_loss.append(loss.numpy())
                        self.epsilon_array.append(self.epsilon.numpy()[0])
                        self.delta_array.append(self.delta.numpy()[0])
                        self.gamma_array.append(self.gamma.numpy()[0])
                        self.u_m_array.append(self.u_mean_var.numpy()[0])
                        self.u_s_array.append(self.u_scale_var.numpy()[0])
                        self.loss_array.append(loss.numpy())
                        self.loss_u_array.append(loss_u.numpy())
                        # Store the sum of the two PDE losses.
                        self.loss_pde_u_array.append(loss_pde_1.numpy() + loss_pde_2.numpy())
                        
                    # --- Loop Control & Logging (Physics phase) ---
                    # Update LR scheduler *after* each batch.
                    self.callbacks.on_batch_end(self.epochs)
                    
                    # Check for max iterations.
                    if self.iterations >= max_iterations: # Use >=
                        self.iterations = -2
                        tf.print("Max iterations reached. Stopping training.")
                        break
                        
                    # Store current weights.
                    self.weights_model_up = self.model_up.get_weights()
                    
                    # Print status periodically.
                    if self.iterations % 200 == 0:
                        tf.print(
                            # Updated format string for clarity
                            "It: %d, Loss: %e, L_Data: %e, L_PDE1: %e, L_PDE2: %e, LR: %2e, eps: %e, w1: %e, w2: %e"
                            % (
                                self.iterations,
                                loss,
                                loss_u,
                                loss_pde_1,
                                loss_pde_2,
                                self.optimizer.lr, # Get current LR
                                self.epsilon.numpy()[0], # Get scalar value of epsilon
                                self.weight_pde_u.numpy(), # Get scalar value of weight
                                self.weight_pde_u_1.numpy() # Get scalar value of weight
                            )
                        )
                    # Plot prediction periodically.
                    if self.iterations % 2000 == 0:
                        model_output = self.model_up(tf.concat([self.X, self.Y], 1))
                        u_pred, _ = tf.split(model_output, 2, axis=1)
                        u_pred_plot = np.reshape(u_pred.numpy(), (self.n, self.n))

                        plt.imshow(u_pred_plot, cmap=plt.cm.RdBu)
                        plt.colorbar(label='Predicted Amplitude (u)')
                        plt.title(f'Prediction at Iteration {self.iterations}')
                        plt.show()
                        
                    # Increment iteration counter *after* checks.
                    self.iterations += 1
                    
                    # Check loss tolerance (optional early stopping, maybe use combined loss or just data loss).
                    # if loss < self.tol:
                    #     tf.print("Loss tolerance met. Stopping training early.")
                    #     self.iterations = -2
                    #     break

                # If max_iterations was reached inside the batch loop, exit outer loop too.
                if self.iterations == -2:
                    break

            # Increment epoch counter after completing all batches for the current phase.
            self.epochs += 1
            
        # --- Training Finished ---
        tf.print("Training loop finished.")
        # Store final learned parameters and loss history.
        self.parameters = [
            self.epsilon_array,
            self.delta_array, # Fixed
            self.gamma_array, # Fixed
            self.u_m_array,   # Auxiliary
            self.u_s_array,   # Auxiliary
        ]
        # Store the very last recorded value of epsilon (might not be the "best" if oscillating).
        if self.epsilon_array: # Check if list is not empty
             self.final_parameters = [
                 self.epsilon_array[-1],
                 self.delta_array[-1],
                 self.gamma_array[-1],
                 self.u_m_array[-1] if self.u_m_array else self.u_mean_var.numpy()[0], # Handle empty lists
                 self.u_s_array[-1] if self.u_s_array else self.u_scale_var.numpy()[0], # Handle empty lists
             ]
        else: # Handle case where training stopped before any history was recorded
             self.final_parameters = [
                 self.epsilon.numpy()[0],
                 self.delta.numpy()[0],
                 self.gamma.numpy()[0],
                 self.u_mean_var.numpy()[0],
                 self.u_scale_var.numpy()[0],
             ]
             tf.print("Warning: Epsilon history is empty. Using current epsilon value as final.")