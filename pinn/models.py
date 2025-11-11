import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from layers import RBFLayer

print("\n*** SUCCESSFULLY LOADED RBF SS MODEL (V3 - PRINCIPLED OPTIMIZATION) ***\n")

class RBF_PINNs(tf.keras.layers.Layer):
    def __init__(
        self,
        units,          # Number of RBF nodes
        x,              # 1D array of x-coordinates
        y,              # 1D array of y-coordinates
        min_lr,         # Minimum learning rate
        max_lr,         # Maximum learning rate
        u,              # 2D numpy array of the input pattern data
        dx,             # Grid spacing
        tol,            # Loss tolerance
        threshold_ep,   # Iteration count for Phase 1
        batchsize,      # Batch size
        sigma2=2,       # RBF layer parameter
        target_data_pde_ratio=15.0,  # *** Stop annealing at this Data:PDE ratio ***
    ):
        super().__init__()
        
        # --- Parameters & Grid Setup ---
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.units = units
        self.dx = dx
        self.batchsize = batchsize
        self.n = len(x)
        self.epochs = 0
        self.iterations = 0

        # --- 3-PHASE TIMELINE ---
        self.threshold_p1 = threshold_ep          
        self.threshold_p2 = threshold_ep + 15000
        
        # --- PRINCIPLED ANNEALING PARAMETERS ---
        self.annealing_step = 4000
        self.pde2_growth_rate = 1.2  # More conservative
        self.data_decay_rate = 0.98   # More conservative
        
        # *** PRINCIPLED STOPPING CONDITION ***
        self.target_data_pde_ratio = target_data_pde_ratio
        self.annealing_active = True
        self.refinement_phase = False  # *** NEW: Track refinement phase ***
        
        # --- Physical Parameters ---
        self.epsilon = tf.Variable([0.5], dtype=tf.float32, trainable=True)
        self.delta = tf.Variable([0.406], dtype=tf.float32, trainable=False)
        self.gamma = tf.Variable([0.196], dtype=tf.float32, trainable=False)
        
        # --- Data & Coordinate Preparation ---
        self.u = tf.constant(u.flatten()[:, None], dtype=tf.float32)
        self.max_val = np.max(x)
        X_np, Y_np = np.meshgrid(x, y)
        self.tot_len = len(X_np.flatten())
        
        self.X = tf.constant(X_np.flatten()[:, None], dtype=tf.float32)
        self.Y = tf.constant(Y_np.flatten()[:, None], dtype=tf.float32)

        # --- Architecture: RBF + Dense ---
        self.model_up = tf.keras.Sequential(
            [
                RBFLayer(self.units, 1 / sigma2, self.max_val),
                tf.keras.layers.Dense(2, input_shape=(self.units,), use_bias=False),
            ]
        )
        
        # --- Separate Optimizers ---
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.max_lr)
        
        # *** CORRECTED LR SCHEDULING ***
        self.initial_eps_lr = 1e-5  # *** CHANGED FROM 1e-4 TO 1e-5 ***
        self.refinement_eps_lr = 1e-6  # *** NEW: Target LR for refinement ***
        self.eps_lr = tf.Variable(self.initial_eps_lr, trainable=False)
        self.optimizer_eps = tf.keras.optimizers.legacy.Adam(learning_rate=self.eps_lr)
        
        self.mse = tf.keras.losses.MeanSquaredError()

        # --- Loss Weights ---
        self.weight_data = tf.Variable(100.0, dtype=tf.float32, trainable=False)
        self.weight_pde1 = tf.Variable(1.0, dtype=tf.float32, trainable=False)
        self.weight_pde2 = tf.Variable(1.0, dtype=tf.float32, trainable=False)
        
        # --- History Tracking ---
        self._init_history()

        # --- Plotting attributes ---
        self.fig = None
        self.ax = None
        self.img = None

    def _init_history(self):
        """Initializes history lists."""
        self._epsilon_array = []
        self._loss_u_array = []
        self._loss_pde1_array = []
        self._loss_pde2_array = []
        self._loss_array = []
        self._delta_array = []
        self._gamma_array = []
        self._u_m_array = []
        self._u_s_array = []
        self._eps_lr_array = []

    # ----------------------------------------------------------------------
    # CORE METHODS
    # ----------------------------------------------------------------------
    
    @tf.function
    def _calculate_residuals(self, x, y):
        """
        Calculates u_pred, p_pred, and all PDE residuals.
        Assumes steady-state (u_t = 0) and q0 = 1.
        """
        with tf.GradientTape(persistent=True) as tape_outer:
            tape_outer.watch([x, y])
            
            with tf.GradientTape(persistent=True) as tape_inner:
                tape_inner.watch([x, y])
                
                model_output = self.model_up(tf.concat([x, y], 1), training=True)
                u_pred, p_pred = tf.split(model_output, 2, axis=1)
            
                u_x = tape_inner.gradient(u_pred, x); u_y = tape_inner.gradient(u_pred, y)
                p_x = tape_inner.gradient(p_pred, x); p_y = tape_inner.gradient(p_pred, y)

            u_xx = tape_outer.gradient(u_x, x); u_yy = tape_outer.gradient(u_y, y)
            p_xx = tape_outer.gradient(p_x, x); p_yy = tape_outer.gradient(p_y, y)
        
            u_xx = tf.zeros_like(x) if u_xx is None else u_xx
            u_y = tf.zeros_like(y) if u_y is None else u_y 
            u_yy = tf.zeros_like(y) if u_yy is None else u_yy
            p_xx = tf.zeros_like(x) if p_xx is None else p_xx
            p_yy = tf.zeros_like(y) if p_yy is None else p_yy
        
        del tape_inner
        del tape_outer
        
        Laplace_u = u_xx + u_yy
        Laplace_p = p_xx + p_yy
        
        pde_residual_1 = p_pred - Laplace_u

        nonlinear_terms = (self.epsilon * u_pred - 
                           self.delta * tf.square(u_pred) - 
                           self.gamma * tf.pow(u_pred, 3))
        
        sh_operator = u_pred + 2.0 * p_pred + Laplace_p
        
        full_sh_residual = nonlinear_terms - sh_operator
        
        return u_pred, p_pred, pde_residual_1, full_sh_residual

    @tf.function
    def train_step_p1(self, x, y, u):
        """PHASE 1: PURE DATA FIT (Network Weights Only)"""
        with tf.GradientTape() as tape:
            u_pred, _ = tf.split(self.model_up(tf.concat([x, y], 1), training=True), 2, axis=1)
            loss_u = self.mse(u_pred, u) / tf.reduce_mean(tf.square(u))
        
        grads = tape.gradient(loss_u, self.model_up.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model_up.trainable_variables))
        return loss_u

    @tf.function
    def train_step_p2(self, x, y, u):
        """PHASE 2: CONSTRAINT-FIT (Network Weights Only, Epsilon is FROZEN)"""
        with tf.GradientTape() as tape:
            u_pred, _, pde1_residual, _ = self._calculate_residuals(x, y) 
            
            loss_u = self.mse(u_pred, u) / tf.reduce_mean(tf.square(u))
            loss_pde1 = tf.reduce_mean(tf.square(pde1_residual))
            
            total_loss = (self.weight_data * loss_u + 
                          self.weight_pde1 * loss_pde1)
                    
        grads = tape.gradient(total_loss, self.model_up.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model_up.trainable_variables))
        return total_loss, loss_u, loss_pde1, tf.constant(0.0)

    @tf.function
    def train_step_p3(self, x, y, u): 
        """PHASE 3: INVERSE SOLVE (Network Weights + Epsilon)"""
        with tf.GradientTape(persistent=True) as tape:
            u_pred, _, pde1_residual, full_sh_residual = self._calculate_residuals(x, y)
            
            loss_u = self.mse(u_pred, u) / tf.reduce_mean(tf.square(u))
            loss_pde1 = tf.reduce_mean(tf.square(pde1_residual))
            loss_pde2 = tf.reduce_mean(tf.square(full_sh_residual))
            
            total_loss = (self.weight_data * loss_u + 
                          self.weight_pde1 * loss_pde1 + 
                          self.weight_pde2 * loss_pde2)
            
        # 1. Update Network
        net_grads = tape.gradient(total_loss, self.model_up.trainable_variables)
        self.optimizer.apply_gradients(zip(net_grads, self.model_up.trainable_variables))
        
        # 2. Update Epsilon
        eps_grads = tape.gradient(total_loss, [self.epsilon])
        self.optimizer_eps.apply_gradients(zip(eps_grads, [self.epsilon]))
        
        self.epsilon.assign(tf.minimum(tf.maximum(self.epsilon, 0.1), 0.9))
        
        del tape
        return total_loss, loss_u, loss_pde1, loss_pde2

    def create_batch(self, batch_size):
        """Samples a batch from the steady-state data."""
        indices = np.random.choice(self.tot_len, batch_size, replace=True)
        x_batch = tf.gather(self.X, indices)
        y_batch = tf.gather(self.Y, indices)
        u_batch = tf.gather(self.u, indices)
        return x_batch, y_batch, u_batch

    def train(self, max_iterations=1e8, step_size=None):
        from clr_callback import CyclicLR
        import matplotlib.pyplot as plt
        plt.ion()
        
        clr_cb = CyclicLR(model_optimizer=self, base_lr=self.min_lr, max_lr=self.max_lr, step_size=step_size)
        self.callbacks = clr_cb
        self.callbacks.on_train_begin()
        
        tf.print(f"Starting 3-PHASE training (Target Data:PDE ratio = {self.target_data_pde_ratio}:1)")

        phase_3_weights_set = False

        while self.iterations < max_iterations:
            
            if self.iterations < self.threshold_p1:
                # --- PHASE 1: DATA-FITTING ---
                x_batch, y_batch, u_batch = self.create_batch(self.batchsize)
                loss_u = self.train_step_p1(x_batch, y_batch, u_batch)
                self.log_and_store(loss_u, "PHASE 1", loss_u, tf.constant(0.0), tf.constant(0.0))

            elif self.iterations < self.threshold_p2:
                # --- PHASE 2: CONSTRAINT-FIT ---
                if self.iterations == self.threshold_p1: 
                    tf.print("--- SWITCHING TO PHASE 2 (CONSTRAINT-FIT) ---")
                
                x_batch, y_batch, u_batch = self.create_batch(self.batchsize)
                total_loss, loss_u, loss_pde1, loss_pde2_ph = self.train_step_p2(x_batch, y_batch, u_batch)
                self.log_and_store(total_loss, "PHASE 2", loss_u, loss_pde1, loss_pde2_ph)
            
            else:
                # --- PHASE 3: INVERSE SOLVE + PRINCIPLED ANNEALING ---
                if not phase_3_weights_set:
                    tf.print("--- SWITCHING TO PHASE 3 (INVERSE SOLVE) ---")
                    tf.print("Principle: Maintain reasonable Data:PDE balance for parameter estimation")
                    self.weight_data.assign(100.0)
                    self.weight_pde1.assign(1.0)
                    self.weight_pde2.assign(1.0)
                    phase_3_weights_set = True
                
                # *** PRINCIPLED ANNEALING WITH RATIO-BASED STOPPING ***
                if (self.annealing_active and 
                    self.iterations > self.threshold_p2 and 
                    self.iterations % self.annealing_step == 0):
                    
                    current_ratio = self.weight_data / self.weight_pde2
                    
                    # Stop when we reach target ratio
                    if current_ratio <= self.target_data_pde_ratio:
                        tf.print(f"--- ANNEALING STOPPED at Iteration {self.iterations} ---")
                        tf.print(f"Reached target Data:PDE ratio: {current_ratio:.1f}:1")
                        tf.print(f"*** SWITCHING ε LR FROM {self.eps_lr.numpy():.2e} TO {self.refinement_eps_lr:.2e} ***")
                        self.annealing_active = False
                        self.refinement_phase = True
                        # *** INSTANT LR REDUCTION TO 1e-6 ***
                        self.eps_lr.assign(self.refinement_eps_lr)
                        self.optimizer_eps.learning_rate = self.eps_lr
                    else:
                        # Conservative annealing toward target ratio
                        tf.print(f"--- Annealing toward {self.target_data_pde_ratio}:1 ratio ---")
                        
                        new_pde2 = tf.minimum(self.weight_pde2 * self.pde2_growth_rate, 50.0)
                        new_data = tf.maximum(self.weight_data * self.data_decay_rate, 50.0)
                        
                        self.weight_pde2.assign(new_pde2)
                        self.weight_data.assign(new_data)
                        
                        tf.print(f"Weights: Data={self.weight_data.numpy():.1f}, PDE2={self.weight_pde2.numpy():.1f}, Ratio={current_ratio:.1f}:1")

                x_batch, y_batch, u_batch = self.create_batch(self.batchsize)
                total_loss, loss_u, loss_pde1, loss_pde2 = self.train_step_p3(x_batch, y_batch, u_batch)
                self.log_and_store(total_loss, "PHASE 3", loss_u, loss_pde1, loss_pde2)

            
            self.callbacks.on_batch_end(self.epochs)
            self.iterations += 1

            # --- Plotting ---
            if self.iterations % 2000 == 0:
                try:
                    tf.print(f"\n[Plotting] Iteration {self.iterations}. Updating plot...")
                    
                    # 1. Get full prediction (outside tape)
                    u_pred_full, _ = tf.split(
                        self.model_up(tf.concat([self.X, self.Y], 1), training=False), 
                        2, axis=1
                    )
                    u_plot = tf.reshape(u_pred_full, (self.n, self.n)).numpy()

                    # 2. Initialize or update plot
                    if not hasattr(self, 'fig') or self.fig is None:
                        # Create figure for the first time
                        self.fig, self.ax = plt.subplots(figsize=(7, 6))
                        self.img = self.ax.imshow(u_plot, cmap='viridis', interpolation='none', origin='lower')
                        self.fig.colorbar(self.img, ax=self.ax)
                    else:
                        # Update existing figure
                        self.img.set_data(u_plot)
                        self.img.set_clim(np.min(u_plot), np.max(u_plot)) # Rescale colors
                        self.fig.canvas.draw()
                    
                    self.ax.set_title(f"Prediction at Iteration {self.iterations}")
                    
                    # 3. Pause to allow GUI update
                    plt.pause(0.01) # Small pause to render
                
                except Exception as e:
                    # Handle if user closes plot window
                    tf.print(f"[Plotting] Error: {e}. Window closed? Re-initializing plot.")
                    self.fig = None

            if self.iterations % 100 == 0: 
                self.epochs += 1
        
        tf.print("Training loop finished.")
        plt.ioff()
        plt.show()

    def log_and_store(self, loss, phase_name, loss_u, loss_pde1, loss_pde2):
        """Consolidates logging, history storage."""
        if loss_pde2 is None: loss_pde2 = tf.constant(0.0)
        
        current_eps_lr = self.eps_lr.numpy()
        
        if self.iterations % 50 == 0:
            self._loss_array.append(loss.numpy())
            self._loss_u_array.append(loss_u.numpy())
            self._loss_pde1_array.append(loss_pde1.numpy())
            self._loss_pde2_array.append(loss_pde2.numpy())
            self._epsilon_array.append(self.epsilon.numpy()[0])
            self._delta_array.append(self.delta.numpy()[0])
            self._gamma_array.append(self.gamma.numpy()[0]) 
            self._u_m_array.append(0.0) 
            self._u_s_array.append(0.0) 
            self._eps_lr_array.append(current_eps_lr)
            
        if self.iterations % 200 == 0:
            tf.print(f"{phase_name} - It: {self.iterations}, Loss: {loss.numpy():.3e}, L_Data: {loss_u.numpy():.3e}, "
                     f"L_PDE1: {loss_pde1.numpy():.3e}, L_PDE2: {loss_pde2.numpy():.3e}, "
                     f"ε: {self.epsilon.numpy()[0]:.6f}, ε_LR: {current_eps_lr:.2e}")
            
    # --- Property methods ---
    
    @property
    def parameters(self):
        return [self._epsilon_array, self._delta_array, self._gamma_array, self._u_m_array, self._u_s_array, self._eps_lr_array]

    @property 
    def loss_pde_u_array(self):
        return [a + b for a, b in zip(self._loss_pde1_array, self._loss_pde2_array)]

    @property
    def loss_u_array(self):
        return self._loss_u_array

    @property
    def loss_array(self):
        return self._loss_array

    @property
    def epsilon_array(self):
        return self._epsilon_array
    
    @property
    def eps_lr_array(self):
        return self._eps_lr_array

    @property
    def final_parameters(self):
        if self._epsilon_array:
            return [self._epsilon_array[-1], self._delta_array[-1], self._gamma_array[-1]]
        else:
            return [self.epsilon.numpy()[0], self.delta.numpy()[0], self.gamma.numpy()[0]]