import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from layers import RBFLayer
from collections import deque

print("\n*** SUCCESSFULLY LOADED RBF SS MODEL (V4 - GRAD NORM) ***\n")

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
        sigma2=5,       # RBF layer parameter
        alpha=2,      # GradNorm asymmetry parameter
    ):
        super().__init__()

        #Parameters & Grid Setup
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.units = units
        self.dx = dx
        self.batchsize = batchsize
        self.n = len(x)
        self.epochs = 0
        self.iterations = 0
        
        #  GradNorm Parameters
        self.alpha = alpha  # Asymmetry parameter
        self.lr_weights = 0.025  # Learning rate for weight updates (from paper)
        self.weight_update_frequency = 100
        
        #Initial Loss Tracking
        self.initial_loss_u = None
        self.initial_loss_pde1 = None  
        self.initial_loss_pde2 = None
        self.initial_losses_set = False

        #Rest of your existing initialization
        self.threshold_p1 = threshold_ep
        self.threshold_p2 = threshold_ep + 30000
        
        
        # Physical Parameters
        self.epsilon = tf.Variable([0.5], dtype=tf.float32, trainable=True)
        self.delta = tf.Variable([0.406], dtype=tf.float32, trainable=False)
        self.gamma = tf.Variable([0.196], dtype=tf.float32, trainable=False)
        
    
        # Data & Architecture
        self.u = tf.constant(u.flatten()[:, None], dtype=tf.float32)
        self.max_val = np.max(x)
        X_np, Y_np = np.meshgrid(x, y)
        self.tot_len = len(X_np.flatten())
        self.u_scale = float(np.mean(u.flatten()**2)) 

        #Batch Selection
        self.permutation = np.arange(self.tot_len)
        np.random.shuffle(self.permutation)
        self.current_idx = 0

        self.X = tf.constant(X_np.flatten()[:, None], dtype=tf.float32)
        self.Y = tf.constant(Y_np.flatten()[:, None], dtype=tf.float32)

        self.model_up = tf.keras.Sequential([
            RBFLayer(self.units, 1 / sigma2, self.max_val),
            tf.keras.layers.Dense(2, input_shape=(self.units,), use_bias=False),
        ])

        # Optimizers - ONLY CHANGE: Lower ε learning rate
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.max_lr)
        self.refinement_eps_lr = 1e-6
        self.initial_eps_lr = 1e-5  # *** CHANGED FROM 1e-4 TO 1e-5 ***
        self.current_eps_lr = float(self.initial_eps_lr)
        self.optimizer_eps = tf.keras.optimizers.legacy.Adam(learning_rate=self.current_eps_lr)
        
        # *** NEW: Separate optimizer for loss weights ***
        self.optimizer_weights = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr_weights)

        self.mse = tf.keras.losses.MeanSquaredError()

        # Loss Weights as Variables
        self.weight_data = tf.Variable(1.0, dtype=tf.float32, trainable=True)  
        self.weight_pde1 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.weight_pde2 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        

        # History Tracking
        self._init_history()

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
        # Weight history
        self._weight_data_array = []
        self._weight_pde1_array = []
        self._weight_pde2_array = []
        self._gradnorm_u_array  = []
        self._gradnorm_pde1_array = []
        self._gradnorm_pde2_array = []
        self._weight_frac_data = []
        self._weight_frac_pde1 = []
        self._weight_frac_pde2 = []


    def apply_gradnorm_weights(self, phase, gradnorms, losses):
        if self.iterations % self.weight_update_frequency != 0:
            return

        # Convert gradnorms and losses to tensors
        gradnorms_tf = [tf.convert_to_tensor(g, dtype=tf.float32) for g in gradnorms]
        losses_tf = [tf.convert_to_tensor(l, dtype=tf.float32) for l in losses]

        # Initialize loss buffers
        num_tasks = len(losses_tf)
        if not hasattr(self, 'loss_buffers') or len(self.loss_buffers) != num_tasks:
            self.loss_buffers = [deque(maxlen=5) for _ in range(num_tasks)]

        # Update buffers
        for i, lt in enumerate(losses_tf):
            self.loss_buffers[i].append(float(lt.numpy()) if hasattr(lt, 'numpy') else float(lt))

        # Compute inverse training rates r_i using moving average
        r = []
        for i, buf in enumerate(self.loss_buffers):
            if len(buf) < 2:
                r.append(tf.constant(1.0, dtype=tf.float32))
            else:
                moving_avg = np.mean(list(buf)[:-1])
                r.append(losses_tf[i] / (moving_avg + 1e-8))
        r_tf = r

        if phase == 'PHASE 1':
            self.weight_data.assign(1.0)
            self.weight_pde1.assign(0.0)
            self.weight_pde2.assign(0.0)
            self.weight_frac_data.append(1.0) 
            self.weight_frac_pde1.append(0.0)
            self.weight_frac_pde2.append(0.0)
            return


        elif phase == 'PHASE 2':
            if self.initial_loss_u is None or self.initial_loss_pde1 is None:
                self.initial_loss_u = losses_tf[0]
                self.initial_loss_pde1 = losses_tf[1]
                self.weight_data.assign(3.0)
                self.weight_pde1.assign(1.0)
                self.weight_pde2.assign(0.0)
                return

            r_avg = (r_tf[0] + r_tf[1]) / 4.0
            r_rel = [r_tf[i] / (r_avg + 1e-8) for i in range(2)]
            G_avg = (gradnorms_tf[0] + gradnorms_tf[1]) / 3.0
            targets = [G_avg * tf.pow(r_rel[i], float(self.alpha)) for i in range(2)]

            # Compute loss_grad entirely in TF
            with tf.GradientTape() as tape:
                tape.watch([self.weight_data, self.weight_pde1])
                actuals = [gradnorms_tf[0] * self.weight_data,
                        gradnorms_tf[1] * self.weight_pde1]
                loss_grad = tf.add_n([tf.square(actuals[i] - targets[i]) for i in range(2)])
            grads = tape.gradient(loss_grad, [self.weight_data, self.weight_pde1])
            self.optimizer_weights.apply_gradients(zip(grads, [self.weight_data, self.weight_pde1]))

            # Optional renormalization
            total = self.weight_data + self.weight_pde1 + 1e-8
            self.weight_data.assign(4.0 * (self.weight_data / total))
            self.weight_pde1.assign(4.0 * (self.weight_pde1 / total))
            self.weight_pde2.assign(0.0)
            self.weight_frac_data.append(self.weight_data/total) 
            self.weight_frac_pde1.append(self.weight_pde1 / total)
            self.weight_frac_pde2.append(0)

        elif phase == 'PHASE 3':
            if not self.initial_losses_set:
                self.initial_loss_u = losses_tf[0]
                self.initial_loss_pde1 = losses_tf[1]
                self.initial_loss_pde2 = losses_tf[2]
                self.initial_losses_set = True
                self.weight_data.assign(6.0)
                self.weight_pde1.assign(1.0)
                self.weight_pde2.assign(3.0)
                tf.print("*** Phase 3 initialized ***")
                return

            r_avg = tf.reduce_mean([r_tf[0], r_tf[1], r_tf[2]])
            r_rel = [r_tf[i] / (r_avg + 1e-8) for i in range(3)]
            G_avg = (gradnorms_tf[0] + gradnorms_tf[1] + gradnorms_tf[2]) / 10
            targets = [G_avg * tf.pow(r_rel[i], float(self.alpha)) for i in range(3)]

            with tf.GradientTape() as tape:
                tape.watch([self.weight_data, self.weight_pde1, self.weight_pde2])
                actuals = [
                    gradnorms_tf[0] * self.weight_data,
                    gradnorms_tf[1] * self.weight_pde1,
                    gradnorms_tf[2] * self.weight_pde2
                ]
                loss_grad = tf.add_n([tf.square(actuals[i] - targets[i]) for i in range(3)])

            grads = tape.gradient(loss_grad, [self.weight_data, self.weight_pde1, self.weight_pde2])
            self.optimizer_weights.apply_gradients(zip(grads, [self.weight_data, self.weight_pde1, self.weight_pde2]))

            # Renormalize to sum=3
            total = self.weight_data + self.weight_pde1 + self.weight_pde2 + 1e-8
            self.weight_data.assign(10 * (self.weight_data / total))
            self.weight_pde1.assign(10 * (self.weight_pde1 / total))
            self.weight_pde2.assign(10 * (self.weight_pde2 / total))
            self.weight_frac_data.append(self.weight_data/total) 
            self.weight_frac_pde1.append(self.weight_pde1 / total)
            self.weight_frac_pde2.append(self.weight_pde2 / total)

    @tf.function
    def _calculate_residuals(self, x, y):
        """Existing residuals calculation - unchanged"""
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
            u_yy = tf.zeros_like(y) if u_yy is None else u_yy
            p_xx = tf.zeros_like(x) if p_xx is None else p_xx
            p_yy = tf.zeros_like(y) if p_yy is None else p_yy

        del tape_inner
        del tape_outer

        Laplace_u = u_xx + u_yy
        Laplace_p = p_xx + p_yy

        pde_residual_1 = p_pred - Laplace_u

        nonlinear_terms = (self.epsilon * u_pred 
                           - self.delta * tf.square(u_pred) 
                           - self.gamma * tf.pow(u_pred, 3))

        sh_operator = u_pred + 2.0 * p_pred + Laplace_p

        full_sh_residual = nonlinear_terms - sh_operator

        return u_pred, p_pred, pde_residual_1, full_sh_residual

    def create_batch(self, batch_size):
        if self.current_idx + batch_size > self.tot_len:
            # Start a new epoch
            np.random.shuffle(self.permutation)
            self.current_idx = 0

        batch_indices = self.permutation[self.current_idx:self.current_idx + batch_size]
        self.current_idx += batch_size

        x_batch = tf.gather(self.X, batch_indices)
        y_batch = tf.gather(self.Y, batch_indices)
        u_batch = tf.gather(self.u, batch_indices)
        return x_batch, y_batch, u_batch

    @tf.function
    def train_step_p1(self, x, y, u):
        """PHASE 1: PURE DATA FIT"""
        with tf.GradientTape() as tape:
            u_pred, _ = tf.split(self.model_up(tf.concat([x, y], 1), training=True), 2, axis=1)
            loss_u = self.mse(u_pred, u) / tf.reduce_mean(tf.square(u) + 1e-8)

        grads = tape.gradient(loss_u, self.model_up.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model_up.trainable_variables))
        del tape
        return loss_u

    @tf.function
    def train_step_p2(self, x, y, u):
        """PHASE 2: CONSTRAINT-FIT (Network Weights Only, Epsilon frozen).
        Returns: total_loss, loss_u, loss_pde1, gradnorm_u, gradnorm_pde1
        """
        with tf.GradientTape(persistent=True) as tape:
            u_pred, _, pde1_residual, _ = self._calculate_residuals(x, y)

            loss_u = self.mse(u_pred, u) / (tf.reduce_mean(tf.square(u)) + 1e-8)
            loss_pde1 = tf.reduce_mean(tf.square(pde1_residual))

            total_loss = (self.weight_data * loss_u +
                        self.weight_pde1 * loss_pde1)

        # gradients for network variables for each loss (separately)
        grads_u = tape.gradient(loss_u, self.model_up.trainable_variables)
        grads_pde1 = tape.gradient(loss_pde1, self.model_up.trainable_variables)

        # compute gradient norms (global L2 norm)
        gradnorm_u = tf.linalg.global_norm(grads_u)
        gradnorm_pde1 = tf.linalg.global_norm(grads_pde1)

        # apply network update using total_loss grads
        net_grads = tape.gradient(total_loss, self.model_up.trainable_variables)
        self.optimizer.apply_gradients(zip(net_grads, self.model_up.trainable_variables))

        del tape
        return total_loss, loss_u, loss_pde1, gradnorm_u, gradnorm_pde1

    @tf.function
    def train_step_phase3(self, x, y, u):
        """PHASE 3: Full physics + inverse solve for epsilon"""
        with tf.GradientTape(persistent=True) as tape:
            u_pred, _, pde1_res, pde2_res = self._calculate_residuals(x, y)
            
            loss_data = tf.reduce_mean(tf.square(u_pred - u)) / (self.u_scale + 1e-8)
            loss_pde1 = tf.reduce_mean(tf.square(pde1_res))
            loss_pde2 = tf.reduce_mean(tf.square(pde2_res))
            
            total_loss = (self.weight_data * loss_data + 
                         self.weight_pde1 * loss_pde1 + 
                         self.weight_pde2 * loss_pde2)
        
        # Compute gradient norms for GradNorm
        grads_data = tape.gradient(loss_data, self.model_up.trainable_variables)
        grads_pde1 = tape.gradient(loss_pde1, self.model_up.trainable_variables)
        grads_pde2 = tape.gradient(loss_pde2, self.model_up.trainable_variables)

        gradnorm_data = tf.linalg.global_norm(grads_data)
        gradnorm_pde1 = tf.linalg.global_norm(grads_pde1)
        gradnorm_pde2 = tf.linalg.global_norm(grads_pde2)
        
        # Update network
        net_grads = tape.gradient(total_loss, self.model_up.trainable_variables)
        self.optimizer.apply_gradients(zip(net_grads, self.model_up.trainable_variables))
        
        # Update epsilon
    
        param_grads = tape.gradient(total_loss, [self.epsilon, self.delta, self.gamma])
        self.optimizer_eps.apply_gradients(zip(param_grads, [self.epsilon, self.delta, self.gamma]))
        self.epsilon.assign(tf.clip_by_value(self.epsilon, 0.0, 1.0))
        self.delta.assign(tf.clip_by_value(self.delta, 0.0, 1.0))
        self.gamma.assign(tf.clip_by_value(self.gamma, 0.0, 1.0))
        del tape
        return total_loss, loss_data, loss_pde1, loss_pde2, gradnorm_data, gradnorm_pde1, gradnorm_pde2
  
    def train(self, max_iterations=int(1e8), step_size=None):
        from clr_callback import CyclicLR
        import matplotlib.pyplot as plt
        plt.ion()
        
        # Initial CLR setup with cycling enabled
        clr_cb = CyclicLR(
            model_optimizer=self, 
            base_lr=self.min_lr, 
            max_lr=self.max_lr, 
            step_size=step_size,
            mode='triangular'
        )
            
        self.callbacks = clr_cb
        self.callbacks.on_train_begin()

        tf.print(f"Starting 3-PHASE training with GRAD NORM WEIGHTING and Cyclic LR")
        tf.print(f"CLR Config: base_lr={self.min_lr:.2e}, max_lr={self.max_lr:.2e}, step_size={step_size}")
        tf.print(f"Initial Net LR: {self.optimizer.learning_rate.numpy():.2e}")

        phase_3_weights_set = False
        logs = {}  # Initialize logs

        while self.iterations < max_iterations:

            # Check if we need to disable CLR cycling at iteration 80000
            if self.iterations == 80000:
                tf.print("Constant LR Implemented")
                clr_cb = CyclicLR(
                    model_optimizer=self, 
                    base_lr=self.min_lr, 
                    max_lr=self.min_lr,  # Same as base_lr = constant learning rate
                    step_size=step_size,
                )
                self.callbacks = clr_cb
                self.callbacks.on_train_begin()
                tf.print(f"LR now fixed at: {self.optimizer.learning_rate.numpy():.2e}")

            if self.iterations < self.threshold_p1:
                x_batch, y_batch, u_batch = self.create_batch(self.batchsize)
                loss_u_tensor = self.train_step_p1(x_batch, y_batch, u_batch)
                logs = {'loss': float(loss_u_tensor.numpy())}
                self.log_and_store(loss_u_tensor, "PHASE 1", loss_u_tensor, tf.constant(0.0), tf.constant(0.0))

            elif self.iterations < self.threshold_p2:
                if self.iterations == self.threshold_p1:
                    tf.print("Phase 2 Active")

                x_batch, y_batch, u_batch = self.create_batch(self.batchsize)
                total_loss_t, loss_u_t, loss_pde1_t, g_u_t, g_p1_t = self.train_step_p2(x_batch, y_batch, u_batch)
                self.apply_gradnorm_weights("PHASE 2", [g_u_t, g_p1_t], [loss_u_t, loss_pde1_t])
                logs = {'loss': float(total_loss_t.numpy())}
                self.log_and_store(total_loss_t, "PHASE 2", loss_u_t, loss_pde1_t, tf.constant(0.0))

            else:
                if not phase_3_weights_set:
                    tf.print("Phase 3 Active")
                    phase_3_weights_set = True

                x_batch, y_batch, u_batch = self.create_batch(self.batchsize)
                total_loss_t, loss_u_t, loss_pde1_t, loss_pde2_t, g_u_t, g_p1_t, g_p2_t = self.train_step_phase3(x_batch, y_batch, u_batch)
                self.apply_gradnorm_weights("PHASE 3", [g_u_t, g_p1_t, g_p2_t], [loss_u_t, loss_pde1_t, loss_pde2_t])
                logs = {'loss': float(total_loss_t.numpy())}
                self.log_and_store(total_loss_t, "PHASE 3", loss_u_t, loss_pde1_t, loss_pde2_t)

            # Update CLR
            self.callbacks.on_batch_end(self.iterations, logs=logs)

            self.iterations += 1
            if self.iterations % 100 == 0:
                self.epochs += 1

        tf.print("Training loop finished.")
        tf.print(f"Final Net LR: {self.optimizer.learning_rate.numpy():.2e}")
        plt.ioff()
        plt.show()

    def log_and_store(self, loss, phase_name, loss_u, loss_pde1, loss_pde2, gradnorm_u=None, gradnorm_pde1=None, gradnorm_pde2=None):
        """Enhanced logging with weight tracking."""
        if loss_pde2 is None: 
            loss_pde2 = tf.constant(0.0)

        current_eps_lr = float(self.current_eps_lr)

        if self.iterations % 50 == 0:
            self._loss_array.append(float(loss.numpy()))
            self._loss_u_array.append(float(loss_u.numpy()))
            self._loss_pde1_array.append(float(loss_pde1.numpy()))
            self._loss_pde2_array.append(float(loss_pde2.numpy()))
            self._epsilon_array.append(float(self.epsilon.numpy()[0]))
            self._delta_array.append(float(self.delta.numpy()[0]))
            self._gamma_array.append(float(self.gamma.numpy()[0]))
            self._u_m_array.append(0.0)
            self._u_s_array.append(0.0)
            self._eps_lr_array.append(current_eps_lr)
            # Store current weights
            self._weight_data_array.append(float(self.weight_data.numpy()))
            self._weight_pde1_array.append(float(self.weight_pde1.numpy()))
            self._weight_pde2_array.append(float(self.weight_pde2.numpy()))
            total = self.weight_data + self.weight_pde1 + self.weight_pde2 + 1e-8
            
            self._weight_frac_data.append(float(self.weight_data.numpy() / total.numpy()))
            self._weight_frac_pde1.append(float(self.weight_pde1.numpy() / total.numpy()))
            self._weight_frac_pde2.append(float(self.weight_pde2.numpy() / total.numpy()))
            
            self._gradnorm_u_array.append(float(gradnorm_u.numpy()) if gradnorm_u is not None else 0.0)
            self._gradnorm_pde1_array.append(float(gradnorm_pde1.numpy()) if gradnorm_pde1 is not None else 0.0)
            self._gradnorm_pde2_array.append(float(gradnorm_pde2.numpy()) if gradnorm_pde2 is not None else 0.0)


        if self.iterations % 200 == 0:
            print(    f"Iter: {self.iterations:6d} | "
                      f"Loss: {float(loss.numpy()):8.3e} | "
                      f"Data: {float(loss_u.numpy()):8.3e} | "
                      f"PDE1: {float(loss_pde1.numpy()):8.3e} | "
                      f"PDE2: {float(loss_pde2.numpy()):8.3e} | "
                      f"ε: {float(self.epsilon.numpy()[0]):.4f} | "
                      f"delta: {float(self.delta.numpy()[0]):.4f} | "
                      f"gamma: {float(self.gamma.numpy()[0]):.4f} | "
                      f"W: ({self.weight_data.numpy():.2f}, "
                      f"{self.weight_pde1.numpy():.2f}, "
                      f"{self.weight_pde2.numpy():.2f}) | "
                      f"LR: {float(self.optimizer.learning_rate.numpy()):.2e} | ")
            
        
    @property
    def weight_data_array(self):
        return self._weight_data_array

    @property
    def weight_pde1_array(self):
        return self._weight_pde1_array

    @property
    def weight_pde2_array(self):
        return self._weight_pde2_array
    
    @property
    def parameters(self):
        return [self._epsilon_array, self._delta_array, self._gamma_array, self._u_m_array, self._u_s_array, self._eps_lr_array]

    @property
    def loss_pde_1_array(self):
        return self._loss_pde1_array

    @property
    def loss_pde_2_array(self):
        return self._loss_pde2_array

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
        
    @property
    def gradnorm_u_array(self):
        return self._gradnorm_u_array
    @property
    def gradnorm_pde1_array(self):
        return self._gradnorm_pde1_array
    @property
    def gradnorm_pde2_array(self):
        return self._gradnorm_pde2_array
    @property 
    def weight_frac_data(self):
        return self._weight_frac_data
    @property
    def weight_frac_pde1(self):
        return self._weight_frac_pde1 
    @property
    def weight_frac_pde2(self):
        return self._weight_frac_pde2 
    
