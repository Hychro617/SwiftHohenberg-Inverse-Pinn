import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from layers import RBFLayer

class RBF_PINNs(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        x,
        y,
        min_lr,
        max_lr,
        u,
        dx,
        tol,
        threshold_ep,
        batchsize,
        sigma2=2,
    ):
        super().__init__()
        self.lr = max_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.tol = tol
        self.units = units
        self.dx = dx
        self.batchsize = batchsize
        self.n = len(x)
        self.epochs = 0
        
        # Only epsilon is trainable, delta and gamma are fixed
        self.epsilon = tf.Variable([0.5], dtype=tf.float32, trainable=True)
        self.delta = tf.Variable([0.406], dtype=tf.float32, trainable=False)
        self.gamma = tf.Variable([0.196], dtype=tf.float32, trainable=False)
        
        self.u_mean_var = tf.Variable([0.9068], dtype=tf.float32, trainable=True)
        self.u_scale_var = tf.Variable([2.5], dtype=tf.float32, trainable=True)
        self.u = u.flatten()[:, None]
        self.u_in = u.flatten()[:, None]

        self.max_val = np.max(x)
        self.threshold = threshold_ep
        
        X, Y = np.meshgrid(x, y)
        self.tot_len = len(X.flatten())
        self.X, self.Y = X.flatten()[:, None], Y.flatten()[:, None]
        self.X_in, self.Y_in = X.flatten()[:, None], Y.flatten()[:, None]
        self.tot_len_in = len(self.X_in.flatten())
        
        self.tot_loss = []
        self.epsilon_array = []
        self.delta_array = []
        self.gamma_array = []
        self.u_m_array = []
        self.u_s_array = []
        self.loss_array = []
        self.loss_u_array = []
        self.loss_pde_u_array = []
        
        self.Data = tf.concat([self.X, self.Y], 1)
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.max_lr)
        self.model_u = tf.keras.Sequential(
            [
                RBFLayer(self.units, 1 / sigma2, self.max_val),
                tf.keras.layers.Dense(1, input_shape=(self.units,), use_bias=False),
            ]
        )
        self.mse = tf.keras.losses.MeanSquaredError()
        self.iterations = 0

        # Initialize dynamic weights
        self.weight_pde_u = tf.Variable(1.0, dtype=tf.float32, trainable=False)
        self.weight_pde_u_1 = tf.Variable(1.0, dtype=tf.float32, trainable=False)

    
    @tf.function
    def _get_losses1(self, x, y, u):
        with tf.GradientTape(persistent=False) as tape1:
            u_pred = self.model_u(tf.concat([x, y], 1), training=True)
            loss_u = self.mse(u_pred, u) / tf.reduce_mean(tf.square(u))
            trainables_u = self.model_u.trainable_variables
            grads_u = tape1.gradient(loss_u, trainables_u)
            del tape1
        self.optimizer.apply_gradients(zip(grads_u, trainables_u))
        loss = loss_u 
        return loss
    
    @tf.function
    def _get_losses2(self, x, y, u):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            u_pred = self.model_u(tf.concat([x, y], 1), training=True)

            # Compute derivatives
            u_x = tape.gradient(u_pred, x)
            u_y = tape.gradient(u_pred, y)  
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)
            Laplace_u = u_xx + u_yy
            lap_x = tape.gradient(Laplace_u, x)
            lap_y = tape.gradient(Laplace_u, y)
            lap_xx = tape.gradient(lap_x, x)
            lap_yy = tape.gradient(lap_y, y)
            Biharmonic_u = lap_xx + lap_yy

            sh_operator = u_pred + 2 * Laplace_u + Biharmonic_u
            
            # CURRICULUM REGULARIZATION 
            if self.iterations < 10000:
                pde_residual = self.epsilon * u_pred - sh_operator
                curriculum_weight = 0.1
            elif self.iterations < 20000:
                pde_residual = (self.epsilon * u_pred - 
                            0.3 * self.delta * tf.square(u_pred) - sh_operator)
                curriculum_weight = 0.3
            elif self.iterations < 30000:
                pde_residual = (self.epsilon * u_pred - 
                            0.7 * self.delta * tf.square(u_pred) - 
                            0.3 * self.gamma * tf.pow(u_pred, 3) - sh_operator)
                curriculum_weight = 0.6
            else:
                pde_residual = (self.epsilon * u_pred - 
                            self.delta * tf.square(u_pred) - 
                            self.gamma * tf.pow(u_pred, 3) - sh_operator)
                curriculum_weight = 1.0

            loss_u = self.mse(u_pred, u) / tf.reduce_mean(tf.square(u))
            loss_pde_u = tf.reduce_mean(tf.square(pde_residual))
            
            def boundary_penalty(param, param_name):
                upper_penalty = tf.maximum(0.0, param - 0.8)**2
                lower_penalty = tf.maximum(0.0, 0.2 - param)**2
                return upper_penalty + lower_penalty
            
            epsilon_bound_penalty = boundary_penalty(self.epsilon, "epsilon")
            
            loss1 = (0.1 * loss_u + 
                    curriculum_weight * self.weight_pde_u * loss_pde_u +
                    0.01 * epsilon_bound_penalty)  
            
            trainables1 = (self.model_u.trainable_variables + 
                        [self.epsilon]) 

        grads1 = tape.gradient(loss1, trainables1)
        if grads1[0] is not None:
            self.optimizer.apply_gradients(zip(grads1, trainables1))
        
        # Reset epsilon if needed 
        if self.iterations % 500 == 0:
            epsilon_near_lower = (self.epsilon <= 0.1)
            epsilon_near_upper = (self.epsilon >= 0.9)
            
            if epsilon_near_lower or epsilon_near_upper:
                if epsilon_near_lower:
                    new_val = 0.3 + 0.1 * tf.random.normal([1])
                else:
                    new_val = 0.5 + 0.1 * tf.random.normal([1])
                
                self.epsilon.assign([tf.maximum(0.1, tf.minimum(0.9, new_val[0]))])
        
        # Apply soft constraints to epsilon
        self.epsilon.assign(tf.minimum(tf.maximum(self.epsilon, 0.1), 0.9))
        
        del tape
        return loss1, loss_u, loss_pde_u

    def update_dynamic_weights(self):
        """Update dynamic weights based on current losses"""
        X1 = tf.Variable(self.X_in, dtype=tf.float32)
        Y1 = tf.Variable(self.Y_in, dtype=tf.float32)
        u1 = tf.Variable(self.u_in, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            u_pred = self.model_u(tf.concat([X1, Y1], 1))

            u_x = tape.gradient(u_pred, X1)
            u_y = tape.gradient(u_pred, Y1)
            u_xx = tape.gradient(u_x, X1)
            u_yy = tape.gradient(u_y, Y1)
            Laplace_u = u_xx + u_yy
            lap_x = tape.gradient(Laplace_u, X1)
            lap_y = tape.gradient(Laplace_u, Y1)
            lap_xx = tape.gradient(lap_x, X1)
            lap_yy = tape.gradient(lap_y, Y1)
            BiLaplace_u = lap_xx + lap_yy

            loss_u = self.mse(u_pred, u1)
            sh_operator = u_pred + 2 * Laplace_u + BiLaplace_u
            Linear_u = self.epsilon * u_pred
            Nonlinear_u = -self.delta * tf.square(u_pred) - self.gamma * tf.pow(u_pred, 3)
            pde_residual = Linear_u + Nonlinear_u - sh_operator
            
            loss_pde_u = tf.reduce_mean(tf.square(pde_residual))
            loss_pde_u_1 = tf.reduce_mean(tf.square(pde_residual))
        
        epsilon = 1e-8
        self.weight_pde_u.assign(loss_pde_u / (loss_pde_u + loss_u +epsilon))
        self.weight_pde_u_1.assign(loss_pde_u_1 / (loss_pde_u_1 + loss_u +epsilon))

        del tape

    def train(self, max_iterations=1e8, step_size=None):
        from clr_callback import CyclicLR
        clr_cb = CyclicLR(
            model_optimizer=self,
            base_lr=self.min_lr,
            max_lr=self.max_lr,
            step_size=step_size,
        )
        self.callbacks = clr_cb
        
        if self.iterations > 1:
            self.model_u.set_weights(self.weights_model_u)
            
        first_pass = True
        self.callbacks.on_train_begin()
        
        while self.iterations > -1:
            if self.iterations < self.threshold:
                len_dat = self.tot_len
                datindexes = tf.data.Dataset.range(len_dat).shuffle(len_dat)
                indexes = list(datindexes.batch(self.batchsize).as_numpy_iterator())
                for batch in indexes:
                    x = tf.Variable(self.X[batch], dtype=tf.float32)
                    y = tf.Variable(self.Y[batch], dtype=tf.float32)
                    u = tf.Variable(self.u[batch], dtype=tf.float32)
                    loss = self._get_losses1(x, y, u)
                    
                    self.weights_model_u = self.model_u.weights
                    if self.iterations % 200 == 0:
                        tf.print(
                            "It: %d, Epoch: %d, Total loss: %e, Learning rate: %2e"
                            % (
                                self.iterations,
                                self.epochs,
                                loss,
                                self.optimizer.lr,
                            )
                        )
                    if self.iterations % 2000 == 0:
                        u_pred = np.reshape(self.model_u(tf.concat([self.X, self.Y], 1)), (self.n, self.n))
                        plt.imshow(u_pred, cmap=plt.cm.RdBu)
                        plt.colorbar(label = 'Amplitude')
                        plt.show()
                    
                    if self.iterations == max_iterations:
                        self.iterations = -2
                        break
                    self.iterations += 1
                    if loss < self.tol:
                        break
                    self.callbacks.on_batch_end(self.epochs)

            else:
                len_dat = self.tot_len_in
                datindexes = tf.data.Dataset.range(len_dat).shuffle(len_dat)
                indexes = list(datindexes.batch(self.batchsize).as_numpy_iterator())

                if self.iterations < (self.threshold + 10000):
                    update_freq = 500
                else:
                    update_freq = 1000

                if first_pass or self.iterations % update_freq == 0:
                    self.update_dynamic_weights()
                    first_pass = False
                            
                for batch in indexes:
                    x = tf.Variable(self.X_in[batch], dtype=tf.float32)
                    y = tf.Variable(self.Y_in[batch], dtype=tf.float32)
                    u = tf.Variable(self.u_in[batch], dtype=tf.float32)
                
                    loss, loss_u, loss_pde_u = self._get_losses2(x, y, u)
                    
                    if self.iterations % 500 == 0:
                        x = tf.Variable(self.X, dtype=tf.float32)
                        y = tf.Variable(self.Y, dtype=tf.float32)
                        u = tf.Variable(self.u, dtype=tf.float32)
                        loss = self._get_losses1(x, y, u)
                        
                    if self.iterations % 50 == 0:
                        self.tot_loss.append(loss.numpy())
                        self.epsilon_array.append(self.epsilon.numpy()[0])
                        self.delta_array.append(self.delta.numpy()[0])
                        self.gamma_array.append(self.gamma.numpy()[0])
                        self.u_m_array.append(self.u_mean_var.numpy()[0])
                        self.u_s_array.append(self.u_scale_var.numpy()[0])
                        self.loss_array.append(loss.numpy())
                        self.loss_u_array.append(loss_u.numpy())
                        self.loss_pde_u_array.append(loss_pde_u.numpy())
                        
                    self.callbacks.on_batch_end(self.epochs)
                    if self.iterations == max_iterations:
                        self.iterations = -2
                        break
                        
                    self.weights_model_u = self.model_u.weights
                    if self.iterations % 200 == 0:
                        tf.print(
                            "It: %d, Epoch: %d, Total loss: %e, Data Loss U %e, Loss_pde_u:%e, Learning rate: %2e, epsilon:%e, delta:%e, gamma:%e, u_scale:%e, u_mean:%e, weight_pde_u:%e"
                            % (
                                self.iterations,
                                self.epochs,
                                loss,
                                loss_u,
                                loss_pde_u,
                                self.optimizer.lr,
                                self.epsilon,
                                self.delta,
                                self.gamma,
                                self.u_scale_var,
                                self.u_mean_var,
                                self.weight_pde_u,
                            )
                        )
                    if self.iterations % 2000 == 0:
                        u_pred = np.reshape(self.model_u(tf.concat([self.X, self.Y], 1)), (self.n, self.n))
                        plt.imshow(u_pred, cmap=plt.cm.RdBu)
                        plt.colorbar(label = 'Amplitude')
                        plt.show()
                    self.iterations += 1
                    if loss_u < self.tol:   
                        break
            self.epochs += 1
            
        self.parameters = [
            self.epsilon_array,
            self.delta_array,
            self.gamma_array,
            self.u_m_array,
            self.u_s_array,
        ]
        self.final_parameters = [
            self.epsilon_array[-1],
            self.delta_array[-1],
            self.gamma_array[-1],
            self.u_m_array[-1],
            self.u_s_array[-1],
        ]