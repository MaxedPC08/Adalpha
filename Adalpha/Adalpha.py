import tensorflow as tf
from keras.optimizers import Optimizer
import numpy as np
import matplotlib.pyplot as plt

class AdalphaBase(Optimizer):
    r"""Base class - do not use (yet)
    """

    def __init__(
            self,
            learning_rate=0.001,
            adjustment_exp=1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False,
            weight_decay=None,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="Adam",
            ema_w=0.9,
            change=1,
            *args,
            **kwargs
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.chaos_punish = adjustment_exp
        self.std = 1.0
        self.loss = 1
        self.ema_w = ema_w
        self.change = change
        self.a = 1
        self.b = 1

    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
          var_list: list of model variables to build Adam variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_loss(self, loss: float):
        self.a = self.ema_w * loss + (1 - self.ema_w) * self.a
        self.b = (1 - self.ema_w) * loss + self.ema_w * self.b
        new_std=(self.change * self.a) / self.b
        self.std = new_std

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        alpha = lr * (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)) * (
                    1 - self.std * self.chaos_punish) ** self.chaos_punish

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config


class Adalpha(AdalphaBase):
    """
    Optimizer for Tensorflow Keras based on the Adam optimizer. This version implements two changes:
    1: Adalpha adjusts the alpha value based on the value passed in through the update_loss method.
    This method is typically implemented in a callback at the end of each batch. Alpha is multiplied by
                            L**adjustment_exp
    adjustment_exp is a value passed into the optimizer on initiation. L is the value passed in through update_loss,
    and should never exceed 1.

    2: Adalpha adjusts the momentum and velocity of all weights using the function
    out = m * (1.91 - (m**2-(0.01*(|mean(m)| + std(m)))/(m**2 + 0.1 * (|mean(m)| + std(m))**2)))
    """
    def __init__(self, *args, **kwargs):
        """
        Initiator function
        :return: None
        """
        super().__init__(*args, **kwargs)

    def _m_activ(self, m):
        """Activate the momentum and velocity of Adam to increase the convergence of low momentum weights.
        :praram m: the value being activated, any Tensorflow.math compatible Tensorflow Tensor
        :return: the activated value - Tensorflow Tensor of same input type
        """
        return m * tf.pow(2 - tf.math.divide_no_nan((tf.square(m) - tf.square(0.01 * tf.abs(tf.abs(tf.reduce_mean(m)) -
                                                                                            tf.math.reduce_std(m)))),
                                                 (tf.square(m) + 0.1 * tf.square(tf.abs(tf.reduce_mean(m)) -
                                                                                 tf.math.reduce_std(m)))), 2)

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]
        alpha = lr * (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)) * (self.std) ** self.chaos_punish

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(self._m_activ(-m * (1 - self.beta_1)))
            m.scatter_add(
                tf.IndexedSlices(
                    self._m_activ(gradient.values * (1 - self.beta_1)), gradient.indices
                )
            )
            v.assign_add(self._m_activ(-v * (1 - self.beta_2)))
            v.scatter_add(
                tf.IndexedSlices(
                    self._m_activ(tf.square(gradient.values) * (1 - self.beta_2)),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            m.assign_add(self._m_activ((gradient - m) * (1 - self.beta_1)))
            v.assign_add(self._m_activ((tf.square(gradient) - v) * (1 - self.beta_2)))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))


class AdalphaCallback(tf.keras.callbacks.Callback):
    """A class that updates the loss of the Max_Adam optimizer.
    Uses a ratio of two weighted exponential moving averages of the loss of the model.
    Experimental"""

    def __init__(self, optimizer: Adalpha):
        super().__init__()
        self.optimizer = optimizer

    def on_train_batch_end(self, batch, logs=None):
        loss = logs["loss"]
        self.optimizer.update_loss(loss)

class AdalphaPlot(AdalphaCallback):
    def __init__(self, optimizer: Adalpha):
        super().__init__(optimizer)
        self.stds = [0]

    def _calculate_loss_std(self, loss):
        self.a = self.optimizer.ema_w * loss + (1-self.optimizer.ema_w) * self.optimizer.a
        self.b = (1 - self.optimizer.ema_w) * loss + self.optimizer.ema_w * self.optimizer.b
        self.stds.append(self.optimizer.learning_rate * (self.optimizer.change * self.a)/self.b)

    def on_train_end(self, logs=None):
        loss = logs["loss"]
        self.optimizer.update_loss(loss)
        self._calculate_loss_std(loss)

        plt.clf()
        plt.title(f"Adalpha Alpha, ema_w = {self.optimizer.ema_w}, change = {self.optimizer.change}")
        plt.plot(self.stds, "r-", label="Adalpha Alpha")
        plt.xlabel("Batch")
        plt.ylabel("Alpha")
        plt.legend()
        plt.show()
        print(f"Ema_w = {self.optimizer.ema_w}, Change = {self.optimizer.change}, r mean = {np.mean(self.stds)/self.optimizer.learning_rate}")
