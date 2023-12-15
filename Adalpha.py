import tensorflow as tf
from keras.optimizers import Optimizer
import numpy as np
import matplotlib.pyplot as plt



class MaxAdam(Optimizer):
    r"""
    Base class for the Adalpha optimizer. Do not use
    """

    def __init__(
            self,
            learning_rate=0.001,  # The initial learning rate.
            chaos_punishment=1,  # The chaos punishment parameter.
            alpha_ema_w=0.9,  # The alpha for the exponential moving average of the loss.
            beta_1=0.9,  # The exponential decay rate for the first moment estimates.
            beta_2=0.999,  # The exponential decay rate for the second moment estimates.
            epsilon=1e-7,  # A constant epsilon used to improve numerical stability.
            amsgrad=False,  # Whether to apply the AMSGrad variant of this algorithm.
            weight_decay=None,  # A constant multiplier applied to the gradient to reduce the weight of unimportant factors.
            clipnorm=None,  # A maximum norm for the gradients.
            clipvalue=None,  # A minimum value for the gradients.
            global_clipnorm=None,  # A maximum norm for all the gradients.
            use_ema=False,  # Whether to use the exponential moving average of the parameters.
            ema_momentum=0.99,  # The momentum for the exponential moving average of the parameters.
            ema_overwrite_frequency=None,  # The frequency with which to overwrite the EMA parameters.
            jit_compile=True,  # Whether to use just-in-time compilation.
            name="Adalpa",  # The name of the optimizer.
            **kwargs  # Additional keyword arguments.
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
        self.chaos_punish = chaos_punishment
        self.std = 0.0
        self.alpha_ema_w = alpha_ema_w

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

    def update_loss(self, new_std: float):
        """
        Set the self.std to the exponential moving average of itself and new_std
        """
        self.std = self.alpha_ema_w*new_std+(1-self.alpha_ema_w)*self.std

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


class AdAlpha_Momentum(MaxAdam):
    """
    Optimizer for Tensorflow Keras based on the Adam optimizer. This version implements two changes:
    1: Adalpha adjusts the alpha value based on the value passed in through the update_loss method.
    This method is typically implemented in a callback at the end of each batch. Alpha is multiplied by
                            (1-L*chaos_punishment)**chaos_punishment
    chaos_punishment is a value passed into the optimizer on initiation. L is the value passed in through update_loss,
    and should never exceed 1.

    2: Adalpha adjusts the momentum and velocity of all weights using the function
    out = m * (1.91 - (m**2-(0.01*(|mean(m)| + std(m)))/(m**2 + 0.1 * (|mean(m)| + std(m))**2)))
    """
    def __init__(self, **kwargs):
        """
        Initiator function
        :return: None
        """
        super().__init__(**kwargs)

    def _m_activ(self, m):
        """Activate the momentum and velocity of Adam to increase the convergence of low momentum weights.
        :praram m: the value being activated, any Tensorflow.math compatible Tensorflow Tensor
        :return: the activated value - Tensorflow Tensor of same input type
        """
        return m * tf.pow(2 - tf.math.divide_no_nan((tf.square(m) - tf.square(0.01 * tf.abs(tf.abs(tf.reduce_mean(m)) - tf.math.reduce_std(m)))),
                                                 (tf.square(m) + 0.1 * tf.square(tf.abs(tf.reduce_mean(m)) - tf.math.reduce_std(m)))), 2)

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
                    1 - self.std) ** self.chaos_punish

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


class MaxAdamCallback(tf.keras.callbacks.Callback):
    # A class that updates the loss of the Max_Adam optimizer

    def __init__(self, optimizer: MaxAdam):
        """
        Initialize the MaxAdamCallback class.

        Args:
            optimizer (MaxAdam): The MaxAdam optimizer to update the loss of.
        """
        super().__init__()
        self.optimizer = optimizer
        self.stds = [0]
        self.losses = []
        self.std = 0.0

    def _calculate_loss_std(self):
        """
        Update the alpha multiplier of the optimizer and save for plotting.
        """
        std = np.cbrt(
            np.divide(np.std(self.losses) / np.mean(self.losses) - self.std, self.std, out=np.zeros_like(self.std),
                      where=self.std != 0))
        self.std = np.std(self.losses) / np.mean(self.losses)
        self.stds.append(self.optimizer.alpha_ema_w * (self.optimizer.lr * (1 - std) ** self.optimizer.chaos_punish) + (
                    1 - self.optimizer.alpha_ema_w) * self.stds[-1])
        self.optimizer.update_loss(std)

    def on_train_end(self, logs=None):
        """
        Plot graphs at the end of training.
        """
        plt.clf()
        plt.plot(self.stds, "r-", label="adalpha learning rate")
        plt.legend()
        plt.show()

class Adalpha_Plot(MaxAdamCallback):
    """
    Similar to Adalpha_Callback in math but it plots the multiplier at the end of training.
    """
    def __init__(self, optimizer: MaxAdam):

        super().__init__(optimizer)
        self.stds = [0]

    def _calculate_loss_std(self):
        """
        Update the alpha multiplier of the optimizer and save for plotting.
        """
        std = np.cbrt(np.divide(np.std(self.losses) / np.mean(self.losses) - self.std, self.std, out=np.zeros_like(self.std),
                        where=self.std != 0))
        self.std = np.std(self.losses) / np.mean(self.losses)
        self.stds.append(self.optimizer.alpha_ema_w*(self.optimizer.lr * (1 - std) ** self.optimizer.chaos_punish) + (1-self.optimizer.alpha_ema_w)*self.stds[-1])
        self.optimizer.update_loss(std)


    def on_train_end(self, logs=None):
        """
        Plot graphs at the end of training.
        """
        plt.clf()
        plt.plot(self.stds, "r-", label="adalpha learning rate")
        plt.legend()
        plt.show()



class OneCallback(tf.keras.callbacks.Callback):
    """A class that updates the loss of the Max_Adam optimizer"""

    def __init__(self, optimizer: MaxAdam, num_to_hold):
        """Initiator
        :param optimizer: The optimizer to update the losses in. Must be an initiated optimizer object.
        :param num_to_hold: Made to keep signature of other callbacks. Has no effect"""
        super().__init__()
        self.optimizer = optimizer
        self.losses = []
        self.hold = num_to_hold
        self.std = 0.0


