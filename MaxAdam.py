import tensorflow as tf
from keras.optimizers import Optimizer
from keras.src.saving.object_registration import register_keras_serializable
from tensorflow.python.util.tf_export import keras_export
import numpy as np
import scipy as sp


@keras_export(
    "keras.optimizers.Adam",
    "keras.optimizers.experimental.Adam",
    "keras.dtensor.experimental.optimizers.Adam",
    v1=[],
)
class MaxAdam(Optimizer):
    r"""Optimizer that implements the Adam algorithm.

    Adam optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments.

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Args:
      learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to `0.001`.
      beta_1: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 1st moment estimates. Defaults to `0.9`.
      beta_2: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 2nd moment estimates. Defaults to
        `0.999`.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        `1e-7`.
      amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond". Defaults to `False`.
      {{base_optimizer_keyword_args}}

    Reference:
      - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
      - [Reddi et al., 2018](
          https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The default value of 1e-7 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since Adam uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    def __init__(
            self,
            learning_rate=0.001,
            chaos_punishment=1,
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
        self.chaos_punish = chaos_punishment
        self.std = 0.0

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
    """A class that updates the loss of the Max_Adam optimizer"""

    def __init__(self, optimizer: MaxAdam, num_to_hold):
        super().__init__()
        self.optimizer = optimizer
        self.losses = []
        self.hold = num_to_hold
        self.std = 0.0

    def _calculate_loss_std(self):
        std = np.divide(np.std(self.losses) / np.mean(self.losses) - self.std, self.std, out=np.zeros_like(self.std),
                        where=self.std != 0)
        self.std = np.std(self.losses) / np.mean(self.losses) - self.std
        self.optimizer.update_loss(std)

    def on_train_batch_end(self, batch, logs=None):
        self.losses.append(logs["loss"])
        self.losses = self.losses[-self.hold:]
        self._calculate_loss_std()


class LossSlopeCallback(tf.keras.callbacks.Callback):
    """A class that updates the loss of the Max_Adam optimizer"""

    def __init__(self, optimizer: MaxAdam, num_to_hold):
        super().__init__()
        self.optimizer = optimizer
        self.losses = []
        self.hold = num_to_hold
        self.std = 0.0
        self.r2 = 0.0

    def _calculate_loss_std(self):
        val, _, _, r2, _ = sp.stats.linregress(np.arange(0, len(self.losses)), np.asarray(self.losses))
        r2 = np.divide(r2, np.mean(self.losses), out=np.zeros_like(r2), where=r2 != 0)
        pc_r2 = np.divide(r2 - self.r2, self.r2, out=np.zeros_like(self.r2), where=self.r2 != 0)
        self.r2 = r2
        self.optimizer.update_loss(-pc_r2)

    def on_train_batch_end(self, batch, logs=None):
        self.losses.append(logs["loss"])
        self.losses = self.losses[-self.hold:]
        self._calculate_loss_std()


class OneCallback(tf.keras.callbacks.Callback):
    """A class that updates the loss of the Max_Adam optimizer"""

    def __init__(self, optimizer: MaxAdam, num_to_hold):
        super().__init__()
        self.optimizer = optimizer
        self.losses = []
        self.hold = num_to_hold
        self.std = 0.0

    def _calculate_loss_std(self):
        self.optimizer.update_loss(0.0)

    def on_train_batch_end(self, batch, logs=None):
        self.losses.append(logs["loss"])
        self.losses = self.losses[-self.hold:]
        self._calculate_loss_std()
