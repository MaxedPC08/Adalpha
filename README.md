Adalpha
=======

Have you ever wondered "Is there a better optimizer than Adam that helps avoid model collapse?"
The answer is yes! Adalpha is an optimizer designed to help avoid model collapse and improve generalization, and it is very closely based on Adam. 
It calculates the change in the loss over an arbitrary amount of time, and updates alpha accordingly, thus the name. 
This scaling and moving averages are controlled is controlled by the parameter `ema_w`. 
Aside from simply adjusting the learning rate, Adalpha gently boosts the momentum and velocity of weights with lower than average momentum and velocity, theoretically encouraging them to learn.
The result is an optimizer that is able to adapt the learning rate to the local geometry of the loss landscape, 
which helps avoid model collapse and improve generalization.
Adalpha is a nearly drop-in replacement for Adam, and is implemented in TensorFlow >= 2.14.

## Usage

To use Adalpha, simply replace `tf.keras.optimizers.Adam` with `adalpha.Adalpha` in your code. 
The default parameters are the same as Adam, so you can use it as a drop-in replacement.
It adds a few parameters, but these are not completely necessary for most use cases.
You do need to pass the model's loss into the optimizer, as Adalpha needs the loss to calculate the change in the loss over time (obviously).
This is a very simple task, and can be done either with a custom callback included in the package or via one of the optimizer's functions.

To use the callback, simply add `adalpha.LossCallback()` to your model's `fit` function, like so:

```python

from Adalpha import Adalpha as aa

my_optimizer = aa.Adalpha()

# Train with Adalpha
callbacks = [aa.AdalphaCallback(my_optimizer)]
model.compile(optimizer=my_optimizer, loss="mse")
history = model.fit(x_data, y_data, callbacks=callbacks)
```
Additionally, you can use the optimizer's `set_loss` function to set the loss manually, like so:

```python

from Adalpha import Adalpha as aa

my_optimizer = aa.Adalpha()
model.compile(optimizer=my_optimizer, loss="mse")
model.train_on_batch(x_data, y_data)
loss = model.loss(y_data, model(x_data))
my_optimizer.set_loss(loss)
```
This is useful if you want to use Adalpha with a custom training loop, or if you want to use a different loss for training and validation.

---
**Parameters**

The Adalpha optimizer has a few parameters that can be adjusted to fit your needs.
The first parameter, `adjustment_exp`, is the exponent used to adjust the learning rate. A value of 1 or 2 usually is sufficient.

The second parameter, `ema_w` is the weight of the exponential moving average of the change in the loss. A value between 0.5 and 1 is required, while a value of 0.9 usually sufficient.
This parameter mostly adjusts the speed at which the learning rate is adjusted.

The third parameter, `change`, is a multiplier on the learning rate that helps it maintaing its normal values. It was designed to be the metric for the change in the loss required for the 
learning rate to be adjusted, which it technically is, but it is not very useful in practice. A value of 1 (meaning the learning rate does not change) is usually sufficient. In testing, this parameter has not been very useful.

---
**Callbacks**

There are two callbacks included in the package. The first, `AdalphaCallback`, is the main callback that is used to update the learning rate.
It simply updates the learning rate after each batch, and is the main callback that should be used when training with Adalpha.
The second callback is the `AdalphaPlot` callback, which is used to plot the change in the loss over time. This is useful for debugging and understanding how Adalpha works, but is not necessary for training.

Installing
----------

Installing Adalpha is simple. Just clone the repository into `site-packages`, then cd into the `Adalpha` folder and run `pip install .` in the root directory.
