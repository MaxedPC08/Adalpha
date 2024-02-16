Adalpha
=======

Have you ever wondered "Is there a better optimizer than Adam that helps avoid model collapse?"
The answer is yes! Adalpha is an optimizer designed to help avoid model collapse and improve generalization, and it is very closely based on Adam. 
It calculates the change in the loss over an arbitrary amount of time, and updates alpha accordingly, thus the name. 
This scaling and moving averages are controlled is controlled by the parameter beta3 and beta4. 
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
import Adalpha as aa
my_optimizer = aa.Adalpha()


# Train with Adalpha
callbacks = [aa.AdalphaCallback(my_optimizer)]
model.compile(optimizer=my_optimizer, loss="mse")
history = model.fit(x_data, y_data, callbacks=callbacks)
```
Additionally, you can use the optimizer's `set_loss` function to set the loss manually, like so:
```python
import Adalpha as aa
my_optimizer = aa.Adalpha()
model.compile(optimizer=my_optimizer, loss="mse")
model.train_on_batch(x_data, y_data)
loss = model.loss(y_data, model(x_data))
my_optimizer.set_loss(loss)
```
This is useful if you want to use Adalpha with a custom training loop, or if you want to use a different loss for training and validation.

---
**Parameters**
The Adalpha optim