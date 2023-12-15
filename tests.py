from typing import Tuple, Any

from numpy import ndarray, dtype, generic

from tests_core import *

FILE_NAME = "SeoulBikeData.csv"
SPLIT = 0.67

def bike_test(
    callback: callable,  # the chaos callback function
    optimizer: callable,  # the chaos optimizer function
    epochs: int = 100,  # the number of epochs to train for
    learning_rate: float = 0.01,  # the learning rate for the optimizer
    ema_w: float = 0.9,  # the exponential moving average weight for the chaos parameter
    chaos_punishment: int = 7,  # the chaos punishment value for the optimizer
) -> tuple[ndarray[Any, dtype[generic | generic | Any]], float]:
    """
    This function trains two different models on the bike dataset and compares their predictions.

    Args:
        callback: the chaos callback function
        optimizer: the chaos optimizer function
        epochs: the number of epochs to train for
        learning_rate: the learning rate for the optimizer
        ema_w: the exponential moving average weight for the chaos parameter
        chaos_punishment: the chaos punishment value for the optimizer

    Returns:
        A tuple containing the r-squared scores for the Adalpha and Adam models, as well as the predicted and actual values for the bike dataset.
    """
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Fitting Results at lr={learning_rate} on Bike Data")
    adalpha_r_2, adalpha_y_pred, adalpha_y_test = adalpha_train_bike(
        callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate, chaos_punishment=chaos_punishment, ema_w=ema_w)
    r_2, y_pred, y_test = adam_train_bike(
        epochs, learning_rate)
    plt.legend()
    plt.show()

    plt.title(f"Predictions vs Actual at lr={learning_rate}\nOver {epochs} epochs")
    plt.xlabel("Training Data")
    plt.ylabel("Model Data")
    plt.plot(y_test, y_pred, "g.", label="Adam Predictions")
    plt.plot(adalpha_y_test, adalpha_y_pred, "r.", label="Adalpha Predictions")
    plt.plot([0, 3500], [0, 3500], "b-", label="Perfect Precictions")
    plt.legend()
    plt.show()
    print(f"Adalpha r squared score: {adalpha_r_2}\nAdam r squared score: {r_2}")
    return adalpha_r_2, r_2

def bike_multiple_test(
    callback: callable,  # the chaos callback function
    optimizer: callable,  # the chaos optimizer function
    epochs: int = 100,  # the number of epochs to train for
    learning_rate: float = 0.01,  # the learning rate for the optimizer
    ema_w: float = 0.9,  # the exponential moving average weight for the chaos parameter
    chaos_punishment: int = 7,  # the chaos punishment value for the optimizer
    tests: int = 10,  # the number of tests to run
    copy: bool = False,  # whether or not to copy the results to the clipboard
) -> None:
    """
    This function trains two different models on the bike dataset and compares their predictions.

    Args:
        callback: the chaos callback function
        optimizer: the chaos optimizer function
        epochs: the number of epochs to train for
        learning_rate: the learning rate for the optimizer
        ema_w: the exponential moving average weight for the chaos parameter
        chaos_punishment: the chaos punishment value for the optimizer
        tests: the number of tests to run
        copy: whether or not to copy the results to the clipboard

    Returns:
        None
    """
    losses = []

    for i in range(tests):
        adalpha_r_2, r_2 = bike_test(
            callback, optimizer, epochs, learning_rate, chaos_punishment)
        losses.append((adalpha_r_2, r_2))

    if copy:
        pd.DataFrame(losses).to_clipboard(excel=True)
    print(np.asarray(losses))

def mnist_test(
    callback: callable,  # the chaos callback function
    optimizer: callable,  # the chaos optimizer function
    epochs: int = 5,  # the number of epochs to train for
    learning_rate: float = 0.01,  # the learning rate for the optimizer
    ema_w: float = 0.9,  # the exponential moving average weight for the chaos parameter
    chaos_punishment: int = 2,  # the chaos punishment value for the optimizer
) -> tuple[float, ndarray[Any, dtype[generic | generic | Any]], ndarray[Any, dtype[generic | generic | Any]]]:
    """
    This function trains two different models on the MNIST dataset and compares their predictions.

    Args:
        callback: the chaos callback function
        optimizer: the chaos optimizer function
        epochs: the number of epochs to train for
        learning_rate: the learning rate for the optimizer
        ema_w: the exponential moving average weight for the chaos parameter
        chaos_punishment: the chaos punishment value for the optimizer

    Returns:
        A tuple containing the accuracy scores for the Adalpha and Adam models, as well as the predicted and actual values for the MNIST dataset.
    """
    adalpha_acc, adalpha_y_pred, adalpha_y_test = adalpha_train_mnist(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate, chaos_punishment=chaos_punishment, ema_w=ema_w)
    acc, y_pred, y_test = adam_train_mnist(epochs, learning_rate)
    plt.show()
    # ====================
    # USE MODEL TO PREDICT and create a scatterplot of the y and y_pred
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, (ax1, ax2) = plt.subplots(1, 2)

    heatmap_max = make_heatmap(np.argmax(adalpha_y_pred, 1), adalpha_y_test)

    im = ax1.imshow(heatmap_max)

    # Show all ticks and label them with the respective list entries
    ax1.set_xticks(np.arange(len(labels)), labels=labels)
    ax1.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.


    ax1.set_title(f"Predictions vs Actual at\nlr={learning_rate} from Adalpha")

    heatmap = make_heatmap(np.argmax(y_pred, 1), y_test)

    im = ax2.imshow(heatmap)

    # Show all ticks and label them with the respective list entries
    ax2.set_xticks(np.arange(len(labels)), labels=labels)
    ax2.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax2.set_title(f"Predictions vs Actual at\nlr={learning_rate} from Adam\nOver {epochs} epochs")
    fig.tight_layout()
    plt.show()
    return adalpha_acc, acc
    
def mnist_multiple_test(
    callback: callable,  # the chaos callback function
    optimizer: callable,  # the chaos optimizer function
    epochs: int = 10,  # the number of epochs to train for
    learning_rate: float = 0.01,  # the learning rate for the optimizer
    ema_w: float = 0.9,  # the exponential moving average weight for the chaos parameter
    chaos_punishment: int = 4,  # the chaos punishment value for the optimizer
    tests: int = 10,  # the number of tests to run
    copy: bool = False,  # whether or not to copy the results to the clipboard
) -> None:
    """
    This function trains two different models on the MNIST dataset and compares their predictions.

    Args:
        callback: the chaos callback function
        optimizer: the chaos optimizer function
        epochs: the number of epochs to train for
        learning_rate: the learning rate for the optimizer
        ema_w: the exponential moving average weight for the chaos parameter
        chaos_punishment: the chaos punishment value for the optimizer
        tests: the number of tests to run
        copy: whether or not to copy the results to the clipboard

    Returns:
        None
    """
    losses = []

    for i in range(tests):
        adalpha_r_2, r_2 = bike_test(
            callback, optimizer, epochs, learning_rate, chaos_punishment)
        losses.append((adalpha_r_2, r_2))

    if copy:
        pd.DataFrame(losses).to_clipboard(excel=True)
    print(np.asarray(losses))


def bike_chaos_test(
    callback: callable,  # the chaos callback function
    optimizer: callable,  # the chaos optimizer function
    epochs: int = 50,  # the number of epochs to train for
    learning_rate: float = 0.01,  # the learning rate for the optimizer
    ema_w: float = 0.9,  # the exponential moving average weight for the chaos parameter
    chaos_punishment: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # the chaos punishment values to test
) -> None:
    """
    This function tests the effect of chaos punishment values on the loss of an Adalpha model trained on the bike dataset.

    Args:
        callback: the callback
        optimizer: the optimizer
        epochs: the number of epochs to train for
        learning_rate: the learning rate for the optimizer
        ema_w: the exponential moving average weight
        chaos_punishment: the chaos punishment values to test

    Returns:
        None
    """
    adalpha_r_2 = []

    for val in chaos_punishment:
        adalpha_r_2.append(adalpha_train_bike(
            callback, optimizer, epochs, learning_rate, chaos_punishment=val, ema_w=ema_w)[0])

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("Chaos Punishment")
    plt.ylabel("Loss")
    plt.title(f"R Squared vs Chaos Punishment\nOver {epochs} epochs")
    plt.plot(chaos_punishment, adalpha_r_2, "r-", label="Adalpha R2")
    plt.legend()
    plt.grid(True)
    plt.show()

def mnist_chaos_test(
    callback: callable,  # the chaos callback function
    optimizer: callable,  # the chaos optimizer function
    epochs: int = 2,  # the number of epochs to train for
    learning_rate: float = 0.01,  # the learning rate for the optimizer
    ema_w: float = 0.9,  # the exponential moving average weight for the chaos parameter
    chaos_punishment: list[float] = [0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # the chaos punishment values to test
) -> None:
    """
    This function tests the effect of chaos punishment values on the loss of an Adalpha model trained on the MNIST dataset.

    Args:
        callback: the chaos callback function
        optimizer: the chaos optimizer function
        epochs: the number of epochs to train for
        learning_rate: the learning rate for the optimizer
        ema_w: the exponential moving average weight
        chaos_punishment: the chaos punishment values to test

    Returns:
        None
    """
    adalpha_acc = []

    for val in chaos_punishment:
        adalpha_acc.append(adalpha_train_mnist(
            callback, optimizer, epochs, learning_rate, chaos_punishment=val, ema_w=ema_w)[0])

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("Chaos Punishment")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Chaos Punishment\nOver {epochs} epochs")
    plt.plot(chaos_punishment, adalpha_acc, "r-", label="Adalpha Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def cifar_test(callback, optimizer, epochs=10, learning_rate=0.01, ema_w=0.9, chaos_punishment=6):
    """
    This function tests the effect of chaos punishment values on the loss of an Adalpha model trained on the CIFAR10 dataset.

    Args:
        callback: the callback
        optimizer: the optimizer
        epochs: the number of epochs to train for
        learning_rate: the learning rate for the optimizer
        ema_w: the exponential moving average weight
        chaos_punishment: the chaos punishment value
    Returns:
        None
    """
    (x_data, y_data), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    input_layer = tf.keras.layers.Input((32, 32, 3))
    model = tf.keras.layers.BatchNormalization()(input_layer)
    model = tf.keras.layers.Conv2D(64, (5, 5), activation="relu")(model)
    model = tf.keras.layers.MaxPool2D(2)(model)
    model = tf.keras.layers.Conv2D(64, (5, 5), activation="relu")(model)
    model = tf.keras.layers.Conv2D(32, (5, 5), activation="relu")(model)
    model = tf.keras.layers.Flatten()(model)
    model = tf.keras.layers.Dense(512, activation="relu")(model)
    output = tf.keras.layers.Dense(10, activation="softmax")(model)

    model = tf.keras.Model(input_layer, output)

    my_optimizer = optimizer(learning_rate=learning_rate, chaos_punishment=chaos_punishment)

    # Train with Adalpha
    callbacks = [callback(my_optimizer, ema_w)]
    model.compile(optimizer=my_optimizer, loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=2048, callbacks=callbacks, validation_split=0.2,
                        verbose=False)
    # Graphing the Adalpha Results
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Fitting Results at lr={learning_rate} on CIFAR")
    plt.plot(history.history["loss"], "r-", label="Adalpha Loss")
    plt.plot(history.history["val_loss"], "y-", label="Adalpha Val Loss")
    adalpha_y_pred = model.predict(x_test, verbose=False)
    print("Evaluating AdAlpha")
    model.evaluate(x_test, y_test)
    # Train with Adam
    model = tf.keras.models.clone_model(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=2048, validation_split=0.2, verbose=False)
    plt.plot(history.history["loss"], "g-", label="Adam Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adam Val Loss")
    plt.legend()
    plt.show()
    y_pred = model.predict(x_test, verbose=False)
    print("Evaluating Adam")
    model.evaluate(x_test, y_test)
    # ====================
    # USE MODEL TO PREDICT and create a scatterplot of the y and y_pred
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, (ax1, ax2) = plt.subplots(1, 2)

    heatmap_max = make_heatmap(np.argmax(adalpha_y_pred, 1), y_test)

    im = ax1.imshow(heatmap_max)

    # Show all ticks and label them with the respective list entries
    ax1.set_xticks(np.arange(len(labels)), labels=labels)
    ax1.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), ha="right", rotation_mode="anchor")

    ax1.set_title(f"Predictions vs Actual at\nlr={learning_rate} from Adalpha")

    heatmap = make_heatmap(np.argmax(y_pred, 1), y_test)

    im = ax2.imshow(heatmap)

    # Show all ticks and label them with the respective list entries
    ax2.set_xticks(np.arange(len(labels)), labels=labels)
    ax2.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax2.set_title(f"Predictions vs Actual at\nlr={learning_rate} from Adam\nOver {epochs} epochs")
    fig.tight_layout()
    plt.legend()
    plt.show()
