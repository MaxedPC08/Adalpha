import matplotlib.pyplot as plt
from tests_core import *
from utils import *

FILE_NAME = "SeoulBikeData.csv"
SPLIT = 0.67

def bike_test(callback, optimizer, epochs=100, learning_rate=0.01, ema_w=0.9, change=0.99, chaos_punishment=7):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Fitting Results at lr={learning_rate} on Bike Data")
    adalpha_r_2, adalpha_y_pred, adalpha_y_test = adalpha_train_bike(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate, chaos_punishment=chaos_punishment, ema_w=ema_w, change=change)
    r_2, y_pred, y_test= adam_train_bike(epochs, learning_rate)
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

def bike_multiple_test(callback, optimizer, epochs=100, learning_rate=0.01, ema_w=0.9, change=0.99,  chaos_punishment=6, tests=10, copy=False):
    """
    Performs multiple tests on the bike data using the given parameters.

    Parameters:
        callback (function): The chaos callback function to use.
        optimizer (function): The chaos optimizer function to use.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use.
        ema_w (float): The EMA weight to use.
        change (float): The change value to use.
        chaos_punishment (int): The chaos punishment value to use.
        tests (int): The number of tests to run.
        copy (bool): Whether to copy the results to the clipboard in Excel format.

    Returns:
        None
    """
    losses = []
    for i in range(tests):
        losses.append(
            bike_test(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate, ema_w=ema_w, change=change, chaos_punishment=chaos_punishment))
    if copy:
        pd.DataFrame(losses).to_clipboard(excel=True)
    print(np.asarray(losses))

def adam_train_mnist(epochs=10, learning_rate=0.01):
    """
    Trains a neural network on the MNIST dataset using Adam optimization and Adalpha optimization.

    Parameters:
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use.

    Returns:
        A tuple containing the accuracy of the Adalpha model, the predictions from the Adalpha model, and the true labels for the test set.
    """
    (x_data, y_data), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_data = np.expand_dims(x_data, 3)
    x_test = np.expand_dims(x_test, 3)

    input_layer = tf.keras.layers.Input((28, 28, 1))
    parallel_1 = tf.keras.layers.Conv2D(32, (7, 7), activation="tanh")(input_layer)
    parallel_1 = tf.keras.layers.MaxPool2D(2)(parallel_1)
    parallel_1 = tf.keras.layers.Conv2D(32, (7, 7), activation="tanh")(parallel_1)
    parallel_1 = tf.keras.layers.Conv2D(32, (5, 5), activation="tanh")(parallel_1)
    parallel_1 = tf.keras.layers.Reshape((32,))(parallel_1)
    output = tf.keras.layers.Dense(10, activation="softmax")(parallel_1)

    model = tf.keras.Model(input_layer, output)

    # Train with Adalpha
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=128, validation_split=0.2,
                        verbose=False)
    # Graphing the Adalpha Results
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Fitting Results at lr={learning_rate} on MNIST")
    plt.plot(history.history["loss"], "r-", label="Adam Loss")
    plt.plot(history.history["val_loss"], "y-", label="Adam Val Loss")
    plt.legend()
    y_pred = model.predict(x_test, verbose=False)
    print("Evaluating Adam")
    return model.evaluate(x_test, y_test)[1], y_pred, y_test

def adalpha_train_mnist(callback, optimizer, epochs=10, learning_rate=0.01, ema_w=0.9, change=0.99, chaos_punishment=2):
    """
    Trains a neural network on the MNIST dataset using Adalpha optimization.

    Parameters:
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use.
        ema_w (float): The EMA weight to use.
        change (float): The change value to use.
        chaos_punishment (int): The chaos punishment value to use.

    Returns:
        A tuple containing the accuracy of the Adalpha model, the predictions from the Adalpha model, and the true labels for the test set.
    """
    (x_data, y_data), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_data = np.expand_dims(x_data, 3)
    x_test = np.expand_dims(x_test, 3)

    input_layer = tf.keras.layers.Input((28, 28, 1))
    parallel_1 = tf.keras.layers.Conv2D(32, (7, 7), activation="tanh")(input_layer)
    parallel_1 = tf.keras.layers.MaxPool2D(2)(parallel_1)
    parallel_1 = tf.keras.layers.Conv2D(32, (7, 7), activation="tanh")(parallel_1)
    parallel_1 = tf.keras.layers.Conv2D(32, (5, 5), activation="tanh")(parallel_1)
    parallel_1 = tf.keras.layers.Reshape((32,))(parallel_1)
    output = tf.keras.layers.Dense(10, activation="softmax")(parallel_1)

    model = tf.keras.Model(input_layer, output)

    my_optimizer = optimizer(learning_rate=learning_rate, chaos_punishment=chaos_punishment)

    # Train with Adalpha
    callbacks = [callback(my_optimizer, ema_w, change)]
    model.compile(optimizer=my_optimizer, loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=128, callbacks=callbacks, validation_split=0.2,
                        verbose=False)
    # Graphing the Adalpha Results
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Fitting Results at lr={learning_rate} on MNIST")
    plt.plot(history.history["loss"], "b-", label="Adalpha Loss")
    plt.plot(history.history["val_loss"], "g-", label="Adalpha Val Loss")
    plt.legend()
    adalpha_y_pred = model.predict(x_test, verbose=False)
    print("Evaluating Adalpha")
    return model.evaluate(x_test, y_test)[1], adalpha_y_pred, y_test

def mnist_test(callback, optimizer, epochs=5, learning_rate=0.01, ema_w=0.9, change=0.99, chaos_punishment=2):
    """
    Trains a neural network on the MNIST dataset using Adalpha optimization.

    Parameters:
        callback (function): The chaos callback function to use.
        optimizer (function): The chaos optimizer function to use.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use.
        ema_w (float): The EMA weight to use.
        change (float): The change value to use.
        chaos_punishment (int): The chaos punishment value to use.

    Returns:
        A tuple containing the accuracy of the Adalpha model, the predictions from the Adalpha model, and the true labels for the test set.
    """

    adalpha_acc, adalpha_y_pred, adalpha_y_test = adalpha_train_mnist(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate, chaos_punishment=chaos_punishment, ema_w=ema_w, change=change)
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

def mnist_multiple_test(callback, optimizer, epochs=10, learning_rate=0.01, ema_w=0.9, change=0.99, chaos_punishment=4, tests=10, copy=False):
    """
    Runs multiple tests of the MNIST_test function.

    Parameters:
        callback (function): The chaos callback function to use.
        optimizer (function): The chaos optimizer function to use.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use.
        ema_w (float): The EMA weight to use.
        change (float): The change value to use.
        chaos_punishment (int): The chaos punishment value to use.
        tests (int): The number of tests to run.
        copy (bool): Whether to copy the results to the clipboard in Excel format.

    Returns:
        None
    """
    losses = []
    for i in range(tests):
        losses.append(
            mnist_test(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate, chaos_punishment=chaos_punishment, ema_w=ema_w, change=change))

    if copy:
        pd.DataFrame(losses).to_clipboard(excel=True)
    print(np.asarray(losses))


def bike_chaos_test(callback, optimizer, epochs=50, learning_rate=0.01, ema_w=0.9, change=0.99, chaos_punishment=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
    """
    Main executable for the program
    :return: None
    """
    adalpha_r_2 = []
    for val in chaos_punishment:
        adalpha_r_2.append(adalpha_train_bike(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate, chaos_punishment=val, ema_w=ema_w, change=change)[0])

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("Chaos Punishment")
    plt.ylabel("Loss")
    plt.title(f"R Squared vs Chaos Punishment\nOver {epochs} epochs")
    plt.plot(chaos_punishment, adalpha_r_2, "r-", label="Adalpha R2")
    plt.legend()
    plt.grid(True)
    plt.show()

def mnist_chaos_test(callback, optimizer, epochs=2, learning_rate=0.01, ema_w=0.9, change=0.99, chaos_punishment=[0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
    """
    Main executable for the program
    :return: None
    """
    adalpha_r_2 = []
    for val in chaos_punishment:
        adalpha_r_2.append(adalpha_train_mnist(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate, chaos_punishment=val, ema_w=ema_w, change=change)[0])

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("Chaos Punishment")
    plt.ylabel("Loss")
    plt.title(f"Accuracy vs Chaos Punishment\nOver {epochs} epochs")
    plt.plot(chaos_punishment, adalpha_r_2, "r-", label="Adalpha R2")
    plt.legend()
    plt.grid(True)
    plt.show()

def bike_ema_w_test(callback, optimizer, epochs=50, learning_rate=0.01, ema_w=[0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], change=0.99, chaos_punishment=2):
    """
    Main executable for the program
    :return: None
    """
    adalpha_r_2 = []
    for val in ema_w:
        adalpha_r_2.append(adalpha_train_bike(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate, chaos_punishment=chaos_punishment, ema_w=val, change=change)[0])

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("ema_w")
    plt.ylabel("Loss")
    plt.title(f"R Squared vs Chaos Punishment\nOver {epochs} epochs")
    plt.plot(ema_w, adalpha_r_2, "r-", label="Adalpha R2")
    plt.legend()
    plt.grid(True)
    plt.show()

def mnist_ema_w_test(callback, optimizer, epochs=50, learning_rate=0.01, ema_w=[0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999], change=0.99, chaos_punishment=2):
    """
    Main executable for the program
    :return: None
    """
    adalpha_r_2 = []
    for val in ema_w:
        adalpha_r_2.append(adalpha_train_mnist(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate, chaos_punishment=chaos_punishment, ema_w=val, change=change)[0])

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("ema_w")
    plt.ylabel("Loss")
    plt.title(f"Accuracy vs Ema_w\nOver {epochs} epochs")
    plt.plot(ema_w, adalpha_r_2, "r-", label="Adalpha Acc")
    plt.legend()
    plt.grid(True)
    plt.show()
def cifar_test(callback, optimizer, epochs=10, learning_rate=0.01, ema_w=0.9, change=0.99, chaos_punishment=6):
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
    callbacks = [callback(my_optimizer, ema_w, change)]
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
