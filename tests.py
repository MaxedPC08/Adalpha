import matplotlib.pyplot as plt

from utils import *

FILE_NAME = "SeoulBikeData.csv"
SPLIT = 0.67

def adam_train_bike(epochs=100, learning_rate=0.01):
    """
        Main executable for the program
        :return: None
        """
    homo_csv(FILE_NAME, FILE_NAME)
    verifier = String_Verifier()
    _, data = csv_to_data(FILE_NAME, (0, 15), verifier=verifier, dtype=str, delimiters=("\n", ","))
    date_data = make_date(np.asarray(data)[:, 0])
    data = np.concatenate((date_data, np.asarray(data)[:, 1:]), axis=1).astype(float)
    np.random.shuffle(data)

    # Remove non-working days
    mask = (data[:, -1] != 0)
    data = data[mask, :]
    # Split data into test and train data
    y_data = np.reshape(np.asarray(data)[:, 3], (data.shape[0], 1))
    x_data = min_max_norm(np.asarray(np.delete(data, 3, 1)))
    y_test = y_data[int(y_data.shape[0] * SPLIT):, :]
    x_test = x_data[int(x_data.shape[0] * SPLIT):, :]
    y_data = y_data[:int(x_data.shape[0] * SPLIT), :]
    x_data = x_data[:int(x_data.shape[0] * SPLIT), :]

    # Create Model
    input_layer = tf.keras.layers.Input(shape=(15))
    normalized_in = tf.keras.layers.BatchNormalization()(input_layer)
    model = tf.keras.layers.Reshape((15, 1))(normalized_in)
    model = tf.keras.layers.Conv1D(96, 5, activation="relu")(model)
    model = tf.keras.layers.Conv1D(64, 5, activation="relu")(model)
    model = tf.keras.layers.Flatten()(model)

    model = tf.keras.layers.concatenate((normalized_in, model))
    model = tf.keras.layers.Dense(1, activation="relu")(model)
    model = tf.keras.Model(input_layer, model)

    # Train with Adalpha
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=128, validation_split=0.2,
                        verbose=False)
    # Graphing the Adalpha Results

    plt.plot(history.history["loss"], "r-", label="Adam Loss")
    plt.plot(history.history["val_loss"], "y-", label="Adam Val Loss")
    y_pred = model.predict(x_test, verbose=False)
    return r2_score(y_pred, y_test), y_pred, y_test

def adalpha_train_bike(callback, optimizer, epochs=100, learning_rate=0.01, chaos_punishment=6):
    """
        Main executable for the program
        :return: None
        """
    homo_csv(FILE_NAME, FILE_NAME)
    verifier = String_Verifier()
    _, data = csv_to_data(FILE_NAME, (0, 15), verifier=verifier, dtype=str, delimiters=("\n", ","))
    date_data = make_date(np.asarray(data)[:, 0])
    data = np.concatenate((date_data, np.asarray(data)[:, 1:]), axis=1).astype(float)
    np.random.shuffle(data)

    # Remove non-working days
    mask = (data[:, -1] != 0)
    data = data[mask, :]
    # Split data into test and train data
    y_data = np.reshape(np.asarray(data)[:, 3], (data.shape[0], 1))
    x_data = min_max_norm(np.asarray(np.delete(data, 3, 1)))
    y_test = y_data[int(y_data.shape[0] * SPLIT):, :]
    x_test = x_data[int(x_data.shape[0] * SPLIT):, :]
    y_data = y_data[:int(x_data.shape[0] * SPLIT), :]
    x_data = x_data[:int(x_data.shape[0] * SPLIT), :]

    my_optimizer = optimizer(learning_rate=learning_rate, chaos_punishment=chaos_punishment)

    # Create Model
    input_layer = tf.keras.layers.Input(shape=(15))
    normalized_in = tf.keras.layers.BatchNormalization()(input_layer)
    model = tf.keras.layers.Reshape((15, 1))(normalized_in)
    model = tf.keras.layers.Conv1D(96, 5, activation="relu")(model)
    model = tf.keras.layers.Conv1D(64, 5, activation="relu")(model)
    model = tf.keras.layers.Flatten()(model)

    model = tf.keras.layers.concatenate((normalized_in, model))
    model = tf.keras.layers.Dense(1, activation="relu")(model)
    model = tf.keras.Model(input_layer, model)

    # Train with Adalpha
    callbacks = [callback(my_optimizer, 20)]
    model.compile(optimizer=my_optimizer, loss="mse")
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=128, callbacks=callbacks, validation_split=0.2,
                        verbose=False)
    # Graphing the Adalpha Results

    plt.plot(history.history["loss"], "g-", label="Adalpha Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adalpha Val Loss")
    y_pred = model.predict(x_test, verbose=False)
    return r2_score(y_pred, y_test), y_pred, y_test

def bike_test(callback, optimizer, epochs=100, learning_rate=0.01, chaos_punishment=6):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Fitting Results at lr={learning_rate} on Bike Data")
    adalpha_r_2, adalpha_y_pred, adalpha_y_test = adalpha_train_bike(callback, optimizer, epochs, learning_rate, chaos_punishment)
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

def bike_multiple_test(callback, optimizer, epochs=100, learning_rate=0.01, chaos_punishment=6, tests=10):
    losses = []
    for i in range(tests):
        losses.append(
            bike_test(callback, optimizer, epochs, learning_rate, chaos_punishment))

    print(np.asarray(losses))

def adam_train_mnist(epochs=10, learning_rate=0.01):
    """
        Main executable for the program
        :return: None
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

def adalpha_train_mnist(callback, optimizer, epochs=10, learning_rate=0.01, chaos_punishment=6):
    """
        Main executable for the program
        :return: None
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
    callbacks = [callback(my_optimizer, 20)]
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

def mnist_test(callback, optimizer, epochs=5, learning_rate=0.01, chaos_punishment=6):
    """
    Main executable for the program
    :return: None
    """
    adalpha_acc, adalpha_y_pred, adalpha_y_test = adalpha_train_mnist(callback, optimizer, epochs, learning_rate, chaos_punishment)
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
    return adalpha_acc
    
def mnist_multiple_test(callback, optimizer, epochs=100, learning_rate=0.01, chaos_punishment=6, tests=10):
    losses = []
    for i in range(tests):
        losses.append(
            mnist_test(callback, optimizer, epochs, learning_rate, chaos_punishment))

    print(np.asarray(losses))


def bike_chaos_test(callback, optimizer, epochs=50, learning_rate=0.01, chaos_punishment=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
    """
    Main executable for the program
    :return: None
    """
    adalpha_r_2 = []
    for val in chaos_punishment:
        adalpha_r_2.append(adalpha_train_bike(callback, optimizer, epochs, learning_rate, val)[0])

    # Graphing the Adalpha Results
    plt.xlabel("Chaos Punishment")
    plt.ylabel("Loss")
    plt.title(f"R Squared vs Chaos Punishment\nOver {epochs} epochs")
    plt.plot(chaos_punishment, adalpha_r_2, "r-", label="Adalpha R2")
    plt.legend()
    plt.grid(True)
    plt.show()

def mnist_chaos_test(callback, optimizer, epochs=2, learning_rate=0.01, chaos_punishment=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
    """
    Main executable for the program
    :return: None
    """
    adalpha_r_2 = []
    for val in chaos_punishment:
        adalpha_r_2.append(adalpha_train_mnist(callback, optimizer, epochs, learning_rate, val)[0])

    # Graphing the Adalpha Results
    plt.xlabel("Chaos Punishment")
    plt.ylabel("Loss")
    plt.title(f"Accuracy vs Chaos Punishment\nOver {epochs} epochs")
    plt.plot(chaos_punishment, adalpha_r_2, "r-", label="Adalpha R2")
    plt.legend()
    plt.grid(True)
    plt.show()

def cifar_test(callback, optimizer, epochs=10, learning_rate=0.01, chaos_punishment=6):
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
    callbacks = [callback(my_optimizer, 20)]
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
