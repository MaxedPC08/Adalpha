from typing import Tuple, Any

from numpy import ndarray, dtype, generic

from utils import *

FILE_NAME = "SeoulBikeData.csv"
SPLIT = 0.67

def adam_train_bike(epochs: int = 100, learning_rate: float = 0.01) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Trains a bike sharing model using the Adam optimizer.

    Args:
        epochs (int, optional): The number of epochs to train for. Defaults to 100.
        learning_rate (float, optional): The learning rate to use for Adam. Defaults to 0.01.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: A tuple containing the R^2 score on the test data, the predicted values, and the true values.
    """
    homo_csv(FILE_NAME, FILE_NAME)
    verifier = String_Verifier()
    _, data = csv_to_data(
        FILE_NAME, (0, 15), verifier=verifier, dtype=str, delimiters=("\n", ",")
    )
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

def adalpha_train_bike(
    callback: callable,
    optimizer: callable,
    epochs: int = 100,
    learning_rate: float = 0.01,
    chaos_punishment: int = 6,
) -> tuple[ndarray[Any, dtype[generic | generic | Any]], Any, ndarray[Any, dtype[generic | generic | Any]]]:
    """
    Trains a bike sharing model using the Adalpha optimizer.

    Args:
        callback callable: A function that returns a Callback object.
        optimizer callable: A function that returns an Optimizer object.
        epochs (int, optional): The number of epochs to train for. Defaults to 100.
        learning_rate (float, optional): The learning rate to use for Adalpha. Defaults to 0.01.
        chaos_punishment (int, optional): The chaos punishment value to use for Adalpha. Defaults to 6.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: A tuple containing the R^2 score on the test data, the predicted values, and the true values.
    """
    homo_csv(FILE_NAME, FILE_NAME)
    verifier = String_Verifier()
    _, data = csv_to_data(
        FILE_NAME, (0, 15), verifier=verifier, dtype=str, delimiters=("\n", ",")
    )
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
    my_optimizer = optimizer()
    model.compile(optimizer=my_optimizer, loss="mse")
    history = model.fit(
        x_data,
        y_data,
        epochs=epochs,
        batch_size=128,
        callbacks=[callback(my_optimizer, chaos_punishment)],
        validation_split=0.2,
        verbose=False,
    )
    # Graphing the Adalpha Results

    plt.plot(history.history["loss"], "g-", label="Adalpha Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adalpha Val Loss")
    y_pred = model.predict(x_test, verbose=False)
    return r2_score(y_pred, y_test), y_pred, y_test

def adalpha_new_train_bike(
    callback: callable,
    optimizer: callable,
    epochs: int = 100,
    learning_rate: float = 0.01,
    chaos_punishment: int = 6,
) -> tuple[ndarray[Any, dtype[generic | generic | Any]], Any, ndarray[Any, dtype[generic | generic | Any]]]:
    """
    Trains a bike sharing model using the Adalpha optimizer.

    Args:
        callback callable: A function that returns a Callback object.
        optimizer callable: A function that returns an Optimizer object.
        epochs (int, optional): The number of epochs to train for. Defaults to 100.
        learning_rate (float, optional): The learning rate to use for Adalpha. Defaults to 0.01.
        chaos_punishment (int, optional): The chaos punishment value to use for Adalpha. Defaults to 6.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: A tuple containing the R^2 score on the test data, the predicted values, and the true values.
    """
    homo_csv(FILE_NAME, FILE_NAME)
    verifier = String_Verifier()
    _, data = csv_to_data(
        FILE_NAME, (0, 15), verifier=verifier, dtype=str, delimiters=("\n", ",")
    )
    date_data = make_date(np.asarray(data)[:, 0])
    data = np.concatenate((date_data, np.asarray(data)[:, 1:]), axis=1).astype(float)
    data = data[int(data.shape[0] * SPLIT):, :]
    data_two = data[:int(data.shape[0] * SPLIT), :]
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
    y_pred = model.predict(x_test, verbose=False)
    plt.plot(history.history["loss"], "g-", label="Adalpha Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adalpha Val Loss")
    orig_r2 = r2_score(model.predict(x_test, verbose=False), y_test)
    return r2_score(y_pred, y_test), y_pred, y_test

def adam_train_mnist(epochs: int = 10, learning_rate: float = 0.01) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Trains a MNIST classifier using the Adam optimizer.

    Args:
        epochs (int, optional): The number of epochs to train for. Defaults to 10.
        learning_rate (float, optional): The learning rate to use for Adam. Defaults to 0.01.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: A tuple containing the accuracy on the test data, the predicted values, and the true values.
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

    plt.plot(history.history["loss"], "r-", label="Adam Loss")
    plt.plot(history.history["val_loss"], "y-", label="Adam Val Loss")
    y_pred = model.predict(x_test, verbose=False)
    return model.evaluate(x_test, y_test)[1], y_pred, y_test

def adalpha_train_mnist(
    callback: callable,
    optimizer: callable,
    epochs: int = 10,
    learning_rate: float = 0.01,
    chaos_punishment: int = 6,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Trains a MNIST classifier using the Adalpha optimizer.

    Args:
        callback (callable): A function that returns a Callback object.
        optimizer (callable): A function that returns an Optimizer object.
        epochs (int, optional): The number of epochs to train for. Defaults to 10.
        learning_rate (float, optional): The learning rate to use for Adalpha. Defaults to 0.01.
        chaos_punishment (int, optional): The chaos punishment value to use for Adalpha. Defaults to 6.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: A tuple containing the accuracy on the test data, the predicted values, and the true values.
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
    my_optimizer = optimizer()
    callbacks = [callback(my_optimizer, chaos_punishment)]
    model.compile(optimizer=my_optimizer, loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=128, callbacks=callbacks, validation_split=0.2,
                        verbose=False)
    # Graphing the Adalpha Results

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Fitting Results at lr={learning_rate} on MNIST")
    plt.plot(history.history["loss"], "g-", label="Adalpha Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adalpha Val Loss")
    adalpha_y_pred = model.predict(x_test, verbose=False)
    return model.evaluate(x_test, y_test)[1], adalpha_y_pred, y_test