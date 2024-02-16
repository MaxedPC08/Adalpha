from numpy import ndarray, dtype, generic
from typing import Tuple, Any

from cartpole import *

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
    callback: AA.AdalphaCallback,
    optimizer: AA.Adalpha,
    epochs: int = 100,
    learning_rate: float = 0.01,
    change: float = 0.99,
    ema_w: float = 0.9,
    adjustment_exp: int = 6,
) -> tuple[ndarray[Any, dtype[generic | generic | Any]], Any, ndarray[Any, dtype[generic | generic | Any]]]:
    """
    Trains a bike sharing model using the Adalpha optimizer.

    Args:
        callback callable: A function that returns a Callback object.
        optimizer callable: A function that returns an Optimizer object.
        epochs (int, optional): The number of epochs to train for. Defaults to 100.
        learning_rate (float, optional): The learning rate to use for Adalpha. Defaults to 0.01.
        adjustment_exp (int, optional): The chaos punishment value to use for Adalpha. Defaults to 6.

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
    my_optimizer = optimizer(learning_rate=learning_rate, adjustment_exp=adjustment_exp)
    callbacks = [callback(my_optimizer, ema_w, change)]

    model.compile(optimizer=my_optimizer, loss="mse")
    history = model.fit(
        x_data,
        y_data,
        epochs=epochs,
        batch_size=128,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=False,
    )
    # Graphing the Adalpha Results

    plt.plot(history.history["loss"], "g-", label="Adalpha Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adalpha Val Loss")
    y_pred = model.predict(x_test, verbose=False)
    return r2_score(y_pred, y_test), y_pred, y_test

def adalpha_new_train_bike(
    callback: AA.AdalphaCallback,
    optimizer: AA.Adalpha,
    epochs: int = 100,
    learning_rate: float = 0.01,
    change: float = 0.99,
    ema_w: float = 0.9,
    adjustment_exp: int = 6,
    plot: bool = False
) -> tuple[ndarray[Any, dtype[generic | generic | Any]], Any, ndarray[Any, dtype[generic | generic | Any]]]:
    """
    Trains a bike sharing model using the Adalpha optimizer.

    Args:
        callback callable: A function that returns a Callback object.
        optimizer callable: A function that returns an Optimizer object.
        epochs (int, optional): The number of epochs to train for. Defaults to 100.
        learning_rate (float, optional): The learning rate to use for Adalpha. Defaults to 0.01.
        adjustment_exp (int, optional): The chaos punishment value to use for Adalpha. Defaults to 6.

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

    my_optimizer = optimizer(learning_rate=learning_rate, adjustment_exp=adjustment_exp)

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
    callbacks = [callback(my_optimizer, 20, ema_w, change)]
    model.compile(optimizer=my_optimizer, loss="mse")
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=128, callbacks=callbacks, validation_split=0.2,
                        verbose=False)
    # Graphing the Adalpha Results
    y_pred = model.predict(x_test, verbose=False)

    plt.plot(history.history["loss"], "g-", label="Adalpha Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adalpha Val Loss")
    plt.title(f"Bike Test Adalpha ema_w={ema_w}, learning rate={learning_rate}")
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
    print(model.summary())
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=128, validation_split=0.2,
                        verbose=False)
    # Graphing the Adalpha Results

    plt.plot(history.history["loss"], "r-", label="Adam Loss")
    plt.plot(history.history["val_loss"], "y-", label="Adam Val Loss")
    y_pred = model.predict(x_test, verbose=False)
    return model.evaluate(x_test, y_test)[1], y_pred, y_test

def adalpha_train_mnist(
        callback: AA.AdalphaCallback,
        optimizer: AA.Adalpha,
        epochs: int = 100,
        learning_rate: float = 0.01,
        change: float = 0.99,
        ema_w: float = 0.9,
        adjustment_exp: int = 6,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Trains a MNIST classifier using the Adalpha optimizer.

    Args:
        callback (callable): A function that returns a Callback object.
        optimizer (callable): A function that returns an Optimizer object.
        epochs (int, optional): The number of epochs to train for. Defaults to 10.
        learning_rate (float, optional): The learning rate to use for Adalpha. Defaults to 0.01.
        adjustment_exp (int, optional): The chaos punishment value to use for Adalpha. Defaults to 6.

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
    my_optimizer = optimizer(learning_rate=learning_rate, adjustment_exp=adjustment_exp)
    callbacks = [callback(my_optimizer, ema_w=ema_w, change=change)]
    model.compile(optimizer=my_optimizer, loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(x_data,
                        y_data,
                        epochs=epochs,
                        batch_size=128,
                        callbacks=callbacks,
                        validation_split=0.2,
                        verbose=False)
    # Graphing the Adalpha Results

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Fitting Results at lr={learning_rate} on MNIST")
    plt.plot(history.history["loss"], "g-", label="Adalpha Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adalpha Val Loss")
    adalpha_y_pred = model.predict(x_test, verbose=False)
    return model.evaluate(x_test, y_test)[1], adalpha_y_pred, y_test

def adam_train_cifar(epochs: int = 10,
                     learning_rate: float = 0.01) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Trains a CIFAR-10 model using the Adam optimizer.

    Args:
        epochs (int): The number of training epochs. Defaults to 10.
        learning_rate (float): The learning rate for the Adam optimizer. Defaults to 0.01.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: A tuple containing the model's accuracy on the test set,
            the predicted labels for the test set, and the true labels for the test set.
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

    my_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Train with Adam
    model.compile(optimizer=my_optimizer,
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(x_data,
                        y_data,
                        epochs=epochs,
                        batch_size=2048,
                        validation_split=0.2,
                        verbose=False)
    plt.plot(history.history["loss"], "g-", label="Adam Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adam Val Loss")
    plt.legend()
    plt.show()
    y_pred = model.predict(x_test, verbose=False)

    # Create a scatterplot of the y and y_pred
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, ax = plt.subplots(1, 1)
    heatmap = make_heatmap(np.argmax(y_pred, 1), y_test)
    im = ax.imshow(heatmap)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f"Predictions vs Actual at\nlr={learning_rate} from Adam\nOver {epochs} epochs")
    fig.tight_layout()
    plt.legend()
    plt.show()
    return model.evaluate(x_test, y_test)[1], y_pred, y_test

def adalpha_train_cifar(
    callback: AA.AdalphaCallback,
    optimizer: AA.Adalpha,
    epochs: int = 10,
    learning_rate: float = 0.01,
    ema_w: float = 0.9,
    change: float = 0.99,
    adjustment_exp: int = 6
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Trains a CIFAR-10 classifier using the AdAlpha algorithm.
    Args:
        callback: Callback function for the optimizer.
        optimizer: Optimizer used for training the model.
        epochs: Number of training epochs. Default is 10.
        learning_rate: Learning rate for the optimizer. Default is 0.01.
        ema_w: Exponential moving average weight. Default is 0.9.
        change: Change factor used in the optimizer. Default is 0.99.
        adjustment_exp: Chaos punishment factor used in the optimizer. Default is 6.
    Returns:
        accuracy: Accuracy of the trained model on the test set.
        y_pred: Predicted labels for the test set.
        y_test: True labels for the test set.
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

    my_optimizer = optimizer(learning_rate=learning_rate, adjustment_exp=adjustment_exp)
    callbacks = [callback(my_optimizer, ema_w=ema_w, change=change)]

    model.compile(optimizer=my_optimizer,
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=2048, validation_split=0.2, verbose=False, callbacks=callbacks)
    plt.plot(history.history["loss"], "g-", label="Adalpha Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adalpha Val Loss")
    plt.legend()
    plt.show()
    y_pred = model.predict(x_test, verbose=False)

    # Create a scatterplot of the y and y_pred
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, ax = plt.subplots(1, 1)
    heatmap = make_heatmap(np.argmax(y_pred, 1), y_test)
    im = ax.imshow(heatmap)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f"Predictions vs Actual at\nlr={learning_rate} from Adalpha\nOver {epochs} epochs")
    fig.tight_layout()
    plt.legend()
    plt.show()
    return model.evaluate(x_test, y_test)[1], y_pred, y_test

def adam_train_cartpole(epochs=50,
                  learning_rate=0.01,
                  memory_size=10000,
                  cycles=30,
                  tests=10,
                  learning_probability=0.7,
                  learning_size=400,
                  rl_learning_rate=0.2,
                  gamma=0.9,
                  exp_decay=0.995,
                  exploration_rate=0.8):
    """
    Executes the cartpole test with the given parameters.

    Args:
        epochs (int, optional): The number of epochs to run the test. Defaults to 50.
        learning_rate (float, optional): The learning rate for the optimizers. Defaults to 0.01.
        memory_size (int, optional): The size of the memory for the optimizer. Defaults to 10000.
        cycles (int, optional): The number of cycles to run the test. Defaults to 30.
        tests (int, optional): The number of tests to run per cycle. Defaults to 10.
        learning_probability (float, optional): The probability of learning during training. Defaults to 0.7.
        learning_size (int, optional): The size of the learning set. Defaults to 400.
        rl_learning_rate (float, optional): The learning rate for the reinforcement learning optimizer. Defaults to 0.2.
        gamma (float, optional): The discount factor for the reinforcement learning optimizer. Defaults to 0.9.
        exp_decay (float, optional): The decay rate for the exploration factor. Defaults to 0.995.
        exploration_rate (float, optional): The exploration rate for the reinforcement learning optimizer. Defaults to 0.8.

    Returns:
        object: The AdAlpha optimizer object.
        object: The Adam optimizer object.
    """
    # Set up your data and model here

    # Create the Adam optimizer
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Create the AdAlpha_Momentum optimizer


    adam_results = train(callback=[],
                         optimizer=adam_optimizer,
                         memory_size=memory_size,
                         cycles=cycles,
                         epochs=epochs,
                         tests=tests,
                         learning_probability=learning_probability,
                         learning_size=learning_size,
                         learning_rate=rl_learning_rate,
                         gamma=gamma,
                         exp_decay=exp_decay,
                         exploration_rate=exploration_rate)

    # Plot the results
    plt.xlabel("Test")
    plt.title(f"Cartpole Test Adam, learning rate={learning_rate}")
    plt.ylabel("Iteration")
    plt.plot(adam_results, label="Adam")
    plt.legend()
    plt.show()

    return adam_results

def adalpha_train_cartpole(callback: AA.AdalphaCallback,
                  optimizer,
                  epochs=50,
                  learning_rate=0.01,
                  ema_w=0.99,
                  change=0.99,
                  adjustment_exp=2,
                  memory_size=10000,
                  cycles=30,
                  tests=10,
                  learning_probability=0.7,
                  learning_size=400,
                  rl_learning_rate=0.2,
                  gamma=0.9,
                  exp_decay=0.995,
                  exploration_rate=0.8):
    """
    Executes the cartpole test with the given parameters.

    Args:
        callback (object): The callback object for the optimizer.
        optimizer (object): The optimizer object.
        epochs (int, optional): The number of epochs to run the test. Defaults to 50.
        learning_rate (float, optional): The learning rate for the optimizers. Defaults to 0.01.
        ema_w (float, optional): The exponential moving average weight for the callback. Defaults to 0.99.
        change (float, optional): The change threshold for the callback. Defaults to 0.99.
        adjustment_exp (int, optional): The punishment factor for chaos in the optimizer. Defaults to 2.
        memory_size (int, optional): The size of the memory for the optimizer. Defaults to 10000.
        cycles (int, optional): The number of cycles to run the test. Defaults to 30.
        tests (int, optional): The number of tests to run per cycle. Defaults to 10.
        learning_probability (float, optional): The probability of learning during training. Defaults to 0.7.
        learning_size (int, optional): The size of the learning set. Defaults to 400.
        rl_learning_rate (float, optional): The learning rate for the reinforcement learning optimizer. Defaults to 0.2.
        gamma (float, optional): The discount factor for the reinforcement learning optimizer. Defaults to 0.9.
        exp_decay (float, optional): The decay rate for the exploration factor. Defaults to 0.995.
        exploration_rate (float, optional): The exploration rate for the reinforcement learning optimizer. Defaults to 0.8.

    Returns:
        list: The results of the test.
    """
    # Create the AdAlpha_Momentum optimizer
    adalpha_optimizer = optimizer(learning_rate=learning_rate, adjustment_exp=adjustment_exp)

    callback = callback(adalpha_optimizer, ema_w=ema_w, change=change)

    adalpha_results = train(callback=callback,
                            optimizer=adalpha_optimizer,
                            memory_size=memory_size,
                            cycles=cycles,
                            epochs=epochs,
                            tests=tests,
                            learning_probability=learning_probability,
                            learning_size=learning_size,
                            learning_rate=rl_learning_rate,
                            gamma=gamma,
                            exp_decay=exp_decay,
                            exploration_rate=exploration_rate)

    # Plot the results
    plt.xlabel("Test")
    plt.ylabel("Iteration")
    plt.title(f"Cartpole Test Adalpha ema_w={ema_w}, learning rate={learning_rate}")
    plt.plot(adalpha_results, label="Adam")
    plt.legend()
    plt.show()
    return adalpha_results
