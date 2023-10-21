from utils import *
import MaxAdam as MA

FILE_NAME = "SeoulBikeData.csv"
SPLIT = 0.67

def bike_dataset(callback, epochs=100, learning_rate=0.01, chaos_punishment=6, mod_mult = 1):
    """
    Main executable for the program
    :return: None
    """
    homo_csv(FILE_NAME, FILE_NAME)
    verifier = String_Verifier()
    _, data = csv_to_data(FILE_NAME, (0, 15), verifier=verifier, dtype=str, delimiters=("\n","," ))
    date_data = make_date(np.asarray(data)[:, 0])
    data = np.concatenate((date_data, np.asarray(data)[:, 1:]), axis=1).astype(float)
    np.random.shuffle(data)

    #Remove non-working days
    mask = (data[:, -1] != 0)
    data = data[mask, :]

    #Split data into test and train data
    y_data = np.reshape(np.asarray(data)[:, 3], (data.shape[0], 1))
    x_data = min_max_norm(np.asarray(np.delete(data, 3, 1)))
    y_test = y_data[int(y_data.shape[0]*SPLIT):, :]
    x_test = x_data[int(x_data.shape[0]*SPLIT):, :]
    y_data = y_data[:int(x_data.shape[0]*SPLIT), :]
    x_data = x_data[:int(x_data.shape[0]*SPLIT), :]

    my_optimizer = MA.MaxAdam(learning_rate=learning_rate, chaos_punishment=chaos_punishment, modifier_multiplier=mod_mult)

    #Create Model
    input_layer = tf.keras.layers.Input(shape=(15))
    normalized_in = tf.keras.layers.BatchNormalization()(input_layer)
    model = tf.keras.layers.Reshape((15, 1))(normalized_in)
    model = tf.keras.layers.Conv1D(96, 5, activation="relu")(model)
    model = tf.keras.layers.Conv1D(64, 5, activation="relu")(model)
    model = tf.keras.layers.Flatten()(model)

    model = tf.keras.layers.concatenate((normalized_in, model))
    model = tf.keras.layers.Dense(1, activation="relu")(model)
    model = tf.keras.Model(input_layer, model)

    #Train with MaxAdam
    callbacks = [callback(my_optimizer, 20)]
    model.compile(optimizer=my_optimizer, loss="mse")
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=128, callbacks=callbacks, validation_split=0.2, verbose=False)
    # Graphing the MaxAdam Results
    plt.xlabel("Epoch")
    plt.ylabel("Loss Magnitude")
    plt.title(f"Model Fitting Results at lr={learning_rate}")
    plt.yscale("log")
    plt.plot(history.history["loss"], "r-", label="MaxAdam Loss")
    plt.plot(history.history["val_loss"], "y-", label="MaxAdam Val Loss")
    max_y_pred = model.predict(x_test, verbose=False)
    max_r_2 = r2_score(max_y_pred, y_test)

    #Train with Adam
    model = tf.keras.models.clone_model(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=128, validation_split=0.2, verbose=False)
    plt.plot(history.history["loss"], "g-", label="Adam Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adam Val Loss")
    plt.legend()
    plt.show()
    # ====================
    # USE MODEL TO PREDICT and create a scatterplot of the y and y_pred

    y_pred = model.predict(x_test, verbose=False)
    r_2 = r2_score(y_pred, y_test)
    plt.title(f"Predictions vs Actual at lr={learning_rate}")
    plt.xlabel("Training Data")
    plt.ylabel("Model Data")
    plt.plot(y_test, y_pred, "g.", label="Adam Predictions")
    plt.plot(y_test, max_y_pred, "r.", label="MaxAdam Predictions")
    plt.plot([0, 3500], [0, 3500], "b-", label="Perfect Precictions")
    plt.legend()
    plt.show()
    print(f"Learning rate: {learning_rate}\nMaxAdam r squared score: {max_r_2}\nAdam  r  squared  score: {r_2}")

def mnist_test(epochs=100, learning_rate=0.01, chaos_punishment=6):
    print(tf.__version__)
    """
    Main executable for the program
    :return: None
    """
    homo_csv(FILE_NAME, FILE_NAME)
    verifier = String_Verifier()
    _, data = csv_to_data(FILE_NAME, (0, 15), verifier=verifier, dtype=str, delimiters=("\n","," ))
    date_data = make_date(np.asarray(data)[:, 0])
    data_2 = np.reshape(np.asarray([[26, 10, 2022, 11, 19.4, 42, 4.5, 2000, 5.5, 0.1, 0, 0, 3, 0, 1]]), (1, 15))
    data = np.concatenate((date_data, np.asarray(data)[:, 1:]), axis=1).astype(float)
    data_2 = min_max_norm_v2(np.asarray(np.delete(data, 3, 1)), data_2)
    print(data.shape)
    np.random.shuffle(data)

    #Remove non-working days
    mask = (data[:, -1] != 0)
    data = data[mask, :]
    print(data.shape)

    #Split data into test and train data
    y_data = np.reshape(np.asarray(data)[:, 3], (data.shape[0], 1))
    x_data = min_max_norm(np.asarray(np.delete(data, 3, 1)))
    y_test = y_data[int(y_data.shape[0]*SPLIT):, :]
    x_test = x_data[int(x_data.shape[0]*SPLIT):, :]
    y_data = y_data[:int(x_data.shape[0]*SPLIT), :]
    x_data = x_data[:int(x_data.shape[0]*SPLIT), :]
    print(y_test.shape)

    my_optimizer = MA.MaxAdam(learning_rate=learning_rate, chaos_punishment=chaos_punishment)

    #Create Model
    input_layer = tf.keras.layers.Input(shape=(15))
    normalized_in = tf.keras.layers.BatchNormalization()(input_layer)
    model = tf.keras.layers.Reshape((15, 1))(normalized_in)
    model = tf.keras.layers.Conv1D(96, 5, activation="relu")(model)
    model = tf.keras.layers.Conv1D(64, 5, activation="relu")(model)
    model = tf.keras.layers.Flatten()(model)

    model = tf.keras.layers.concatenate((normalized_in, model))
    model = tf.keras.layers.Dense(1, activation="relu")(model)
    model = tf.keras.Model(input_layer, model)
    model.summary()

    #Train with MaxAdam
    callbacks = [MA.MaxAdamCallback(my_optimizer, 20)]
    model.compile(optimizer=my_optimizer, loss="mse")
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=128, callbacks=callbacks, validation_split=0.2, verbose=False)
    # Graphing the MaxAdam Results
    plt.xlabel("Epoch")
    plt.ylabel("Loss Magnitude")
    plt.title(f"Model Fitting Results at lr={learning_rate}")
    plt.yscale("log")
    plt.plot(history.history["loss"], "r-", label="MaxAdam Loss")
    plt.plot(history.history["val_loss"], "y-", label="MaxAdam Val Loss")
    max_y_pred = model.predict(x_test)
    max_r_2 = r2_score(max_y_pred, y_test)

    #Train with Adam
    model = tf.keras.models.clone_model(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=128, validation_split=0.2, verbose=False)
    plt.plot(history.history["loss"], "g-", label="Adam Loss")
    plt.plot(history.history["val_loss"], "b-", label="Adam Val Loss")
    plt.legend()
    plt.show()
    # ====================
    # USE MODEL TO PREDICT and create a scatterplot of the y and y_pred

    y_pred = model.predict(x_test)
    r_2 = r2_score(y_pred, y_test)
    plt.title(f"Predictions vs Actual at lr={learning_rate}")
    plt.xlabel("Training Data")
    plt.ylabel("Model Data")
    plt.plot(y_test, y_pred, "g.", label="Adam Predictions")
    plt.plot(y_test, max_y_pred, "r.", label="MaxAdam Predictions")
    plt.plot([0, 3500], [0, 3500], "b-", label="Perfect Precictions")
    plt.legend()
    plt.show()
    print(f"MaxAdam r squared score: {max_r_2} \n Adam  r  squared  score: {r_2}")
