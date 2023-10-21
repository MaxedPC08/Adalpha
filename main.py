"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""
import MaxAdam as MA
from utils import *

FILE_NAME = "SeoulBikeData.csv"
SPLIT = 0.67
FILE_LOAD_NAME = "model.h5"

FILE_LOAD = False


def main():
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

    my_optimizer = MA.MaxAdam(learning_rate=0.01, chaos_punishment=4)

    #Create
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

    #----IMPORTANT---- If the network has not yet been trained comment the following line of code
    if FILE_LOAD:
        model.load_weights(FILE_LOAD_NAME)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(FILE_LOAD_NAME, "val_loss", save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(patience=50),
                 MA.MaxAdamCallback(my_optimizer, 20)]
    model.compile(optimizer=my_optimizer, loss="mse")
    history = model.fit(x_data, y_data, epochs=100, batch_size=128, callbacks=callbacks, validation_split=0.2)
    model.load_weights(FILE_LOAD_NAME)
    # Graphing the
    print("Done with ANN fitting.")
    plt.xlabel("Fitting 'Epoch' Number")
    plt.ylabel("Loss Magnitude")
    plt.title("Model Fitting Results")
    plt.yscale("log")
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()
    # ====================
    # USE MODEL TO PREDICT and create a scatterplot of the y and y_pred
    print(x_test)
    y_pred = model.predict(x_test)
    r_2 = r2_score(y_pred, y_test)
    print("R^2:", r_2)
    plt.title("MPG Multifeature Modeling Result")
    plt.xlabel("Training Data")
    plt.ylabel("Model Data")
    plt.plot(y_test, y_pred, "g.")
    plt.plot([0, 2000], [0, 2000], "b-")
    plt.grid(True)
    plt.show()
    #Highest r2 = 91.68
    print(model(data_2))


if __name__ == "__main__":
    main()
 