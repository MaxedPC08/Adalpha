"""
Description of data set is found hereLinks to an external site..

Data file is attached to this assignment.

Determine your best ANN model for the dataset to predict bike demand (rented bike count) based on the features provided. You may have to convert some text feature information into numerical values to use in your algorithm.

Use a random 67% of the data for your training set, and the rest as your test set.

Be sure to report your R2 and adjusted R2 values. Feel free to use complex features and feature normalization. It might be useful to parse the date into month, day, and year to use as features.

Report the accuracy of the test set.

Max Clemetsen
PY 3.10
10/16/23
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import utils


class String_Verifier:
    """
    Class to hold a verifier and a runtime definied constant
    """
    def __init__(self, sym: str = "?"):
        """
        Initiator for the class
        :param sym: Symbol that causes the validator to return false
        """
        self.sym = sym

    def __call__(self, data: list):
        """
        Call function for the validator to check the data
        :param data: input data to check
        :return: T/F if the data is good
        """
        for i in range(len(data)):
            data[i] = data[i].replace("Winter", "0").replace("Spring", "1").replace("Summer", "2").replace("Autumn", "3")\
            .replace("No Holiday", "0").replace("Holiday", "1").replace("Yes", "1").replace("No", "0")
        return data
def csv_to_data(cf_name: str,
                row_limits: tuple,
                delimiters: tuple = ("\n", " "),
                dtype=float,
                name: bin = True,
                verifier: callable = None):
    """
    Opens csv file and reads to 2d nd array, deleting rows the verifier determines unfit.
    :param cf_name: File name of CSV file
    :param row_limits: Tuple of range of the rows to return
    :param delimiters: Tuple of delimiters for reading csv file
    :param dtype: data type to return
    :param name: Bin whether to return name row (row[0])
    :param verifier: Defaults to None. Function to validate rows - delete row if verifier does not return True
    :return: If name: (NDArray, shape (1, row length), NDArray, shape (n, row length)), If not name: NDArray, shape (n, row length)
    """
    #Create an emtpy list and open file in context manager
    outer_list = []
    with open(cf_name) as file:
        #Separate the rows
        cvr = csv.reader(file, delimiter=delimiters[0])
        for row in cvr:
            #Separate the columns
            cvr_2 = csv.reader(row, delimiter=delimiters[1])
            inner_list = []
            #Check if the row is valid
            if verifier:
                if verifier(row):
                    for row_2 in cvr_2:
                        inner_list.append(verifier(row_2))
                    outer_list.append(inner_list[0][row_limits[0]:row_limits[1]])
            else:  #If no verifier return all rows
                for row_2 in cvr_2:
                    inner_list.append(row_2)
                outer_list.append(inner_list[0][row_limits[0]:row_limits[1]])

    if name:  #return the data
        names = np.asarray(outer_list[:name], dtype=str)
        data = np.asarray(outer_list[name+1:], dtype=dtype)
        return names, data
    return np.asarray(outer_list, dtype=dtype)

def homo_csv(file_name, out_name):
    """
    Homogenize CSV file and write to a new file (can be the same as the first name)
    :param file_name: Name of the file to read
    :param out_name: Name of the file to write to
    :return: None
    """
    #Read the file to string
    file = open(file_name, "r").read()

    #Replace tabs with space
    while "\t" in file:
        file = file.replace("\t", " ")

    #Replace double space with single
    while "  " in file:
        file = file.replace("  ", " ")

    with open(out_name, 'w') as out:
        out.write(file)

def r2_score(y: npt.NDArray, h: npt.NDArray) -> npt.NDArray:
    """
    Calculate r squared value
    :param y: Actual values
    :param h: Pred Values
    :return: r^2 score
    """
    return 1-np.sum((y-h)**2)/np.sum((y-np.mean(y))**2)

def min_max_norm(data: npt.NDArray) -> npt.NDArray:
    """
    Normalize the data between 0, 1
    :param data:
    :return:
    """
    return (data-np.min(data))/(np.max(data)-np.min(data))

def min_max_norm_v2(data: npt.NDArray, data_2) -> npt.NDArray:
    """
    Normalize the data between 0, 1
    :param data:
    :return:
    """
    return (data_2-np.min(data))/(np.max(data)-np.min(data))
def make_date(data: npt.NDArray) -> npt.NDArray:
    """
    Make date data out of string
    :return: NDArray
    """
    data_out = []
    for i in data:
        data_out.append([i[0:1], i[3:4], i[6:10]])

    return np.asarray(data_out)
FILE_NAME = "SeoulBikeData.csv"
SPLIT = 0.67
FILE_LOAD_NAME = "model.h5"

FILE_LOAD = False

def main():
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

    my_optimizer = utils.MaxAdam(learning_rate=0.01, chaos_punishment=4)

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
                 utils.MaxAdamCallback(my_optimizer, 20)]
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
 