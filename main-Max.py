"""
Write a simple ANN program that uses the previously used AUTO MPG dataset provided below
(y=mpg, with features (x’s): cylinders, displacement, horsepower, weight, model year, and acceleration).
Note that you may have to perform screening of the data to eliminate spurious data rows and extraneous data.
You should use normalization on the input data (x's). Calculate and print the R2 value. Adjust the ANN
parameters to get the highest R2 value possible. Plot your final model y values (y-axis) vs. given mpg
(x-axis) with the one-to-one trend line. Print the model structure (layers with number of nodes and activations).

Max Clemetsen
Fall 2023
TF 2.13 PY
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import metrics


from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())



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


def min_max_norm(data: npt.NDArray) -> npt.NDArray:
    """
    Normalize the data between 0, 1
    :param data:
    :return:
    """
    return (data-np.min(data))/(np.max(data)-np.min(data))
def make_date(data: npt.NDArray) -> npt.NDArray:
    data_out = []
    for i in data:
        data_out.append([i[0:1], i[3:4], i[6:10]])

    return np.asarray(data_out)
FILE_NAME = "SeoulBikeData.csv"
SPLIT = 0.33

def main():
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
    print(data.shape)

    y_data = np.reshape(np.asarray(data[:, 3]), (data.shape[0], 1))
    x_data = np.delete(data, 3, 1)
    print(y_data.shape, x_data.shape)
    y_test = y_data[int(y_data.shape[0]*SPLIT):, :]
    x_test = x_data[int(x_data.shape[0]*SPLIT):, :]
    y_data = y_data[:int(x_data.shape[0]*SPLIT), :]
    x_data = x_data[:int(x_data.shape[0]*SPLIT), :]
    print(y_data.shape, x_data.shape)


    input_layer = tf.keras.layers.Input(shape=(15,))
    model = tf.keras.layers.Reshape((15, 1))(input_layer)
    model = tf.keras.layers.Conv1D(64, 7, activation="relu")(model)
    model = tf.keras.layers.Conv1D(32, 5, activation="relu")(model)
    model = tf.keras.layers.Flatten()(model)

    model = tf.keras.layers.concatenate((model, input_layer))
    model = tf.keras.layers.Dense(12, activation='relu')(model)
    model = tf.keras.layers.Dense(1, activation='relu')(model)
    #model = tf.keras.layers.Dense(1, activation='relu')(model)
    model = tf.keras.Model(input_layer, model)
    #model.load_weights("model-1.h5")
    model.summary()
    callbacks = [tf.keras.callbacks.ModelCheckpoint("model-1.h5", "loss", save_best_only=True),
                 tf.keras.callbacks.ReduceLROnPlateau("loss")
                 #tf.keras.callbacks.EarlyStopping("loss", patience=50)
                 ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99), loss="mse")
    history = model.fit(x_data, y_data, epochs=1000, batch_size=128, callbacks=callbacks, validation_split=0.1)
    print(model.evaluate(x_data, y_data))
    print(model.evaluate(x_data, y_data))

    print("Done with ANN fitting.")
    plt.xlabel("Fitting 'Epoch' Number")
    plt.ylabel("Loss Magnitude")
    plt.title("Model Fitting Results")
    plt.yscale("log")
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()
    # ====================
    # USE MODEL TO PREDICT
    y_pred = model.predict(x_test).flatten()
    r_2 = metrics.r2_score(y_pred, y_test)
    print("R^2:", r_2)
    plt.title("MPG Multifeature Modeling Result")
    plt.xlabel("Training MPG")
    plt.ylabel("Model MPG")
    plt.plot(y_test, y_pred, "r.")
    plt.plot([0, 2000], [0, 2000], "b-")
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    main()



