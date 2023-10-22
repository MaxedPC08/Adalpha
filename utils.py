import csv
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import matplotlib.pyplot as plt

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

def make_heatmap(x:npt.NDArray, y:npt.NDArray) -> npt.NDArray:
    """
    Makes a 2D heatmap from 2 1D NDArrays
    Naive implementation. Needs vectorization.
    :param x: This is the first of 2 1D NDArrays. Must have dtype == int
    :param y: This is the second 1D NDArrays. Must have dtype == int
    :return: This is the output value. Has dtype == int
    """
    heatmap = np.zeros((np.max(y)+1, np.max(y)+1))
    for i in range(len(x)):
        heatmap[x[i], y[i]] = heatmap[x[i], y[i]]+1
    return heatmap


