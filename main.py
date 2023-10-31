"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""

from tests import *
import MaxAdam as MA

if __name__ == "__main__":
    max_data = [None for _ in range(3)]
    data = [None for _ in range(3)]
    max_data[0], data[0] = mnist_test(MA.OneCallback, MA.AdAlpha_Momentum, epochs=7, learning_rate=0.01)
    max_data[1], data[1] = mnist_test(MA.OneCallback, MA.AdAlpha_Momentum, epochs=5, learning_rate=0.01)
    max_data[2], data[2] = mnist_test(MA.OneCallback, MA.AdAlpha_Momentum, epochs=2, learning_rate=0.01)
    plt.plot(max_data)
    plt.plot(data)
    plt.show()
