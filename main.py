"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""

from tests import *
import MaxAdam as MA

if __name__ == "__main__":
    mnist_test(MA.OneCallback, MA.AdAlpha_Momentum, epochs=10, learning_rate=0.001)
