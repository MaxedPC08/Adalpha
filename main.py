"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""

from tests import *

if __name__ == "__main__":
    mnist_test(MA.MaxAdamCallback, learning_rate=0.01, epochs=1, chaos_punishment=8)

 