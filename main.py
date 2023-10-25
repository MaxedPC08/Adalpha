"""
Test file for Max_Adam



Max Clemetsen
PY 3.10
10/16/23
"""

from tests import *

if __name__ == "__main__":
    mnist_chaos_test(MA.MaxAdamCallback, MA.AdAlpha_Momentum, epochs=5)
    #bike_dataset(MA.MaxAdamCallback, MA.AdAlpha_Momentum, epochs=100, chaos_punishment=8)
    mnist_test(MA.MaxAdamCallback, MA.AdAlpha_Momentum, chaos_punishment=8)
 