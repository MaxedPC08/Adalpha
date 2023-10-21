"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""

from tests import *

if __name__ == "__main__":
    bike_dataset(MA.MaxAdamCallback, learning_rate=0.01, epochs=50, chaos_punishment=8)
    bike_dataset(MA.MaxAdamCallback, learning_rate=0.1, epochs=50, chaos_punishment=8)
    bike_dataset(MA.MaxAdamCallback, learning_rate=1, epochs=50, chaos_punishment=8)
    bike_dataset(MA.MaxAdamCallback, learning_rate=10, epochs=50, chaos_punishment=8)
 