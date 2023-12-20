"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""

from tests import *
import Adalpha as AA

if __name__ == "__main__":
    learning_rate = 0.1
    bike_multiple_test(AA.Adalpha_Callback, AA.AdAlpha_Momentum, epochs=20, learning_rate=learning_rate, chaos_punishment=2, change=0.99, copy=True)

