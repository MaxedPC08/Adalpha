"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""

from tests import *
import MaxAdam as MA

if __name__ == "__main__":
    for _ in range(10):
        bike_dataset(MA.LossSlopeCallback, MA.AdAlpha_Momentum, learning_rate=0.1, chaos_punishment=6, epochs=20)

