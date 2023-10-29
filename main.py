"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""

from tests import *
import MaxAdam as MA

if __name__ == "__main__":
    bike_dataset(MA.LossSlopeCallback, MA.AdAlpha_Momentum, learning_rate=1, chaos_punishment=12)
 