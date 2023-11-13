"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""

from tests import *
import MaxAdam as MA
from gan_test import *

if __name__ == "__main__":
    bike_multiple_test(MA.MaxAdamCallback, MA.AdAlpha_Momentum, epochs=20, learning_rate=0.001, chaos_punishment=6)
