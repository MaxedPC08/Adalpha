"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""

from tests import *
import Adalpha as AA

if __name__ == "__main__":
    adalpha_train_mnist(AA.Adalpha_Plot, AA.AdAlpha_Momentum, epochs=1, learning_rate=0.01, chaos_punishment=2)

