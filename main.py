"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""

from tests import *
import Adalpha as AA

if __name__ == "__main__":
    mnist_multiple_test(AA.Adalpha_Plot, AA.AdAlpha_Momentum, epochs=1, learning_rate=0.01, chaos_punishment=2, ema_w=0.995)

