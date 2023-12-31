"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""
from tests import *
import Adalpha as AA

if __name__ == "__main__":
    cartpole_multiple_test(AA.Adalpha_Callback, AA.Adalpha_Momentum, exploration_rate=0.4, epochs=10, learning_rate=0.0001, chaos_punishment=2, ema_w=0.8, change=0.95, cycles=10, copy=True)
