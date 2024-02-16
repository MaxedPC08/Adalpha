"""
Test file for Max_Adam

Max Clemetsen
PY 3.10
10/16/23
"""
from Adalpha.test_files.tests import mnist_ema_w_test
import Adalpha as AA

mnist_ema_w_test(AA.AdalphaPlot, AA.Adalpha, epochs=50, learning_rate=0.01, adjustment_exp=2, ema_w=[0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99], tests=1, change=1)