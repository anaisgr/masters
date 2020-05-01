import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, peak_widths
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def lorentzian(x, x0, a, gam):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)

def lorentzian_fixed_w(gam):
    def lo(x, x0, a):
        return lorentzian(x, x0, a, gam)
    return lo

def n_lorentzian(x, n, *p):
    if n == 1:
        return lorentzian(x, *p)
    else:
        return lorentzian(x, *p[:3]) + n_lorentzian(x, n-1, *p[3:])
    
def n_lorentzian_fixed_w(x, n, gam, *p):
    if n == 1:
        return lorentzian_fixed_w(12)(x, *p)
    else:
        return lorentzian_fixed_w(12)(x, *p[:2]) + n_lorentzian_fixed_w(x, n-1, 12, *p[2:])
    
def func(n):
    def lo(x, *p):
        return n_lorentzian(x, n, *p)
    return lo

def func_2d(n):
    def lo(x, *p):
        return n_lorentzian_fixed_w(x, n, 12, *p)
    return lo

def get_rough_max(x,y, low=None, high=None, height=500, width=10):
    peaks, _ = find_peaks(y, height=height, width=width)
    indices = peaks+x.index[0]
    plt.plot(x,y)
    plt.plot(x[indices], y[indices], "x")
    return x[indices], y[indices]

class spec_data:
    
    def __init__(self, d_peak, g_peak, d2_peak):
        self.d_peak = d_peak
        self.g_peak = g_peak
        self.d2_peak = d2_peak
        self.id_ig = d_peak["intensity"]/g_peak["intensity"]
        self.i2d_ig = d2_peak["intensity"]/g_peak["intensity"]

