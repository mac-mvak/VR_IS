import numpy as np
import scipy.linalg as sla
import scipy as sp
#from distributions import Funnel
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import torch

def q(y, x, grad_log_pi, tau): #q(y | x)
    w = y - x - tau * grad_log_pi(x)
    return np.exp(-(w**2).sum(axis=1)/(4 * tau))


def mala(x, grad_log_pi, pi, tau, k=3):
    acceptances = []
    for _ in range(k):
        grad_log = grad_log_pi(x)
        new_x = x + tau * grad_log + np.sqrt(2 * tau) * np.random.normal(size=x.size).reshape(x.shape)
        a1 = (pi(new_x) * q(x, new_x, grad_log_pi, tau))
        a2 = (pi(x) * q(new_x, x, grad_log_pi, tau))
        alpha = a1/a2
        alpha = np.minimum(np.ones_like(alpha), alpha)
        u = np.random.rand(alpha.shape[0])
        acceptance = u <= alpha
        acceptances.append(acceptance.mean())
        x = x * (~acceptance).reshape(-1, 1) + new_x * acceptance.reshape(-1, 1)
    return x, acceptances




