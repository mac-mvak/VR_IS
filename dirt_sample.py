from sirt2 import SIRT_sample
import tt, tt.cross
from scipy.special import erf, erfinv
import numpy as np

SIGMA = []

def generate(type, sizes):
    q = np.random.uniform(size=sizes)
    if type != 'u':
        cdf_factor = erf(SIGMA/np.sqrt(2))/0.5
        q = erfinv((q - 0.5) * cdf_factor) * np.sqrt(2)
    return q


class DIRT_sampler:
    def __init__(self, N):
        self.N = N
        self.Fs = []

def DIRT_construct(dots, steps, d, gridder, log_dens, betas, ref):
    tt_grids, grids = gridder(dots, d, steps)
    N = betas.shape[0]
    DIRT = DIRT_sampler(N)
    for i in range(N):
        print(f'DIRT_level={i}')
        if i == 0:
            func = lambda x: np.exp(log_diff(log_dens, betas[i])(x)/2)
            tt_tens = tt.multifuncrs2(tt_grids, func,
                                eps=1e-10, nswp=10, do_qr=True, verb=1, rmax=15)
        else:
            tens_prev = tt_tens
            tt_tens = tt.multifuncrs2(tt_grids, dual_beta_fun(DIRT, log_dens, betas[i],  betas[i-1], ref),
                                eps=1e-10, nswp=10, do_qr=True, verb=1, y0 = tens_prev, rmax=15)
        DIRT.Fs.append((tt_tens.to_list(tt_tens), grids))
    return DIRT


def DIRT_sample(DIRT, q, ref, logfunc = None):
    N = len(DIRT.Fs) - 1
    log_dens_fin = 0.
    z = q
    if ref != 'u':
        cdf_factor = 0.5/erf(SIGMA/np.sqrt(2))
    for i in range(N, -1, -1):
        if ref != 'u':
            z = erf(z/np.sqrt(2)) * cdf_factor + 0.5
        z, log_dens = SIRT_sample(DIRT.Fs[i], z)
        log_dens_fin += log_dens
    if logfunc is not None:
        exact_dens = logfunc(z)
        return z, log_dens_fin, exact_dens
    return z, log_dens_fin

def Banana_log(a, b):
    def density(z):
        # n = self.dim/2
        d = z.shape[1]
        even = np.arange(0, d, 2)
        odd = np.arange(1, d, 2)

        ll = (
                -0.5 * (z[..., odd] - b * z[..., even] ** 2 + (a**2)*b) ** 2
                - ((z[..., even]) ** 2) / (2 * a**2)
        )
        return  ll.sum(-1)/2
    return density

def dual_beta_fun(DIRT, logfun, beta_max, beta_min, ref):
    def fun(x):
        z, _ = DIRT_sample(DIRT, x, ref)
        F = log_diff(logfun, beta_max, beta_min)(z)
        if ref[0] != 'u':
            F -= (x**2).sum(-1)/2
        return np.exp(F/2)
    return fun

def log_diff(logfun, beta_max = 1.0, beta_min=0.):
    def f(z):
        return logfun(z) * (beta_max - beta_min)
    return f

def make_meshgrids_lin(dots, d, steps):
    A = [np.ones((1, step, 1)) for step in steps]
    tt_grids, grids = [], []
    for k in range(d):
        B = A.copy()
        start, end = dots[k]
        grid = np.linspace(start, end, num=steps[k])
        B[k] = grid.reshape((1, -1, 1))
        grids.append(grid)
        tt_grids.append(
            tt.vector.from_list(B)
        )
    return tt_grids, grids

def Funnel_log(a, b):

    def density(x):
        d = x.shape[1]
        part1 = x[:, 0]**2/(2 * a**2)
        part2 = (x[:, 1:]**2).sum(axis=1)
        part2 *= np.exp(-2 * b * x[:, 0])/2
        part2 += (d-1) * b * x[:, 0]
        ans = part1 + part2#  + d/10 * np.log(2 * np.pi) + d/10 * np.log(a)
        return -ans

    return density

if __name__ == '__main__':
    d = 5
    a = 2.0
    b = 0.5
    dots = [(-18., 18.)] * d
    SIGMA = np.array([18] * d)
    SIGMA[0] = 6
    dots[0] = (-6., 6.)
    steps = [50] * d
    q = generate('n', (2000, d))
    betas = np.array([0.5, 0.75, 1])
    target = Funnel_log(a=a, b=b)  # normal_density_general(np.zeros(d), Sigma) #
    DIRT = DIRT_construct(dots, steps, d, make_meshgrids_lin, target, betas, 'n')
    z, dens = DIRT_sample(DIRT, q, target)
    np.save(f'funnel_{d}.npy', z)

