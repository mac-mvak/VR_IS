import numpy as np
import scipy.linalg as sla
import tt, tt.cross
#from distributions import Funnel
from tqdm import tqdm
import pickle
import torch
from distributions import Funnel


#w = tt.cross.rect_cross

def make_meshgrids(dots, d, steps):
    A = [np.ones((1, step, 1)) for step in steps]
    tt_grids, alphas = [], []
    for k in range(d):
        B = A.copy()
        start, end = dots[k]
        B[k], alpha = np.linspace(start, end, num=steps[k], retstep=True)
        B[k] = B[k].reshape((1, -1, 1))
        alphas.append(alpha)
        tt_grids.append(
            tt.vector.from_list(B)
        )
    return tt_grids, alphas


def TT_cross_density(density, dots, steps, d, y=None):
    tt_grids, alpha = make_meshgrids(dots, d, steps)
    print(y)
    tt_cross = tt.multifuncrs2(tt_grids, density, eps=1e-9, nswp=30, do_qr=True, verb=1)
    return tt_cross, alpha


def linear_core_integration(alpha, block):
    summation = (block[:, 1:(block.shape[1] - 1), :].sum(axis=1) +
                 (block[:, (0, block.shape[1] - 1), :].sum(axis=1))) / 2
    return alpha * summation


def SIRT_integration(blocks, alphas):
    # We want to have number of Choletsky decompositions.
    d = len(blocks)
    Ps = [np.array([1])] * d  # build P_k
    Rprev = np.array(1.).reshape((1,1))
    Ps[-1] = Rprev
    for i in range(d-1, 0, -1):
        block = blocks[i].copy()
        block = np.einsum('abi, ij -> abj', block, Rprev)
        weight = np.ones(block.shape[1]) * alphas[i]
        weight[0] = weight[-1] = alphas[i]/2
        weight = np.sqrt(weight)
        block = np.einsum('abi, b-> abi', block, weight)
        block = block.reshape((block.shape[0], -1)).T # M = block.T block Now let's do QR
        R = sla.qr(block, mode='economic')[1].T
        Ps[i-1] = R
        Rprev = R
    return Ps



def SIRT_sampling(blocks, seeds, dots, steps, alphas, core_integration_method):
    d = len(blocks)
    Ps = SIRT_integration(blocks, alphas)
    phis = [np.zeros(1)] * (d + 1)
    phis[0] = np.ones((seeds.shape[0], 1))
    # grid, distance = np.linspace(begin, end, k * (steps-1), retstep=True, endpoint=False)
    ans = np.zeros_like(seeds)
    #   print(linear_summation)
    for i in tqdm(range(d)):
        begin, end = dots[i]
        grid = np.linspace(begin, end, num=steps[i])
        block = blocks[i]
        assert np.all(~np.isnan(block))
        G = np.einsum('ai, ibc -> abc', phis[i], block)
        new_phi = np.empty((seeds.shape[0], block.shape[-1]))
        for l in range(seeds.shape[0]):
            marginal_pdf = ((G[l, ...] @ Ps[i])**2).sum(axis=-1)
            sub_integral = alphas[i] * (marginal_pdf[1:] + marginal_pdf[:marginal_pdf.shape[0] - 1]) / 2
            marginal_cdf = np.cumsum(sub_integral)
            normalizing_constant = marginal_cdf[-1]
            marginal_cdf /= normalizing_constant
            marginal_pdf /= normalizing_constant
            if l == 0 and i == 0:  # сохранение плотности для теста
                np.save(f'normal_density_{i}.npy', np.concatenate([grid.reshape((-1, 1)),
                                                                   marginal_pdf.reshape(-1, 1)], axis=1))
            if False:
                print(grid[
                      np.searchsorted(marginal_cdf, [0.1, 0.5, 0.9], side='left')
                  ])
            q = seeds[l, i]
            sort_pos = np.searchsorted(marginal_cdf, q, side='right')
            i1, i2 = sort_pos, sort_pos + 1
            pdf1, pdf2 = marginal_pdf[i1], marginal_pdf[i2]
            cdf1, cdf2 = marginal_cdf[i1-1], marginal_cdf[i1]
            x1, x2 = grid[i1], grid[i2]
            C = (q - cdf1)
            D = 2 * C * (pdf1 - pdf2) + pdf1**2 * (x1 - x2)
            D *= (x1-x2)
            h = pdf1 - pdf2
            if np.abs(h) >= 1e-10:
                new_x = (pdf1 * x2 - pdf2 * x1 - np.sqrt(np.abs(D)))/h
            else:
                new_x = x1 + C/pdf1
            if new_x > x2:
                new_x = x2
            elif new_x < x1:
                new_x = x1
            linear_pi = block[:, i1, :] * (x2 - new_x) / (x2 - x1) + block[:, i2, :] * (new_x - x1) / (x2 - x1)
            # new_x = grid[sort_pos]
            ans[l, i] = new_x
            new_phi[l, :] = (phis[i][l, :]) @ (linear_pi)

        phis[i + 1] = new_phi
    # print(new_phi)
    return ans, phis[-1]


def exp_density_general(lambdas):
    def density(x):
        mask = np.all(x > 0, axis=1)
        ans = np.zeros(x.shape[0], dtype=float)
        ans[mask] = (lambdas * np.exp(-lambdas * x[mask])).prod(axis=1)
        return ans

    return density


def normal_density_general(mu, Sigma):
    LU = sla.lu_factor(Sigma)
    det = sla.det(Sigma)

    def density(x):
        normalised_x = x - mu
        y = (normalised_x.T * sla.lu_solve(LU, normalised_x.T)).sum(axis=0)
        ans = np.exp(-y / 2) / ((2 * np.pi) ** (mu.shape[0] / 2) * np.sqrt(det))
        return np.sqrt(ans)

    return density

def Funnel1(a, b):

    def density(x):
        d = x.shape[1]
        part1 = x[:, 0]**2/(2 * a**2)
        part2 = (x[:, 1:]**2).sum(axis=1)
        part2 *= np.exp(-2 * b * x[:, 0])/2
        part2 += (d-1) * b * x[:, 0]
        ans = part1 + part2  + d/4 * np.log(2 * np.pi) + d/4 * np.log(a)
        ans = np.exp(-ans/2)
        return ans

    return density


def Banana(a, b):
    def density(z):
        # n = self.dim/2
        d = z.shape[1]
        even = np.arange(0, d, 2)
        odd = np.arange(1, d, 2)

        ll = (
                -0.5 * (z[..., odd] - b * z[..., even] ** 2 + (a**2)*b) ** 2
                - ((z[..., even]) ** 2) / (2 * a**2)
        ) - d/5
        return np.exp(ll.sum(-1)/2)

    return density

# 6.3 -> 1.1588 -> 3.1588
#d = 10
#mu = np.zeros(d)
#Sigma = np.eye(d)
#start, end = -10, 10
#steps = 10000
#lambdas = np.arange(1, d + 1)
#cross, alpha = TT_cross_density(
#    normal_density_general(mu, Sigma), start, end, steps, d, 2
#)

#ans, pdfs = TT_sampling(cross.to_list(cross), np.random.uniform(size=(2, d)),
                        #start, end, steps, alpha, linear_core_integration)

#print(ans)

class tensor:
    def __init__(self, f):
        self.core = f.core
        self.ps = f.ps
        self.n = f.n
        self.r = f.r
        self.d = f.d


n_steps = 1
dist = "Funnel"
d = 4
a = 2.0
b = 0.5
target = Funnel1(a=a, b=b) #normal_density_general(np.zeros(d), np.eye(d))
#target1 = Funnel(a=a, b=b, dim=d)


def dens_f(target, x):
    prob = target.prob(torch.Tensor(x))
    prob[prob < 1e-8] = 0.
    return prob


dots = [(-20., 20.)] * d
dots[0] = (-6, 6)
steps = [50] * d
lambdas = np.arange(1, d + 1)


#grid = np.linspace(dots[0][0], dots[0][1], steps[0])
#dens = tt.vector.from_list([np.exp(-grid**2/(2 * a**2 * d/4)).reshape(1,-1,1)])
#y = dens
#for i in range(1, d):
#    grid = np.linspace(dots[i][0], dots[i][1], steps[i])
#    dens = tt.vector.from_list([np.exp((grid - 1) ** 2 / (2 * a ** 2 * d / 4)).reshape(1,-1,1)])
#    y = tt.kron(y, dens)

#grid = np.zeros(steps[0], dtype=float).reshape(1,-1,1)
#grid[:, 850,:] = 1.

final = tt.rand(50, d=15, r=41)
cross, alpha = TT_cross_density(target, dots, steps, d)

def check_approx(cross, l, N):
    ans, max_diff = 0., 0.
    grids = [
        np.linspace(dots[k][0], dots[k][1],
                    num=steps[k]) for k in range(len(dots))
    ]
    for _ in range(l):
        perm = np.random.choice(
            np.arange(0, 20), d
        )
        perm[1:] = 24
        lin_dots = np.zeros(d)
        for k in range(lin_dots.shape[0]):
            lin_dots[k] = grids[k][perm[k]]
        target_val = target(lin_dots.reshape(1, -1))[0]
        cross_val = cross[perm.tolist()]
        ans += (target_val - cross_val)**2
        max_diff = max(max_diff, np.abs((target_val - cross_val)))
    return np.sqrt(ans / l), max_diff

u, m_diff = check_approx(cross, 10000, 50)
print(u)
print(m_diff)

#f = tensor(cross)
#with open('cross.pickle', 'wb') as file:
#    pickle.dump(f, file)
ans, pdfs = SIRT_sampling(cross.to_list(cross), np.random.uniform(size=(2000, d)),
                        dots, steps, alpha, linear_core_integration)

np.save(f'ans{d}.npy', ans)
#print(ans)

