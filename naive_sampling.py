import numpy as np
import scipy.linalg as sla
#import tt, tt.cross
#from distributions import Funnel
from tqdm import tqdm
import pickle
import torch
import matplotlib.pyplot as plt
#from distributions import Funnel


#w = tt.cross.rect_cross


def make_meshgrids_cheb(dots, d, steps):
    grids = []
    for k in range(d):
        w = dots[k][1]
        N = steps[k] - 1
        grid = -np.cos(np.arange(0, N + 1) * np.pi / N) * w
        grids.append(grid)
    return grids

def make_meshgrids_lin(dots, d, steps):
    grids = []
    for k in range(d):
        start, end = dots[k]
        grid = np.linspace(start, end, num=steps[k])
        grids.append(grid)
    return grids

def make_meshgrids_log(dots, d, steps):
    grids = []
    for k in range(d):
        start, end = dots[k]
        if k == 0:
            grid = np.linspace(start, end, num=steps[k])
        else:
            grid1 = -10.**np.arange(-3, -10, step=-1)
            grid2 = np.linspace(dots[k][0], -1e-3, num=30, endpoint=False)
            grid = np.concatenate([grid2, grid1, np.flip(-grid1), np.flip(-grid2)])
        grids.append(grid)
    return grids



def linear_core_integration(alpha, block):
    summation = (block[:, 1:(block.shape[1] - 1), :].sum(axis=1) +
                 (block[:, (0, block.shape[1] - 1), :].sum(axis=1))) / 2
    return alpha * summation


def SIRT_integration(blocks, grids):
    # We want to have number of Choletsky decompositions.
    d = len(blocks)
    Ps = [np.array([1])] * d  # build P_k
    Rprev = np.array(1.).reshape((1,1))
    Ps[-1] = Rprev
    for i in range(d-1, 0, -1):
        block = blocks[i].copy()
        block = np.einsum('abi, ij -> abj', block, Rprev)
        weight = np.zeros(block.shape[1])
        alphas = grids[i][1:] - grids[i][:-1]
        weight[1:] += alphas
        weight[:-1] += alphas
        weight /= 2
        weight = np.sqrt(weight)
        block = np.einsum('abi, b-> abi', block, weight)
        block = block.reshape((block.shape[0], -1)).T # M = block.T block Now let's do QR
        R = sla.qr(block, mode='economic')[1].T
        Ps[i-1] = R
        Rprev = R
    return Ps



def make_dens_array(grids, density):
    d = len(grids)
    Us = np.meshgrid(*grids, indexing='ij')
    W = np.stack([U.reshape(-1, 1) for U in Us]).reshape(d, -1).T
    W_dens = density(W)
    W_dens = W_dens.reshape([grid.shape[0] for grid in grids])
    return W_dens

def integrate_last_coord(vec, grids, k):
    grid = grids[k]
    alphas = grid[1:] - grid[:-1]
    sub_integral = np.zeros_like(vec)
    sub_integral[..., 1:] += vec[...,:-1]
    sub_integral[..., 1:] += vec[..., 1:]
    sub_integral[..., 1:] = np.einsum('...i, i-> ...i', sub_integral[..., 1:], alphas)
    integral = sub_integral.sum(-1) / 2
    return integral

def prepare_vec(vec, dots):
    now_vec = vec
    for l in range(len(dots)):
        i1, i2, x1, x2, new_x = dots[l]
        now_vec = now_vec[i1, ...] * (x2 - new_x) / (x2 - x1) + now_vec[i2, ...] * (new_x - x1) / (x2 - x1)
    return now_vec


def naive_sampling(vec, seeds, dots, grids):
    d = len(vec.shape)
    phis = [np.zeros(1)] * (d + 1)
    phis[0] = np.ones((seeds.shape[0], 1))
    # grid, distance = np.linspace(begin, end, k * (steps-1), retstep=True, endpoint=False)
    ans = np.zeros_like(seeds)
    vec_lists = [[] for _ in range(seeds.shape[0])]
    for i in range(d):
        begin, end = dots[i]
        grid = grids[i]
        alphas = grid[1:] - grid[:-1]
        for l in tqdm(range(seeds.shape[0]), desc=f'{i + 1}/{d}'):
            now_vec = prepare_vec(vec, vec_lists[l])
            now_vec1 = now_vec
            k = d-1
            if i > 0 or l == 0:
                while len(now_vec1.shape) != 1:
                    now_vec1 = integrate_last_coord(now_vec1, grids, k)
                    k -= 1
            if i == 0 and l == 0:
                now_vec11 = now_vec1
            elif i == 0:
                now_vec1 = now_vec11
            marginal_pdf = now_vec1
            sub_integral = np.zeros(marginal_pdf.shape[0])
            sub_integral[1:] += marginal_pdf[:-1]
            sub_integral[1:] += marginal_pdf[1:]
            sub_integral[1:] *= alphas/2
            marginal_cdf = np.cumsum(sub_integral)
            normalizing_constant = marginal_cdf[-1]
            marginal_cdf /= normalizing_constant
            marginal_pdf /= normalizing_constant
            if i == 0 and l == 0:  # сохранение плотности для теста
                np.save(f'density_{i}.npy', np.concatenate([grid.reshape((-1, 1)),
                                                                   marginal_pdf.reshape(-1, 1)], axis=1))
            q = seeds[l, i]
            sort_pos = np.searchsorted(marginal_cdf, q)
            i1, i2 = sort_pos-1, sort_pos
            pdf1, pdf2 = marginal_pdf[i1], marginal_pdf[i2]
            cdf1, cdf2 = marginal_cdf[i1], marginal_cdf[i2]
            x1, x2 = grid[i1], grid[i2]
            pos = np.searchsorted(grid, x1)
            if q < cdf1:
                print(q, cdf1)
            elif q > cdf2:
                print(q, cdf2)
            assert cdf1 == marginal_cdf[pos]
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
            vec_lists[l].append((i1, i2, x1, x2, new_x))
            # new_x = grid[sort_pos]
            ans[l, i] = new_x
    # print(new_phi)
    return ans, np.array(vec).reshape(-1)


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
        print('0/6')
        part1 = x[:, 0]**2/(2 * a**2)
        print('1/6')
        part2 = (x[:, 1:]**2).sum(axis=1)
        print('2/6')
        part2 *= np.exp(-2 * b * x[:, 0])/2
        print('3/6')
        part2 += (d-1) * b * x[:, 0]
        print('4/6')
        ans = part1 + part2#  + d/10 * np.log(2 * np.pi) + d/10 * np.log(a)
        print('5/6')
        ans = np.exp(-(ans))
        print('6/6')
        return ans

    return density

def Funnel2(a, b):

    def prob(z):
        # pdb.set_trace()
        d = z.shape[1]
        z1 = z[..., 0]
        logprob1 = -z1**2/(2 * a**2)
        # logprob2 = self.distr2(z[...,0])
        logprob2 = (
            -0.5*(z[..., 1:] ** 2).sum(-1) * np.exp(-2 * b * z1)
        #    - np.log(self.dim)
            - (d-1) * b * z1
        )
        return np.exp(logprob1 + logprob2 - d)

    return prob

def Test_Dens(x):
    return x[:, 1:].sum(-1)

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



a, b = 2.0, 0.5
d=4
dens = Funnel2(a, b)
dots = [(-17., 17.)] * d
dots[0] = (-7., 7.)
steps = [51] *d
grids = make_meshgrids_log(dots, d, steps)
W = make_dens_array(grids, dens)
seeds = np.random.uniform(size=(2000, d))
ans, _ = naive_sampling(W, seeds, dots, grids)
np.save(f'ans{d}.npy', ans)
#true = np.load(f'true{d}.npy')
plt.scatter(ans[:, 0], ans[:, 1], label='tt')
#plt.scatter(true[:, 0], true[:, 1], label='true', color='orange')
plt.legend()
plt.show()
