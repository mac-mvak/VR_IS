import numpy as np
import scipy.linalg as sla
import scipy as sp
#from distributions import Funnel
import matplotlib.pyplot as plt
#from MALA import mala
import torch
import tntorch as tn
import pickle
#from distributions import Funnel


#w = tt.cross.rect_cross

def make_meshgrids_lin(dots, d, steps):
    grids = []
    for k in range(d):
        start, end = dots[k]
        grid = torch.linspace(start, end, steps[k], dtype=torch.float64)
        grids.append(grid)
    return grids


def TT_cross_density1(domain, density, y=None):
    cross = tn.cross(function=density, domain=domain, function_arg="matrix",
                     rmax=20, kickrank=3, max_iter=10, device='cpu')
    return cross


def linear_core_integration(alpha, block):
    summation = (block[:, 1:(block.shape[1] - 1), :].sum(axis=1) +
                 (block[:, (0, block.shape[1] - 1), :].sum(axis=1))) / 2
    return alpha * summation


def SIRT_integration(blocks, grids):
    # We want to have number of Choletsky decompositions.
    d = len(blocks)
    Ps = [torch.tensor([1])] * d  # build P_k
    Rprev = torch.tensor([1.]).reshape((1,1))
    Ps[-1] = Rprev
    for i in range(d-1, 0, -1):
        block = blocks[i].clone().double()
        block = torch.einsum('abi, ij -> abj', block, Rprev)
        weight = torch.zeros(block.shape[1])
        alphas = grids[i][1:] - grids[i][:-1]
        weight[1:] += alphas
        weight[:-1] += alphas
        weight /= 2
        weight = torch.sqrt(weight)
        block = torch.einsum('abi, b-> abi', block, weight)
        block = block.reshape((block.shape[0], -1)).T # M = block.T block Now let's do QR
        R = torch.linalg.qr(block)[1].T
        Ps[i-1] = R
        Rprev = R
    return Ps



def SIRT_sampling(blocks, seeds, grids):
    d = len(blocks)
    Ps = SIRT_integration(blocks, grids)
    phis = [torch.zeros(1)] * (d + 1)
    phis[0] = torch.ones((seeds.shape[0], 1), dtype=torch.float64)
    # grid, distance = np.linspace(begin, end, k * (steps-1), retstep=True, endpoint=False)
    ans = torch.zeros_like(seeds, dtype=torch.float64)
    log_dens = torch.zeros(seeds.shape[0])
    #   print(linear_summation)
    for i in range(d):
        grid = grids[i]
        block = blocks[i].double()
        #assert np.all(~np.isnan(block))
        G = torch.einsum('ai, ibc -> abc', phis[i].double(), block)
        new_phi = torch.empty((seeds.shape[0], block.shape[-1]))
        alphas = grid[1:] - grid[:-1]
        for l in range(seeds.shape[0]):
            marginal_pdf = ((G[l, ...] @ Ps[i].double())**2).sum(axis=-1)
            sub_integral = torch.zeros(marginal_pdf.shape[0])
            sub_integral[1:] += marginal_pdf[:-1]
            sub_integral[1:] += marginal_pdf[1:]
            sub_integral[1:] *= alphas/2
            marginal_cdf = torch.cumsum(sub_integral, 0)
            normalizing_constant = marginal_cdf[-1].item()
            marginal_cdf /= normalizing_constant
            marginal_pdf /= normalizing_constant
            if l == 0 and i == 0:  # сохранение плотности для теста
                np.save(f'normal_density_{i}.npy', np.concatenate([grid.reshape((-1, 1)),
                                                                   marginal_pdf.reshape(-1, 1)], axis=1))
            if False:
                print(grid[
                      torch.searchsorted(marginal_cdf, [0.1, 0.5, 0.9], side='left')
                  ])
            q = seeds[l, i]
            sort_pos = torch.searchsorted(marginal_cdf, q)
            i1, i2 = sort_pos-1, sort_pos
            pdf1, pdf2 = marginal_pdf[i1], marginal_pdf[i2]
            cdf1, cdf2 = marginal_cdf[i1], marginal_cdf[i2]
            x1, x2 = grid[i1], grid[i2]
            pos = torch.searchsorted(grid, x1)
            #if q < cdf1:
            #    print(q, cdf1)
            #elif q > cdf2:
            #    print(q, cdf2)
            #assert cdf1 == marginal_cdf[pos]
            C = (q - cdf1)
            D = 2 * C * (pdf1 - pdf2) + pdf1**2 * (x1 - x2)
            D *= (x1-x2)
            h = pdf1 - pdf2
            if np.abs(h) >= 1e-10:
                new_x = (pdf1 * x2 - pdf2 * x1 - torch.sqrt(np.abs(D)))/h
            else:
                new_x = x1 + C/pdf1
            if new_x > x2:
                new_x = x2
            elif new_x < x1:
                new_x = x1
            linear_pi = block[:, i1, :] * (x2 - new_x) / (x2 - x1) + block[:, i2, :] * (new_x - x1) / (x2 - x1)
            lin_pdf = pdf1 * (x2 - new_x) / (x2 - x1) + pdf2 * (new_x - x1) / (x2 - x1)
            log_dens += torch.log(lin_pdf)
            ans[l, i] = new_x
            new_phi[l, :] = (phis[i][l, :].double()) @ (linear_pi)

        phis[i + 1] = new_phi
    return ans, log_dens

def SIRT_sample(F, q):
    blocks = F[0]
    grids = F[1]
    return SIRT_sampling(blocks, q, grids)

def Banana(a, b):
    def density(z):
        d = z.shape[1]
        even = np.arange(0, d, 2)
        odd = np.arange(1, d, 2)

        ll = (
                -0.5 * (z[..., odd] - b * z[..., even] ** 2 + (a**2)*b) ** 2
                - ((z[..., even]) ** 2) / (2 * a**2)
        )
        ll = (ll.sum(-1))/2 #+ 1
        final = torch.exp(ll.double())
        #print(final.max())
        #print(ll.max(), ll.min())
        return final

    return density

def log_grad_Funnel1(a, b):
    def pi(x):
        d = x.shape[1]
        ans = np.empty_like(x)
        x0 = x[:, 0]
        ans[:, 1:] = -x[:, 1:] * np.exp(-2 * b * x0).reshape(-1, 1)
        ans[:, 0] = -x0/a**2 + b * np.exp(-2 * b * x0) * (x[:, 1:]**2).sum(axis=1) - (d-1)*b
        return ans
    return pi

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

def get_centers(a, n, d):
    cores_coords = np.arange(n)
    x = a * np.cos(2 * np.pi * cores_coords/n).reshape(-1, 1)
    y = a * np.sin(2 * np.pi * cores_coords/n).reshape(-1,1)
    concat = np.concatenate([x, y], axis=1)
    ans = np.random.uniform(-3, 3, (n, d)) #np.zeros((n, d-2)) #    np.random.normal(0, 0.2, (n, d-2)) #
    #ans = np.concatenate([concat, w], axis=1)
    return ans

if __name__ == '__main__':
    n_steps = 1
    dist = "Funnel"
    d = 30
    a = 5.0
    b = 0.02
    num_centers = 50
    sigma = 0.2
    mus = get_centers(5., num_centers, d)
    Sigmas = [sigma * np.eye(d)] * len(mus)  #np.array([[2,0,0,1],[0,2,0,0],[0,0,2,0],[1,0,0,2]])
    target = Banana(a, b) # normal_density_general(np.zeros(d), Sigma) #
    #target1 = Funnel(a=a, b=b, dim=d)





    dots = [(-4., 6.)] * d
    dots[0] = (-15, 15)
    steps = [50] * d
    #lambdas = np.arange(1, d + 1)


    #grid = np.linspace(dots[0][0], dots[0][1], steps[0])
    #dens = tt.vector.from_list([np.exp(-grid**2/(2 * a**2 * d/4)).reshape(1,-1,1)])
    #y = dens
    #for i in range(1, d):
    #    grid = np.linspace(dots[i][0], dots[i][1], steps[i])
    #    dens = tt.vector.from_list([np.exp((grid - 1) ** 2 / (2 * a ** 2 * d / 4)).reshape(1,-1,1)])
    #    y = tt.kron(y, dens)

    #grid = np.zeros(steps[0], dtype=float).reshape(1,-1,1)
    #grid[:, 850,:] = 1.
    domain = make_meshgrids_lin(dots, d, steps)
    cross = TT_cross_density1(domain, target)
    print(cross)
    #lists = cross.to_list(cross)
    ans, pdfs = SIRT_sampling(cross.cores, torch.rand(size=(2000, d)),
                             domain)
    #np.save(f'ans{d}.npy', ans)
    #ans, accs = mala(ans, log_grad_Funnel1(a, b), Funnel1(a, b), 0.3, k=0)
    plt.scatter(ans[:, 0].numpy(), ans[:, 1].numpy())
    #plt.title(f'sqrt Sampling, d={d}')
    #plt.scatter(mus[:, 0], mus[:, 1], c='yellow', label='centers')
    #plt.legend()
    plt.show()
    #plt.scatter(ans[])
    #print(ans)

