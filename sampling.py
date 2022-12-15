import numpy as np
import scipy.linalg as sla
import tt


def make_meshgrids(begin, end, d, steps):
    linspace, alpha = np.linspace(begin, end, num=steps, retstep=True)
    A = [linspace.reshape(1, -1, 1)] + [np.ones((1, len(linspace), 1))] * (d - 1)
    tt_grids = []
    for k in range(d):
        B = A.copy()
        B[0], B[k] = B[k], B[0]
        tt_grids.append(
            tt.vector.from_list(B)
        )
    return tt_grids, alpha


def TT_cross_density(density, begin, end, steps, d, tt_rank):
    tt_grids, alpha = make_meshgrids(begin, end, d, steps)
    tt_cross = tt.multifuncrs2(tt_grids, density, eps=1e-8, do_qr=True, verb=1)
    return tt_cross, alpha


def linear_core_integration(alpha, block):
    summation = (block[:, 1:(block.shape[1] - 1), :].sum(axis=1) +
                 (block[:, (0, block.shape[1] - 1), :].sum(axis=1))) / 2
    return alpha * summation


def TT_sampling(blocks, seeds, begin, end, steps, alpha, core_integration_method):
    d = len(blocks)
    grid = np.linspace(begin, end, num=steps)
    Ps = [np.array([1])] * d  # build P_k
    for i in range(d - 1, 0, -1):
        integrate_block = core_integration_method(alpha, blocks[i])
        Ps[i - 1] = integrate_block @ Ps[i]
    phis = [np.zeros(1)] * (d + 1)
    phis[0] = np.ones((seeds.shape[0], 1))
    # grid, distance = np.linspace(begin, end, k * (steps-1), retstep=True, endpoint=False)
    ans = np.zeros_like(seeds)
    #   print(linear_summation)
    for i in range(d):
        block = blocks[i]
        assert np.all(~np.isnan(block))
        psi_i = np.einsum('kij, j->ki', block, Ps[i])  # psi estimation
        new_phi = np.empty((seeds.shape[0], block.shape[-1]))
        for l in range(seeds.shape[0]):
            marginal_pdf = np.abs(phis[i][l, :] @ psi_i)
            sub_integral = alpha * (marginal_pdf[1:] + marginal_pdf[:marginal_pdf.shape[0] - 1]) / 2
            marginal_cdf = np.cumsum(sub_integral)
            normalizing_constant = marginal_cdf[-1]
            marginal_cdf /= normalizing_constant
            marginal_pdf /= normalizing_constant
            if l == i:  # сохранение плотности для теста
                np.save(f'normal_density_{i}.npy', np.concatenate([grid.reshape((-1, 1)),
                                                                   marginal_pdf.reshape(-1, 1) / normalizing_constant], axis=1))
            print(grid[
                      np.searchsorted(marginal_cdf, [0.1, 0.5, 0.9], side='left')
                  ])
            sort_pos = np.searchsorted(marginal_cdf, seeds[l, i], side='left')
            i1, i2 = sort_pos - 1, sort_pos
            pdf1, pdf2 = marginal_pdf[i1], marginal_pdf[i2]
            cdf1 = marginal_cdf[i1]
            x1, x2 = grid[i1], grid[i2]
            A = 0.5 * (pdf2 - pdf1) / (x2 - x1)
            D = pdf1 ** 2 + 4 * A * (seeds[l, i] - cdf1)
            if A != 0:
                new_x = x1 + (-pdf1 + np.sqrt(np.abs(D))) / (2 * A)
            else:
                new_x = (seeds[l, i] - cdf1) / pdf1
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
        return ans

    return density


d = 10
mu = np.zeros(d)
Sigma = np.eye(d)
start, end = -10, 10
steps = 10000
lambdas = np.arange(1, d + 1)
cross, alpha = TT_cross_density(
    normal_density_general(mu, Sigma), start, end, steps, d, 2
)

ans, pdfs = TT_sampling(cross.to_list(cross), np.random.uniform(size=(2, d)),
                        start, end, steps, alpha, linear_core_integration)

print(ans)
