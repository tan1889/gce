import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar


def line_search(buffer, d, gamma_max, verbose):
    """
    Line search for greedy algorithms. This solves argmin_{gamma} Loss(z + gamma*d, y), where z is output
    of f(x) and direction d=d(x) is a direction parameterized by a basic NN module i.e. a neuron or NN2, etc.
    Checked on multiple datasets loss={mse,nll} n_sections={1,10} => brent 1 section is sufficient!
    """
    if type(buffer.criterion) is nn.MSELoss and hasattr(buffer, 'sum_net'):  # closed form solution
        gamma = line_search_mse(buffer, d, gamma_max)
        if verbose >= 6:
            print('   line_search (closed form):  gamma*={:.5f}  n_evals=1'.format(gamma))
        return gamma

    r = bounded_line_search(buffer.eval_loss, d, gamma_max)
    if verbose >= 6:
        print('   line_search (brent):  gamma*={:.5f}  min_loss={:.5f}  n_evals={}'.format(
            r['gamma'], r['f_min'], r['n_evals']
        ))
    return r['gamma']


# def line_search_test(buffer, d, alpha_max, verbose):
#     # compare between closed form, golden, brent (with 1, 4 sections)
#     sols = []
#     for method in ['brent', 'golden']:
#         for n_sections in [1, 10]:
#             r = bounded_line_search(buffer.eval_loss, d, alpha_max, method=method, n_sections=n_sections)
#             sols.append(r)
#
#     if type(buffer.criterion) is nn.MSELoss and hasattr(buffer, 'sum_net'):  # closed form solution
#         alpha_clf = line_search_mse(buffer, d, alpha_max)
#         loss_clf = buffer.eval_loss(alpha_clf, d)
#         sols.append({'alpha': alpha_clf, 'f_min': loss_clf, 'n_evals': 1, 'method': 'MSE closed form'})
#
#     sols.sort(key=lambda x: x['f_min'])
#     for sol in sols:
#         print('alpha={:.5f}\tloss={:.5f}\tn_evals={}\tdiff={:.5f}\tmethod={}'.format(
#             sol['alpha'], sol['f_min'], sol['n_evals'], sol['f_min'] - sols[0]['f_min'], sol['method']
#         ))
#
#     return sols[0]['alpha']


def bounded_line_search(f, d, gamma_max, method='brent', n_sections=1, tol=1e-4):
    """find global minimizer of f(alpha) in range [0, gamma_max] using Brent's or golden section line search.
    As there could be multiple optima, we divide the search range into n_sections and find min of each section.
    Then the global optimum is the min of all optimum. More n_sections => more iterations, better minimum """
    assert method in {'brent', 'bounded', 'golden'}, 'method must be Brent or Golden!'

    while n_sections > 1 and tol * n_sections >= gamma_max:  # range too small -> reduce n_sections
        n_sections = n_sections // 2

    b = -tol
    best = {'gamma': float('inf'), 'f_min': float('inf'), 'n_evals': 0}
    for k in range(n_sections):
        a = b + tol
        b = gamma_max * (k + 1) / n_sections
        if method in {'brent', 'bounded'}:
            res = minimize_scalar(f, args=d, bounds=[a, b], method='bounded', tol=tol)
        else:
            res = minimize_scalar(f, args=d, bracket=[a, b], method='golden', tol=tol)
            # golden method does not guarantee the solution is in [amin, amax], need to force this
            if res.x < 0:
                res.x = 0
                res.fun = f(0)
                res.nfev += 1
            elif res.x > gamma_max:
                res.x = gamma_max
                res.fun = f(gamma_max, d)
                res.nfev += 1
        best['n_evals'] += res.nfev
        if res.fun < best['f_min']:
            best['f_min'] = res.fun
            best['gamma'] = res.x
        best['method'] = '{} {}-sections'.format(method, n_sections)
    return best


def line_search_mse(fw_buffer, d, gamma_max):
    r"""closed form solution for argmin_{gamma} Loss(z + gamma*d, y), only for MSE loss and WGN FW, not for GRN!
    Loss(a,b) = 1/n 1/k sum_{ij} (a_ij - b_ij)^2"""
    sum_d2 = sum_d_yz = 0.
    for x, y, z, _, _ in fw_buffer:
        d_batch = d(x, z)
        sum_d_yz += torch.mean(d_batch * (y - z)).item()
        sum_d2 += torch.mean(d_batch ** 2).item()
    if -1e-7 <= sum_d2 <= 1e-7:
        gamma = 0.
    else:
        gamma_opt = sum_d_yz / sum_d2
        gamma = min(max(0., gamma_opt), gamma_max)
    return gamma
