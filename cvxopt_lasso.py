import numpy as np
from cvxopt import matrix, solvers
import pdb
solvers.options['show_progress'] = False
def monotonic_lasso(A, b, constraint=1, no_lasso=[], param_ranks=[], rank_coeff=[], start = None):
    m,n = A.shape
    lasso_inds = [ii for ii in range(n) if ii not in no_lasso]

    G_upper = np.hstack((np.eye(n)[lasso_inds], np.zeros((n,n))[lasso_inds][:,lasso_inds]))
    G_lower = np.hstack((np.zeros((n,n))[lasso_inds],-np.eye(n)[lasso_inds][:,lasso_inds]))

    summation = np.ones(n)
    summation[no_lasso] = 0
    abs_sum = np.hstack((-summation, summation[lasso_inds]))
    ranking = np.zeros((np.max([len(param_ranks)-1,0]), G_upper.shape[1]))
    if len(rank_coeff)>0:
        rank_coeff = rank_coeff[param_ranks]
    else:
        rank_coeff = np.ones(len(param_ranks))

    for row, ii, jj, c1, c2 in zip(range(ranking.shape[0]), param_ranks[:-1], param_ranks[1:], rank_coeff[:-1], rank_coeff[1:]):
        ranking[row, ii] = c1
        ranking[row, jj] = -c2
        if ii in lasso_inds:
            ranking[row, lasso_inds[ii]+n] = c1
        if jj in lasso_inds:
            ranking[row, lasso_inds[jj]+n] = -c2


    G = np.vstack((G_upper, G_lower, abs_sum, ranking))
    h = np.hstack((np.zeros(G_upper.shape[0] + G_lower.shape[0]), constraint, np.zeros(ranking.shape[0])))

    dA = np.hstack((A,A[:,lasso_inds]))
    P = dA.T.dot(dA)
    q = -2.0 * dA.T.dot(b)

    if start is None:
        ret = solvers.qp(matrix(P), matrix(q), G = matrix(G), h = matrix(h))
    else:
        ret = solvers.qp(matrix(P), matrix(q), G = matrix(G), h = matrix(h), init_vals=start)

    fit_x = np.array(ret['x'])
    fit_x[lasso_inds] += fit_x[n:]
    return fit_x[:n]

def monotonic_fit(A, b, param_ranks=[], rank_coeff=[]):
    m,n = A.shape
    ranking = np.zeros((np.max([len(param_ranks)-1,0]), n))
    rank_coeff = rank_coeff[param_ranks]
    for row, ii, jj, c1, c2 in zip(range(ranking.shape[0]), param_ranks[:-1], param_ranks[1:], rank_coeff[:-1], rank_coeff[1:]):
        ranking[row, ii] = c1
        ranking[row, jj] = -c2

    G = ranking
    h = np.zeros(ranking.shape[0])

    P = A.T.dot(A)
    q = -2.0 * A.T.dot(b)
    ret = solvers.qp(matrix(P), matrix(q), G = matrix(G), h = matrix(h))
    fit_x = np.array(ret['x'])
    return fit_x[:n]

if __name__ == '__main__':
    m, n = 500, 20

    A = np.random.randn(m,n)
    x = np.random.randn(n)
    x[5:] *= np.abs(x[5:])>1.
    x[0] = -3
    x[1] = -2
    x[2] = -1
    x[3] = -2
    x[4] = -1
    b = A.dot(x)

    xhat = monotonic_lasso(A, b, constraint=10, param_ranks = [1,0,2,3,4,5,6])
    for xhat1, x1 in zip(xhat, x):
        print '%f %f'%(xhat1, x1)

    xhat = monotonic_lasso(A, b, constraint=10, param_ranks = [1,0,2,3,4,5,6], no_lasso=range(3))
    for xhat1, x1 in zip(xhat, x):
        print '%f %f'%(xhat1, x1)

    xhat = monotonic_lasso(A, b, constraint=10, param_ranks = [1,0,2,3,4,5,6], no_lasso=range(5))
    for xhat1, x1 in zip(xhat, x):
        print '%f %f'%(xhat1, x1)
