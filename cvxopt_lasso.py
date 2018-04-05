import numpy as np
from cvxopt import matrix, solvers
import time
import pdb
solvers.options['show_progress'] = False
def monotonic_lasso(A, b, constraint=1, no_lasso=[], param_ranks=[], rank_coeff=[], start = None):
    #def monotonic_lasso(A, b, constraint=1, no_lasso=[], param_ranks=[], rank_coeff=[], start = None)
    #performs a lasso fit with constraints allowing you to remove lasso penalty on certain parameters,
    #enforce rank of certain parameters
    #A x = b
    #constraint- sum(abs(x)) <= constraint
    #no lasso - indices for x that do not have lasso penalty
    #param_ranks - enforce certain ordering in x
    #rank_coeff - x*rank_coeff must follow param_ranks
    #start - initial guess for x. Has minimal effect on run time in simulations.
    
    #get problem dimensions
    m,n = A.shape
    
    #find coefficients to subject to lasso penalty
    lasso_inds = [ii for ii in range(n) if ii not in no_lasso]

    #split x into positive and negative portions. G_upper forces first set of x to be negative
    #G_lower forces second set of x to be positive
    G_upper = np.hstack((np.eye(n)[lasso_inds], np.zeros((n,n))[lasso_inds][:,lasso_inds]))
    G_lower = np.hstack((np.zeros((n,n))[lasso_inds],-np.eye(n)[lasso_inds][:,lasso_inds]))
    
    #inequality matrix where sum of all the x must be lass than "constraint"
    summation = np.ones(n)
    summation[no_lasso] = 0
    abs_sum = np.hstack((-summation, summation[lasso_inds]))
    
    #make ranking constraint matrices
    ranking = np.zeros((np.max([len(param_ranks)-1,0]), G_upper.shape[1]))
    if len(rank_coeff)>0:
        rank_coeff = rank_coeff[param_ranks]
    else:
        rank_coeff = np.ones(len(param_ranks))
    
    #map lasso_indices to their counted index, make matrices enforcing rank_coeff * x 
    #have the order specified in param_ranks
    ind2lasso_inds = {ind:ii for ii, ind in enumerate(lasso_inds)}
    for row, ii, jj, c1, c2 in zip(range(ranking.shape[0]), param_ranks[:-1], param_ranks[1:], rank_coeff[:-1], rank_coeff[1:]):
        ranking[row, ii] = c1
        ranking[row, jj] = -c2
        if ii in lasso_inds:
            ranking[row, ind2lasso_inds[ii]+n] = c1
        if jj in lasso_inds:
            ranking[row, ind2lasso_inds[jj]+n] = -c2

    #inequality constraints
    G = np.vstack((G_upper, G_lower, abs_sum, ranking))
    h = np.hstack((np.zeros(G_upper.shape[0] + G_lower.shape[0]), constraint*2., np.zeros(ranking.shape[0])))
    
    #linear algebra formulation of basic problem
    dA = np.hstack((A,A[:,lasso_inds]))
    P = dA.T.dot(dA)
    q = -2. * dA.T.dot(b)
    
    #if you know a good solution use that, otherwise just solve
    if start is None:
        ret = solvers.qp(matrix(P), matrix(q), G = matrix(G), h = matrix(h))
    else:
        x0 = np.zeros(G.shape[1])
        x0[:n] = start.flatten()
        pos_x = np.zeros(G.shape[1]-n)
        lasso_x = x0[lasso_inds]
        pos_x = lasso_x * (lasso_x > 0)
        x0[lasso_inds] *= x0[lasso_inds] < 0
        x0[n:] = pos_x
        ret = solvers.qp(matrix(P), matrix(q), G = matrix(G), h = matrix(h), init_vals=x0*2)
    
    #add up the negative and positive x and return the solutions
    fit_x = np.array(ret['x'])
    fit_x[lasso_inds] += fit_x[n:]
    return fit_x[:n]/2.

def monotonic_fit(A, b, param_ranks=[], rank_coeff=[]):
    #monotonic_fit(A, b, param_ranks=[], rank_coeff=[])
    #solves Ax=b 
    #with ranking constraints defined by param_ranks. 
    #param_ranks - rank of selected x
    #rank_coeff - param_rank describes x*rank_coeff. Should be empty or same size 
    #as param_rank
    #This function is useful for finding the upper limit of lasso constraints,
    #or could be used to find a solution with known ranks.
    m,n = A.shape
    ranking = np.zeros((np.max([len(param_ranks)-1,0]), n))
    if len(rank_coeff)>0:
        rank_coeff = rank_coeff[param_ranks]
    else:
        rank_coeff = np.ones(len(param_ranks))
    
    for row, ii, jj, c1, c2 in zip(range(ranking.shape[0]), param_ranks[:-1], param_ranks[1:], rank_coeff[:-1], rank_coeff[1:]):
        ranking[row, ii] = c1
        ranking[row, jj] = -c2

    G = ranking
    h = np.zeros(ranking.shape[0])

    P = A.T.dot(A)
    q = -2. * A.T.dot(b)
    ret = solvers.qp(matrix(P), matrix(q), G = matrix(G), h = matrix(h))
    fit_x = np.array(ret['x'])
    return fit_x[:n]/2.

if __name__ == '__main__':
    m, n = 500, 100

    A = np.random.randn(m,n)
    x = np.random.randn(n)
    x[5:] *= np.abs(x[5:])>1.
    x[0] = -3
    x[1] = -2
    x[2] = -1
    x[3] = -2
    x[4] = -1
    b = A.dot(x)
    #correct way to do this
    print 'correct fit:'
    xhat = monotonic_lasso(A, b, constraint=10, param_ranks = np.argsort(x[:6]))
    for xhat1, x1 in zip(xhat, x):
        print '%f %f'%(xhat1, x1)#compare fit and true parameters
    print 'SSE=%f'%(np.sum((x-xhat)**2))
    print ' '
    #trying to break the algorithm now
    print 'trying to break the fit:'
    xhat = monotonic_lasso(A, b, constraint=10, param_ranks = [1,0,2,3,4,5,6], no_lasso=range(3))
    print 'SSE=%f'%(np.sum((x-xhat)**2))

    print ' '
    
    m, n = 2000, 1000

    A = np.random.randn(m,n)
    x = np.random.randn(n)
    x[5:] *= np.abs(x[5:])>1.
    x[0] = -3
    x[1] = -2
    x[2] = -1
    x[3] = -2
    x[4] = -1
    b = A.dot(x)
    
    #get an upper limit solution
    xhat = monotonic_fit(A, b, param_ranks = [1,0,2,3,4,5,6])
    xhat2 = monotonic_lasso(A, b, constraint=np.sum(np.abs(xhat)), param_ranks = [1,0,2,3,4,5,6], no_lasso=range(5))
    print 'lasso upper limit solution compared to unconstrainted solution,'
    print 'SSE difference in coefficients:%e'%(np.sum((xhat-xhat2)**2))
    
    #compare the time it takes with a known solution as a starting point versus 
    #no specificed starting point. The differences are small
    t = time.time()
    xhat = monotonic_lasso(A, b, constraint=10, param_ranks = [1,0,2,3,4,5,6], no_lasso=range(5))
    print 'fit without good start, time:%f'%(time.time()-t)
    
    t = time.time()
    xhat = monotonic_lasso(A, b, constraint=10, param_ranks = [1,0,2,3,4,5,6], no_lasso=range(5), start=xhat)
    print 'fit with good start, time:%f'%(time.time()-t)
