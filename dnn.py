#####################################################
# Distributional Nearest Neighbors implementations  #
# Currently supports:                               #
#   Wasserstein2^2 distance with item-item nn and   #
#   row estimation                                  #
#   MMD_k^2 metric with item-item nn and row        #
#   estimation. Can be characterized by             #
#     Linear kernel, square kernel, exponential     #
#     kernel.                                       #
#####################################################

# imports
import numpy as np
import itertools
import string

# MMD specific functions
def sqmmd_est2(dat1, dat2, kernel) :
    """
    Computes U-statistics estimate of squared MMD_k, when number of samples from each distribution are different

    Input
        dat 1 : (m * d) data matrix coming from first d dim distribution
        dat 2 : (n * d) data matrix coming from second d dim distribution
        kernel : k(x, y) that defines MMD_k^2
    
    Output
        MMD_k^2 estimator

    Author: Kyuseong Choi
    """
    #print(dat1.shape)
    #print(dat2.shape)
    m = dat1.shape[0]
    n = dat2.shape[0]

    if dat1.shape[1] != dat2.shape[1] : 
        print("Data dimension do not match!")
        return
    
    d = dat1.shape[1]

    XX = np.matmul(dat1, np.transpose(dat1)) # m by m matrix with x_i^Tx_j
    YY = np.matmul(dat2, np.transpose(dat2)) # n by n matrix with y_i^Ty_j
    XY = np.matmul(dat1, np.transpose(dat2)) # m by n matrix with x_i^Ty_j  

    if kernel == "linear" :
        kXX, kYY, kXY = XX, YY, XY
    if kernel == "square" :
        kXX, kYY, kXY = (XX + np.ones( (m, m) ))**2, (YY + np.ones( (n, n) ))**2, (XY + np.ones( (m, n) ))**2
    if kernel == "exponential" :
        dXX_mm = np.vstack((np.diag(XX), )*m)
        dYY_nn = np.vstack((np.diag(YY), )*n)
        dXX_mn = np.vstack((np.diag(XX), )*n).T
        dYY_mn = np.vstack((np.diag(YY), )*m)

        kXX = np.exp( -0.5*( dXX_mm + np.transpose(dXX_mm) - 2*XX ) ) 
        kYY = np.exp( -0.5*( dYY_nn + np.transpose(dYY_nn) - 2*YY ) )
        kXY = np.exp( -0.5*( dXX_mn + dYY_mn - 2*XY ) )
        
    val = (kXX.sum() - np.diag(kXX).sum())/(m*(m - 1)) + (kYY.sum() - np.diag(kYY).sum())/(n*(n - 1)) - 2*kXY.sum()/(n*m)
    if val < 0 : 
        val = 0

    return(val)

def avg_mmd_distance(A_hat : list, A : np.array, inds, kernel : string):
    """
    Compute the average U-statistics estimate of squared MMD_k over inds cells

    Parameters:
    -----------
    A_hat : a list of size T with estimates specified by inds
    A : a np array of length T with values specified by inds
    inds : list-like, represents the entries of A_hat and A to compute the
           estimate of MMD_k^2 for
    kernel : string, the kernel that defines MMD_k^2 (e.g. exponential, square, linear)
    """
    err = 0
    for val in inds:
        est_val = A_hat[val]
        err_onecell = sqmmd_est2(est_val, A[val], kernel = kernel)
        err += err_onecell
    return err / len(inds)

def sqmmd_cols(Z, M, kernel):
    """
    Computes U-statistic estimate of MMD_k^2 when the number of data measurements
    are the same between columns of the tensor Z (masked by M)

    Z : N x T x n x d tensor
    M : N x T masking matrix

    WORK IN PROGRESS -> vectorizing MMD implementation between columns
    """
    N, T, n, d = Z.shape

    # want a T x T matrix, where each entry is the MMD^2 difference between the columns
    # to do this, need a T x T x N matrix where each entry is a vector of entry differences for each pair of columns
    # to do this, need to broadcast the matmul
    interm = Z.swapaxes(0, 1)[:, None, :, :] @ Z.swapaxes(0, 1).swapaxes(2, 3) # this is T x T x N x n x n
    
    if kernel == "linear":
        # the column mat mul between same columns is diagnonal along first two axes
        # for columns 0 and 1
        # kXX = interm[0, 0, :, :] n x n matrix with x_i^Tx_j
        # kYY = interm[1, 1, :, :] n x n matrix with y_i^Tx_j
        # kXY = interm[0, 1, :, :] n x n matrix with x_i^Ty_j
        diag = np.diagonal(interm, axis1 = 0, axis2 = 1).swapaxes(3, 2).swapaxes(2, 1).swapaxes(1, 0) # this is T x N x n x n and represents products within the same column

        # sum of x_i^Tx_j where i not equal j
        # T x N - T x N = T x N
        sm_col = (np.nansum(diag, axis=(2, 3))) - (np.nansum(np.diagonal(diag, axis1 = 2, axis2 = 3), axis = 2))
        # x_i^Tx_j + y_i^y_j
        sum_sm_col = sm_col[:, None] + sm_col # this is T x T x N
        # now subtract 2 * non-diagnonal entries
        # T x T x N - T x N
        nondiag = (np.nansum(interm, axis = (3, 4)) / 2) - (np.nansum(diag, axis=(2, 3)) / 2) # first is T x T x N, second is T x N

        res_units = np.nanmean(sum_sm_col - (2 * nondiag), axis = 2)
        return res_units


def exponential_kernel(x, y):
    """
    Exponential kernel

    Parameters:
    -----------
    x : N x T x d
    y : N x T x d

    Returns:
    -----------
    exp : T x T x N matrix where exp_ij = the exp kernel computation between column i
    and column j
    """
    exp = np.exp(-0.5 * np.linalg.norm(x.swapaxes(0, 1)[:, None, :] - y.swapaxes(0, 1), axis = 3))
    return exp

def square_kernel(x, ys):
    """
    Square kernel

    Parameters:
    -----------
    x : N x T x d
    ys : N x T x d

    Returns:
    -----------
    sq : T x T x N array of where sq_ij = the square kernel computation between
    the column i and the column j.
    """
    return (np.einsum('ntd,ncd->tcn', x, ys) + 1)**2



def linear_kernel(x, y):
    """
    Linear kernel

    Parameters:
    -----------
    x : N x T x d
    y : N x T x d

    Returns:
    ----------
    lk : T x T x N array where lk_ij = the linear kernel computation between column
    i and column j 
    """
    return np.einsum('ntd,ncd->tcn', x, y)


# Wasserstein specific functions
def avg_wasserstein2_dist(Y_i, Y_j):
    """
    Return the average Wasserstein2^2 distance between two sets of measurements

    Assume that Y_i and Y_j are sorted
    """
    return np.nanmean((np.nanmean((Y_i - Y_j)**2, axis = 1)), axis = 0)



# function handles either metric

def get_col_differences(Z, M, metric, kernel):
    """
    Z : a N x T x n x d tensor. If metric = "wasserstein2", then d = 1
    M : a N x T masking matrix
    metric : distance metric between columns, either "wasserstein2" or "mmd"
    kern : kernel function to use if metric = "mmd", None by default

    Assumes Z is sorted along the last axis (for Wasserstein)

    Returns:
    diffs : a T x T matrix
    """
    Z_cp = Z.copy()
    N, T, n, d = Z_cp.shape
    Z_cp[M == 0] = np.nan
    Z_cp[M == 2] = np.nan
    diffs = 0
    if metric == "wasserstein2":
        Z_cp = Z_cp.reshape((N, T, n))
        diffs = np.nanmean((Z_cp.swapaxes(0, 1)[:, None, :,  :] - Z_cp.swapaxes(0, 1))**2, axis = -1)
        # above is shape T x T x N
        # take mean across users
        diffs = np.nanmean(diffs, axis = -1)
    elif metric == "mmd":
        # TODO: see if this can be vectorized
        if kernel == "exponential":
            kern = exponential_kernel
        elif kernel == "square":
            kern = square_kernel
        elif kernel == "linear":
            kern = linear_kernel
        else:
            # default is linear kernel for now
            kern = linear_kernel
        sum_h = np.zeros([T, T, N])
        for l, l_p in itertools.combinations(range(n), r=2):
            if l != l_p:
                x = Z_cp[:, :, l] # this is N x T x d
                x_p = Z_cp[:, :,  l_p]
                y = Z_cp[:, :, l] # this is N x T x d 
                y_p = Z_cp[:, :, l_p]
                sum_h += kern(x, x_p) + kern(y, y_p) - kern(x, y_p) - kern(x_p, y) # these results are T x T x N
    
        diffs =  (1 / (n * (n - 1)) * (np.nanmean(sum_h, axis = 2))) # average over users, unobserved entries are already nan
        # since MMD_hat is estimate, could be negative. if negative, assume that the MMD distance is 0
        print(np.sum(diffs == 0))
        print(np.sum(diffs > 0))
        diffs = np.clip(diffs, a_min = 0, a_max = None)
    np.fill_diagonal(diffs, np.inf)
    return diffs

def get_col_differences_all_kernels(Z, M, metric, kernel):
    """
    Z : a N x T x n x d tensor. If metric = "wasserstein2", then d = 1
    M : a N x T masking matrix
    metric : distance metric between columns, either "wasserstein2" or "mmd"
    kern : kernel function to use if metric = "mmd", None by default

    Assumes Z is sorted along the last axis (for Wasserstein)

    FOR EXPERIMENTS: returns the column differences for all kernel types

    Returns:
    diffs : a T x T matrix
    """
    Z_cp = Z.copy()
    N, T, n, d = Z_cp.shape
    Z_cp[M == 0] = np.nan
    Z_cp[M == 2] = np.nan
    if metric == "wasserstein2":
        Z_cp = Z_cp.reshape((N, T, n))
        diffs = np.nanmean((Z_cp.swapaxes(0, 1)[:, None, :,  :] - Z_cp.swapaxes(0, 1))**2, axis = -1)
        # above is shape T x T x N
        # take mean across users
        diffs = np.nanmean(diffs, axis = -1)
        np.fill_diagonal(diffs, np.inf)
        return diffs
    elif metric == "mmd":
        # TODO: see if this can be vectorized
        if kernel == "exponential":
            kern = exponential_kernel
        elif kernel == "square":
            kern = square_kernel
        elif kernel == "linear":
            kern = linear_kernel
        else:
            # default is linear kernel for now
            kern = linear_kernel
        sum_exp = np.zeros([T, T, N])
        sum_square = np.zeros([T, T, N])
        for l, l_p in itertools.combinations(range(n), r=2):
            if l != l_p:
                x = Z_cp[:, :, l] # this is N x T x d
                x_p = Z_cp[:, :,  l_p]
                y = Z_cp[:, :, l] # this is N x T x d 
                y_p = Z_cp[:, :, l_p]
                sum_exp += exponential_kernel(x, x_p) + exponential_kernel(y, y_p)
                - exponential_kernel(x, y_p) - exponential_kernel(x_p, y) # these results are T x T x N
                sum_square += square_kernel(x, x_p) + square_kernel(y, y_p)
                - square_kernel(x, y_p) - square_kernel(x_p, y)
        exp_diffs =  (1 / (n * (n - 1)) * (np.nanmean(sum_exp, axis = 2)))
        sq_diffs = (1 / (n * (n - 1)) * (np.nanmean(sum_square, axis = 2)))
        # since MMD_hat is estimate, could be negative. if negative, assume that the MMD distance is 0
        exp_diffs = np.clip(exp_diffs, a_min = 0, a_max = None)
        sq_diffs = np.clip(sq_diffs, a_min = 0, a_max = None)
        np.fill_diagonal(exp_diffs, np.inf)
        np.fill_diagonal(sq_diffs, np.inf)
        return exp_diffs, sq_diffs

    

# TODO: see if the loops can be eliminated in this case
def estimate_row(Z, M, i, cols, col_dists, eta, metric = "wasserstein2"):
    """
    Estimate a row i of observations using eta-radius neighborhood and col_dists
    distances between columns. The columns of interest are indexed by cols

    Returns:
    row_ests : a numpy array of size T x n if metric is "wasserstein2" and
    a (potentially) ragged list of length T otherwise.
    """
    N, T, n, d = Z.shape
    Z_cp = Z.copy()
    #M_cp = M.copy()
    Z_cp[M == 0] = np.nan
    Z_cp[M == 2] = np.nan
    # create a table of indices to slice and average over
    neighborhoods = col_dists <= eta
    if metric == "wasserstein2":
        row_ests = np.full([T, n], np.nan)
        for j in cols:
            j_nn = neighborhoods[j]
            #print("neighborhood")
            #print(j_nn)
            #print(np.sum(j_nn))
            if np.sum(j_nn) > 0:
                #print(np.nansum(Z[i, :]))
                est = np.nanmean(Z_cp[i, j_nn], axis = 0).flatten()
            else:
                est = np.nanmean(Z_cp[i, :], axis = 0).flatten()
            row_ests[j] = est
    elif metric == "mmd":
        #print("in here")
        row_ests = [None] * T
        for j in cols:
            j_nn = neighborhoods[j]
            j_nn_nonan = Z_cp[i, j_nn][M[i, j_nn] == 1]
            if np.sum(j_nn_nonan) > 0:
                # Z_cp[i, j_nn] is len j_nn x n x d
                #print("Num neighbors: " + str(np.sum(j_nn)))
                #print(j_nn_nonan.shape)
                est = np.concatenate(j_nn_nonan, axis = 0)
            else:
                ann_nonan =  Z_cp[i, :][M[i, :] == 1]
                est = np.concatenate(ann_nonan, axis = 0)
            row_ests[j] = est
    # this is np array of size T x n if wasserstein, 
    # list of length T with jagged np arrays if mmd
    return row_ests
