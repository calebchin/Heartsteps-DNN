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

import itertools
import string

# imports
import numpy as np


# MMD specific functions
def sqmmd_est2(dat1, dat2, kernel, sigma=None):
    """
    Computes U-statistics estimate of squared MMD_k,
    when number of samples from each distribution are different

    Input
        dat 1 : (m * d) data matrix coming from first d dim distribution
        dat 2 : (n * d) data matrix coming from second d dim distribution
        kernel : k(x, y) that defines MMD_k^2

    Output
        MMD_k^2 estimator
    """
    # print(dat1.shape)
    # print(dat2.shape)
    m = dat1.shape[0]
    n = dat2.shape[0]

    if dat1.shape[1] != dat2.shape[1]:
        print("Data dimension do not match!")
        return

    # d = dat1.shape[1]

    XX = np.matmul(dat1, np.transpose(dat1))  # m by m matrix with x_i^Tx_j
    YY = np.matmul(dat2, np.transpose(dat2))  # n by n matrix with y_i^Ty_j
    XY = np.matmul(dat1, np.transpose(dat2))  # m by n matrix with x_i^Ty_j

    if kernel == "linear":
        kXX, kYY, kXY = XX, YY, XY
    if kernel == "square":
        kXX, kYY, kXY = (
            (XX + np.ones((m, m))) ** 2,
            (YY + np.ones((n, n))) ** 2,
            (XY + np.ones((m, n))) ** 2,
        )
    if kernel == "exponential":
        dXX_mm = np.vstack((np.diag(XX),) * m)
        dYY_nn = np.vstack((np.diag(YY),) * n)
        dXX_mn = np.vstack((np.diag(XX),) * n).T
        dYY_mn = np.vstack((np.diag(YY),) * m)

        kXX = np.exp(-0.5 * (dXX_mm + np.transpose(dXX_mm) - 2 * XX))
        kYY = np.exp(-0.5 * (dYY_nn + np.transpose(dYY_nn) - 2 * YY))
        kXY = np.exp(-0.5 * (dXX_mn + dYY_mn - 2 * XY))

    val = (
        (np.sum(kXX) - np.sum(np.diag(kXX))) / (m * (m - 1))
        + (np.sum(kYY) - np.sum(np.diag(kYY))) / (n * (n - 1))
        - 2 * np.sum(kXY) / (n * m)
    )
    # since we use unbiased estimator, mmd_hat can be negative, so we clip to 0
    if val < 0:
        val = 0

    return val


def sqmmd_est2_col(dat1, dat2, kernel, sigma=None):
    """
    Computes U-statistics estimate of squared MMD_k,
    when number of samples from each distribution are different

    Input
        dat 1 : N x m x d data matrix coming from first d dim distribution
        dat 2 : N x n x d data matrix coming from second d dim distribution
        kernel : k(x, y) that defines MMD_k^2

    Output
        Mean MMD_k^2 estimator between datasets with N entries (N = 1 is a single cell)
    """

    # print(dat1.shape)
    # print(dat2.shape)
    N, m, d1 = dat1.shape
    N, n, d2 = dat2.shape

    if d1 != d2:
        print("Data dimension do not match!")
        return

    XX = np.matmul(
        dat1, np.transpose(dat1, axes=(0, 2, 1))
    )  # m by m matrix with x_i^Tx_j
    YY = np.matmul(
        dat2, np.transpose(dat2, axes=(0, 2, 1))
    )  # n by n matrix with y_i^Ty_j
    XY = np.matmul(
        dat1, np.transpose(dat2, axes=(0, 2, 1))
    )  # m by n matrix with x_i^Ty_j

    if kernel == "linear":
        kXX, kYY, kXY = XX, YY, XY
    if kernel == "square":
        kXX, kYY, kXY = (
            (XX + np.ones((m, m))) ** 2,
            (YY + np.ones((n, n))) ** 2,
            (XY + np.ones((m, n))) ** 2,
        )
    if kernel == "exponential":
        dXX_mm = np.broadcast_to(np.diagonal(XX.T)[:, None], (N, m, m))
        dYY_nn = np.broadcast_to(np.diagonal(YY.T)[:, None], (N, n, n))
        dXX_mn = np.broadcast_to(np.diagonal(XX.T)[:, :, None], (N, m, n))
        dYY_mn = np.broadcast_to(np.diagonal(YY.T)[:, None], (N, m, n))
        kXX = np.exp(-0.5 * (dXX_mm + np.transpose(dXX_mm, axes=(0, 2, 1)) - 2 * XX))
        kYY = np.exp(-0.5 * (dYY_nn + np.transpose(dYY_nn, axes=(0, 2, 1)) - 2 * YY))
        kXY = np.exp(-0.5 * (dXX_mn + dYY_mn - 2 * XY))

    val_all = (
        (np.sum(kXX, axis=(1, 2)) - np.sum(np.diagonal(kXX.T), axis=1)) / (m * (m - 1))
        + (np.sum(kYY, axis=(1, 2)) - np.sum(np.diagonal(kYY.T), axis=1))
        / (n * (n - 1))
        - 2 * np.sum(kXY, axis=(1, 2)) / (n * m)
    )
    val = np.nanmean(val_all)
    if val < 0:
        val = 0

    return val


def avg_mmd_distance(A_hat: list, A: np.array, inds, kernel: string, sigma=None):
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
    num_val = len(inds)
    for val in inds:
        est_val = A_hat[val]
        err_onecell = sqmmd_est2(est_val, A[val], kernel=kernel, sigma=sigma)
        if ~np.isnan(err_onecell):
            err += err_onecell
        else:
            num_val -= 1
    if num_val == 0:
        return np.nan
    return err / num_val


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
    # to do this, need a T x T x N matrix where each entry is a
    # vector of entry differences for each pair of columns
    # to do this, need to broadcast the matmul
    interm = Z.swapaxes(0, 1)[:, None, :, :] @ Z.swapaxes(0, 1).swapaxes(
        2, 3
    )  # this is T x T x N x n x n

    if kernel == "linear":
        # the column mat mul between same columns is diagnonal along first two axes
        # for columns 0 and 1
        # kXX = interm[0, 0, :, :] n x n matrix with x_i^Tx_j
        # kYY = interm[1, 1, :, :] n x n matrix with y_i^Tx_j
        # kXY = interm[0, 1, :, :] n x n matrix with x_i^Ty_j
        diag = (
            np.diagonal(interm, axis1=0, axis2=1)
            .swapaxes(3, 2)
            .swapaxes(2, 1)
            .swapaxes(1, 0)
        )  # this is T x N x n x n and represents products within the same column

        # sum of x_i^Tx_j where i not equal j
        # T x N - T x N = T x N
        sm_col = (np.nansum(diag, axis=(2, 3))) - (
            np.nansum(np.diagonal(diag, axis1=2, axis2=3), axis=2)
        )
        # x_i^Tx_j + y_i^y_j
        sum_sm_col = sm_col[:, None] + sm_col  # this is T x T x N
        # now subtract 2 * non-diagnonal entries
        # T x T x N - T x N
        nondiag = (np.sum(interm, axis=(3, 4)) / 2) - (
            np.sum(diag, axis=(2, 3)) / 2
        )  # first is T x T x N, second is T x N

        res_units = np.nanmean(sum_sm_col - (2 * nondiag), axis=2)
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
    exp = np.exp(
        -0.5
        * (np.linalg.norm(x.swapaxes(0, 1)[:, None, :] - y.swapaxes(0, 1), axis=3)) ** 2
    )
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
    return (np.einsum("ntd,ncd->tcn", x, ys) + 1) ** 2


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
    return np.einsum("ntd,ncd->tcn", x, y)


def wasserstein2_dist(Y_i, Y_j):
    """
    Returns: 2-Wasserstein^2 distance between two n sample empirical distributions

    Parameters:
    -----------
    Y_i : n x 1 vector
    Y_j : n x 1 vector
    """
    return np.nanmean((Y_i - Y_j) ** 2)


# Wasserstein specific functions
def avg_wasserstein2_dist(Y_i, Y_j):
    """
    Return the average Wasserstein2^2 distance between two sets of measurements

    Assume that Y_i and Y_j are sorted
    """
    return np.nanmean((np.nanmean((Y_i - Y_j) ** 2, axis=1)), axis=0)


# function handles either metric


def get_col_differences(Z, M, metric, kernel, sigma=None):
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
        diffs = np.nanmean(
            (Z_cp.swapaxes(0, 1)[:, None, :, :] - Z_cp.swapaxes(0, 1)) ** 2, axis=-1
        )
        # above is shape T x T x N
        # take mean across users
        diffs = np.nanmean(diffs, axis=-1)
    elif metric == "mmd":
        # TODO: see if this can be vectorized
        diffs = np.full([T, T], 0.0)
        for i in range(T - 1):
            for j in range(i + 1, T):
                col_diffs = sqmmd_est2_col(Z_cp[:, i], Z_cp[:, j], kernel, sigma)
                diffs[i, j] = col_diffs
        diffs = diffs + diffs.T
        diffs = np.clip(diffs, a_min=0, a_max=None)
    np.fill_diagonal(diffs, np.inf)
    return diffs


def get_col_differences_all_kernels(Z, M, metric, kernel):
    """
    Z : an N x T x n x d tensor. If metric = "wasserstein2", then d = 1
    M : an N x T masking matrix
    metric : distance metric between columns, either "wasserstein2" or "mmd"
    kern : kernel function to use if metric = "mmd", None by default

    Assumes Z is sorted along the last axis (for Wasserstein)

    FOR EXPERIMENTS: returns the column differences for all kernel types

    Returns:
    diffs : a T x T matrix
    """
    print("getting diffs")
    Z_cp = Z.copy()
    N, T, n, d = Z_cp.shape
    Z_cp[M == 0] = np.nan
    Z_cp[M == 2] = np.nan
    if metric == "wasserstein2":
        Z_cp = Z_cp.reshape((N, T, n))
        diffs = np.nanmean(
            (Z_cp.swapaxes(0, 1)[:, None, :, :] - Z_cp.swapaxes(0, 1)) ** 2, axis=-1
        )
        # above is shape T x T x N
        # take mean across users
        diffs = np.nanmean(diffs, axis=-1)
        np.fill_diagonal(diffs, np.inf)
        return diffs
    elif metric == "mmd":
        # # TODO: see if this can be vectorized
        # if kernel == "exponential":
        #     kern = exponential_kernel
        # elif kernel == "square":
        #     kern = square_kernel
        # elif kernel == "linear":
        #     kern = linear_kernel
        # else:
        #     # default is linear kernel for now
        #     kern = linear_kernel
        sum_exp = np.zeros([T, T, N])
        sum_square = np.zeros([T, T, N])
        for n1, n2 in itertools.combinations(range(n), r=2):
            if n1 != n2:
                x = Z_cp[:, :, n1]  # this is N x T x d
                x_p = Z_cp[:, :, n2]
                y = Z_cp[:, :, n1]  # this is N x T x d
                y_p = Z_cp[:, :, n2]
                sum_exp += exponential_kernel(x, x_p) + exponential_kernel(y, y_p)
                -exponential_kernel(x, y_p) - exponential_kernel(
                    x_p, y
                )  # these results are T x T x N
                sum_square += square_kernel(x, x_p) + square_kernel(y, y_p)
                -square_kernel(x, y_p) - square_kernel(x_p, y)
        exp_diffs = 1 / (n * (n - 1)) * (np.nanmean(sum_exp, axis=2))
        sq_diffs = 1 / (n * (n - 1)) * (np.nanmean(sum_square, axis=2))
        # since MMD_hat is estimate, could be negative.
        # if negative, assume that the MMD distance is 0
        exp_diffs = np.clip(exp_diffs, a_min=0, a_max=None)
        sq_diffs = np.clip(sq_diffs, a_min=0, a_max=None)
        np.fill_diagonal(exp_diffs, np.inf)
        np.fill_diagonal(sq_diffs, np.inf)
        return exp_diffs, sq_diffs


# TODO: see if the loops can be eliminated in this case
def estimate_row(Z, M, i, cols, col_dists, eta, metric="wasserstein2", ret_nn=False):
    """
    Estimate a row i of observations using eta-radius neighborhood and col_dists
    distances between columns. The columns of interest are indexed by cols

    Returns:
    row_ests : a numpy array of size T x n if metric is "wasserstein2" and
    a (potentially) ragged list of length T otherwise.
    """
    N, T, n, d = Z.shape
    Z_cp = Z.copy()
    # M_cp = M.copy()
    Z_cp[M == 0] = np.nan
    Z_cp[M == 2] = np.nan

    nn_count = np.full([T], np.nan)

    list_neighbors = [None] * T

    # create a table of indices to slice and average over
    neighborhoods = col_dists <= eta
    if metric == "wasserstein2":
        row_ests = np.full([T, n], np.nan)
        for j in cols:
            j_nn = neighborhoods[j]
            # print("neighborhood")
            # print(j_nn)
            # print(np.sum(j_nn))
            neighbors = Z_cp[i, j_nn][M[i, j_nn] == 1]
            if np.size(neighbors) > 0:
                # print(np.nansum(Z[i, :]))
                est = np.nanmean(neighbors, axis=0).flatten()
                nn_count_row = neighbors.shape[0]
                list_neighbors_row = np.nonzero(j_nn & (M[i, :] == 1))
            else:
                # est = np.nanmean(Z_cp[i, :], axis = 0).flatten()
                est = np.full([n], -np.inf)
                nn_count_row = 0
                list_neighbors_row = None
            row_ests[j] = est
            nn_count[j] = nn_count_row
            list_neighbors[j] = list_neighbors_row
    elif metric == "mmd":
        # print("in here")
        row_ests = [None] * T
        for j in cols:
            j_nn = neighborhoods[j]
            nan_logic = ~np.any(np.isnan(Z[i, j_nn]), axis=1)
            # if d == 1:
            #     print(np.all(nan_logic, axis = 1).shape)
            #     print(M[i, j_nn].shape)
            #     print(Z_cp[i, j_nn].shape)
            j_nn_nonan = Z_cp[i, j_nn][(M[i, j_nn] == 1) & np.all(nan_logic, axis=1)]
            if np.size(j_nn_nonan) > 0:
                # Z_cp[i, j_nn] is len j_nn x n x d
                # print("Num neighbors: " + str(np.sum(j_nn)))
                est = np.concatenate(j_nn_nonan, axis=0)
                nn_count_row = j_nn_nonan.shape[0]
                nan_logic_full = ~np.any(np.isnan(Z[i, :]), axis=1)
                if d == 1:
                    nan_logic[:, None]
                list_neighbors_row = np.nonzero(
                    j_nn & (M[i, :] == 1) & np.all(nan_logic_full, axis=1)
                )
            else:
                # ann_nonan =  Z_cp[i, :][M[i, :] == 1]
                # est = np.concatenate(ann_nonan, axis = 0)
                est = np.full([n, d], -np.inf)
                nn_count_row = 0
                list_neighbors_row = None
            row_ests[j] = est
            nn_count[j] = nn_count_row
            list_neighbors[j] = list_neighbors_row
    # this is np array of size T x n if wasserstein,
    # list of length T with jagged np arrays if mmd
    if ret_nn:
        return row_ests, nn_count, list_neighbors
    return row_ests


# -----------------------------------------------------------#
# Functions for python package


def tune_and_estimate(
    Z: np.array,
    M: np.array,
    axis="row",
    metric="wasserstein2",
    k=None,
    eta_spec="axis",
    kernel=None,
    ret_nn=False,
):
    """
    Returns matrix of estimates Z_hat, where entries Z_hat[i,j] = np.nan if M[i, j] == 1
    and is the DNN estimate if M[i, j] == 0.

    To estimate specific values more efficiently, see `tune_and_estimate_specific`.

    Parameters:
    -----------
    Z : N x T x n x d np array
    M : N x T np array
    axis : the axis over which to compare. If "row", then user-user nearest
           neighbors is computed (find similar users), if "col", then time-time
           nearest neighbors is used (find similar cols/time).
    metric : the DNN method to use. "wasserstein2" by default.
             If "mmd", then a kernel type must be specified.
    k : the number of folds to be used in k-fold
        cross validation when searching for eta.
        By default, None. When None, LOOCV is used.
    eta_spec : if "axis", then an eta is tuned for every `axis`. if "all",
               then eta is tuned for every entry (not recommended).
    kernel : the kernel type to use. Currently supported: "square" and
             "exponential", where "exponential" refers to the Gaussian kernel.
    ret_nn : if True, returns estimates, number of neighbors, list of neighbors.
             If False, just returns estimates.
             False by default.
    """
    # need to add assertions for compatability and data types

    # type errors
    assert len(Z.shape) == 4
    assert len(M.shape) == 2
    N, T, n, d = Z.shape
    assert M.shape[0] == N
    assert (metric == "wasserstein2") | (metric == "mmd")


# should allow use of internal funcs for custom tuning
# strats - need to make docs more accessible.
