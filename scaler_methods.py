# include implementations for SoftImpute, USVT, LI 2019NN

# for softimpute, use fancyimpute package

from fancyimpute import SoftImpute, BiScaler
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# test mse for all methods
def mse(A_hat, A):
    """
    Returns the mean squared error of estimate A_hat from ground truth A
    """
    return np.nanmean((A_hat - A)**2)

def softimpute_inds(A, test_inds):
  """
  Uses SoftImpute to impute the nan values in A (masked by M).

  Parameters:
  -----------
  A : a N x T matrix
  M : a N x T masking matrix for A
  test_inds : the indices of A that we are interested in testing

  Returns:
  -----------
  mse : the mse of imputations made by softimpute versus the true value
  """
  A_cp = A.copy()
  A_cp[test_inds] = np.nan

  A_normed = BiScaler().fit_transform(A_cp)
  A_hat = SoftImpute().fit_transform(A_normed)
  ests = A_hat[test_inds]
  ground_truth = A[test_inds]
  return mse(ests, ground_truth)

def softimpute_end_testset(Z, M, i, testset_size):
    """
    Test soft impute on last testset_size chunk of matrix
    """
    # TODO: do we treat test set as monolith or individual points?
    # current treatment is a monolith -> could cause problems as
    # there is a chunk of missing at the end
    Z_cp = Z.copy()
    Z_cp[M != 1] = np.nan
    test_size = int(testset_size * len(Z_cp[0]))
    test_inds = np.nonzero(M[i] == 1)[0]
    test_inds = test_inds[test_inds >= (len(Z[i]) - test_size)]
    row_inds = np.repeat(i, len(test_inds))
    final_inds = (row_inds, test_inds)
    return softimpute_inds(Z, final_inds)

# usvt



# Li 2019 (S-NN)
def all_dissims(O, M, ax):
    """
    Returns all dissimilarities in matrix O masked by M

    Parameters
    ----------
    O : a matrix of size a x b
    M : a masking matrix of size a x b that
        dictates which entries of O are known
    ax : specifies which dimension of O to compute similarities over
        (either 0 -> rows, or 1 -> cols).

    Returns: 
    -----------
    Returns a x a dissimilarity matrix if ax = 0 (rows), b x b if ax = 1 (cols).
    """
    # apply mask to the original matrix -> if something is not observed, then
    # should not include in the mean calcs
    O[M != 1] = np.nan 
    if ax == 1:
        O = O.T
        M = M.T
    O_br = O[:, None] # add extra dim for broadcast operation
    # all row dissims between pairs of rows
    dis = (O_br - O)**2
    # take mean over the sample dimension (now the 2nd dim)
    mean_row_dis = np.nanmean(dis, axis = 2)
    # overlap between every
    overlap = np.sum(M[:, None] * M, axis = 2).astype('float64')
    zs = np.nonzero(overlap == 0)
    overlap[zs] = np.nan
    
    # rows with 0 overlap will have nan in this matrix
    # entry cannot be own neighbor, so dist is infinite
    mean_ovr = mean_row_dis / overlap
    np.fill_diagonal(mean_ovr, np.inf)
    return mean_ovr

def estimate_row(Z, M, u, inds, dissims, eta, ax):
    """
    Estimate values at entries inds for row u in z.

    Parameters:
    -----------
    Z : N x T matrix masked by M
    M : N x T matrix that masks Z
    dissims : dissimilarity matrix between cols (T x T) or rows (N x N). 
              Cols or rows depends on ax
    ax : if ax = 0, then the dissimilarities are rows. If ax = 1, then the
         dissimilariteis are columns.

    Returns:
    -----------
    ests : length T array with nn estimates in entries specifed by inds
    """
    Z_cp = Z.copy
    N, T = Z_cp.shape
    Z_cp[M != 1] = np.nan
    if ax == 0:
        # nn_inds already accounts for no overlap entries
        nn_inds = dissims[u] <= eta

        ests = np.full([T], np.nan)
        ests[inds] = np.nanmean(Z[nn_inds, inds] * (M[nn_inds, inds] == 1), axis = 0)
        return ests
    else: 
        ests = np.full([T], np.nan)
        for i in inds:
          nn_inds = dissims[i] <= eta 
          ests[i] = np.nanmean(Z[u,nn_inds] * (M[u, nn_inds] == 1), axis = 1)
        return ests

def cross_validate_row_snn(Z, M, i, k, eta, col_dists, negate_mask = False):
    """
    Performs k-fold cross validation on row i in dataset `Z` with mask `M` using
    parameters `k` and `eta`, distributional estimation function `estimate`, 
    and error function `error_metric`. `error_metric` can either be `wasserstein2`
    or `mmd`.

    Z : N x T x n array
    M : N x T array
    """
    # find observed indices of row i and create splits off of those
    obvs_inds_x = np.nonzero(M[i] == 1)[0]
    
    # shuffle the x coordinates
    np.random.seed(seed = 42)
    np.random.shuffle(obvs_inds_x)
    
    # split coordinates into k-folds
    folds = np.array_split(obvs_inds_x, k, axis = 0)
    tot_error = 0
    final_k = k

    for j in range(k):
        # indices of the observed outcomes in the validation set in this iteration
        folds_ind = folds[j]

        ground_truth = Z[i, folds_ind]
        est_mask = M.copy()
        Z_folds = Z.copy()
        # hide observed values in the validation set
        # CURRENT STRATEGY: mask row for each fold indice
        # TODO: this breaks with k = 1
        est_mask[i, folds_ind] = 0
        Z_folds[i, folds_ind] = np.nan
        A_hat = estimate_row(Z_folds, est_mask, i, folds_ind, col_dists, eta, ax = 1)
    
        # compute error wrt given metric
        err = mse(A_hat[folds_ind], ground_truth)
        if np.isnan(err):
            final_k -= 1
        else:
          tot_error += err
    # return average error over k folds
    return tot_error / final_k

def search_eta(data_table, mask, k, row, col_dists, negate_mask = False):
    """
    Search for an optimal eta using cross validation on
    the observed data.
    """
    def obj(eta):
    #     return evaluate_eta(data_table, mask, k, eta)
        return cross_validate_row_snn(data_table, mask, row, k, eta, col_dists, negate_mask = negate_mask)
    trials = Trials()
    best_eta = fmin(fn=obj, verbose = True, space=hp.uniform('eta', -0.3, 0.3), algo=tpe.suggest, max_evals=200, trials = trials)
    return best_eta['eta'], trials

def test_snn_whole(Z, Msk, testset_size, k, ax, row):
    """
    Compute the test error of the test set (testset_size proportion of the end of the matrix)

    Test set strategy: tune eta using train set, find neighbors for elements in the
    test set using the entire matrix

    Parameters:
    -----------
    Z : a N x T matrix
    Msk : a N x T masking matrix for Z
    testset_size : the size of the test set
    k : number of folds for cv
    ax : row nn (ax = 0) or col nn (ax = 1)
    row : the row of interest
    """
    test_ind = int(testset_size * len(Z[0]))
    
    # test_Z = Z[:, -test_ind:]
    # test_Msk = Msk[:, -test_ind:]
    train_Z = Z[:, :test_ind]
    train_Msk = Msk[:, :test_ind]

    # sample split
    dist_train_msk = train_Msk.copy()
    if ax == 1:
      dist_train_msk[row] = 0
    # TODO: this breaks with unit-unit sample splitting 
    dissims = all_dissims(train_Z, dist_train_msk, ax)
    eta_star, trials = search_eta(train_Z, train_Msk, k, row, dissims)

    test_inds = np.nonzero(Msk[row] == 1)[0]
    test_inds = test_inds[test_inds >= (len(Msk[row]) - test_ind)]
    dist_Msk = Msk.copy()
    if ax == 1:
      dist_Msk[row] = 0 # mask out row when computing distance
    dissims = all_dissims(Z, dist_Msk, ax)
    #print(col_dists.shape)
    estimates = estimate_row(Z, Msk, row, test_inds, dissims, eta_star, ax = ax)
    est_test_inds = estimates[test_inds]
    ground_truth = Z[row, test_inds]

    error = mse(est_test_inds, ground_truth)
    print("Avg MSE: " + str(error))
    return error, trials

def test_snn_end(Z, Msk, testset_size, k, ax, row):
    """
    Compute the test error of the test set (testset_size proportion of the end of the matrix)

    Test set strategy: tune eta using train set, find neighbors for elements in the
    test set using only the test set
    Parameters:
    -----------
    Z : a N x T matrix
    Msk : a N x T masking matrix for Z
    testset_size : the size of the test set
    k : number of folds for cv
    ax : row nn (ax = 0) or col nn (ax = 1)
    """
    test_ind = int(testset_size * len(Z[0]))
    
    test_Z = Z[:, -test_ind:]
    test_Msk = Msk[:, -test_ind:]
    train_Z = Z[:, :test_ind]
    train_Msk = Msk[:, :test_ind]

    # sample split
    dist_train_msk = train_Msk.copy()
    # TODO: this breaks with unit-unit sample splitting
    dist_train_msk[row] = 0
    
    dissims = all_dissims(train_Z, dist_train_msk, ax)
    eta_star, trials = search_eta(train_Z, train_Msk, k, row, dissims)

    test_inds = np.nonzero(test_Msk[row] == 1)[0]
    dist_Msk = test_Msk.copy()
    dist_Msk[row] = 0 # mask out row when computing distance
    dissims = all_dissims(test_Z, dist_Msk, ax)
    #print(col_dists.shape)
    estimates = estimate_row(test_Z, test_Msk, row, test_inds, dissims, eta_star, ax = ax)
    est_test_inds = estimates[test_inds]
    ground_truth = test_Z[row, test_inds]

    error = mse(est_test_inds, ground_truth)
    print("Avg MSE: " + str(error))
    return error, trials
