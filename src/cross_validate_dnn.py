import dnn
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def cross_validate_row_dnn(Z, M, i, k, eta, col_dists, estimate, error_metric, kernel, negate_mask = False):
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
    #k = len(obvs_inds_x)
    # split coordinates into k-folds
    if len(obvs_inds_x) < 1:
        return np.inf
    folds = np.array_split(obvs_inds_x, k, axis = 0)
    tot_error = 0
    final_k = k

    for j in range(k):
        # indices of the observed outcomes in the validation set in this iteration
        folds_ind = folds[j]

        ground_truth = Z[i, folds_ind]
        ground_truth_row = Z[i]
        est_mask = M.copy()
        Z_folds = Z.copy()
        # hide observed values in the validation set
        # CURRENT STRATEGY: mask row for each fold indice
        # TODO: this breaks with k = 1
        est_mask[i, folds_ind] = 0
        Z_folds[i, folds_ind] = np.nan
        A_hat = estimate(Z_folds, est_mask, i, folds_ind, col_dists, eta, metric = error_metric)
    
        # compute error wrt given metric
        err = 0
        if error_metric == 'wasserstein2':
            est_val = A_hat[folds_ind]
            #est_val = A_hat
            ground_truth = np.squeeze(ground_truth, axis = -1)
            #print(est_val)
            #print(ground_truth)
            err = dnn.avg_wasserstein2_dist(est_val, ground_truth)
            #print(err)
        elif error_metric == 'mmd':
            # in this case A_hat is a (potentially) ragged list
            # TODO: can this be vectorized?
            err = dnn.avg_mmd_distance(A_hat, ground_truth_row, folds_ind, kernel)
            # for val in folds_ind:
            #     est_val = A_hat[val]
            #     err_onecell = dnn.sqmmd_est2(est_val, ground_truth_row[val], kernel = kernel)
            #     err += err_onecell
            # err = err / len(folds_ind)
        tot_error += err
    # return average error over k folds
    if np.isnan(tot_error):
        return np.inf
    return tot_error / final_k

def search_eta(data_table, mask, k, row, col_dists, est_func, metric, range_min, range_max, kernel = None, negate_mask = False):
    """
    Search for an optimal eta using cross validation on
    the observed data.
    """
    def obj(eta):
    #     return evaluate_eta(data_table, mask, k, eta)
        return cross_validate_row_dnn(data_table, mask, row, k, eta, col_dists, est_func, error_metric = metric, kernel = kernel, negate_mask = negate_mask)
    trials = Trials()
    best_eta = fmin(fn=obj, verbose = True, space=hp.uniform('eta', range_min, range_max), algo=tpe.suggest, max_evals=200, trials = trials)
    return best_eta['eta'], trials

# TODO: make sure these are general to any dataset, not just HS
# TODO: should provide test error computation for non-end positions
#       in the matrix
def test_w2_hs_whole(M_hs, A_hs, testset_size, k, row):
    """
    Compute the test error of the test set (testset_size proportion of the end of the matrix)

    Test set strategy: tune eta using train set, find neighbors for elements in the
    test set using the entire matrix
    """
    test_ind = int(testset_size * len(M_hs[0]))
    M_hs = np.sort(M_hs, axis = -1)
    
    test_M = M_hs[:, -test_ind:]
    test_A = A_hs[:, -test_ind:]
    train_M = M_hs[:, :test_ind]
    train_A = A_hs[:, :test_ind]

    # sample split
    dist_train_A = train_A.copy()
    dist_train_A[row] = 0
    col_dists = dnn.get_col_differences(train_M, dist_train_A, metric = "wasserstein2", kernel = None)
    eta_star, trials = search_eta(train_M, train_A, k, row, col_dists, dnn.estimate_row, "wasserstein2")

    test_inds = np.nonzero(A_hs[row] == 1)[0]
    test_inds = test_inds[test_inds >= (len(A_hs[row]) - test_ind)]
    dist_A = A_hs.copy()
    dist_A[row] = 0 # mask out row when computing distance
    col_dists = dnn.get_col_differences(M_hs, dist_A, "wasserstein2", kernel = None)
    #print(col_dists.shape)
    estimates = dnn.estimate_row(M_hs, A_hs, row, test_inds, col_dists, eta_star, "wasserstein2")
    est_test_inds = estimates[test_inds]
    ground_truth = M_hs[row, test_inds].squeeze(axis = -1)

    ws2_error = dnn.avg_wasserstein2_dist(est_test_inds, ground_truth)
    print("Avg Wasserstein Error: " + str(ws2_error))
    return ws2_error, trials

def test_w2_hs_end(M_hs, A_hs, testset_size, k, row):
    """
    Compute the test error of the test set (testset_size proportion of the end of the matrix)

    Current strategy: tune eta using train set, only find neighbors within the test set 
    """
    test_ind = int(testset_size * len(M_hs[0]))
    M_hs = np.sort(M_hs, axis = -1)
    
    test_M = M_hs[:, -test_ind:]
    test_A = A_hs[:, -test_ind:]
    train_M = M_hs[:, :test_ind]
    train_A = A_hs[:, :test_ind]

    # sample split
    dist_train_A = train_A.copy()
    dist_train_A[row] = 0
    col_dists = dnn.get_col_differences(train_M, dist_train_A, metric = "wasserstein2", kernel = None)
    eta_star, trials = search_eta(train_M, train_A, k, row, col_dists, dnn.estimate_row, "wasserstein2")

    test_inds = np.nonzero(test_A[row] == 1)[0]
    dist_A = test_A.copy()
    dist_A[row] = 0 # mask out row when computing distance
    col_dists = dnn.get_col_differences(test_M, dist_A, "wasserstein2", kernel = None)
    #print(col_dists.shape)
    estimates = dnn.estimate_row(test_M, test_A, row, test_inds, col_dists, eta_star, "wasserstein2")
    est_test_inds = estimates[test_inds]
    ground_truth = test_M[row, test_inds].squeeze(axis = -1)

    ws2_error = dnn.avg_wasserstein2_dist(est_test_inds, ground_truth)
    print("Avg Wasserstein Error: " + str(ws2_error))
    return ws2_error, trials

def test_mmd_hs_whole(M_hs, A_hs, testset_size, k, row, kernel):
    """
    Compute the test error of the test set (testset_size proportion of the end of the matrix)
    
    Current strategy: tune eta using train set, only find neighbors within the test set 
    """
    test_ind = int(testset_size * len(M_hs[0]))
    M_hs = np.sort(M_hs, axis = -1)
    
    test_M = M_hs[:, -test_ind:]
    test_A = A_hs[:, -test_ind:]
    train_M = M_hs[:, :test_ind]
    train_A = A_hs[:, :test_ind]

    # sample split
    dist_train_A = train_A.copy()
    dist_train_A[row] = 0
    col_dists = dnn.get_col_differences(train_M, dist_train_A, metric = "mmd", kernel = kernel)
    eta_star, trials = search_eta(train_M, train_A, k, row, col_dists, dnn.estimate_row, "mmd")

    test_inds = np.nonzero(A_hs[row] == 1)[0]
    test_inds = test_inds[test_inds >= (len(A_hs[row]) - test_ind)]
    if len(test_inds) < 1:
      return np.nan, trials
    dist_A = A_hs.copy()
    dist_A[row] = 0 # mask out row when computing distance
    col_dists = dnn.get_col_differences(M_hs, dist_A, metric = "mmd", kernel = kernel)
    #print(col_dists.shape)
    estimates = dnn.estimate_row(M_hs, A_hs, row, test_inds, col_dists, eta_star, "mmd")
    #est_test_inds = estimates[test_inds]
    ground_truth = M_hs[row]

    mmd_error = dnn.avg_mmd_distance(estimates, ground_truth, test_inds, kernel = kernel)
    print("Avg MMD Error: " + str(mmd_error))
    return mmd_error, trials

def test_mmd_hs_end(M_hs, A_hs, testset_size, k, row, kernel):
    """
    Compute the test error of the test set (testset_size proportion of the end of the matrix)

    Current strategy: tune eta using train set, only find neighbors within the test set 
    """
    test_ind = int(testset_size * len(M_hs[0]))
    #M_hs = np.sort(M_hs, axis = -1)
    
    test_M = M_hs[:, -test_ind:]
    test_A = A_hs[:, -test_ind:]
    train_M = M_hs[:, :test_ind]
    train_A = A_hs[:, :test_ind]

    # sample split
    dist_train_A = train_A.copy()
    dist_train_A[row] = 0
    # TODO: fix kernel name vs kernel function discrepancy
    col_dists = dnn.get_col_differences(train_M, dist_train_A, metric = "mmd", kernel = kernel)
    eta_star, trials = search_eta(train_M, train_A, k, row, col_dists, dnn.estimate_row, "mmd")

    test_inds = np.nonzero(test_A[row] == 1)[0]
    print("Test Inds")
    print(test_inds)
    print(len(test_inds))
    if len(test_inds) < 1:
        return np.nan, trials
    dist_A = test_A.copy()
    dist_A[row] = 0 # mask out row when computing distance
    col_dists = dnn.get_col_differences(test_M, dist_A, metric = "mmd", kernel = kernel)
    #print(col_dists.shape)
    estimates = dnn.estimate_row(test_M, test_A, row, test_inds, col_dists, eta_star, "mmd")
    #est_test_inds = estimates[test_inds]
    ground_truth = test_M[row]

    mmd_error = dnn.avg_mmd_distance(estimates, ground_truth, test_inds, kernel = kernel)
    print("Avg MMD Error: " + str(mmd_error))
    return mmd_error, trials


# TODO: find a way to find neighbors using the inverse of mask (i.e. match with
# masked entries to find neighbors).