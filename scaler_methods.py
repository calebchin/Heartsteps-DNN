# include implementations for SoftImpute, USVT, LI 2019NN

# for softimpute, use fancyimpute package

from fancyimpute import SoftImpute, BiScaler
import numpy as np
# test mse for SoftImpute
def mse(A_hat, A):
    """
    Returns the mean squared error of estimate A_hat from ground truth A
    """
    return np.nanmean((A_hat - A)**2)

def test_softimpute(A, M, test_inds):
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
