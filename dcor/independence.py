"""
Functions for testing independence of several distributions.

The functions in this module provide methods for testing if
the samples generated from two random vectors are independent.
"""
from __future__ import annotations

from typing import TypeVar

import numpy as np
import scipy.stats

from ._dcor import u_distance_correlation_sqr

## Additional modules for Multivariate dcov-based test of independence------------
import math
from ._dcor import u_distance_covariance_sqr, gamma_ratio, rndm_projection  
from .distances import dist_sum
from ._rowwise import rowwise
from scipy.special import gammainc
# from mpmath import*




from ._dcor_internals import (
    _check_same_n_elements,
    _distance_matrix_generic,
    _u_distance_matrix,
    double_centered,
    mean_product,
    u_complementary_projection,
    u_product,
)
from ._hypothesis import HypothesisTest, _permutation_test_with_sym_matrix
from ._utils import (
    ArrayType,
    RandomLike,
    _random_state_init,
    _sqrt,
    _transform_to_2d,
)

Array = TypeVar("Array", bound=ArrayType)


def distance_covariance_test(
    x: Array,
    y: Array,
    *,
    num_resamples: int = 0,
    exponent: float = 1,
    random_state: RandomLike = None,
    n_jobs: int = 1,
) -> HypothesisTest[Array]:
    """
    Test of distance covariance independence.

    Compute the test of independence based on the distance
    covariance, for two random vectors.

    The test is a permutation test where the null hypothesis is that the two
    random vectors are independent.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        num_resamples: Number of permutations resamples to take in the
            permutation test.
        random_state: Random state to generate the permutations.
        n_jobs: Number of jobs executed in parallel by Joblib.

    Returns:
        Results of the hypothesis test.

    See Also:
        distance_covariance

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2, 3, 4],
        ...               [5, 6, 7, 8],
        ...               [9, 10, 11, 12],
        ...               [13, 14, 15, 16]])
        >>> b = np.array([[1, 0, 0, 1],
        ...               [0, 1, 1, 1],
        ...               [1, 1, 1, 1],
        ...               [1, 1, 0, 1]])
        >>> dcor.independence.distance_covariance_test(a, a)
        HypothesisTest(pvalue=1.0, statistic=208.0)
        >>> dcor.independence.distance_covariance_test(a, b)
        ...                                      # doctest: +ELLIPSIS
        HypothesisTest(pvalue=1.0, statistic=11.75323056...)
        >>> dcor.independence.distance_covariance_test(b, b)
        HypothesisTest(pvalue=1.0, statistic=1.3604610...)
        >>> dcor.independence.distance_covariance_test(a, b,
        ... num_resamples=5, random_state=0)
        HypothesisTest(pvalue=0.8333333333333334, statistic=11.7532305...)
        >>> dcor.independence.distance_covariance_test(a, b,
        ... num_resamples=5, random_state=13)
        HypothesisTest(pvalue=0.5..., statistic=11.7532305...)
        >>> dcor.independence.distance_covariance_test(a, a,
        ... num_resamples=7, random_state=0)
        HypothesisTest(pvalue=0.125, statistic=208.0)

    """
    x, y = _transform_to_2d(x, y)

    _check_same_n_elements(x, y)

    random_state = _random_state_init(random_state)

    # Compute U-centered matrices
    u_x = _distance_matrix_generic(
        x,
        centering=double_centered,
        exponent=exponent,
    )
    u_y = _distance_matrix_generic(
        y,
        centering=double_centered,
        exponent=exponent,
    )

    # Use the dcov statistic
    def statistic_function(distance_matrix: Array) -> Array:
        return u_x.shape[0] * mean_product(
            distance_matrix,
            u_y,
        )

    return _permutation_test_with_sym_matrix(
        u_x,
        statistic_function=statistic_function,
        num_resamples=num_resamples,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def partial_distance_covariance_test(
    x: Array,
    y: Array,
    z: Array,
    *,
    num_resamples: int = 0,
    exponent: float = 1,
    random_state: RandomLike = None,
    n_jobs: int | None = 1,
) -> HypothesisTest[Array]:
    """
    Test of partial distance covariance independence.

    Compute the test of independence based on the partial distance
    covariance, for two random vectors conditioned on a third.

    The test is a permutation test where the null hypothesis is that the first
    two random vectors are independent given the third one.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        z: Observed random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        num_resamples: Number of permutations resamples to take in the
            permutation test.
        random_state: Random state to generate the permutations.
        n_jobs: Number of jobs executed in parallel by Joblib.

    Returns:
        Results of the hypothesis test.

    See Also:
        partial_distance_covariance

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2, 3, 4],
        ...               [5, 6, 7, 8],
        ...               [9, 10, 11, 12],
        ...               [13, 14, 15, 16]])
        >>> b = np.array([[1, 0, 0, 1],
        ...               [0, 1, 1, 1],
        ...               [1, 1, 1, 1],
        ...               [1, 1, 0, 1]])
        >>> c = np.array([[1000, 0, 0, 1000],
        ...               [0, 1000, 1000, 1000],
        ...               [1000, 1000, 1000, 1000],
        ...               [1000, 1000, 0, 1000]])
        >>> dcor.independence.partial_distance_covariance_test(a, a, b)
        ...                                       # doctest: +ELLIPSIS
        HypothesisTest(pvalue=1.0, statistic=142.6664416...)
        >>> test = dcor.independence.partial_distance_covariance_test(a, b, c)
        >>> test.pvalue # doctest: +ELLIPSIS
        1.0
        >>> np.allclose(test.statistic, 0, atol=1e-6)
        True
        >>> test = dcor.independence.partial_distance_covariance_test(a, b, c,
        ... num_resamples=5, random_state=0)
        >>> test.pvalue # doctest: +ELLIPSIS
        0.1666666...
        >>> np.allclose(test.statistic, 0, atol=1e-6)
        True

    """
    random_state = _random_state_init(random_state)

    # Compute U-centered matrices
    u_x = _u_distance_matrix(x, exponent=exponent)
    u_y = _u_distance_matrix(y, exponent=exponent)
    u_z = _u_distance_matrix(z, exponent=exponent)

    # Compute projections
    proj = u_complementary_projection(u_z)

    p_xz = proj(u_x)
    p_yz = proj(u_y)

    # Use the pdcor statistic
    def statistic_function(distance_matrix: Array) -> Array:
        return u_x.shape[0] * u_product(
            distance_matrix,
            p_yz,
        )

    return _permutation_test_with_sym_matrix(
        p_xz,
        statistic_function=statistic_function,
        num_resamples=num_resamples,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def distance_correlation_t_statistic(
    x: Array,
    y: Array,
) -> Array:
    """
    Statistic used in :func:`distance_correlation_t_test`.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.

    Returns:
        T statistic.

    See Also:
        distance_correlation_t_test

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2, 3, 4],
        ...               [5, 6, 7, 8],
        ...               [9, 10, 11, 12],
        ...               [13, 14, 15, 16]])
        >>> b = np.array([[1, 0, 0, 1],
        ...               [0, 1, 1, 1],
        ...               [1, 1, 1, 1],
        ...               [1, 1, 0, 1]])
        >>> with np.errstate(divide='ignore'):
        ...     dcor.independence.distance_correlation_t_statistic(a, a)
        inf
        >>> dcor.independence.distance_correlation_t_statistic(a, b)
        ...                                      # doctest: +ELLIPSIS
        -0.4430164...
        >>> with np.errstate(divide='ignore'):
        ...     dcor.independence.distance_correlation_t_statistic(b, b)
        inf

    """
    bcdcor = u_distance_correlation_sqr(x, y)

    n = x.shape[0]
    v = n * (n - 3) / 2

    return np.sqrt(v - 1) * bcdcor / _sqrt(1 - bcdcor**2)


def distance_correlation_t_test(
    x: Array,
    y: Array,
) -> HypothesisTest[Array]:
    """
    Test of independence for high dimension.

    It is based on convergence to a Student t distribution.
    The null hypothesis is that the two random vectors are
    independent.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.

    Returns:
        Results of the hypothesis test.

    See Also:
        distance_correlation_t_statistic

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2, 3, 4],
        ...               [5, 6, 7, 8],
        ...               [9, 10, 11, 12],
        ...               [13, 14, 15, 16]])
        >>> b = np.array([[1, 0, 0, 1],
        ...               [0, 1, 1, 1],
        ...               [1, 1, 1, 1],
        ...               [1, 1, 0, 1]])
        >>> with np.errstate(divide='ignore'):
        ...     dcor.independence.distance_correlation_t_test(a, a)
        ...                                      # doctest: +ELLIPSIS
        HypothesisTest(pvalue=0.0, statistic=inf)
        >>> dcor.independence.distance_correlation_t_test(a, b)
        ...                                      # doctest: +ELLIPSIS
        HypothesisTest(pvalue=0.6327451..., statistic=-0.4430164...)
        >>> with np.errstate(divide='ignore'):
        ...     dcor.independence.distance_correlation_t_test(b, b)
        ...                                      # doctest: +ELLIPSIS
        HypothesisTest(pvalue=0.0, statistic=inf)

    """
    t_test = distance_correlation_t_statistic(x, y)

    n = x.shape[0]
    v = n * (n - 3) / 2
    df = v - 1

    p_value = 1 - scipy.stats.t.cdf(t_test, df=df)

    return HypothesisTest(pvalue=p_value, statistic=t_test)

#-----------------------------------------------------------------------------------------------------------------

"""
A Statistically and Numerically Efficient Independence Test Based on Random Projections and Distance Covariance

:cite:`b-dcov_random_projection`.

References
----------
.. bibliography:: ../refs.bib
   :labelprefix: B
   :keyprefix: b-
"""


def gamma_cdf(x, shape,  scale):
    # mp.dps = 25; mp.pretty = True
    # return gammainc(shape, a = 0, b = float(x / scale)) / np.exp(gammaln(shape))
    return gammainc(shape, float(x / scale))

def u_dist_cov_sqr_mv_test(X, Y, n_projs= 500, method='mergesort'):
    """

     Parameters
     ----------
     X : N x p, array of arrays, where p > 1
     Y : N x q, array of arrays, where q >= 1
     where p and q are the number of dimensions of X and Y, respectively and N: number of samples

     n_projs : Number of projections (integer type), optional
         DESCRIPTION. The default is 500. (paper suggested to consider: n_projs < N/logN, larger n_projs provides better results)
     method : fast computation method either 'mergesort' or 'avl', optional
         DESCRIPTION. The default is 'mergesort'.

    Returns
    -------
    Results of the hypothesis test.
    
    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> from scipy.stats import multivariate_normal
        >>> mean_vector = [2, 3, 5, 3, 2, 1]
        >>> matrix_size = 6
        >>> A = 0.5 * np.random.rand(matrixSize, matrix_size)
        >>> B = np.dot(A, A.transpose())
        >>> n_samples = 3000
        >>> X = multivariate_normal.rvs(mean_vector, B, size = n_samples)
        >>> X1 = X.T[:4]
        >>> X2 = X.T[4:] 
        >>> print(f"Test of independence using fast distance covariance = {u_dist_cov_sqr_mv_test(X1.T, X2.T)}")
        
    """

    n_samples = np.shape(X)[0]
    p = np.shape(X)[1]
    if Y.T.ndim == 1:
        q = 1
    else:
        q = np.shape(Y)[1]
      
      
    sqrt_pi_value = math.sqrt(math.pi)
    C_p = sqrt_pi_value * gamma_ratio(p)
    C_q = sqrt_pi_value * gamma_ratio(q)

    X_proj_1 = np.empty(( n_projs, n_samples))    
    Y_proj_1 = np.empty(( n_projs, n_samples))
    X_proj_2 = np.empty(( n_projs, n_samples))    
    Y_proj_2 = np.empty(( n_projs, n_samples))
    S2_n = 0
    S3_n = 0
    
    for i in range(n_projs):
        X_proj_1[ i, :] = rndm_projection(X, p)
        Y_proj_1[ i, :] = rndm_projection(Y, q)
        S2_n += (2 * dist_sum(X_proj_1[ i, :]))
        S3_n += (2 * dist_sum(Y_proj_1[ i, :]))
        X_proj_2[ i, :] = rndm_projection(X, p)
        Y_proj_2[ i, :] = rndm_projection(Y, q)
        
    omega1_ = rowwise(u_distance_covariance_sqr, 
                      X_proj_1, Y_proj_1, rowwise_mode= method)
    omega1_bar = C_p * C_q * np.mean(omega1_)
    
    S11_ =  np.array(rowwise(u_distance_covariance_sqr, 
                             X_proj_1, X_proj_1, rowwise_mode= method))
    S12_ =  np.array(rowwise(u_distance_covariance_sqr,
                             Y_proj_1, Y_proj_1, rowwise_mode= method))   
    S1_bar = C_p * C_q * np.mean(S11_* S12_)
    
    S2_bar = (C_p * S2_n) / (n_projs * n_samples * (n_samples-1))
    S3_bar = (C_q * S3_n) / (n_projs * n_samples * (n_samples-1))
    
    omega2_ = rowwise(u_distance_covariance_sqr, 
                      X_proj_1, X_proj_2, rowwise_mode = method)
    omega2_bar = (C_p ** 2) * np.mean(omega2_)
    
    omega3_ = rowwise(u_distance_covariance_sqr, 
                      Y_proj_1, Y_proj_2, rowwise_mode = method)
    omega3_bar = (C_q ** 2) * np.mean(omega3_)
    
    # calculate alpha and beta--------------------------------------
    denom = (((n_projs-1) * omega2_bar * omega3_bar) + S1_bar) / n_projs
    alpha = (0.5 * ((S2_bar * S3_bar) ** 2)) / denom
    beta = (0.5 * S2_bar * S3_bar) / denom

    # calculate test statistic and the p-value--------------
    Test_statistic = ((n_samples * omega1_bar) + (S2_bar * S3_bar))
    
    p_val = 1 - gamma_cdf(Test_statistic, 
                          shape = alpha,  scale = float(1 / beta))
    
    if p_val < 0: p_val = 0 # Adjust the output of numerical integration as produced by gammainc
    
    
    return HypothesisTest(pvalue = p_val, statistic = Test_statistic)

