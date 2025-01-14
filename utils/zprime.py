import logging
from typing import Optional,Tuple

import numpy as np
import pandas as pd

import math
import statistics
from scipy.stats import median_abs_deviation

def robust_stat(
    x: np.ndarray,
    c: Optional[float] = 1.5,
    beta: Optional[float] = 0.778,
    max_iter: Optional[int] = 1000,
    robust_tol: Optional[float] = 1e-4,
) -> Tuple[float, float]:
    """Calculate robust mean and robust standard deviation for an array

    The simple robust iterative approach incorporates all observations, treating them
    as "reliable", while also downweighting them until both the robust mu and the
    robust sigma estimates converge. The median and median absolute deviation (MAD) are
    good starting values for estimating the robust mean and robust standard deviation.

    In an example data set of {1, 2, 3, 4, 25}, the sample median is 3 and
    the normalized MAD is 1.483. After iteratively downweighting outliers,
    the robust mean becomes 3.307 while the robust sigma becomes 2.408.

    For more information, please refer to:
    Committee, Analytical Methods. ‘Robust Statistics–How Not to
    Reject Outliers. Part 1. Basic Concepts’. Analyst, vol. 114,
    The Royal Society of Chemistry, 1989, pp. 1693–1697,
    https://doi.org/10.1039/AN9891401693.

    Parameters
    ----------
    x : np.ndarray
        Array of data values of interest with type np.float64 or np.int64
    c : float, optional
        Cut-off point, by default 1.5
        See referenced article above for more information on c.
    beta : float, optional
        Correction factor dependent upon c, by default 0.778
        See referenced article above for more information on beta and
        recommended beta values for different c values.
    max_iter : int, optional
        Maximum number of iterations for finding robust estimates, by default 1000
    robust_tol : float, optional
        Tolerance for the difference between two consecutive iterations, by default 1e-4

    Returns
    -------
    Tuple[float, float]
        Robust mu and robust sigma
    """
    assert isinstance(x, np.ndarray), "Provided data is not of type np.ndarray"

    assert all(isinstance(val, (np.float64, np.int64)) for val in x), (
        "At least one value in the provided np.ndarray is not of type np.float64"
        " or np.int64"
    )

    if len(x) == 0:
        raise ValueError("Array is empty")
    elif len(x) == 1:
        return float(x[0]), 0.0

    n = len(x)
    c_adj = c * math.sqrt((n - 1) / n)

    new_mu = np.median(x)
    new_sigma = median_abs_deviation(x, scale="normal")

    for _ in range(max_iter):
        prev_mu = new_mu
        prev_sigma = new_sigma
        pseudo_x = [0] * n
        c_thresh = c_adj * new_sigma

        upper_thresh = new_mu + c_thresh
        lower_thresh = new_mu - c_thresh
        pseudo_x = np.maximum(x, lower_thresh)
        pseudo_x = np.minimum(pseudo_x, upper_thresh)

        new_mu = np.mean(pseudo_x)
        new_sigma = math.sqrt(statistics.variance(pseudo_x) / beta)

        if (
            max(
                abs(new_mu - prev_mu) / (prev_sigma + 1e-99),
                abs(new_sigma / (prev_sigma + 1e-99) - 1),
            )  # addeed 1e-99 in the denominator to allow for cases where sigma is 0
            < robust_tol
        ):
            break

    return new_mu, new_sigma




def _zprime(
    control_mu: float, control_sigma: float, sample_mu: float, sample_sigma: float
) -> float:
    """Calculate zprime given control and sample statistics

    Zprime formula is from:
    Zhang, Ji-Hu, et al. ‘A Simple Statistical Parameter for Use
    in Evaluation and Validation of High Throughput Screening Assays’.
    SLAS Discovery, vol. 4, no. 2, 1999, pp. 67–73,
    https://doi.org10.1177/108705719900400206.
    """
    return 1 - 3 * (control_sigma + sample_sigma) / abs(control_mu - sample_mu)


def robust_zprime(
    origin: np.ndarray,
    target: np.ndarray,
    c: Optional[float] = 1.5,
    beta: Optional[float] = 0.778,
    max_iter: Optional[int] = 1000,
    robust_tol: Optional[float] = 1e-4,
    zprime_tol: Optional[float] = 1e-99,
    return_stats: Optional[bool] = False,
) -> np.ndarray:
    """Calculate robust zprime between origin and target arrays

    Parameters
    ----------
    origin : np.ndarray
        Control data array containing either np.float64 values or np.int64 values
    target : np.ndarray
        Target sample data array containing either np.float64 values or np.int64 values
    c : float, optional
        Cut-off point, by default 1.5
        See referenced article under robust_stat in stats.py for more information on c.
    beta : float, optional
        Correction factor dependent upon c, by default 0.778
        See referenced article under robust_stat in stats.py for more information on
        beta and recommended beta values for different c values.
    max_iter : int, optional
        Maximum number of iterations for finding robust estimates, by default 1000
    robust_tol : float, optional
        Tolerance for the difference between two consecutive iterations in the
        robust_stat algorithm, by default 1e-4
    zprime_tol : float, optional
        Tolerance for calculating difference in mu for zprime, by default 1e-99
        The zprime_tol is added to target_mu when calculating zprime to avoid
        dividing by zero for cases when origin mu equals the target mu.
    return_stats: bool, optional
        Parameter for user to designate desired output, by default False
        See Returns description below for more details.

    Returns
    -------
    np.ndarray
        If return_stats is True, then the returned array is of type
        np.ndarray[np.float64, np.float64, np.float64, np.float64, np.float64]
        containing the zprime value, robust origin mu, robust origin sigma,
        robust target mu, and robust target sigma, respectively.

        If return_stats is False, then the returned array is the zprime value of type
        np.ndarray[np.float64].
    """
    origin_mu, origin_sigma = robust_stat(origin, c, beta, max_iter, robust_tol)
    target_mu, target_sigma = robust_stat(target, c, beta, max_iter, robust_tol)

    if origin_mu == target_mu:
        logging.warning(
            f"Origin mu and target mu are equal with a value of: {round(origin_mu, 3)}"
        )
        logging.warning(
            f"Adding zprime_tol of {zprime_tol} to target mu to continue calculations."
        )
        target_mu += zprime_tol

    zprime_val = _zprime(origin_mu, origin_sigma, target_mu, target_sigma)

    if return_stats:
        robust_array = [zprime_val, origin_mu, origin_sigma, target_mu, target_sigma]

        return np.array(robust_array)

    return np.array([zprime_val])


def robust_zprime_df(
    df: pd.DataFrame,
    value_col: str,
    ref_col: str,
    ref_origin: str,
    c: Optional[float] = 1.5,
    beta: Optional[float] = 0.778,
    max_iter: Optional[int] = 1000,
    robust_tol: Optional[float] = 1e-4,
    zprime_tol: Optional[float] = 1e-99,
) -> pd.DataFrame:
    """Calculate robust zprime between origin and multiple targets in a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    value_col : str
        Name of column that contains either floats or integers to be used in
        the zprime calculation
    ref_col : str
        Reference metadata column name containing data labels as strings
    ref_origin : str
        Element in ref_col to be used as origin group label
    c : float, optional
        Cut-off point, by default 1.5
        See referenced article under robust_stat in stats.py for more information on c.
    beta : float, optional
        Correction factor dependent upon c, by default 0.778
        See referenced article under robust_stat in stats.py for more information on
        beta and recommended beta values for different c values.
    max_iter : int, optional
        Maximum number of iterations for finding robust estimates, by default 1000
    robust_tol : float, optional
        Tolerance for the difference between two consecutive iterations in the
        robust statistics algorithm, by default 1e-4
    zprime_tol : float, optional
        Tolerance for calculating difference in mu for zprime, by default 1e-99
        The zprime_tol is added to target_mu when calculating zprime to avoid
        dividing by zero for cases when origin mu equals the target mu.

    Returns
    -------
    pd.DataFrame
        ref_col, zprime, robust mu, and robust sigma
        The origin group will always be in the first row with a zprime of -np.inf.
    """
    assert isinstance(df, pd.DataFrame), "df argument is not of type DataFrame"
    assert isinstance(value_col, str), "value_col argument is not of type str"
    assert isinstance(ref_col, str), "ref_col argument is not of type str"
    assert isinstance(ref_origin, str), "ref_origin argument is not of type str"
    assert value_col in df.columns, "value_col argument is not a column in the dataset"
    assert all(
        isinstance(val, (float, int)) for val in df[value_col]
    ), "at least one value in the value column is not of type float or int"
    assert ref_col in df.columns, "ref_col argument is not a column in the dataset"
    assert (
        ref_origin in df[ref_col].values
    ), "ref_origin is not found in the reference column"

    zprime_df = (
        (
            df.groupby(by=ref_col, sort=False)[value_col].agg(
                lambda x: robust_stat(
                    x=x.to_numpy(),
                    c=c,
                    beta=beta,
                    max_iter=max_iter,
                    robust_tol=robust_tol,
                )
            )
        )
        .apply(pd.Series)
        .reset_index()
        .rename({0: "robust_mu", 1: "robust_sigma"}, axis=1)
    )

    origin_mu = zprime_df.loc[zprime_df[ref_col] == ref_origin]["robust_mu"].item()
    origin_sigma = zprime_df.loc[zprime_df[ref_col] == ref_origin][
        "robust_sigma"
    ].item()

    equal_mu_targets = zprime_df.loc[
        zprime_df["robust_mu"] == origin_mu, ref_col
    ].to_list()

    equal_mu_targets.remove(ref_origin)

    if len(equal_mu_targets) > 0:
        logging.warning(
            f"Origin mu and target mu are equal with a value of {round(origin_mu, 3)} "
            f"for the target samples: {equal_mu_targets}"
        )
        logging.warning(
            f"Adding zprime_tol of {zprime_tol} to target mu to continue calculations."
        )

        zprime_df.loc[
            zprime_df[ref_col].isin(equal_mu_targets),
            "robust_mu",
        ] += zprime_tol

    zprime_df["zprime"] = _zprime(
        origin_mu, origin_sigma, zprime_df["robust_mu"], zprime_df["robust_sigma"]
    )

    zprime_df = zprime_df[
        [ref_col, "zprime", "robust_mu", "robust_sigma"]
    ]  # Moving zprime column to be the second column

    zprime_df = pd.concat(
        [
            zprime_df.loc[zprime_df[ref_col] == ref_origin],
            zprime_df.loc[zprime_df[ref_col] != ref_origin],
        ]
    )  # Moving the origin group to be the first row

    return zprime_df