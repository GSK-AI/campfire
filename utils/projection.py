from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _scalar_proj(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Scalar projection of x onto y"""
    return np.dot(x, y) / np.linalg.norm(y)


def _vector_proj(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector projection of x onto y"""
    norm_y = y / np.linalg.norm(y)
    return norm_y * _scalar_proj(x, y)


def _normalized_scalar_proj(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Scalar projection of x onto y normalized by magnitude of y"""
    return _scalar_proj(x, y) / np.linalg.norm(y)


def _vector_rej(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector rejection of x onto y"""
    return x - _vector_proj(x, y)


def _scalar_rej(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Scalar rejection of x onto y"""
    return np.linalg.norm(_vector_rej(x, y))


def _normalized_scalar_rej(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Scalar rejection of x onto y normalized by magnitude of y"""
    return _scalar_rej(x, y) / np.linalg.norm(y)


def _scalar_projection(
    df: pd.DataFrame,
    feature_col: List[str],
    ref_col: str,
    ref_origin: str,
    ref_target: str,
    ref_sample_with_replacement: Optional[bool] = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    df = df.copy()

    origin = (
        df.loc[df[ref_col] == ref_origin, feature_col]
        .sample(
            frac=1.0,
            replace=ref_sample_with_replacement,
            random_state=random_state,
            axis=0,
        )
        .mean(axis=0)
        .values
    )
    df.loc[:, feature_col] = df.loc[:, feature_col].apply(lambda x: x - origin, axis=1)
    y = (
        df.loc[df[ref_col] == ref_target, feature_col]
        .sample(
            frac=1.0,
            replace=ref_sample_with_replacement,
            random_state=random_state,
            axis=0,
        )
        .mean(axis=0)
        .values
    )
    scalar_proj = df.loc[:, feature_col].apply(_normalized_scalar_proj, axis=1, y=y)
    scalar_rej = df.loc[:, feature_col].apply(_normalized_scalar_rej, axis=1, y=y)
    scalar_rej -= scalar_rej[df[df[ref_col] == ref_target].index.tolist()].mean(axis=0)
    return scalar_proj, scalar_rej


def scalar_projection(
    df: pd.DataFrame,
    feature_col: List[str],
    ref_col: str,
    ref_origin: str,
    ref_target: str,
    num_bootstrap: Optional[int] = 10,
    random_state: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate scalar projection score based on reference vector

    Using num_bootstrap > 0 will take bootstrap samples of ref_origin and
    ref_target. The final score in this case is will be the mean projection
    score on each bootstrapped ref vector.


    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_col : List[str]
        List of feature column names
    ref_col : str
        Reference metadata column name
    ref_origin : str
        Element in ref_col to be used as origin for reference vector
    ref_target : str
        Element in ref_col to be used as target for reference vector
    num_bootstrap : int, optional
        Number of bootstrap samples, by default 10
    random_state : int, optional
        Random state for resampling reproducibility, by default 0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Scalar projection and scalar rejection
    """
    if num_bootstrap < 0:
        raise ValueError("num_bootstrap must be >= 0.")

    scalar_proj = []
    scalar_rej = []
    replace = True if num_bootstrap else False
    num_runs = num_bootstrap or 1
    for _ in range(num_runs):
        s_proj, s_rej = _scalar_projection(
            df=df,
            feature_col=feature_col,
            ref_col=ref_col,
            ref_origin=ref_origin,
            ref_target=ref_target,
            ref_sample_with_replacement=replace,
            random_state=random_state,
        )
        scalar_proj.append(s_proj)
        scalar_rej.append(s_rej)
    return np.mean(scalar_proj, axis=0), np.mean(scalar_rej, axis=0)