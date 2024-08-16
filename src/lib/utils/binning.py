import numpy as np
import pandas as pd

def val_log_bin(
    val: np.ndarray,
    n_bin_per_decade: int = 20,
    n_bin_under_start: int = 5,
    start: float = 0.1
) -> np.ndarray:
    linbins = np.linspace(
        0,
        start,
        num=n_bin_under_start,
        endpoint=False
    )
    labels = [
        val[np.digitize(val, linbins, right=True)==i].mean() 
        for i in range(1, n_bin_under_start)
    ]
    logbins = np.logspace(
        np.log10(start), 
        np.log10(val.max()), 
        int(np.ceil(np.log10(val.max()) - np.log10(start)) * n_bin_per_decade)+1
    )
    labels.extend([
        10**(np.log10(val[np.digitize(val, logbins, right=True)==i]).mean())
        for i in range(1, len(logbins))
    ])
    bins = np.concatenate([linbins, logbins])
    return bins, labels

def df_log_bin(
    df: pd.DataFrame,
    col: str,
    label_col: str = "labels",
    start: float = 0.1,
    n_bin_per_decade: int = 20,
    n_bin_under_start: int = 5,
    level: str = None,
) -> pd.DataFrame:
    if not level:
        val = df[col].to_numpy()
        bins, labels = val_log_bin(
            val, 
            start=start,
            n_bin_per_decade = n_bin_per_decade,
            n_bin_under_start =  n_bin_under_start,
        )
        temp = np.digitize(val, bins, right=True) - 1
        temp[temp>=len(labels)] = len(labels) - 1
        df[label_col] = [labels[i] for i in temp]
        return df
    else:
        dfs = []
        for _, x in list(df.groupby(level)):
            dfs.append(
                df_log_bin(
                    x.dropna(),
                    col,
                    label_col=label_col,
                    start=start,
                    n_bin_per_decade=n_bin_per_decade,
                    n_bin_under_start= n_bin_under_start,
                )
            )
        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        return dfs

def val_lin_bin(
    val: np.ndarray,
    n_bin: int = 25,
) -> np.ndarray:
    bins = np.linspace(
        val.min(),
        val.max(),
        num=n_bin+1,
    )
    labels = [
        val[np.digitize(val, bins, right=True)==i].mean() 
        for i in range(1, len(bins))
    ]
    return bins, labels

def df_lin_bin(
    df: pd.DataFrame,
    col: str,
    n_bin: int = 25,
    label_col: str = "labels",
    level: str = None,
) -> pd.DataFrame:
    if not level:
        val = df[col].to_numpy()
        bins, labels = val_lin_bin(val, n_bin=n_bin)
        df[label_col] = [labels[i] for i in np.digitize(val, bins, right=True) - 1]
        return df
    else:
        dfs = []
        for _, x in list(df.groupby(level)):
            dfs.append(
                df_lin_bin(
                    x.dropna(),
                    col,
                    n_bin=n_bin,
                    label_col=label_col,
                )
            )
        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        return dfs

def val_quantile_bin(
    val: np.ndarray,
    n_bin: int = 50,
    x_arithmetic_mean: bool = True,
) -> np.ndarray:
    val = np.sort(val)
    bins = np.linspace(
        0,
        len(val)-1,
        num=n_bin+1,
    )
    bins = val[np.round(bins).astype(int)]
    if x_arithmetic_mean:
        labels = [
            np.nanmean(val[np.digitize(val, bins, right=True)==i])
            for i in range(1, len(bins))
        ]
    else:
        labels = [
            np.exp(np.nanmean(np.log(val[np.digitize(val, bins, right=True)==i])))
            for i in range(1, len(bins))
        ]
    return bins, labels

def df_quantile_bin(
    df: pd.DataFrame,
    col: str,
    n_bin: int | float = 50,
    label_col: str = "labels",
    level: str = None,
    x_arithmetic_mean: bool = True,
) -> pd.DataFrame:
    if not level:
        val = df[col].to_numpy()
        bins, labels = val_quantile_bin(val, n_bin=n_bin, x_arithmetic_mean=x_arithmetic_mean,)
        df[label_col] = [labels[i] for i in np.digitize(val, bins, right=True) - 1]
        return df
    else:
        dfs = []
        for _, x in list(df.groupby(level)):
            if isinstance(n_bin, int):
                temp_n_bin = n_bin
            else:
                temp_n_bin = np.max((2, round(len(x) * n_bin)))
            dfs.append(
                df_quantile_bin(
                    x.dropna(),
                    col,
                    n_bin=temp_n_bin,
                    label_col=label_col,
                    x_arithmetic_mean=x_arithmetic_mean,
                )
            )
        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        return dfs


def log_mean(arr):
    m = 10**(np.log10(arr).mean())
    return m

def log_sd(arr):
    t = 10**(np.log10(arr).std())
    return (log_mean(arr)/t, log_mean(arr)*t)


def pos_mean(arr):
    m = arr[arr>0].mean()
    return m