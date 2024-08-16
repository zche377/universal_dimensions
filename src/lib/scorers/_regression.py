import warnings
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(level=logging.INFO)
import torch
import numpy as np
import xarray as xr

from torchmetrics.functional import pearson_corrcoef

from bonner.caching import cache
from bonner.computation.metrics import pearson_r, covariance, spearman_r
from bonner.computation.regression._utilities import Regression
from bonner.computation.regression import (
    create_splits,
    LinearRegression,
    PLSRegression,
    SGDLinearRegression,
)

from lib.scorers._definition import Scorer, CrossPredictorScorer


def _load_regression(regression: str) -> callable:
    match regression:
        case "ols" | "ridgecv":
            return LinearRegression
        case "pls":
            return PLSRegression
        case "sgd":
            return SGDLinearRegression
        case _:
            raise ValueError("regression must be 'ols', 'pls', 'sgd', or 'ridgecv'")         

def _compute_metric(
    metric: str,
    y_true: list[torch.Tensor],
    y_predicted: list[torch.Tensor],
) -> torch.Tensor:
    match metric:
        case "correlation":
            metric = pearson_r
        case "spearman":
            metric = spearman_r
        case "covariance":
            metric = covariance
        case _:
            raise ValueError("metric must be 'correlation' or 'covariance'")

    scores = []
    n_folds = len(y_true)
    for _ in range(n_folds):
        y_true_, y_predicted_ = y_true.pop(), y_predicted.pop()
        
        scores.append(metric(y_true_, y_predicted_).cpu())

    # output scores-per-fold
    scores = torch.stack(scores)
    return scores

def _best_alpha_per_column(
        x: torch.Tensor, y: torch.Tensor, alphas: list[float], device: str,
    ) -> torch.Tensor:
        helper = RidgeGCVRegression(alphas=alphas, device=device)
        helper.fit(x, y)
        return helper.alpha_.cpu()
   

class RegressionScorer(Scorer):
    def __init__(
        self,
        regression: str,  # "ols" or "ridgecv" or "pls" or "sgd"
        n_folds: int,
        metric: str = "correlation",
        shuffle: bool = True,
        seed: int = 11,
        device: torch.device | str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        alphas: list[float] = np.concatenate(([0], np.logspace(-3, 4, 8))).tolist(),
        **regression_kwargs,
    ) -> None:
        self.regression = regression
        self.metric = metric
        self.n_folds = n_folds
        self.shuffle: bool = shuffle
        self.seed: int = seed
        self.regression_kwargs = regression_kwargs
        self.device = device
        self.alphas = alphas
        identifier = f"regression.{regression}"
        if len(regression_kwargs) > 0:
            identifier += "".join([f".{k}={v}" for k, v in regression_kwargs.items()])
        identifier += (
            f".n_folds={self.n_folds}"
            f".metric={self.metric}"
            f".shuffle={self.shuffle}"
            f".seed={self.seed}"
        )
        super().__init__(identifier=identifier)

    def _compute_predictions(
        self,
        *,
        predictor: xr.DataArray,
        target: xr.DataArray,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        x = torch.tensor(predictor.values).to(self.device)
        y = torch.tensor(target.values).to(self.device)
        splits = create_splits(n=y.shape[-2], n_folds=self.n_folds, shuffle=self.shuffle, seed=self.seed)
        
        y_true, y_predicted = [], []
        
        for indices_test in splits:
            indices_train = np.setdiff1d(np.arange(y.shape[-2]), np.array(indices_test))
            x_train, x_test = x[..., indices_train, :], x[..., indices_test, :]
            y_train, y_test = y[..., indices_train, :], y[..., indices_test, :]
            
            if self.regression == "ridgecv":
                model = RidgeGCVRegression(alphas=self.alphas, device=self.device)
                model.fit(x_train, y_train)
            else:
                model = _load_regression(self.regression)(**self.regression_kwargs)
                model.fit(x=x_train, y=y_train)
            
            y_true.append(y_test)
            y_predicted.append(model.predict(x_test))
            
        return y_true, y_predicted
    
    # not called directly, but by the parent class
    def _score(
        self,
        *,
        predictor: xr.DataArray,
        target: xr.DataArray,
        target_dim: str,
    ) -> xr.DataArray:
        y_true, y_predicted = self._compute_predictions(
            predictor=predictor,
            target=target,
        )
        scores = _compute_metric(metric=self.metric, y_true=y_true, y_predicted=y_predicted)
        if scores.dim() == 1:
            scores = scores.unsqueeze(-1)
        return xr.DataArray(
            name="score",
            data=scores.cpu().numpy(),
            dims=("fold", target_dim),
            coords={
                "fold": range(self.n_folds),
                target_dim: target[target_dim].values,
            },
            attrs=(
                {
                    "predictor": predictor.name,
                    "target": target.name,
                    "regression": self.regression,
                    "n_folds": self.n_folds,
                    "metric": self.metric,
                    "shuffle": int(self.shuffle),
                    "seed": self.seed,
                }
                | self.regression_kwargs
            ),
        )


class CrossPredictorRegressionScorer(CrossPredictorScorer):
    def __init__(
        self,
        regression: str,  # "ols" or "ridgecv" or "pls" or "sgd"
        n_folds: int,
        metric: str = "correlation",
        shuffle: bool = True,
        seed: int = 11,
        device: torch.device | str = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        alphas: list[float] = np.concatenate(([0], np.logspace(-3, 4, 8))).tolist(),
        **regression_kwargs,
    ) -> None:
        self.regression = regression
        self.metric = metric
        self.n_folds = n_folds
        self.shuffle: bool = shuffle
        self.seed: int = seed
        self.regression_kwargs = regression_kwargs
        self.device = device
        self.alphas = alphas
        identifier = f"cross_predictor_regression.{regression}"
        if len(regression_kwargs) > 0:
            identifier += "".join([f".{k}={v}" for k, v in regression_kwargs.items()])
        identifier += (
            f".n_folds={self.n_folds}"
            f".metric={self.metric}"
            f".shuffle={self.shuffle}"
            f".seed={self.seed}"
        )
        super().__init__(identifier=identifier)
        
    def _compute_predictions(
        self,
        *,
        predictor_train_rep: xr.DataArray, 
        predictor_test_rep: xr.DataArray, 
        target: xr.DataArray,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        x_train_rep = torch.tensor(predictor_train_rep.values).to(self.device)
        x_test_rep = torch.tensor(predictor_test_rep.values).to(self.device)
        y = torch.tensor(target.values).to(self.device)
        splits = create_splits(n=y.shape[-2], n_folds=self.n_folds, shuffle=self.shuffle, seed=self.seed)
        
        y_true, y_predicted_train_rep, y_predicted_test_rep = [], [], []
        
        for indices_test in splits:
            indices_train = np.setdiff1d(np.arange(y.shape[-2]), np.array(indices_test))
            x_train_train_rep, x_test_train_rep = x_train_rep[..., indices_train, :], x_train_rep[..., indices_test, :]
            x_test_test_rep = x_test_rep[..., indices_test, :]
            y_train, y_test = y[..., indices_train, :], y[..., indices_test, :]
            
            if self.regression == "ridgecv":
                self.regression_kwargs["l2_penalty"] = _best_alpha_per_column(
                    x=x_train_train_rep, y=y_train, alphas=self.alphas, device=self.device
                )
                
            model = _load_regression(self.regression)(**self.regression_kwargs)
            if self.regression == "ridgecv":
                self.regression_kwargs.pop("l2_penalty")
            model.fit(x=x_train_train_rep, y=y_train)
            
            y_true.append(y_test)
            y_predicted_train_rep.append(model.predict(x_test_train_rep))
            y_predicted_test_rep.append(model.predict(x_test_test_rep))
        
        return y_true, y_predicted_train_rep, y_predicted_test_rep

    def _score(
        self,
        *,
        predictor_train_rep: xr.DataArray, 
        predictor_test_rep: xr.DataArray, 
        target: xr.DataArray,
        target_dim: str,
    ) -> xr.DataArray:
        y_true, y_predicted_train_rep, y_predicted_test_rep = self._compute_predictions(
            predictor_train_rep=predictor_train_rep,
            predictor_test_rep=predictor_test_rep,
            target=target,
        )
        reliability_scores = _compute_metric(
            metric=self.metric,
            y_true=y_predicted_train_rep, 
            y_predicted=y_predicted_test_rep,
        )
        similarity_scores = _compute_metric(
            metric=self.metric,
            y_true=y_true, 
            y_predicted=y_predicted_train_rep,
        )
        
        def _squeeze(scores):
            if scores.dim() == 1:
                scores = scores.unsqueeze(-1)
                return scores
            
        reliability_scores, similarity_scores = _squeeze(reliability_scores), _squeeze(similarity_scores)
        
        def _to_dataarray(scores):
            return xr.DataArray(
                name="score",
                data=scores.cpu().numpy(),
                dims=("fold", target_dim),
                coords={
                    "fold": range(self.n_folds),
                    target_dim: target[target_dim].values,
                },
                
            )
        
        return xr.Dataset(
            data_vars={
                "reliability_score": _to_dataarray(reliability_scores),
                "similarity_score": _to_dataarray(similarity_scores),
            },
            attrs=(
                {
                    "predictor_train_rep": predictor_train_rep.name,
                    "predictor_test_rep": predictor_test_rep.name,
                    "target": target.name,
                    "regression": self.regression,
                    "n_folds": self.n_folds,
                    "metric": self.metric,
                    "shuffle": int(self.shuffle),
                    "seed": self.seed,
                }
                | self.regression_kwargs
            ),
        )


def unify_dtypes(*args, target_dtype=None, precision='lowest'):
    dtypes_order = [torch.float16, torch.float32, torch.float64]
    dtypes = [arg.dtype for arg in args if isinstance(arg, torch.Tensor)]
    if not dtypes:
        return args
    if not target_dtype:
        if precision == 'highest':
            target_dtype = max(dtypes, key=dtypes_order.index)
        elif precision == 'lowest':
            target_dtype = min(dtypes, key=dtypes_order.index)
    result = tuple(arg.clone().to(dtype=target_dtype) 
                   if isinstance(arg, torch.Tensor) else arg for arg in args)
    return result[0] if len(result) == 1 else result

def convert_tensor_backend(data, backend='torch'):
    assert backend == "torch"
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return torch.utils.dlpack.from_dlpack(data.toDlpack())

def convert_to_tensor(*args, dtype=None, device=None, copy=False):
    def convert_item(arg):
        def process_tensor(tensor):
            tensor = tensor.clone() if copy else tensor
            if dtype is not None:
                tensor = tensor.to(dtype)
            if device is not None:
                tensor = tensor.to(device)
            return tensor

        if isinstance(arg, torch.Tensor):
            return process_tensor(arg)
        elif isinstance(arg, np.ndarray):
            arg = convert_tensor_backend(arg, 'torch')
            return process_tensor(arg)
        elif isinstance(arg, list):
            return [convert_item(item) for item in arg]
        elif isinstance(arg, dict):
            return {key: convert_item(val) for key, val in arg.items()}
        return arg

    outputs = [convert_item(arg) for arg in args]
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


# adapted from coco's deepjuice/alignment.py - TorchEstimator and TorchRidgeGCV
class RidgeGCVRegression(Regression):
    def __init__(
        self,
        alphas: list[float] = np.logspace(-3, 4, 8).tolist(),
        fit_intercept: bool = True,
        scale_x: bool = False,
        scoring: str = "pearsonr",
        alpha_per_target: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.scale_x = scale_x
        self.scoring = scoring
        self.device = device
        self.alpha_per_target = alpha_per_target
        self.dtype = dtype
        
        self.is_fitted = False

    def __repr__(self):
        if not self.is_fitted:
            return "TorchRidgeGCV (No Fit)"
        if self.is_fitted:
            return "TorchRidgeGCV (Fitted)"

    @staticmethod
    def _decomp_diag(v_prime, Q):
        return (v_prime * Q**2).sum(axis=-1)

    @staticmethod
    def _diag_dot(D, B):
        if len(B.shape) > 1:
            D = D[(slice(None),) + (None,) * (len(B.shape) - 1)]
        return D * B

    @staticmethod
    def _find_smallest_angle(query, vectors):
        abs_cosine = torch.abs(torch.matmul(query, vectors))
        return torch.argmax(abs_cosine).item()

    def _compute_gram(self, x, sqrt_sw):
        x_mean = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        return x.matmul(x.T), x_mean

    def _eigen_decompose_gram(self, x, y, sqrt_sw):
        K, x_mean = self._compute_gram(x, sqrt_sw)
        if self.fit_intercept:
            K += torch.outer(sqrt_sw, sqrt_sw)
        eigvals, Q = torch.linalg.eigh(K)
        QT_y = torch.matmul(Q.T, y)
        return x_mean, eigvals, Q, QT_y

    def _solve_eigen_gram(self, alpha, y, sqrt_sw, x_mean, eigvals, Q, QT_y):
        w = 1.0 / (eigvals + alpha)
        if self.fit_intercept:
            normalized_sw = sqrt_sw / torch.linalg.norm(sqrt_sw)
            intercept_dim = self._find_smallest_angle(normalized_sw, Q)
            w[intercept_dim] = 0  # cancel regularization for the intercept

        c = torch.matmul(Q, self._diag_dot(w, QT_y))
        G_inverse_diag = self._decomp_diag(w, Q)
        # handle case where y is 2-d
        if len(y.shape) != 1:
            G_inverse_diag = G_inverse_diag[:, None]
        return G_inverse_diag, c

    def to(self, device):
        self.device = device
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__setattr__(attr, value.to(device))
                
    def cuda(self):
        self.to('cuda')

    def cpu(self):
        self.to('cpu')
        
    def remove_from_gpu(self):
        self.to('cpu')
    
    def parse_input_data(self, *args, copy=False):
        args = convert_to_tensor(*args, copy=copy, device=self.device)
        if isinstance(args, torch.Tensor):
            args = (args, )
        args = unify_dtypes(*args, target_dtype=self.dtype)
        return args
        
    def preprocess_data(self, x, y, center=[], scale=[], output=None, save_to_class=False, **kwargs):
        
            stats = {f'{var}_{stat}': None for stat in ['mean','std','offset','scale'] for var in ['x', 'y']}

            x, y = self.parse_input_data(x, y)
            
            def parse_preprocessing_args(*args):
                parsed_args = []
                for arg in args:
                    if arg is None or len(arg) == 0:
                        parsed_args.append('none')
                    elif isinstance(arg, list):
                        parsed_args.append(''.join(arg))
                    else:
                        parsed_args.append(arg)
                return tuple(parsed_args)

            center, scale, output = parse_preprocessing_args(center, scale, output)
            
            if kwargs.get('fit_intercept', False):
                center += 'x'

            if 'x' in center.lower():
                stats['x_mean'] = x.mean(dim = 0)
            if 'y' in center.lower():
                stats['y_mean'] = y.mean(dim = 0) if y.ndim > 1 else y.mean()
                
            if 'x' in scale.lower():
                stats['x_std'] = x.std(dim=0, correction=1)
                stats['x_std'][stats['x_std'] == 0.0] = 1.0  
            if 'y' in scale.lower():
                stats['y_std'] = y.std(dim=0, correction=1)
                stats['y_std'][stats['y_std'] == 0.0] = 1.0 
            
            if 'x' in center.lower():
                x -= stats['x_mean']
            if 'y' in center.lower():
                y -= stats['y_mean']
                
            if 'x' in scale.lower():
                x /= stats['x_std']
            if 'y' in scale.lower():
                y /= stats['y_std']

            if output == 'mean_std':
                if stats['x_mean'] is None:
                    stats['x_mean'] = x.mean(dim=0)
                if stats['y_mean'] is None:
                    stats['y_mean'] = y.mean(dim = 0) if y.ndim > 1 else y.mean()
                if stats['x_std'] is None:
                    stats['x_std'] = torch.ones(x.shape[1], dtype=x.dtype,  device=x.device)
                if stats['y_std'] is None:
                    stats['y_std'] = torch.ones(y.shape[1], dtype=y.dtype,  device=y.device)
                    
            if output == 'offset_scale':
                stats['x_offset'] = stats.pop('x_mean', None)
                stats['y_offset'] = stats.pop('y_mean', None)
                stats['x_scale'] = stats.pop('x_std', None)
                if stats['x_offset'] is None:
                    stats['x_offset'] = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
                if stats['y_offset'] is None:
                    stats['y_offset'] = torch.zeros(y.shape[1], dtype=y.dtype, device=y.device)
                if stats['x_scale'] is None:
                    stats['x_scale'] = torch.ones(x.shape[1], dtype=x.dtype,  device=x.device)

            if save_to_class:
                for stat, value in stats.items():
                    if value is not None:
                        setattr(self, stat, value)

                return x, y

            if not save_to_class:
                if output == 'offset_scale':
                    return x, y, stats['x_offset'], stats['y_offset'], stats['x_scale']

                if output == 'mean_std':
                    return x, y, stats['x_mean'], stats['y_mean'], stats['x_std'], stats['y_std']
                
                return x, y
    
    def fit(self, x, y):
        self.alphas = torch.as_tensor(self.alphas, dtype=torch.float32)

        preprocessing_kwargs = {"output": "offset_scale"}
        if self.fit_intercept:
            preprocessing_kwargs["center"] = "xy"
        if self.scale_x:
            preprocessing_kwargs["scale"] = "x"

        x, y, x_offset, y_offset, x_scale = self.preprocess_data(
            x, y, **preprocessing_kwargs
        )

        decompose = self._eigen_decompose_gram
        solve = self._solve_eigen_gram

        sqrt_sw = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)

        x_mean, *decomposition = decompose(x, y, sqrt_sw)

        n_y = 1 if len(y.shape) == 1 else y.shape[1]
        n_alphas = 1 if self.alphas.ndim == 0 else len(self.alphas)

        best_alpha, best_coef, best_score, best_y_pred = [None] * 4

        for i, alpha in enumerate(torch.atleast_1d(self.alphas)):
            G_inverse_diag, coef = solve(
                float(alpha), y, sqrt_sw, x_mean, *decomposition
            )
            y_pred = y - (coef / G_inverse_diag)

            score = pearson_corrcoef(y, y_pred)
            if not self.alpha_per_target:
                score = score.mean()

            # Keep track of the best model
            if best_score is None:
                best_alpha = alpha
                best_coef = coef
                best_score = score
                best_y_pred = y_pred
                if self.alpha_per_target and n_y > 1:
                    best_alpha = torch.full((n_y,), alpha)

            else:
                if self.alpha_per_target and n_y > 1:
                    to_update = score > best_score
                    best_alpha[to_update] = alpha
                    best_coef[:, to_update] = coef[:, to_update]
                    best_score[to_update] = score[to_update]
                    best_y_pred[:, to_update] = y_pred[:, to_update]

                elif score > best_score:
                    best_alpha, best_coef, best_score, best_y_pred = (
                        alpha,
                        coef,
                        score,
                        y_pred,
                    )

        self.alpha_ = best_alpha
        self.score_ = best_score
        self.dual_coef_ = best_coef
        self.coef_ = self.dual_coef_.T.matmul(x)
        self.cv_y_pred_ = best_y_pred

        self.is_fitted = True

        x_offset += x_mean * x_scale
        if self.fit_intercept:
            self.coef_ = self.coef_ / x_scale
            self.intercept_ = y_offset - torch.matmul(x_offset, self.coef_.T)
        else:
            self.intercept_ = torch.zeros(1, device=self.coef_.device)

        return self

    def predict(self, x):
        x = self.parse_input_data(x)
        return x.matmul(self.coef_.T) + self.intercept_

    def score(self, x, y):
        x = self.parse_input_data(x)
        y = self.parse_input_data(y)
        return pearson_corrcoef(y, self.predict(x))

