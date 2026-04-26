"""
BayesianOptimizer - 贝叶斯优化策略选择
Lightweight Bayesian optimization for strategy parameter tuning.
Uses Gaussian Process surrogate + Expected Improvement acquisition.
No GPU needed; pure NumPy.
"""

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class HyperparameterPoint:
    """A single hyperparameter configuration."""
    params: dict[str, float]
    score: float
    timestamp: str


class BayesianOptimizer:
    """
    Bayesian optimization for strategy parameter tuning.
    
    Uses a Gaussian Process surrogate model with:
    - RBF (Radial Basis Function) kernel
    - Expected Improvement (EI) acquisition function
    - Random restart optimization for EI
    
    Designed for tuning continuous float parameters (e.g., temperature, learning_rate).
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]],
        maximize: bool = True,
        noise_variance: float = 0.01,
        kernel_variance: float = 1.0,
        kernel_lengthscale: float = 0.5,
    ):
        """
        Args:
            param_bounds: Dict of param_name -> (min, max) bounds
            maximize: If True, maximize; else minimize
            noise_variance: Observation noise variance
            kernel_variance: GP kernel variance (sigma^2)
            kernel_lengthscale: GP kernel lengthscale (l)
        """
        self.param_names = list(param_bounds.keys())
        self.param_bounds = param_bounds
        self.maximize = maximize
        self.noise_variance = noise_variance
        self.kernel_variance = kernel_variance
        self.kernel_lengthscale = kernel_lengthscale
        
        self.observations: list[HyperparameterPoint] = []
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def _params_to_vector(self, params: dict[str, float]) -> np.ndarray:
        vec = []
        for name in self.param_names:
            lo, hi = self.param_bounds[name]
            normalized = (params[name] - lo) / (hi - lo) if hi > lo else 0.5
            vec.append(normalized)
        return np.array(vec)

    def _vector_to_params(self, vec: np.ndarray) -> dict[str, float]:
        params = {}
        for i, name in enumerate(self.param_names):
            lo, hi = self.param_bounds[name]
            params[name] = float(vec[i]) * (hi - lo) + lo
        return params

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel: k(x,y) = sigma^2 * exp(-0.5 * ||x-y||^2 / l^2)"""
        dist_sq = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return self.kernel_variance * np.exp(-0.5 * dist_sq / (self.kernel_lengthscale ** 2))

    def _gp_predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GP posterior mean and variance at X_new."""
        if len(self.observations) == 0:
            # Prior: mean=0.5, var=1.0
            mean = np.full(len(X_new), 0.5)
            var = np.full(len(X_new), 1.0)
            return mean, var
        
        X = self._X
        y = self._y
        
        K = self._rbf_kernel(X, X)
        K += self.noise_variance * np.eye(len(X))
        K_new = self._rbf_kernel(X_new, X)
        K_new_new = self._rbf_kernel(X_new, X_new)
        
        # Cholesky for numerical stability
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            v = np.linalg.solve(L, K_new.T)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            K_inv = np.linalg.pinv(K)
            alpha = K_inv @ y
            v = K_new.T @ K_inv
        
        mean = K_new @ alpha
        
        if len(X_new) == 1:
            var = float(K_new_new - (K_new @ np.linalg.solve(L.T, np.linalg.solve(L, K_new.T))))
        else:
            var = np.diag(K_new_new - (K_new @ np.linalg.solve(L.T, np.linalg.solve(L, K_new.T))))
        
        var = np.maximum(var, 1e-6)
        return mean, var

    def _expected_improvement(self, X_new: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition function."""
        if len(self.observations) == 0:
            return np.ones(len(X_new))  # Uniform when no data
        
        mean, var = self.gp_predict(X_new)
        std = np.sqrt(var)
        
        y_best = float(np.max(self._y) if self.maximize else np.min(self._y))
        
        if self.maximize:
            z = (mean - y_best) / std
        else:
            z = (y_best - mean) / std
        
        pdf_z = np.exp(-0.5 * z ** 2) / np.sqrt(2 * math.pi)
        cdf_z = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        
        ei = std * (z * cdf_z + pdf_z)
        ei[std < 1e-6] = 0.0
        return ei

    def gp_predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._gp_predict(X_new)

    def suggest(self) -> dict[str, float]:
        """Suggest the next hyperparameter configuration to evaluate."""
        if len(self.observations) < 2:
            # Random restart
            import random
            params = {name: random.uniform(lo, hi) for name, (lo, hi) in self.param_bounds.items()}
            return params
        
        # Optimize EI using random restarts
        best_ei = -1.0
        best_x = None
        
        for _ in range(20):
            # Random starting point
            x0 = np.random.rand(len(self.param_names))
            
            # Local optimization (gradient-free: Nelder-Mead style)
            x = x0.copy()
            for _ in range(50):
                # Evaluate at current
                ei_current = self._expected_improvement(x.reshape(1, -1))[0]
                
                # Random step
                step = np.random.randn(len(self.param_names)) * 0.1
                x_new = np.clip(x + step, 0.0, 1.0)
                ei_new = self._expected_improvement(x_new.reshape(1, -1))[0]
                
                if ei_new > ei_current:
                    x = x_new
            
            ei = self._expected_improvement(x.reshape(1, -1))[0]
            if ei > best_ei:
                best_ei = ei
                best_x = x
        
        return self._vector_to_params(best_x)

    def observe(self, params: dict[str, float], score: float):
        """Record an observation."""
        from datetime import datetime
        
        point = HyperparameterPoint(
            params=params,
            score=score,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.observations.append(point)
        
        vec = self._params_to_vector(params)
        y_val = score if self.maximize else -score
        
        if self._X is None:
            self._X = vec.reshape(1, -1)
            self._y = np.array([y_val])
        else:
            self._X = np.vstack([self._X, vec.reshape(1, -1)])
            self._y = np.append(self._y, y_val)

    def get_best(self) -> tuple[dict[str, float], float]:
        """Return the best observed parameters and score."""
        if not self.observations:
            return {}, 0.0
        
        if self.maximize:
            best_idx = int(np.argmax(self._y))
            score = float(self._y[best_idx])
        else:
            best_idx = int(np.argmin(self._y))
            score = float(-self._y[best_idx])
        
        return self.observations[best_idx].params, score

    def save(self, path: str):
        """Persist optimizer state to JSON."""
        data = {
            "observations": [
                {"params": o.params, "score": o.score, "timestamp": o.timestamp}
                for o in self.observations
            ],
            "param_bounds": self.param_bounds,
            "maximize": self.maximize,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str):
        """Load optimizer state from JSON."""
        data = json.loads(Path(path).read_text())
        self.observations = [
            HyperparameterPoint(**o) for o in data["observations"]
        ]
        if self.observations:
            self._X = np.array([self._params_to_vector(o.params) for o in self.observations])
            self._y = np.array([o.score if self.maximize else -o.score for o in self.observations])
