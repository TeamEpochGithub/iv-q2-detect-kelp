from dataclasses import dataclass
from typing import Any

from dask.array.core import Array
from numpy import ndarray
from src.pipeline.ensemble.ensemble_base import EnsembleBase


@dataclass
class DLEnsemble(EnsembleBase):
    

    def transform(self, X: Array) -> ndarray[Any, Any]:
        # TODO: Implement transform
        return super().transform(X)
    
    def fit(self, X: Array, y: Array) -> Any:
        # TODO: Implement fit
        return super().fit(X, y)

    def fit_transform(self, X: Array, y: Array, **fit_params: str) -> ndarray[Any, Any]:
        # TODO: Implement fit_transform
        return super().fit_transform(X, y, **fit_params)