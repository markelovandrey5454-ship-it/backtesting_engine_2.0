from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BasePortfolioStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        pass
