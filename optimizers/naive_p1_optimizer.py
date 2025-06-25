import numpy as np
import pandas as pd
from typing import List, Tuple, Union

from optimizers.weights_optimizer import WeightsOptimizer, Status

class NaiveP1Optimizer(WeightsOptimizer):
    """
    Naive implementation of WeightsOptimizer where weights are set to 1 for part 1 items and 0 otherwise.
    """

    def __init__(self, data: pd.DataFrame, items: List[str], version: str):
        super().__init__(name = "MDS-UPDRS Part 1", data = data, items = items, version=version)

    
    def calc_weights(self) -> Tuple[np.ndarray, str]:
        """
        Naively set weights to 1 for items starting with 'NP1' and 0 otherwise.
        
        Returns:
            Tuple[np.ndarray, str]: weights and the calculation status.
        """
        # Create weights: 1 for items starting with 'NP1', 0 otherwise
        naive_weights = np.array([1 if item.startswith('NP1') else 0 for item in self.items])
        self.status = Status.DONE
        self.weights = naive_weights
        return self.weights, self.status
