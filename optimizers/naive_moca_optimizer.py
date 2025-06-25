import numpy as np
import pandas as pd
from typing import List, Tuple, Union

from optimizers.weights_optimizer import WeightsOptimizer, Status

class NaiveMocaOptimizer(WeightsOptimizer):
    """
    Naive implementation of WeightsOptimizer where weights are set to 1 for MoCA items and 0 otherwise.
    """

    def __init__(self, data: pd.DataFrame, items: List[str], version: str):
        super().__init__(name = "MoCA", data = data, items = items, version=version)

    
    def calc_weights(self) -> Tuple[np.ndarray, str]:
        """
        Naively set weights to 1 for MoCA items and 0 otherwise.
        
        Returns:
            Tuple[np.ndarray, str]: weights and the calculation status.
        """
        # excluding 'MCAVFNUM' items (item per number of words in verbal fluency part) which do not appear in original MoCA
        naive_weights = np.array([1 if (item.startswith('MCA') and ('MCAVFNUM' not in item)) else 0 for item in self.items])
        self.status = Status.DONE
        self.weights = naive_weights
        return self.weights, self.status
