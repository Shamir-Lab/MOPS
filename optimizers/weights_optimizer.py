import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from enum import Enum

class Status(Enum):
    NOT_STARTED = "Not Started"
    DONE = "Done"
    ERROR = "Error"
    TIMED_OUT = "Timed Out"


class WeightsOptimizer(ABC):
    """
    Abstract class for a weights optimization approach.
    
    Attributes:
        name (str): Name of the optimizer.
        data (pd.DataFrame): The input data containing features for optimization.
        items (List[str]): List of items to optimize.
        version (str): Version identifier for cache files.
    """
    
    def __init__(self, name: str, data: pd.DataFrame, items: List[str], version: str):
        self.name = name
        self.data = data
        self.items = items
        self.version = version
        self.status = Status.NOT_STARTED
        self.cache_dir = 'optimizers/cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_name(self) -> str:
        """
        Get the name of the optimizer.
        
        Returns:
            str: The name of the optimizer.
        """
        return self.name

    def get_status(self) -> Status:
        """
        Get the optimization status.
        
        Returns:
            Status: The status of optimization process.
        """
        return self.status

    def get_weights(self) -> np.ndarray:
        """
        Get the optimized weights.
        
        Returns:
            Status: The status of optimization process.
        """
        if self.status == Status.NOT_STARTED:
            raise Exception("Optimzation hasn't run yet")
        elif self.status == Status.ERROR:
            raise Exception("Error occured in optimzation process")
            
        return self.weights
        
    @abstractmethod
    def calc_weights(self) -> Tuple[np.ndarray, str]:
        """
        Abstract method to calculate optimal weights.
        
        Returns:
            Tuple[np.ndarray, str]: Optimized weights and the optimization status.
        """
        pass
