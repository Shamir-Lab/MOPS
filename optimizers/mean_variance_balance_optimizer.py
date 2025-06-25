import numpy as np
import pandas as pd
import cvxpy as cp
from typing import List, Tuple, Optional
from optimizers.weights_optimizer import WeightsOptimizer, Status

class MeanVarianceBalanceOptimizer(WeightsOptimizer):
    """
    An optimizer that maximizes the mean of score differences while penalizing their variance and regularizes weights.
    """

    def __init__(self, data: pd.DataFrame, items: List[str], version: str, var_penalty: float = 0.01, 
                 reg_type: str = 'l1', reg_strength: float = 0.01, name: Optional[str] = None):
        """
        Initialize the optimizer with data, items, penalty factor, and optional regularization.
    
        Args:
            data (pd.DataFrame): Input data for optimization.
            items (List[str]): List of items to optimize.
            var_penalty (float): Penalty factor for variance.
            reg_type (str): Type of regularization ('l1', 'l2' or None). Default is 'l1'.
            reg_strength (float): Strength of the regularization. Default is 0.01.
            name (Optional[str]): Custom name for the optimizer. If None, a default name is generated.
        """
        # Build the default name if not provided
        reg_type_name = f", Reg: {reg_type}" if reg_type else ""
        default_name = f"Mean-Variance Balance{reg_type_name}"
        
        super().__init__(name=name if name else default_name, data=data, items=items, version=version)
        self.var_penalty = var_penalty
        self.reg_type = reg_type
        self.reg_strength = reg_strength

    def calc_weights(self) -> Tuple[np.ndarray, str]:
        """
        Calculate the optimal weights using quadratic programming to maximize mean score differences
        while minimizing variance and applying optional regularization.

        Returns:
            Tuple[np.ndarray, str]: The optimized weights and the calculation status.
        """
        # Define the weight variables for each item
        weights = cp.Variable(len(self.items), nonneg=True)

        # List to hold score differences
        score_diffs = []

        # Calculate score differences for each pair of visits
        for patient_id, group in self.data.groupby('PATNO'):
            visits = group.sort_values('visit_month')
            n_visits = len(visits)

            # Calculate the sum of month differences for normalization
            total_month_diff = sum(visits.iloc[j]['visit_month'] - visits.iloc[i]['visit_month']
                                   for i in range(n_visits) for j in range(i + 1, n_visits))

            for i in range(n_visits):
                for j in range(i + 1, n_visits):
                    visit_i = visits.iloc[i]
                    visit_j = visits.iloc[j]
                    month_diff = visit_j['visit_month'] - visit_i['visit_month']

                    score_i = sum(weights[k] for k in range(len(self.items)) if visit_i[self.items[k]] == 1)
                    score_j = sum(weights[k] for k in range(len(self.items)) if visit_j[self.items[k]] == 1)
                    score_diff = score_j - score_i

                    # Apply normalization factor by month difference
                    normalization_factor = month_diff * (n_visits / total_month_diff)

                    weighted_score_diff = score_diff * normalization_factor
                    # Append weighted score difference to the list
                    score_diffs.append(weighted_score_diff)

        # Convert score_diffs to a CVXPY variable
        score_diffs = cp.hstack(score_diffs)

        # Define the mean and variance of the score differences
        n_diffs = score_diffs.shape[0]
        mean_diff = cp.sum(score_diffs) / n_diffs
        variance_diff = cp.sum_squares(score_diffs - mean_diff) / n_diffs

        # Regularization term
        regularization_term = 0
        if self.reg_type == 'l1':
            regularization_term = self.reg_strength * cp.norm(weights, 1)
        elif self.reg_type == 'l2':
            regularization_term = self.reg_strength * cp.sum_squares(weights)

        # Set the objective function to maximize mean_diff and minimize variance_diff
        objective = cp.Maximize(mean_diff - self.var_penalty * variance_diff - regularization_term)

        # Define and solve the problem
        prob = cp.Problem(objective, [weights <= 1])
        prob.solve(verbose=True)

        # Retrieve the optimized weights
        self.weights = weights.value
        self.status = Status.DONE if prob.status == cp.OPTIMAL else Status.ERROR

        return self.weights, self.status
