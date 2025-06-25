import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import cvxpy as cp

from optimizers.weights_optimizer import WeightsOptimizer, Status

class QuadraticProgrammingOptimizer(WeightsOptimizer):
    """
    Quadratic programming-based implementation of WeightsOptimizer that optimizes weights 
    based on score differences, with a quadratic penalty for negative score differences.
    """

    def __init__(self, data: pd.DataFrame, items: List[str], version: str, lambda_reg: float = 0.01, 
                 penalty_factor: float = 1.0, name: Optional[str] = None):
        default_name = f"Quadratic Programming Optimizer, Lambda={lambda_reg}, Penalty={penalty_factor}"
        super().__init__(name=name if name else default_name, data=data, items=items, version=version)
        self.lambda_reg = lambda_reg
        self.penalty_factor = penalty_factor

    def calc_weights(self) -> Tuple[np.ndarray, str]:
        """
        Calculate the optimal weights using quadratic programming.

        Returns:
            Tuple[np.ndarray, str]: The optimized weights and the calculation status.
        """

        # Define the weight variables for each item with bounds [0, 1]
        weights = cp.Variable(len(self.items), nonneg=True)
        constraints = [weights <= 1]  # Upper bound constraint

        # Objective function components
        objective_terms = []

        # Regularization term (L1 norm)
        regularization_term = self.lambda_reg * cp.norm(weights, 1)

        # Calculate score differences for each pair of visits
        for patient_id, group in self.data.groupby('PATNO'):
            visits = group.sort_values('visit_month')
            n_visits = len(visits)

            # Calculate the sum of month differences for normalization
            total_month_diff = sum(
                visits.iloc[j]['visit_month'] - visits.iloc[i]['visit_month']
                for i in range(n_visits) for j in range(i + 1, n_visits)
            )

            for i in range(n_visits):
                for j in range(i + 1, n_visits):
                    visit_i = visits.iloc[i]
                    visit_j = visits.iloc[j]
                    month_diff = visit_j['visit_month'] - visit_i['visit_month']

                    # Calculate the score differences
                    score_i = sum(weights[k] for k in range(len(self.items)) if visit_i[self.items[k]] == 1)
                    score_j = sum(weights[k] for k in range(len(self.items)) if visit_j[self.items[k]] == 1)
                    score_diff = score_j - score_i

                    # Penalize negative score differences quadratically
                    penalty = cp.square(cp.pos(-score_diff))

                    # Combine linear score difference and penalty
                    combined_score = score_diff - self.penalty_factor * penalty

                    # Apply normalization factor
                    normalization_factor = month_diff * (n_visits / total_month_diff)

                    # Add the weighted combined score to the objective terms
                    weighted_combined_score = combined_score * normalization_factor
                    objective_terms.append(weighted_combined_score)

        # Set the objective function
        objective = cp.Maximize(cp.sum(objective_terms) - regularization_term)

        # Define and solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=True)  # Enable verbose output to see progress

        # Retrieve the optimized weights
        optimized_weights = np.array(weights.value).flatten()
        status = Status.DONE if prob.status == cp.OPTIMAL else Status.ERROR

        self.status = status
        self.weights = optimized_weights

        return self.weights, self.status
