import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import pulp

from optimizers.weights_optimizer import WeightsOptimizer, Status

class WeightedLinearProgrammingOptimizer(WeightsOptimizer):
    """
    Linear programming-based implementation of WeightsOptimizer that optimizes weights based on weighted score differences.
    """

    def __init__(self, data: pd.DataFrame, items: List[str], version: str, lambda_reg: float = 0.01, 
                 penalty_factor: float = 2.0, name: Optional[str] = None):
        default_name = f"Weighted Linear Programming Optimizer, Lambda={lambda_reg}, Penalty={penalty_factor}"
        super().__init__(name=name if name else default_name, data=data, items=items, version=version)
        self.lambda_reg = lambda_reg
        self.penalty_factor = penalty_factor

    def calc_weights(self) -> Tuple[np.ndarray, str]:
        """
        Calculate the optimal weights using linear programming with weighted score differences.

        Returns:
            Tuple[np.ndarray, str]: The optimized weights and the calculation status.
        """

        # Initialize the linear programming problem
        prob = pulp.LpProblem("Maximize_Score_Difference", pulp.LpMaximize)

        # Define the weight variables for each item
        weights = {item: pulp.LpVariable(f'weight_{item}', lowBound=0, upBound=1) for item in self.items}

        # Define auxiliary variables for positive and negative score differences
        pos_diffs = {}
        neg_diffs = {}

        # Objective function components
        objective_terms = []
        regularization_term = self.lambda_reg * pulp.lpSum(weights[item] for item in self.items)

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

                    # Calculate scores for visits i and j using the weights
                    score_i = pulp.lpSum(weights[item] for item in self.items if visit_i[item] == 1)
                    score_j = pulp.lpSum(weights[item] for item in self.items if visit_j[item] == 1)
                    score_diff = score_j - score_i

                    # Define auxiliary variables for positive and negative parts of score_diff
                    pos_diffs[(patient_id, i, j)] = pulp.LpVariable(f'pos_diff_{patient_id}_{i}_{j}', lowBound=0)
                    neg_diffs[(patient_id, i, j)] = pulp.LpVariable(f'neg_diff_{patient_id}_{i}_{j}', lowBound=0)

                    # Add constraints to relate the auxiliary variables with the score difference
                    prob += score_diff == pos_diffs[(patient_id, i, j)] - neg_diffs[(patient_id, i, j)]

                    # Weight the score difference by the normalized month difference
                    normalization_factor = month_diff * (n_visits / total_month_diff)
                    
                    # Apply penalty to negative score differences
                    weighted_score_diff = (pos_diffs[(patient_id, i, j)] - self.penalty_factor * neg_diffs[(patient_id, i, j)]) * normalization_factor
                    objective_terms.append(weighted_score_diff)

        # Set the objective function
        prob += pulp.lpSum(objective_terms) - len(self.data) * regularization_term

        # Solve the problem
        prob.solve()

        # Retrieve the optimized weights
        optimized_weights = np.array([weights[item].varValue for item in self.items])
        status = pulp.LpStatus[prob.status]

        self.status = status
        self.weights = optimized_weights
        
        return self.weights, self.status
