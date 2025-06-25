import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import gurobipy as gp

from optimizers.weights_optimizer import WeightsOptimizer, Status
from optimizers import utils

class MixedIntegerProgrammingOptimizer(WeightsOptimizer):
    """
    Mixed Integer Programming-based implementation of WeightsOptimizer using binary variables 
    to enforce sparsity and optimize the number of increases.
    """
    
    def __init__(self, data: pd.DataFrame, items: List[str], version: str, time_limit: int = 600, 
                 lambda_reg: float = 10, name: Optional[str] = None):
        """
        Initialize the Mixed Integer Programming optimizer.
    
        Args:
            data (pd.DataFrame): Input data for optimization.
            items (List[str]): List of items to optimize.
            version (str): Version identifier for the optimization run.
            time_limit (int): Time limit for the optimization process in seconds. Default is 600.
            lambda_reg (float): Regularization strength parameter. Default is 10.
            name (Optional[str]): Custom name for the optimizer. If None, a default name is generated.
        """
        default_name = f"Mixed Integer Programming, Lambda={lambda_reg}"
        super().__init__(name=name if name else default_name, data=data, items=items, version=version)
        self.time_limit = time_limit  # Time limit in seconds
        self.lambda_reg = lambda_reg
        self.model_path = os.path.join(self.cache_dir, f'{self.version}_mip_lambda_{self.lambda_reg}.mps')
        self.solution_path = os.path.join(self.cache_dir, f'{self.version}_mip_lambda_{self.lambda_reg}.sol')
        self.variables = {}  # To store individual variables (v[i])
        
    def calc_weights(self) -> Tuple[np.ndarray, str]:
        # Get the matrix of score differences
        A = utils.get_diffs_matrix(self.data, self.items)
        n = A.shape[1]
    
        # Dynamically set M based on the number of items
        eps = 0.01
        M = len(self.items) * 100 + 10

        try:
            # Check if a saved model state exists and load it
            if os.path.exists(self.model_path):
                print(f"Loading existing model state from {self.model_path}")
                model = gp.read(self.model_path)  # Use .mps format for model
                if os.path.exists(self.solution_path):
                    model.read(self.solution_path)  # Read solution if it exists
                # Retrieve individual variables from the loaded model
                for i in range(n):
                    self.variables[i] = model.getVarByName(f"v[{i}]")
            else:
                # Create a new Gurobi model
                model = gp.Model()

                # Add variables as individual components
                v = [model.addVar(lb=0, ub=10, vtype=gp.GRB.CONTINUOUS, name=f"v[{i}]") for i in range(n)]
                z = [model.addVar(vtype=gp.GRB.BINARY, name=f"z[{i}]") for i in range(n)]
                is_positive = [model.addVar(vtype=gp.GRB.BINARY, name=f"is_positive[{i}]") for i in range(len(A))]
                is_negative = [model.addVar(vtype=gp.GRB.BINARY, name=f"is_negative[{i}]") for i in range(len(A))]

                # Store the variables for later use
                self.variables = {i: v[i] for i in range(n)}

                # Add constraints
                for i in range(len(A)):
                    model.addConstr(is_positive[i] + is_negative[i] <= 1, name=f"mutual_exclusive[{i}]")
                    model.addConstr(A[i] @ v >= eps - (1 - is_positive[i]) * M, name=f"positive_constraint[{i}]")
                    model.addConstr(A[i] @ v <= M * is_positive[i], name=f"positive_upper_bound[{i}]")
                    model.addConstr(A[i] @ v <= -eps + (1 - is_negative[i]) * M, name=f"negative_constraint[{i}]")
                    model.addConstr(A[i] @ v >= -M * is_negative[i], name=f"negative_lower_bound[{i}]")
                for i in range(n):
                    model.addConstr(v[i] <= M * z[i], name=f"sparsity_control[{i}]")

                # Set objective: Maximize the number of positive values minus the regularization term
                model.setObjective(gp.quicksum(is_positive) - self.lambda_reg * gp.quicksum(z), gp.GRB.MAXIMIZE)

            # Set Gurobi parameters
            model.setParam('TimeLimit', self.time_limit)
            model.setParam('OutputFlag', 1)
            model.setParam('MIPFocus', 1)
            model.setParam('Presolve', 2)
            model.setParam('Cuts', 2)

            # Optimize the model
            model.optimize()

            # Check the solver status
            if model.Status == gp.GRB.OPTIMAL:
                self.status = Status.DONE
            elif model.Status == gp.GRB.TIME_LIMIT:
                self.status = Status.TIMED_OUT
            else:
                self.status = Status.ERROR

            # Save the model and solution if optimization succeeded or timed out
            if self.status in [Status.DONE, Status.TIMED_OUT]:
                model.write(self.model_path)  # Save the model in .mps format
                model.write(self.solution_path)  # Always save the solution, even if timed out
                print(f"Saved Gurobi model to {self.model_path}")
                print(f"Saved Gurobi solution to {self.solution_path}")

            # Retrieve the optimized weights if the solution is available
            if self.status != Status.ERROR:
                self.weights = np.array([self.variables[i].X for i in range(n)])  # Use .X to get variable values
            else:
                self.weights = None
        except gp.GurobiError as e:
            print(f"Gurobi solver error: {e}")
            self.status = Status.ERROR
            return None, self.status
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.status = Status.ERROR
            return None, self.status

        return self.weights, self.status
