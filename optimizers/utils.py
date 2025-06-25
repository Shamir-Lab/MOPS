import numpy as np
import pandas as pd
from typing import List

def get_diffs_matrix(df: pd.DataFrame, items: List[str]) -> np.ndarray:
    """
    Generate the matrix of score differences for all pairs of visits.

    Args:
        df (pd.DataFrame): DataFrame containing participant data.
        items (List[str]): List of item names.

    Returns:
        np.ndarray: Matrix of score differences.
    """
    matrix_A_rows = []

    # Iterate over each participant
    for patient_id, group in df.groupby('PATNO'):
        visits = group.sort_values('visit_month')
        n_visits = len(visits)

        for i in range(n_visits):
            for j in range(i + 1, n_visits):
                visit_i = visits.iloc[i]
                visit_j = visits.iloc[j]

                # Create a row for the matrix A
                row = [visit_j[item] - visit_i[item] for item in items]
                
                # Append the row to the list
                matrix_A_rows.append(row)

    # Convert the list of rows to a numpy array (matrix A)
    A = np.array(matrix_A_rows)
    return A
