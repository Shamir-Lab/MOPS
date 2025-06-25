import numpy as np
import pandas as pd
from typing import List

class Evaluator:
    """
    Class for evaluating the model with given weights using a test DataFrame.
    
    Attributes:
        test_df (pd.DataFrame): The test data to evaluate.
        all_items (List[str]): The list of item names to calculate scores.
    """
    
    def __init__(self, test_df: pd.DataFrame, all_items: List[str]):
        """
        Initialize the Evaluator with test DataFrame and item list.

        Args:
            test_df (pd.DataFrame): The test data for evaluation.
            all_items (List[str]): The list of items to consider in the evaluation.
        """
        self.test_df = test_df.copy()
        self.all_items = all_items

    def get_num_pairs(self) -> pd.Series:
        """
        Compute the number of patient pairs for each month difference.
    
        Returns:
            pd.Series: Series containing the number of patient pairs for each month difference.
        """
        temp_df = self.test_df.copy()
        
        month_diffs = []
        for patient_id, group in temp_df.groupby('PATNO'):
            visits = group.sort_values('visit_month')
            n_visits = len(visits)
    
            for i in range(n_visits):
                for j in range(i + 1, n_visits):
                    visit_i = visits.iloc[i]
                    visit_j = visits.iloc[j]
                    month_diff = visit_j['visit_month'] - visit_i['visit_month']
                    month_diffs.append(month_diff)
    
        # Create DataFrame and count occurrences per month_diff
        month_diffs_df = pd.DataFrame(month_diffs, columns=['month_diff'])
        num_pairs = month_diffs_df['month_diff'].value_counts().sort_index()
    
        return num_pairs

    def get_increase_percentages(self, weights: np.ndarray) -> pd.Series:
        """
        Calculate the percentage of positive score differences for each month difference using the current weights.

        Args:
            weights (np.ndarray): The weights to apply to the items.

        Returns:
            pd.Series: Series containing the percentage of positive score differences for each month difference.
        """
        # Calculate total scores using a temporary copy of `test_df`
        temp_df = self.test_df.copy()
        temp_df['total_score'] = temp_df[self.all_items].dot(weights)

        # Calculate score differences dynamically
        score_diffs = []
        for patient_id, group in temp_df.groupby('PATNO'):
            visits = group.sort_values('visit_month')
            n_visits = len(visits)

            for i in range(n_visits):
                for j in range(i + 1, n_visits):
                    visit_i = visits.iloc[i]
                    visit_j = visits.iloc[j]
                    month_diff = visit_j['visit_month'] - visit_i['visit_month']
                    score_diff = visit_j['total_score'] - visit_i['total_score']
                    score_diffs.append({'month_diff': month_diff, 'score_diff': score_diff})

        # Create DataFrame for score differences
        score_diffs_df = pd.DataFrame(score_diffs)

        # Group by month_diff and filter groups with at least 10 entries
        grouped_filtered = score_diffs_df.groupby('month_diff').filter(lambda x: len(x) >= 10)

        # Calculate the percentage of strictly positive score differences
        percentage_scores = grouped_filtered.groupby('month_diff')['score_diff'].apply(lambda x: (x > 0).mean() * 100)
        
        return percentage_scores

    def calculate_weighted_scores(self, weights: np.ndarray) -> pd.DataFrame:
        """
        Calculate total scores based on the provided weights.

        Args:
            weights (np.ndarray): The weights applied to each item.

        Returns:
            pd.DataFrame: A DataFrame containing PATNO, INFODT, and total weighted score.
        """
        temp_df = self.test_df.copy()

        # Ensure the columns in all_items are numeric
        temp_df[self.all_items] = temp_df[self.all_items].apply(pd.to_numeric, errors='coerce')

        # Ensure the number of items matches the number of weights
        if temp_df[self.all_items].shape[1] != len(weights):
            raise ValueError(f"Mismatch in number of items and weights: {temp_df[self.all_items].shape[1]} items, {len(weights)} weights")

        # Calculate total scores using the provided weights
        total_score = temp_df[self.all_items].dot(weights)

        # Return the DataFrame with total score column added
        return pd.concat([temp_df[['PATNO', 'INFODT']], total_score.rename('total_score')], axis=1)

    def calculate_all_weighted_scores(self, df_weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the `calculate_weighted_scores` function for each method in df_weights.

        Args:
            df_weights (pd.DataFrame): DataFrame containing different sets of weights.

        Returns:
            pd.DataFrame: A DataFrame containing PATNO, INFODT, total_score, and the method used.
        """
        weighted_scores = pd.DataFrame()
        for method in df_weights.columns[1:]:
            weights = df_weights[method].values
            weighted_scores_method = self.calculate_weighted_scores(weights)
            weighted_scores_method['method'] = method
            weighted_scores = pd.concat([weighted_scores, weighted_scores_method], axis=0)
        return weighted_scores
