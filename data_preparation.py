import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Tuple

from data import ppmi_data_loader


# Configs
PARTS = [1, 2, 3]  # MDS-UPDRS parts to use
COHORTS = ["Parkinson's Disease"]  # Latest diagnosis of patients to include
REMOVE_SCREENING_AND_BASELINE = True  # Whether to remove screening and baseline visits


def flip_values(series: pd.Series) -> pd.Series:
    """Flip values in the series such that higher values indicate more severe symptoms.

    Args:
        series (pd.Series): The series to flip.

    Returns:
        pd.Series: The flipped series with the mapping applied.
    """
    unique_values = sorted(series.unique())
    value_map = {v: unique_values[-i - 1] for i, v in enumerate(unique_values)}
    return series.map(value_map)


def drop_redundant_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """Drop specified redundant columns from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        columns_to_drop (List[str]): List of column names to drop.

    Returns:
        pd.DataFrame: DataFrame with specified columns removed.
    """
    return df.drop(columns=columns_to_drop)


def load_and_prepare_ppmi_data(include_moca: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load PPMI data using the data loader.

    Returns:
        Tuple containing DataFrames for part 1 patient, part 1 rater, part 2, part 3, and MoCA.
    """
    dl = ppmi_data_loader.PpmiDataLoader(ppmi_dir='data/PPMI')
    p1_p_df = dl.get_mds_updrs_part1_patient_df()
    p1_r_df = dl.get_mds_updrs_part1_rater_df()
    p2_df = dl.get_mds_updrs_part2_df()
    p3_df = dl.get_mds_updrs_part3_df()
    moca_df = dl.get_moca_df() if include_moca else pd.DataFrame()
    return p1_p_df, p1_r_df, p2_df, p3_df, moca_df


def filter_participants(status_df: pd.DataFrame) -> List[int]:
    """Filter participants based on cohort definition.

    Args:
        status_df (pd.DataFrame): DataFrame containing participant status.

    Returns:
        List[int]: List of valid participant IDs.
    """
    status_df = status_df[~((status_df['CONCOHORT'].notnull()) & (status_df['CONCOHORT'] != status_df['COHORT']))]
    valid_patnos = status_df[status_df.COHORT_DEFINITION.isin(COHORTS)].PATNO.unique().tolist()
    return valid_patnos


def create_merged_df(include_moca: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Create and return a merged DataFrame from multiple PPMI datasets.

    Args:
        include_moca (boolean): Whether to include MoCA data in addition to MDS-UPDRS.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Merged DataFrame and a list of item columns.
    """
    p1_p_df, p1_r_df, p2_df, p3_df, moca_df = load_and_prepare_ppmi_data(include_moca)
    status_df = ppmi_data_loader.PpmiDataLoader(ppmi_dir='data/PPMI').get_participant_status()

    valid_patnos = filter_participants(status_df)
    p3_df = p3_df[p3_df.PATNO.isin(valid_patnos)]

    p3_df = p3_df[(p3_df.PDMEDYN != 1) | (~p3_df.PDSTATE.isna())]
    p3_df = p3_df[~((p3_df.PDMEDYN == 0) & (~p3_df.PDSTATE.isna()))]

    if REMOVE_SCREENING_AND_BASELINE:
        p3_df = p3_df[~p3_df.EVENT_ID.isin(['SC', 'BL'])]

    p3_df = p3_df.drop_duplicates(subset=['PATNO', 'EVENT_ID', 'PDSTATE'], keep='last')
    p3_df = p3_df[p3_df.PDSTATE != 'OFF']
    p3_df = p3_df[(p3_df.NHY.notna()) & (p3_df.NHY != 101)]
    p3_df = p3_df[(p3_df.DYSKPRES != 1) & (p3_df.DYSKIRAT != 1)]

    COMMON_COLS_TO_REMOVE = ['REC_ID', 'PAG_NAME', 'ORIG_ENTRY', 'LAST_UPDATE']
    p1_p_df = drop_redundant_columns(p1_p_df, COMMON_COLS_TO_REMOVE + ['NUPSOURC', 'NP1PTOT'])
    p1_r_df = drop_redundant_columns(p1_r_df, COMMON_COLS_TO_REMOVE + ['NUPSOURC', 'NP1RTOT'])
    p2_df = drop_redundant_columns(p2_df, COMMON_COLS_TO_REMOVE + ['NUPSOURC', 'NP2PTOT'])
    p3_df = drop_redundant_columns(p3_df, COMMON_COLS_TO_REMOVE + [
        'EXAMDT', 'PDTRTMNT', 'PDSTATE', 'HRPOSTMED', 'HRDBSON', 'HRDBSOFF',
        'ONOFFORDER', 'OFFEXAM', 'OFFNORSN', 'DBSOFFTM', 'DBSYN', 'ONEXAM', 'ONNORSN', 'DBSONTM',
        'PDMEDDT', 'PDMEDTM', 'PDMEDYN', 'DYSKPRES', 'DYSKIRAT', 'NP3TOT', 'EXAMTM'
    ])

    if include_moca and not moca_df.empty:
        moca_df = drop_redundant_columns(moca_df, COMMON_COLS_TO_REMOVE + ['MCATOT'])

    df = pd.merge(p1_p_df, p1_r_df, on=['PATNO', 'EVENT_ID', 'INFODT'], how='inner')
    df = pd.merge(df, p2_df, on=['PATNO', 'EVENT_ID', 'INFODT'], how='inner')
    df = pd.merge(df, p3_df, on=['PATNO', 'EVENT_ID', 'INFODT'], how='inner')
    if include_moca and not moca_df.empty:
        df = pd.merge(df, moca_df, on=['PATNO', 'EVENT_ID', 'INFODT'], how='inner')

    patno_counts = df['PATNO'].value_counts()
    df = df[df['PATNO'].isin(patno_counts[patno_counts > 1].index)]

    df = df.dropna()

    mds_updrs_items = [c for c in df.columns if c.startswith('NP')]
    moca_items = [c for c in df.columns if c.startswith('MCA')]
    items = mds_updrs_items + (moca_items if include_moca else [])

    if include_moca:
        df[moca_items] = df[moca_items].apply(flip_values)

    df[items + ['NHY']] = df[items + ['NHY']].astype(int)
    df = df[~(df[mds_updrs_items] == 101).any(axis=1)]

    df['INFODT'] = pd.to_datetime(df['INFODT'], format='%m/%Y')
    df['earliest_visit'] = df.groupby('PATNO')['INFODT'].transform('min')
    df['visit_month'] = (df['INFODT'].dt.year - df['earliest_visit'].dt.year) * 12 + \
                        (df['INFODT'].dt.month - df['earliest_visit'].dt.month)
    # Round to the closest multiple of 12
    df['visit_month'] = df['visit_month'].apply(lambda x: int(Decimal(x / 12.0).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * 12))

    # Remove pairs with the same visit month
    df = df.drop_duplicates(subset=['PATNO', 'visit_month'], keep='last')
    df = df.drop(columns='earliest_visit').reset_index(drop=True)

    return df, items


def thermometer_encode_column(column: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
    """Encode a column using thermometer encoding.

    Args:
        column (pd.Series): The column to encode.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Encoded DataFrame and column names.
    """
    max_value = int(column.max())
    encoded = np.zeros((len(column), max_value))
    for i, val in enumerate(column):
        encoded[i, :int(val)] = 1
    cols = [f"{column.name}_th{i + 1}" for i in range(max_value)]
    return pd.DataFrame(encoded, columns=cols), cols


def encode_df(merged_df: pd.DataFrame, items: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Thermometer encode the specified items in the DataFrame.

    Args:
        merged_df (pd.DataFrame): The merged DataFrame to encode.
        items (List[str]): List of item columns to encode.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Encoded DataFrame and list of encoded item names.
    """
    encoded_items = []
    df = merged_df.copy()

    for item in items:
        thermometer_encoded, cols = thermometer_encode_column(merged_df[item])
        encoded_items.extend(cols)
        df = df.drop(columns=[item])
        df = pd.concat([df, thermometer_encoded], axis=1)

    return df, encoded_items


def get_encoded_df(include_moca: bool = False) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Generate and return the encoded DataFrame.

    Args:
        include_moca (boolean): Whether to include MoCA data in addition to MDS-UPDRS.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Encoded DataFrame and list of encoded item names.
    """
    merged_df, items = create_merged_df(include_moca=include_moca)
    encoded_df, encoded_items = encode_df(merged_df, items)
    return encoded_df, items, encoded_items

