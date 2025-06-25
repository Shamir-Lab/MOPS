"""
Utilities for reading the PPMI data tables.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd

DATA_DOWNLOAD_DATE = datetime(2024, 8, 7)

class PpmiDataLoader:
    def __init__(self, ppmi_dir: Path):
        ppmi_dir = Path(ppmi_dir)
        # Check if the path exists and is a directory.
        if not ppmi_dir.exists() or not ppmi_dir.is_dir():
            raise ValueError(f"Invalid directory path: {ppmi_dir}")

        self.ppmi_dir = ppmi_dir
        self.date_str = DATA_DOWNLOAD_DATE.strftime("%d%b%Y")

    def _get_df(self, df_path: Path):
        return pd.read_csv(self.ppmi_dir / df_path)

    def get_ledd_df(self):
        return self._get_df(
            Path("medical_history")
            / f"LEDD_Concomitant_Medication_Log_{self.date_str}.csv"
        )

    def get_dbs_df(self):
        return self._get_df(
            Path("medical_history") / f"Surgery_for_PD_Log_{self.date_str}.csv"
        )

    def get_diagnosis_history_df(self):
        return self._get_df(
            Path("medical_history") / f"PD_Diagnosis_History_{self.date_str}.csv"
        )

    def get_primary_diagnosis_df(self):
        return self._get_df(
            Path("medical_history") / f"Primary_Clinical_Diagnosis_{self.date_str}.csv"
        )

    def get_dopa_start_df(self):
        return self._get_df(
            Path("medical_history")
            / f"Initiation_of_Dopaminergic_Therapy_{self.date_str}.csv"
        )

    def get_participant_status(self):
        return self._get_df(
            Path("subject_characteristics")
            / f"Participant_Status_{self.date_str}.csv"
        )

    def get_participant_age(self):
        return self._get_df(
            Path("subject_characteristics") / f"Age_at_visit_{self.date_str}.csv"
        )

    def get_participant_demographics(self):
        return self._get_df(
            Path("subject_characteristics") / f"Demographics_{self.date_str}.csv"
        )

    def get_mds_updrs_part1_patient_df(self):
        return self._get_df(
            Path("motor_assessments")
            / f"MDS-UPDRS_Part_I_Patient_Questionnaire_{self.date_str}.csv"
        )

    def get_mds_updrs_part1_rater_df(self):
        return self._get_df(
            Path("motor_assessments")
            / f"MDS-UPDRS_Part_I_{self.date_str}.csv"
        )

    def get_mds_updrs_part2_df(self):
        return self._get_df(
            Path("motor_assessments")
            / f"MDS_UPDRS_Part_II__Patient_Questionnaire_{self.date_str}.csv"
        )

    def get_mds_updrs_part3_df(self):
        return self._get_df(
            Path("motor_assessments") / f"MDS-UPDRS_Part_III_{self.date_str}.csv"
        )

    def get_mds_updrs_part4_df(self):
        return self._get_df(
            Path("motor_assessments")
            / f"MDS-UPDRS_Part_IV__Motor_Complications_{self.date_str}.csv"
        )

    def get_moca_df(self):
        return self._get_df(
            Path("non_motor_assessments")
            / f"Montreal_Cognitive_Assessment__MoCA__{self.date_str}.csv"
        )

    def get_se_adl_df(self):
        return self._get_df(
            Path("motor_assessments")
            / f"Modified_Schwab___England_Activities_of_Daily_Living_{self.date_str}.csv"
        )