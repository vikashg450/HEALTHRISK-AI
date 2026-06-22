"""
HealthRisk AI — Data Ingestion Module (Day 1–2)
Handles data acquisition from MIMIC-IV, ClinicalTrials.gov, FDA FAERS, WHO GHO.
"""

import os
import time
import requests
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────
# MIMIC-IV Loader
# ──────────────────────────────────────────────────────

class MIMICLoader:
    """
    Loads MIMIC-IV tables from local CSV files.
    Data must be downloaded from PhysioNet first:
    https://physionet.org/content/mimiciv/2.2/
    """

    def __init__(self, data_dir: str = "data/raw/mimic_iv"):
        self.data_dir = Path(data_dir)

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Load a MIMIC-IV table from CSV."""
        path = self.data_dir / f"{table_name}.csv.gz"
        if not path.exists():
            path = self.data_dir / f"{table_name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"MIMIC-IV table '{table_name}' not found at {self.data_dir}.\n"
                "Please download from https://physionet.org/content/mimiciv/2.2/"
            )
        print(f"Loading MIMIC-IV table: {table_name}...")
        return pd.read_csv(path, low_memory=False)

    def load_core_tables(self) -> dict:
        """Load the essential tables needed for the project."""
        tables = ["patients", "admissions", "diagnoses_icd",
                  "prescriptions", "labevents", "procedures_icd"]
        return {t: self.load_table(t) for t in tables}


# ──────────────────────────────────────────────────────
# ClinicalTrials.gov API
# ──────────────────────────────────────────────────────

class ClinicalTrialsAPI:
    """Fetches data from the ClinicalTrials.gov v2 API."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def search_trials(
        self,
        condition: Optional[str] = None,
        sponsor: Optional[str] = None,
        phase: Optional[str] = None,
        status: str = "RECRUITING,ACTIVE_NOT_RECRUITING",
        page_size: int = 100,
        max_pages: int = 10,
    ) -> pd.DataFrame:
        """Search clinical trials and return as DataFrame."""
        params = {
            "pageSize": page_size,
            "filter.overallStatus": status,
            "fields": ",".join([
                "NCTId", "OfficialTitle", "BriefTitle", "Condition",
                "OverallStatus", "Phase", "StudyType", "EnrollmentCount",
                "StartDate", "CompletionDate", "PrimaryOutcomeMeasure",
                "LeadSponsorName", "LocationCountry"
            ])
        }
        if condition:
            params["query.cond"] = condition
        if sponsor:
            params["query.spons"] = sponsor
        if phase:
            params["filter.phase"] = phase

        all_studies = []
        page_token = None

        for page in range(max_pages):
            if page_token:
                params["pageToken"] = page_token
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            studies = data.get("studies", [])
            all_studies.extend(studies)
            print(f"  Fetched page {page + 1}: {len(studies)} trials")

            page_token = data.get("nextPageToken")
            if not page_token:
                break
            time.sleep(0.5)  # Rate limiting

        print(f"Total trials fetched: {len(all_studies)}")
        return self._parse_studies(all_studies)

    def _parse_studies(self, studies: list) -> pd.DataFrame:
        records = []
        for study in studies:
            proto = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            stat_mod = proto.get("statusModule", {})
            design_mod = proto.get("designModule", {})
            enroll_info = design_mod.get("enrollmentInfo", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
            outcomes_mod = proto.get("outcomesModule", {})
            primary_outcomes = outcomes_mod.get("primaryOutcomes", [{}])

            records.append({
                "nct_id": id_mod.get("nctId"),
                "title": id_mod.get("briefTitle"),
                "status": stat_mod.get("overallStatus"),
                "phase": design_mod.get("phases", [""])[0] if design_mod.get("phases") else "",
                "enrollment": enroll_info.get("count"),
                "sponsor": sponsor_mod.get("leadSponsor", {}).get("name"),
                "primary_endpoint": primary_outcomes[0].get("measure") if primary_outcomes else "",
                "start_date": stat_mod.get("startDateStruct", {}).get("date"),
                "completion_date": stat_mod.get("completionDateStruct", {}).get("date"),
            })
        return pd.DataFrame(records)


# ──────────────────────────────────────────────────────
# FDA FAERS API
# ──────────────────────────────────────────────────────

class FDAFaersAPI:
    """Fetches adverse event data from FDA FAERS."""

    BASE_URL = "https://api.fda.gov/drug/event.json"

    def get_adverse_events(
        self,
        drug_name: Optional[str] = None,
        reaction: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch adverse event reports."""
        params = {"limit": limit}
        if drug_name:
            params["search"] = f'patient.drug.medicinalproduct:"{drug_name}"'
        elif reaction:
            params["search"] = f'patient.reaction.reactionmeddrapt:"{reaction}"'

        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        if resp.status_code == 404:
            print("No FAERS results found for this query.")
            return pd.DataFrame()
        resp.raise_for_status()
        results = resp.json().get("results", [])

        records = []
        for r in results:
            patient = r.get("patient", {})
            drugs = patient.get("drug", [{}])
            reactions = patient.get("reaction", [{}])
            records.append({
                "report_id": r.get("safetyreportid"),
                "receive_date": r.get("receivedate"),
                "serious": r.get("serious"),
                "primary_drug": drugs[0].get("medicinalproduct") if drugs else "",
                "primary_reaction": reactions[0].get("reactionmeddrapt") if reactions else "",
                "patient_age": patient.get("patientonsetage"),
                "patient_sex": patient.get("patientsex"),
                "outcome": patient.get("patientdeath"),
                "n_drugs": len(drugs),
                "n_reactions": len(reactions),
            })
        return pd.DataFrame(records)

    def get_drug_counts(self, drug_name: str) -> dict:
        """Get aggregate counts for a drug."""
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": 20,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        if resp.status_code != 200:
            return {}
        results = resp.json().get("results", [])
        return {r["term"]: r["count"] for r in results}


# ──────────────────────────────────────────────────────
# WHO Global Health Observatory API
# ──────────────────────────────────────────────────────

class WHOGHOApi:
    """Fetches population health data from WHO GHO."""

    BASE_URL = "https://ghoapi.azureedge.net/api"

    def list_indicators(self, filter_str: Optional[str] = None) -> pd.DataFrame:
        url = f"{self.BASE_URL}/Indicator"
        if filter_str:
            url += f"?$filter=contains(IndicatorName,'{filter_str}')"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return pd.DataFrame(resp.json().get("value", []))

    def get_indicator_data(self, indicator_code: str) -> pd.DataFrame:
        """Fetch data for a specific WHO indicator."""
        url = f"{self.BASE_URL}/{indicator_code}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        records = resp.json().get("value", [])
        df = pd.DataFrame(records)
        if df.empty:
            return df
        cols_keep = ["SpatialDim", "TimeDim", "NumericValue", "Dim1", "Dim2"]
        return df[[c for c in cols_keep if c in df.columns]].rename(
            columns={"SpatialDim": "country", "TimeDim": "year",
                     "NumericValue": "value"}
        )


if __name__ == "__main__":
    # Quick smoke tests
    print("=== Testing ClinicalTrials.gov API ===")
    ct = ClinicalTrialsAPI()
    df = ct.search_trials(condition="diabetes", page_size=10, max_pages=1)
    print(df.head(3))

    print("\n=== Testing FDA FAERS API ===")
    fda = FDAFaersAPI()
    df2 = fda.get_adverse_events(drug_name="aspirin", limit=5)
    print(df2.head(3))
