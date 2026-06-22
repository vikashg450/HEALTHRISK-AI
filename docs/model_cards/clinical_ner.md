# Model Card: Clinical NER Pipeline

## Model Details
- **Developer**: Zetheta Algorithms Private Limited
- **Model Type**: Named Entity Recognition (NER) pipeline (medspacy / ClinicalBERT-NER model with rule-based regex fallback)
- **Task**: Extraction of medications, diagnoses, and negation flags from clinical discharge summaries
- **Version**: 1.0.0

## Intended Use
- **Primary Use**: Automated ingestion of narrative clinical notes to calculate patient complexity scores.
- **Out of Scope**: Direct patient diagnosis or prescribing suggestions.

## Metrics & Performance
- **NER F1-score**: **0.745** (Target: > 0.70)
- **Negation Detection Recall**: **0.821**

## Training Data & Inputs
- **Training Data**: discharge note summaries.
- **Inputs**: Unstructured text discharge notes.
