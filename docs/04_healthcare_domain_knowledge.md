# 04. Healthcare Domain Knowledge

This report provides the biomedical and clinical context behind the data structures and risk scoring algorithms embedded in the HealthRisk AI platform.

## Clinical Foundations

### 1. MIMIC-IV Database
The system is modeled around the data schema of **MIMIC-IV (Medical Information Mart for Intensive Care)**, a de-identified, comprehensive database containing hospital records for patients admitted to the Beth Israel Deaconess Medical Center.
- Key tables utilized include `patients` (demographics), `admissions` (timestamps, discharge dispositions), `diagnoses_icd` (coded conditions), `labevents` (objective diagnostic values), and `prescriptions` (therapeutics).

### 2. Charlson Comorbidity Index (CCI)
The CCI is a validated method of classifying comorbid conditions of patients to predict 1-year mortality. Each comorbidity category is assigned a weight (1, 2, 3, or 6) based on the adjusted relative risk of mortality:
- **Weight 1**: Myocardial infarction, congestive heart failure, peripheral vascular disease, dementia, chronic pulmonary disease, diabetes without complications.
- **Weight 2**: Diabetes with end-organ damage, hemiplegia/paraplegia, moderate-to-severe renal disease, localized solid tumors.
- **Weight 3**: Moderate-to-severe liver disease.
- **Weight 6**: Metastatic solid tumors, AIDS/HIV.
- The final score is the sum of comorbidity weights, serving as a powerful baseline indicator for patient frailty and readmission probability.

### 3. Clinical Trial Phase Transition Probabilities
Pharma valuation models depend heavily on the **Probability of Success (PoS)** of drug development pipelines. Development is structured into sequential phases:
- **Phase I**: Evaluates safety and dosage in small cohorts. Historical success probability to transition: **~63%**.
- **Phase II**: Evaluates efficacy and side effects in larger patient groups. Historical success probability to transition: **~31%**.
- **Phase III**: Confirms efficacy and monitors adverse reactions in large, randomized trials. Historical success probability to transition: **~49.3%**.
- **Biomarker Selection**: In modern clinical trials, utilizing patient stratification biomarkers is statistically shown to double success rates in target indications (e.g., oncology).

### 4. Adverse Event Reporting (FDA FAERS)
The **FDA Adverse Event Reporting System (FAERS)** is a database that contains information on adverse event and medication error reports submitted to the FDA. The ingestion pipeline tracks drug safety signals by querying FAERS APIs, identifying sudden spikes in adverse event reports that could impact clinical utility and portfolio equity valuations.
