"""
HealthRisk AI — ClinicalBERT NLP Module (Day 4–5)
Fine-tunes ClinicalBERT on discharge summaries for:
  1. Risk classification (low / medium / high)
  2. Named Entity Recognition (medications, diagnoses, procedures)
  3. Clinical complexity scoring (cost prediction from notes)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments, Trainer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List, Optional, Dict


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────
# DATASET CLASSES
# ──────────────────────────────────────────────────────

class ClinicalNoteDataset(Dataset):
    """Dataset for clinical note classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


class ClinicalEmbeddingExtractor:
    """Extracts [CLS] embeddings from ClinicalBERT for downstream tasks."""

    def __init__(self, model_path: str = MODEL_NAME, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(DEVICE)
        self.model.eval()
        self.max_length = max_length

    def embed(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Embed a list of texts → (n, 768) numpy array."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Embedded {i + len(batch)}/{len(texts)} texts")
        return np.vstack(all_embeddings)


# ──────────────────────────────────────────────────────
# RISK CLASSIFICATION MODEL
# ──────────────────────────────────────────────────────

class ClinicalRiskClassifier:
    """
    Fine-tuned ClinicalBERT for patient risk stratification:
    0 = Low Risk, 1 = Medium Risk, 2 = High Risk
    Target: AUROC > 0.75
    """

    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = 3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.num_labels = num_labels

    def prepare_data(
        self,
        notes_df: pd.DataFrame,
        label_col: str = "risk_label",
        text_col: str = "note_text",
        test_size: float = 0.20,
        seed: int = 42,
    ):
        texts = notes_df[text_col].tolist()
        labels = notes_df[label_col].tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=seed
        )
        self.train_dataset = ClinicalNoteDataset(X_train, y_train, self.tokenizer)
        self.test_dataset = ClinicalNoteDataset(X_test, y_test, self.tokenizer)
        self.y_test = y_test
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    def fine_tune(
        self,
        output_dir: str = "models/clinicalbert_risk",
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_dir=f"{output_dir}/logs",
            seed=42,
            fp16=torch.cuda.is_available(),
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )
        trainer.train()
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate AUROC on test set."""
        self.model.eval().to(DEVICE)
        loader = DataLoader(self.test_dataset, batch_size=32)
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch["labels"].numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # One-vs-rest AUROC
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(all_labels, classes=list(range(self.num_labels)))
        auroc = roc_auc_score(y_bin, all_probs, multi_class="ovr", average="macro")
        print(f"AUROC (macro OvR): {auroc:.4f}  (target > 0.75)")
        return {"auroc": auroc}


# ──────────────────────────────────────────────────────
# CLINICAL NER PIPELINE
# ──────────────────────────────────────────────────────

class ClinicalNERPipeline:
    """
    Named Entity Recognition for clinical notes using medspacy.
    Extracts: medications, diagnoses, procedures, lab values, negations.
    Target: F1 > 0.70
    """

    def __init__(self):
        try:
            import medspacy
            self.nlp = medspacy.load()
            self._use_medspacy = True
        except ImportError:
            print("medspacy not installed. Using rule-based fallback.")
            self._use_medspacy = False
            self._init_fallback()

    def _init_fallback(self):
        """Simple regex-based NER fallback."""
        import re
        self._rx_patterns = {
            "MEDICATION": re.compile(
                r"\b(aspirin|metformin|lisinopril|atorvastatin|metoprolol|"
                r"warfarin|furosemide|amlodipine|omeprazole|insulin)\b", re.I
            ),
            "DIAGNOSIS": re.compile(
                r"\b(diabetes|hypertension|heart failure|pneumonia|sepsis|"
                r"COPD|acute MI|stroke|CKD|atrial fibrillation)\b", re.I
            ),
        }

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract clinical entities from text."""
        entities = []
        if self._use_medspacy:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "negated": getattr(ent._, "is_negated", False),
                    "uncertain": getattr(ent._, "is_uncertain", False),
                    "historical": getattr(ent._, "is_historical", False),
                })
        else:
            for label, pattern in self._rx_patterns.items():
                for m in pattern.finditer(text):
                    entities.append({
                        "text": m.group(),
                        "label": label,
                        "start": m.start(),
                        "end": m.end(),
                        "negated": "no " + m.group().lower() in text.lower(),
                        "uncertain": False,
                        "historical": "history of " + m.group().lower() in text.lower(),
                    })
        return entities

    def process_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Process a batch of clinical notes."""
        return [self.extract_entities(t) for t in texts]

    def build_entity_features(self, texts: List[str]) -> pd.DataFrame:
        """Convert NER output to feature matrix."""
        records = []
        for i, text in enumerate(texts):
            ents = self.extract_entities(text)
            labels = [e["label"] for e in ents]
            records.append({
                "idx": i,
                "n_medications": labels.count("MEDICATION"),
                "n_diagnoses": labels.count("DIAGNOSIS"),
                "n_procedures": labels.count("PROCEDURE"),
                "n_negated": sum(1 for e in ents if e.get("negated")),
                "n_uncertain": sum(1 for e in ents if e.get("uncertain")),
                "n_historical": sum(1 for e in ents if e.get("historical")),
                "total_entities": len(ents),
            })
        return pd.DataFrame(records)


# ──────────────────────────────────────────────────────
# CLINICAL COMPLEXITY SCORER
# ──────────────────────────────────────────────────────

class ClinicalComplexityScorer:
    """
    Predicts total cost from clinical note embeddings alone.
    Target: R² > 0.15 (demonstrating notes contain info beyond structured codes)
    """

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        self.extractor = ClinicalEmbeddingExtractor()
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(DEVICE)
        self.is_fitted = False

    def fit(
        self,
        texts: List[str],
        costs: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
    ):
        print("Extracting embeddings...")
        embeddings = self.extractor.embed(texts)
        costs_log = np.log1p(costs)  # Log transform for skewed cost distribution

        X = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(costs_log, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        optimizer = torch.optim.AdamW(self.regressor.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.regressor.train()
            idx = torch.randperm(X.size(0))
            epoch_loss = 0
            for i in range(0, X.size(0), batch_size):
                batch_idx = idx[i: i + batch_size]
                pred = self.regressor(X[batch_idx])
                loss = criterion(pred, y[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

        self.is_fitted = True
        self.embeddings_ = embeddings

    def predict(self, texts: List[str]) -> np.ndarray:
        embeddings = self.extractor.embed(texts)
        X = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
        self.regressor.eval()
        with torch.no_grad():
            preds_log = self.regressor(X).cpu().numpy().flatten()
        return np.expm1(preds_log)  # Back-transform

    def evaluate(self, texts: List[str], costs: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import r2_score, mean_absolute_percentage_error
        preds = self.predict(texts)
        r2 = r2_score(costs, preds)
        mape = mean_absolute_percentage_error(costs, preds)
        print(f"R²: {r2:.4f}  (target > 0.15)")
        print(f"MAPE: {mape:.4f}  (target < 0.15)")
        return {"r2": r2, "mape": mape}


if __name__ == "__main__":
    print("Testing ClinicalNERPipeline...")
    ner = ClinicalNERPipeline()
    sample = ("Patient admitted with heart failure exacerbation. "
              "No chest pain. History of hypertension. "
              "Started furosemide 40mg IV and aspirin 81mg PO.")
    entities = ner.extract_entities(sample)
    print(f"Found {len(entities)} entities:")
    for e in entities:
        print(f"  [{e['label']}] '{e['text']}' | negated={e['negated']}")
