# ComplaintOps: Banking Intent MLOps Platform

An enterprise-ready, end-to-end MLOps platform for automated customer-support intent classification in retail banking using the real-world **BANKING77** benchmark (77 fine-grained intents across 13,083 queries).

---

## 🏛️ System Architecture

```text
BANKING77 Ingestion (10,003 Train / 3,080 Test)
       │
       ▼
Feature Pipeline (Sublinear TF-IDF 1-2 n-grams, 10,000 features)
       │
       ▼
Regularized Classifier (Multinomial Logistic Regression, C=10.0, balanced)
       │
       ▼
Evaluation Harness (Macro-F1, Accuracy, Top Confused Pairs, Confusion Matrix)
       │
       ▼
MLflow 3 Tracking & Model Registry (Parameters, Per-Class Metrics, Version 1)
```

---

## 📊 Phase 1 & 2 Benchmark Results

Evaluated strictly on the official, unseen 3,080 test queries (balanced 40 queries per intent across all 77 classes):

| Metric | Baseline Score | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **0.8935 - 0.8951** | ~2,755 of 3,080 test queries accurately classified |
| **Macro-F1** | **0.8939 - 0.8954** | High unweighted mean F1 across 77 fine-grained classes |
| **Weighted-F1** | **0.8939 - 0.8954** | Balanced across classes |
| **Macro-Precision** | **0.8978 - 0.8989** | Low cross-class false-positive rate |
| **Macro-Recall** | **0.8935 - 0.8951** | Comprehensive intent recovery |
| **Train Duration** | **~4.6 - 7.0 seconds** | Rapid, reproducible CPU training on 10,003 samples |
| **Inference Latency** | **~0.026 ms / query** | Sub-millisecond single-query throughput |

---

## 🚀 Getting Started

### 1. Setup Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps
```

### 2. Train Model with MLflow Tracking & Model Registry
To execute training, evaluate on official test split, log all 77 per-class metrics to MLflow, and register version 1 of `BankingIntentClassifier`:

```powershell
python -m complaint_ops.models.train --use-mlflow --register-model
```

### 3. Launching the MLflow UI & Viewing Experiments
To launch the interactive MLflow UI server locally:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Then open your web browser to **[http://localhost:5000](http://localhost:5000)**.

In the MLflow UI you can:
- **Experiments**: View the `ComplaintOps-Banking77` experiment and all execution runs.
- **Parameters**: Inspect TF-IDF n-gram ranges, max features, regularization $C$, and sample volume.
- **Metrics**: Track overall `accuracy`, `macro_f1`, and per-class metrics (`class_f1/<category>`).
- **Artifacts**: Download `evaluation/baseline_eval_report.json`, `evaluation/per_class_metrics.csv`, and view `figures/confusion_matrix.png`.
- **Model Registry**: Inspect the registered model `BankingIntentClassifier`, its versions (Version 1, Version 2, etc.), lineage run ID, and transition stages.

### 4. Querying Registered Models in Python
```python
from complaint_ops.tracking import load_registered_model, get_model_registry_history

# View version ledger
history = get_model_registry_history("BankingIntentClassifier")
print("Latest registered version:", history[0]["version"])

# Load production model directly from MLflow
pipeline = load_registered_model(model_name="BankingIntentClassifier", version="1")
prediction = pipeline.predict(["When will my replacement debit card arrive?"])
print("Predicted intent:", prediction[0])
```

---

## 🧪 Automated Testing

ComplaintOps includes an automated test suite covering dataset schemas, preprocessing, pipeline serialization, probability calibration, low-confidence routing, and MLflow tracking:

```powershell
pytest tests/ -v
```

**Test Breakdown (16 passed):**
- `tests/test_data.py`: Text cleaning, control character stripping, query validation, and dataset distribution inspection (4 tests).
- `tests/test_pipeline.py`: Pipeline construction, fit/predict, prediction schema, low-confidence fallback trigger, artifact serialization, metric calculations, and real baseline bundle validation (7 tests).
- `tests/test_tracking.py`: MLflow experiment initialization, per-class CSV export, parameter/metric/artifact tracking, model registration & versioning, and end-to-end model artifact prediction consistency (5 tests).

---

## 🗺️ Project Roadmap
- [x] **Phase 1**: Dataset Ingestion, Baseline Modeling, Multi-Class Evaluation & Artifact Serialization
- [x] **Phase 2**: MLflow Experiment Tracking, Artifact Logging & Model Versioning
- [ ] **Phase 3**: FastAPI Production Prediction Service with Low-Confidence & Fallback Routing
- [ ] **Phase 4**: Automated Model Release Gate (Champion vs. Challenger) & Rollback Engine
- [ ] **Phase 5**: Interactive Operations Dashboard (Streamlit + Plotly)
- [ ] **Phase 6**: Statistical Input Drift Detection & Controlled Synthetic Drift Scenarios
- [ ] **Phase 7**: Containerization (Docker) & Continuous Integration / Deployment (GitHub Actions)
- [ ] **Phase 8**: Production Runbook & Verification
