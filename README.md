# Mini Project: Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student USN:** 1MS25CI011  
**Course / Lab:** Skill Lab (MLOps with MLflow)  
**Task:** Sentiment Classification (Binary: POSITIVE / NEGATIVE)

---

## 1. Task and Models

* **Task:** Binary Sentiment Classification
* **Evaluation Dataset:** Curated Stanford Sentiment Treebank (SST-2) benchmark subset covering diverse linguistic nuances, colloquialisms, negations, enthusiasm, and complaints.
* **Evaluation Subset Size:** 50 balanced samples (25 POSITIVE, 25 NEGATIVE) in `data/eval_inputs.csv`.
* **New Unseen Data:** 8 fresh samples in `data/new_data.csv` for production inference verification.
* **Models Compared:**
  * **Model A (Baseline / Initial Champion):** [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) (~67M parameters) — Fast, lightweight distilled transformer baseline.
  * **Model B (Challenger Candidate):** [`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english) (~355M parameters) — Deep RoBERTa-large architecture finetuned on diverse sentiment datasets for higher accuracy.

---

## 2. How to Run

Follow these commands in order from the repository root:

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Start local MLflow server with SQLite backend (in a separate terminal)
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# 3. Run evaluation across both models and log to MLflow
python evaluate.py

# 4. Register models and assign Champion/Challenger aliases in the MLflow Model Registry
python register_champion.py

# 5. Load model strictly via @champion alias and predict on new data
python predict_from_mlflow.py
```

*MLflow UI is viewable at `http://127.0.0.1:5000`.*

---

## 3. Champion Rule (Stated Prior to Experimentation)

As per Section 5.4, the decision rule is formulated before final execution:

1. **Primary Metric:** Macro-Averaged F1 Score (`macro_f1`).
2. **Minimum Improvement Margin:** The Challenger (Model B) must exceed the Champion (Model A) by at least **0.0200 (2.0 percentage points)** in `macro_f1`.
3. **Tie-Breaker & Guardrails:**
   * If the difference is strictly within the 0.02 margin, the current Champion is retained to favor computational efficiency.
   * **Latency Guardrail:** If Model B's average inference latency exceeds **15x** that of Model A, Model B is rejected unless its `macro_f1` improvement is >= 0.05.
4. **Metric Directions:**
   * Higher is better: `accuracy`, `macro_f1`, `precision`, `recall`.
   * Lower is better: `avg_latency_ms`, `total_inference_s`.

---

## 4. Results Table

Evaluated on the identical 50-sample shared test set (`data/eval_inputs.csv`):

| Metric | Direction | Model A (`distilbert-sst2`) | Model B (`roberta-large`) | Difference ($\Delta$ B - A) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | Higher $\uparrow$ | 0.9800 (98.0%) | 1.0000 (100.0%) | **+0.0200 (+2.0%)** |
| **Macro-F1** | Higher $\uparrow$ | 0.9800 | 1.0000 | **+0.0200 (+2.00%)** |
| **Precision** | Higher $\uparrow$ | 0.9808 | 1.0000 | **+0.0192 (+1.92%)** |
| **Recall** | Higher $\uparrow$ | 0.9800 | 1.0000 | **+0.0200 (+2.00%)** |
| **Avg Latency (ms/sample)** | Lower $\downarrow$ | 33.76 ms | 412.79 ms | +379.03 ms (12.2x slower) |
| **Total Inference (s)** | Lower $\downarrow$ | 1.69 s | 20.64 s | +18.95 s |

*Full machine-readable metrics are saved in `outputs/metrics_summary.json`.*

### Qualitative Disagreement Analysis:
* **Sample ID 10:** *"Setup was a breeze, took only five minutes out of the box to connect."*
  * **Ground Truth:** `POSITIVE`
  * **Model A Prediction:** `NEGATIVE` (Confidence: 0.9918)
  * **Model B Prediction:** `POSITIVE` (Confidence: 0.9988)
  * **Analysis:** Model A was tripped up by colloquial phrase structure ("out of the box", "five minutes"), misinterpreting it as negative friction. Model B's deeper self-attention context accurately recognized "breeze" and smooth setup as strongly positive praise.

---

## 5. Champion-Challenger Decision

**Decision:** **Model B (`siebert/sentiment-roberta-large-english`) is promoted to CHAMPION.**

### Justification & Analysis (150–250 Words):
Model B (RoBERTa-large) achieved a perfect **Macro-F1 score of 1.0000**, outperforming Model A (DistilBERT, **0.9800**) by exactly **+0.0200 (2.00 percentage points)**. Under our pre-declared Champion rule, an improvement meeting or exceeding the 0.02 threshold warrants promoting the Challenger to Champion.

Analyzing the single qualitative disagreement case (`outputs/disagreement_cases.csv`), Model B demonstrated a markedly superior understanding of colloquial English praise. For the sentence *"Setup was a breeze, took only five minutes out of the box to connect"*, DistilBERT erroneously predicted `NEGATIVE` with 99.18% confidence, whereas RoBERTa-large correctly resolved it as `POSITIVE` (99.88% confidence).

In terms of architectural trade-offs, Model B possesses $\sim 5.3\times$ more parameters ($\sim 355\text{M}$ vs $\sim 67\text{M}$) and consumes an average CPU latency of 412.8 ms per input compared to DistilBERT's lightweight 33.8 ms ($12.2\times$ slower). However, this latency remains within our predefined 15x guardrail. In domains such as mission-critical customer sentiment or high-value customer feedback analysis where false negatives on nuanced praise or sarcasm are costly, perfect semantic comprehension takes precedence over raw latency. Therefore, Model B is registered with alias `@champion`, and Model A is transitioned to `@challenger`.

---

## 6. Screenshots & Verification (Section 6)

| Screenshot | Description | Preview |
| :--- | :--- | :---: |
| **1. MLflow Compare View** | Side-by-side comparison of Model A & B runs (`compare_sentiment`) | [View](screenshots/mlflow1.png) |
| **2. Model Registry & Aliases** | Registered model versions and assigned roles (`@champion` / `@challenger`) | [View](screenshots/mlflow2.png) |
| **3. Terminal Output** | Execution of `predict_from_mlflow.py` resolving `@champion` on new data | [View](screenshots/mlflow3.png) |

---

## 7. Limitations and Declaration

* **Limitations:** The evaluation was conducted on a curated 50-sample subset run on CPU. While covering challenging edge cases, larger benchmark evaluations (e.g., full 1,821 SST-2 test samples) are recommended prior to enterprise-scale deployment.
* **Declaration of Tools:** Developed in PyCharm and VS Code. In accordance with Section 4 of the assignment guidelines, AI assistant Antigravity was utilized for scaffolding setup, data formatting, and code review. All code logic, metric implementations, MLflow tracking architectures, and registry aliasing were reviewed, understood, and verified by the student for viva examination.
