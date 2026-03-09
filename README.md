#  News Bias Detector

A fine-tuned RoBERTa model for detecting bias in news sentences. Achieves **82% accuracy** and **0.81 macro F1** on a held-out test set, significantly outperforming a zero-shot BART-large-mnli baseline (64% accuracy).

---

## Results

| Model | Accuracy | Macro F1 | Biased F1 | Non-biased F1 |
|-------|----------|----------|-----------|---------------|
| Zero-Shot (BART-large-mnli) | 0.6463 | 0.39 | 0.75 | 0.42 |
| **Fine-tuned (RoBERTa-base)** | **0.8232** | **0.81** | **0.86** | **0.75** |

### Classification Report (Test Set)

```
              precision    recall  f1-score   support

      biased       0.88      0.84      0.86       204
  non-biased       0.72      0.79      0.75       107

    accuracy                           0.82       311
   macro avg       0.80      0.81      0.81       311
weighted avg       0.83      0.82      0.82       311
```

---

## Dataset

Uses the [BABE (Bias Annotations By Experts)](https://github.com/media-bias-group/BABE) dataset — a collection of ~1,700 labeled news sentences from outlets across the political spectrum (Breitbart, Alternet, USA Today, NBC, etc.), annotated for bias by domain experts.

**Label distribution after cleaning:**
- `biased`: 1,018 sentences
- `non-biased`: 533 sentences

**Splits:** 80% train / 20% test, with 15% of train used for validation.

---

## How It Works

The pipeline has two main stages:

### 1. Feature Engineering (VADER Sentiment)
- Sentences are lightly cleaned (HTML stripped, unicode normalized) for the transformer
- A separate heavy-cleaned + lemmatized version is used for VADER sentiment scoring
- VADER compound sentiment scores are stored as a feature alongside the model predictions

### 2. Fine-tuning
- `roberta-base` is fine-tuned for binary sequence classification (`biased` / `non-biased`)
- Class-weighted cross-entropy loss compensates for the ~2:1 class imbalance
- Dynamic padding, gradient checkpointing, and fp16 keep RAM usage manageable on free Colab

---

## Project Structure

```
bias-detector/
├── bias_detector_v2.ipynb   # Main notebook (all cells self-contained)
├── README.md
└── labeled_dataset.xlsx     # Source dataset (place in Google Drive)
```

---

## Setup

### Requirements

```bash
pip install vaderSentiment lxml openpyxl
pip install --upgrade transformers datasets accelerate
python -m spacy download en_core_web_sm
```

### Google Drive structure

Place your dataset at:
```
MyDrive/Colab_Notebooks/archive/labeled_dataset.xlsx
```

The trained model will be saved to:
```
MyDrive/Colab_Notebooks/bias_detector_roberta/
```

### Running

1. Open `bias_detector_v2.ipynb` in Google Colab
2. Run **Cell 1** (installs) and **restart the runtime**
3. Run all remaining cells top to bottom
4. Training takes ~15–30 minutes on a free Colab GPU

---

## Inference

After training, use the built-in `predict_bias()` function to classify any sentence:

```python
predict_bias("The radical left continues to destroy everything America stands for.")
# → {'label': 'biased', 'confidence': 0.9312, 'scores': {'biased': 0.9312, 'non-biased': 0.0688}}

predict_bias("The committee voted 7-3 in favour of the new infrastructure bill.")
# → {'label': 'non-biased', 'confidence': 0.8901, 'scores': {'biased': 0.1099, 'non-biased': 0.8901}}
```

To reload the model in a new session without retraining:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

SAVE_PATH = '/content/drive/MyDrive/Colab_Notebooks/bias_detector_roberta'
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH)
model.eval()
```

---

## Technical Details

### Model
- **Base model:** `roberta-base` (125M parameters)
- **Task:** Binary sequence classification
- **Labels:** `biased` (0), `non-biased` (1)

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 2e-5 |
| Batch size | 8 (effective 16 with gradient accumulation) |
| Max epochs | 4 (early stopping patience=2) |
| Max token length | 256 |
| Warmup steps | ~10% of total steps |
| Weight decay | 0.01 |
| Loss function | CrossEntropyLoss (class-weighted) |

### RAM Optimizations

Free-tier Colab has limited RAM. The following techniques keep it within bounds:

| Technique | Effect |
|-----------|--------|
| Drop unused DataFrame columns immediately | ~30–50% CPU RAM reduction |
| Dynamic padding (`DataCollatorWithPadding`) | ~40% tokenization RAM reduction |
| Gradient checkpointing | ~40% GPU RAM during training |
| fp16 training | ~50% GPU RAM |
| Gradient accumulation (steps=2) | Halves activation memory |
| Delete DataFrames after HF Dataset creation | Eliminates redundant copies |
| `save_total_limit=1` | Only best checkpoint kept on disk |

---

## Limitations

- Trained on English-language news sentences only
- Works best on sentence-level input (not full articles)
- Non-biased precision (0.72) reflects the training set imbalance — more non-biased examples would improve this
- Political bias framing reflects the dataset's U.S.-centric news sources

---

## Potential Improvements

- **More non-biased data** or augmentation (back-translation, synonym replacement) to address class imbalance
- **`roberta-large`** for higher accuracy at the cost of more RAM
- **Multi-label classification** to also predict opinion level (`Label_opinion` column in dataset)
- **Article-level prediction** by aggregating sentence-level scores

---

## License

This project is for research and educational purposes. The BABE dataset has its own license — refer to the [original repository](https://github.com/media-bias-group/BABE) for usage terms.
