---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
- natural-language-inference
- nli
repo: https://github.com/VP1606/NLU-CW

---

# Model Card for GroupCW47-ModernBERT-Ensemble-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a weighted ensemble of two fine-tuned `answerdotai/ModernBERT-large` models designed for binary Natural Language Inference (NLI). It predicts whether a given hypothesis is entailed by a premise (1 = ENTAILMENT, 0 = NOT_ENTAILMENT) by combining the probability outputs of both models with a dynamically optimised threshold.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model was developed for a university shared task on Natural Language Understanding. It takes pairs of sentences (a premise and a hypothesis) and predicts a binary relationship: 1 (ENTAILMENT) or 0 (NOT_ENTAILMENT).

The ensemble combines two independently fine-tuned ModernBERT-large checkpoints:
- **modernBerta**: Fine-tuned directly from `answerdotai/ModernBERT-large`
- **taskSource**: Fine-tuned from a tasksource-initialised ModernBERT-large checkpoint

The final prediction is produced by a weighted sum of the softmax probabilities from both models (weight 0.4 for modernBerta, 0.6 for taskSource), followed by a dynamically tuned threshold of 0.43 to maximise binary F1.

- **Developed by:** Group CW47 — Premakantha Varun, Kaan Oktem, Munir Emre Tanatas
- **Language(s):** English
- **Model type:** Transformer-based Sequence Classifier (Weighted Ensemble)
- **Model architecture:** ModernBERT-Large (395M parameters × 2)
- **Finetuned from models:**
  - `modernBerta`: `answerdotai/ModernBERT-large`
  - `taskSource`: `tasksource/ModernBERT-large-nli`

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://github.com/VP1606/NLU-CW
- **Base model:** https://huggingface.co/answerdotai/ModernBERT-large

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

Both models were fine-tuned on the provided training dataset of 24,432 NLI sentence pairs (`train.csv`). Rows with NaN values in premise, hypothesis, or label columns were dropped. Labels were strictly mapped to integers (0 and 1) to prevent CUDA device-side assertion failures on the GPU.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

**modernBerta** (fine-tuned from `answerdotai/ModernBERT-large`):

      - learning_rate: 2e-05
      - train_batch_size: 4 (per device)
      - gradient_accumulation_steps: 4 (effective batch size of 16)
      - eval_batch_size: 8 (per device)
      - num_epochs: 3
      - weight_decay: 0.01
      - mixed_precision: fp16=True
      - best model selection: metric = F1 (load_best_model_at_end=True)

**taskSource** (fine-tuned from `tasksource/ModernBERT-large-nli`):

      - learning_rate: 1e-05
      - train_batch_size: 4 (per device)
      - gradient_accumulation_steps: 4 (effective batch size of 16)
      - eval_batch_size: 8 (per device)
      - num_epochs: 3
      - weight_decay: 0.01
      - mixed_precision: bf16=True
      - best model selection: metric = F1 (load_best_model_at_end=True)

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->

      - model size: ~395M parameters per model (~1.5 GB per safetensors file)
      - training time: ~35 minutes per model on NVIDIA T4 (Google Colab)
      - total training steps: 4,581 (3 epochs over 24,432 samples)
      - best checkpoint: epoch 2 / 3 (selected by highest dev F1)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

The development set (`dev.csv`) consisting of 6,736 NLI sentence pairs with ground-truth labels.

#### Metrics

<!-- These are the evaluation metrics being used. -->

      - Binary F1-Score
      - Accuracy

### Results

#### Standalone Models (dev set)
| Model | F1 Score |
|---|---|
| modernBerta (fine-tuned ModernBERT-large) | 92.83% |
| taskSource (tasksource fine-tuned ModernBERT-large) | 94.00% |

#### Ensemble Sweep (dev set)
| orig weight | tasksource weight | Default F1 | Best F1 | Threshold |
|---|---|---|---|---|
| 0.2 | 0.8 | 94.10% | 94.20% | 0.24 |
| 0.3 | 0.7 | 94.13% | 94.18% | 0.56 |
| **0.4** | **0.6** | **94.20%** | **94.22%** | **0.43** |
| 0.5 | 0.5 | 94.17% | 94.17% | 0.50 |
| 0.6 | 0.4 | 93.01% | 93.66% | 0.40 |

**Best configuration:** modernBerta=0.4 + taskSource=0.6 → **F1: 94.22%** @ threshold 0.43

#### Training Set Results (modernBerta)
| Metric | Score |
|---|---|
| Train Loss | 0.6102 |
| Train F1 | — |

> Note: Training F1 was not logged separately during the run. Dev F1 was used as the primary optimisation metric via `load_best_model_at_end=True`.

## Technical Specifications

### Hardware

      - RAM: at least 16 GB
      - GPU: Requires at least 15–16 GB of VRAM per model (e.g., NVIDIA T4, L4, or V100) for training. Inference can run on Apple Silicon (MPS) but will be slower.

### Software

      - Python 3.9+
      - Transformers (Hugging Face)
      - PyTorch
      - scikit-learn
      - pandas, numpy

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

**Domain Specificity:** Both models are fine-tuned exclusively on the 24K coursework NLI dataset and may not generalise well to other domains.
**Threshold Sensitivity:** Final predictions rely on the dynamically tuned threshold of 0.43; performance may vary if the label distribution shifts.
**Language Limitation:** Trained exclusively on English text.
**Ensemble Dependency:** Both model checkpoints (~1.5 GB each) must be available at inference time.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

**Ensemble Strategy:** Rather than picking a single best model, a grid search over ensemble weights (0.2–0.6 for each model) was conducted on the dev set. The weighted softmax probabilities were combined and a threshold sweep from 0.10 to 0.90 was applied to find the configuration maximising binary F1.

**Dynamic Threshold Optimisation:** Instead of relying on a standard 0.50 confidence cut-off, a custom loop was implemented post-training to extract raw logits, convert them to probabilities via Softmax, and test all thresholds between 0.10 and 0.90 to scientifically maximise the target metric.
