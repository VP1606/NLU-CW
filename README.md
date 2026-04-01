# OracleNet — COMP34812 NLU Coursework

**Group:** CW47
**Chosen Categories**: Category B, C

## 1. Category B: Deep Learning (Non-Transformer)
**Model:** ResESIM (Residual ESIM) with 1477-dimensional input embeddings

### Model: OracleNet

We propose OracleNet: a model built on the ResESIM Architecture (Chen et al. (2017)), with further enhancements:

- Multi-Head Attention
- Highway Input Projection (Srivastava et al. (2015))
- Attention Weighted Pooling (Bahdanau et al. (2015))
- Intra-Sentence Self Attention (Parikh et al. (2016))
- and more...

### Results

| Split | Macro F1 |
|---|---|
| Dev set | 70.41% |
| Training set | 72.6% |
| Trial set (50 examples) | 75.85% |

**Best model:** `ff2f02d4`
[Hugging Face](https://huggingface.co/VP1606/OracleNet)

### Code Structure

```
NLU-CW/
├── demo.ipynb                  # Demo notebook — run this to generate predictions
├── res_esim/
│   ├── run/
│   │   └── train.py            # Training entry point
│   ├── model_layers/
│   │   └── oracle_net.py       # OracleNet model definition (ResESIM)
│   ├── loader/
│   │   └── res_esim_dataset.py # Dataset loader for pre-computed embeddings
│   └── evaluation/             # Evaluation code (dev set F1)
├── precomputeClasses.py        # EmbeddingPrecomputer — 1477d embedding pipeline
├── build_meta.py               # Builds meta.pt (vocab, GloVe, char2idx, pos2idx)
├── final_model_versions/
│   └── ff2f02d4/
│       ├── best_model.pt       # Trained model weights (stored via Git LFS)
│       └── meta.json           # Model configuration
├── notebook_data/
│   ├── meta.pt                 # Vocab + GloVe embeddings (stored via Git LFS)
│   └── elmo_model/             # ELMo weights (stored via Git LFS)
├── data/
│   ├── train.csv               # Training data
│   ├── dev.csv                 # Development data (with labels)
│   ├── test.csv                # Test data (no labels)
│   └── NLI_trial.csv           # Trial data (50 examples, with labels)
└── require310.txt              # Python dependencies
```

**Code organisation:**
- **Training code:** `res_esim/run/train.py` and `precomputeClasses.py`
- **Evaluation code:** `res_esim/evaluation/` — evaluates on dev set
- **Demo code:** `demo.ipynb` — loads saved model and generates predictions

### Running the Demo

#### Requirements
- Python 3.10
- Git LFS (to pull model weights)

#### Setup
```bash
# Pull model weights via Git LFS
git lfs install
git lfs pull

# Install dependencies
pip install -r require310.txt
./venv310/bin/pip install 'tokenizers==0.13.3' 'allennlp==2.10.1' 'allennlp-models==2.10.1' 'numpy<2.0'

```

### Generate Predictions
Open and run `demo.ipynb` top to bottom. It will:
1. Pre-compute 1477d embeddings (GloVe + ELMo + CharCNN + POS + Negation)
2. Load the trained OracleNet model
3. Run inference on `data/test.csv`
4. Save predictions to `Group_16_B.csv`

### Training

```bash
# From the root directory
python -m res_esim.run.train
```

Note: ELMo pre-computation requires the `venv310` environment (Python 3.10 with AllenNLP).

---

### Attribution

This work builds on the following:

| Resource | Reference |
|---|---|
| ESIM architecture | Chen et al. (2017), "Enhanced LSTM for Natural Language Inference" |
| Highway Input Projection | Srivastava et al. (2015), "Training Very Deep Networks" |
| Attention Weighted Pooling | dos Santos et al. (2016), "Attentive Pooling Networks" |
| Intra-Sentence Self-Attention | Parikh et al. (2016), "A Decomposable Attention Model for NLI" |
| ELMo embeddings | Peters et al. (2018), "Deep contextualized word representations" — weights from [AllenNLP](https://allennlp.org/) |
| GloVe embeddings | Pennington et al. (2014), "GloVe: Global Vectors for Word Representation" — 300d vectors from [Stanford NLP](https://nlp.stanford.edu/projects/glove/) |
| CharCNN | Kim et al. (2016), "Character-Aware Neural Language Models" |
| POS tags | Universal Dependencies via [spaCy](https://spacy.io/) `en_core_web_sm` |
| AllenNLP library | Gardner et al. (2018), [https://github.com/allenai/allennlp](https://github.com/allenai/allennlp) |

---

### Model Files (Cloud Storage)

Large files are stored via Git LFS. If Git LFS is unavailable, download manually:

| File | Description |
|---|---|
| `notebook_data/meta.pt` | Vocab + GloVe embedding matrix (1.0 MB) |
| `final_model_versions/ff2f02d4/best_model.pt` | Trained OracleNet weights (86.9 MB) |
| `notebook_data/elmo_model/weights.hdf5` | ELMo weights (357 MB) |

*(Note: OracleNet model weights are also available on [Hugging Face](https://huggingface.co/VP1606/OracleNet))*

---
---

## 2. Category C: Deep Learning (Transformer)

# ModernBERT-Large NLI — COMP34812 NLU Coursework
**Group:** CW47
**Category:** C — Deep Learning (Transformer)
**Model:** Ensemble of two fine-tuned `answerdotai/ModernBERT-large` models (395M parameters each) with dynamic threshold optimisation

---

## Results

### Standalone Models (dev set)
| Model | F1 Score |
|---|---|
| ModernBERT-large (fine-tuned) | 92.83% |
| ModernBERT-large (tasksource fine-tuned) | 94.00% |

### Ensemble Sweep (dev set)
| orig weight | tasksource weight | Default F1 | Best F1 | Threshold |
|---|---|---|---|---|
| 0.2 | 0.8 | 94.10% | 94.20% | 0.24 |
| 0.3 | 0.7 | 94.13% | 94.18% | 0.56 |
| **0.4** | **0.6** | **94.20%** | **94.22%** | **0.43** |
| 0.5 | 0.5 | 94.17% | 94.17% | 0.50 |
| 0.6 | 0.4 | 93.01% | 93.66% | 0.40 |

**Best configuration:** orig=0.4 + tasksource=0.6 → **F1: 94.22%** @ threshold 0.43

**Models:** `final_model_versions/modernBerta` + `final_model_versions/taskSource`
**Ensemble weights:** 0.4 (modernBerta) + 0.6 (taskSource)
**Optimal threshold:** 0.43

---

## Code Structure

```
NLU-CW/
├── transformer_3_large.ipynb       # Training pipeline: data cleaning, fine-tuning, threshold tuning
├── inference_demo_transformer.ipynb # Inference notebook — run this to generate predictions
├── Group_CW47_NLI_C.csv            # Final test set predictions (ready for submission)
├── final_model_versions/
│   ├── modernBerta/                # Fine-tuned ModernBERT-large checkpoint (stored via Git LFS)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── taskSource/                 # Second fine-tuned ModernBERT-large checkpoint (stored via Git LFS)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── data/
    ├── train.csv                   # Training data
    ├── dev.csv                     # Development data (with labels)
    ├── test.csv                    # Test data (no labels)
    └── NLI_trial.csv               # Trial data (with labels)
```

**Code organisation:**
- **Training code:** `transformer_3_large.ipynb` — data cleaning, fine-tuning, threshold tuning
- **Inference code:** `inference_demo_transformer.ipynb` — ensemble inference on test set

---

## Running the Demo

### Requirements
- Python 3.9+
- `transformers`, `torch`, `scikit-learn`, `pandas`, `numpy`
- **Hardware:** GPU with at least 15GB VRAM recommended (e.g. NVIDIA T4, L4, V100). Can run on Apple Silicon (MPS) but will be slow.

### Setup
```bash
# Pull model weights via Git LFS
git lfs install
git lfs pull

# Install dependencies
pip install transformers torch scikit-learn pandas numpy
```

### Generate Predictions
Open and run `inference_demo_transformer.ipynb`. It will:
1. Load both fine-tuned ModernBERT-large models from `final_model_versions/`
2. Tokenize `data/test.csv`
3. Generate probability logits from each model
4. Combine via weighted ensemble (0.4 / 0.6)
5. Apply threshold 0.43 and save predictions to `Group_CW47_C.csv`

---

## Training

Training is executed by running `transformer_3_large.ipynb` in Google Colab (GPU required).

**Key optimisations:**
- Mixed precision (`fp16=True`) to reduce GPU memory usage
- Gradient accumulation (4 steps, effective batch size 16) to prevent OOM errors
- Strict NaN scrubbing on labels to prevent CUDA `device-side assert` crashes
- Dynamic threshold tuning (0.10–0.90) to maximise F1 independently of default 0.50

---

## Attribution

| Resource | Reference |
|---|---|
| ModernBERT architecture | Answer.AI / Hugging Face: `answerdotai/ModernBERT-large` |
| Transformers library | Wolf et al. (2020), "Transformers: State-of-the-Art Natural Language Processing" |
| Dataset | Coursework-provided NLI datasets (`train`, `dev`, `test`) |

---

## Use of Generative AI Tools

During the development of this project, Large Language Models (LLMs) were utilized as coding assistants. Specifically, AI tools were used to:
1. Debug PyTorch `CUDA: device-side assert` memory errors by suggesting robust data-scrubbing techniques for formatting labels as strict integers.
2. Recommend hardware-optimization strategies for Large models, such as implementing `fp16` mixed precision and gradient accumulation steps to prevent GPU memory overflow.
3. Assist in writing the Python iteration loops for the dynamic threshold optimization script.
4. Format the final output arrays into the strict pandas CSV structure required by the local scorer and submission guidelines.
5. Map our project details into the provided Jinja model card template format.

