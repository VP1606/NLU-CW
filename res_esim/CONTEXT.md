# Res-ESIM Implementation Context

## Project Overview
Implementation of the **Residual Connected Enhanced Sequential Inference Model** (res-ESIM) for Natural Language Inference (NLI), based on the paper by Li et al. (2019).

**Task**: Classify sentence pairs (premise, hypothesis) into three categories:
- Entailment
- Neutral
- Contradiction

---

## Architecture Overview (from Paper)

The res-ESIM model consists of 4 main components:

### 1. Embedding Layer
- **Current**: BERT-based contextual embeddings
- **Future**: Will be replaced with ELMo embeddings
- **Purpose**: Convert tokens to dense vector representations
- **Abstraction Point**: The model accepts pre-computed embeddings as input, making it agnostic to the embedding source

### 2. Residual Connected Block (N × ESIM-blocks)
Each ESIM block contains:
- **Encoding Layer** (BiLSTM): Encodes premise and hypothesis separately using a shared BiLSTM
- **Interaction Layer** (Soft Attention): Cross-sentence attention mechanism
  - Premise attends to hypothesis → h'_p
  - Hypothesis attends to premise → h'_h
- **Enhancement Layer**: Creates rich representations via concatenation
  - `m = [h, h_att, h - h_att, h * h_att]` (4 × hidden_dim)
- **FFN Layer**: Projects enhanced representation back to hidden_dim
- **Residual Connection + LayerNorm**: `output = LayerNorm(input + FFN(enhancement))`

**Key Innovation**: Stacking N blocks with residual connections prevents degradation in deep networks.

### 3. Aggregation Layer (BiLSTM)
- Separate BiLSTMs for premise and hypothesis
- Captures sequential dependencies after all ESIM blocks
- Outputs: `v_p` and `v_h`

### 4. Classification Layer
- **Max Pooling**: Extract most salient features (with masking for padding)
- **Concatenation**: `v = [v_p_max; v_h_max; v_p_max - v_h_max; v_p_max * v_h_max]`
- **FFN**: Two-layer network with ReLU activation
  - W4: (4 × hidden_dim) → hidden_dim
  - W5: hidden_dim → num_classes (3)
- **Output**: Class logits (softmax applied in loss function)

---

## Current Implementation Status

### File Structure
```
res_esim/
├── model_layers/
│   ├── esim_block.py          # Single ESIM block
│   ├── res_esim_block.py      # Stacked ESIM blocks with residual connections
│   ├── stock_classifier.py    # Aggregation + classification layers
│   └── oracle_net.py          # Unified model (encoder + classifier)
├── trainer/
│   ├── training.py            # Training loop with warmup/decay scheduler
│   ├── evaluation.py          # Evaluation metrics (loss, accuracy, F1)
│   └── inference.py           # Prediction interface
└── Residual_Connected_Enhanced_Sequential_Inference_Model_for_Natural_Language_Inference.pdf
```

### Implementation Details

#### 1. **ESIMBlock** (`esim_block.py`)
**Purpose**: Single ESIM block with attention, enhancement, and residual connection

**Current Issues**:
- ❌ Line 12: Defines `self.shared_bilstm` but calls `self.bilstm` (line 112)
- ❌ Line 33: Defines `_soft_dot_attention` but calls `_cross_attention` (line 116)
- ❌ Line 78: `_enhance` is @staticmethod but has `self` parameter (incorrect)
- ❌ Line 119: Calls `_enhance(enc_p, att_p)` but method signature expects `(self, h, h_att)`

**Expected Flow**:
```
Input: (h_p, h_h) → BiLSTM → Attention → Enhancement → FFN → Residual + LayerNorm → Output
```

#### 2. **ResESIM** (`res_esim_block.py`)
**Purpose**: Stack N ESIM blocks with residual connections

**Current Issues**:
- ❌ Line 4: Import path `from esim_block import ESIMBlock` should be `from .esim_block import ESIMBlock`

**Features**:
✅ Input projection layer (adapts input_dim to hidden_dim)
✅ Padding mask generation
✅ Sequential processing through N ESIM blocks
✅ Dropout for regularization

#### 3. **StockClassifier** (`stock_classifier.py`)
**Purpose**: Aggregation BiLSTM + max pooling + FFN classification

**Current Issues**:
- ❌ Line 44, 52: LSTM parameter should be `hidden_size` not `hidden_dim`

**Features**:
✅ Separate BiLSTMs for premise and hypothesis (as per paper)
✅ Masked max pooling (prevents padding tokens from affecting pooling)
✅ Enhancement concatenation `[v_p; v_h; v_p - v_h; v_p * v_h]`
✅ Two-layer FFN (W4 + ReLU + W5)
✅ Returns logits (not probabilities)

#### 4. **OracleNet** (`oracle_net.py`)
**Purpose**: Unified end-to-end model combining encoder and classifier

**Current Issues**:
- ❌ Line 11: Import typo `stock_classifer` should be `stock_classifier`
- ❌ Line 45: Typo `self.classifer` should be `self.classifier`

**Architecture**:
```python
Input (premise_emb, hyp_emb, lengths)
    ↓
ResESIM (encoder) → (h_p, h_h, mask_p, mask_h)
    ↓
StockClassifier → logits
```

#### 5. **Training Infrastructure** (`trainer/`)

**training.py**:
✅ Warmup + linear decay scheduler (as per paper)
✅ Gradient clipping (max_norm=1.0)
✅ One-epoch training function
✅ Metrics: loss, accuracy, macro-F1

**evaluation.py**:
✅ Evaluation on dev set
✅ Returns loss, accuracy, macro-F1

**inference.py**:
✅ Prediction function
✅ Label mapping (IDX2LABEL)
⚠️ Note: Label key in batch is inconsistent (`'label'` in training.py vs `'labels'` in evaluation.py)

---

## Key Design Decisions

### 1. **Embedding Abstraction**
The model accepts pre-computed embeddings as input rather than raw text:
- **Input**: `(batch, seq_len, input_dim)` embedding tensors
- **Benefit**: Decouples embedding method (BERT, ELMo, GloVe) from core architecture
- **Flexibility**: Can swap embedding sources without changing model code

### 2. **Padding Mask Handling**
- Masks are `True` for padding positions
- Applied as `-inf` before softmax (attention, max pooling)
- Prevents padding tokens from influencing attention weights or pooling results

### 3. **Residual Connections**
- Formula: `output = LayerNorm(input + transformation(input))`
- Enables gradient flow in deep networks
- Prevents degradation as network depth increases

### 4. **Shared vs. Independent BiLSTMs**
- **ESIM Blocks**: Shared BiLSTM for premise and hypothesis (parameter efficiency)
- **Aggregation Layer**: Separate BiLSTMs (allows specialization)

---

## Hyperparameters (from Paper)

- **hidden_dim**: 300 (for SNLI experiments)
- **num_blocks**: 1-6 (paper experiments with different depths)
- **dropout_rate**: 0.2 (SNLI), 0.1 (MultiNLI)
- **batch_size**: 64
- **learning_rate**: Warmup then linear decay to 0
- **gradient_clipping**: max_norm = 1.0
- **optimizer**: Adam (β1=0.9, β2=0.999, weight_decay=0.01)

---

## Data Format

Expected batch structure:
```python
{
    'premise_embedding': Tensor(batch, len_p, input_dim),
    'hyp_embedding': Tensor(batch, len_h, input_dim),
    'premise_length': Tensor(batch,),  # actual lengths (for masking)
    'hyp_length': Tensor(batch,),
    'label': Tensor(batch,)  # 0=entailment, 1=neutral, 2=contradiction
}
```

---

## Current Git Status

**Branch**: `esim/res-esim`
**Main Branch**: `main`

**Recent Commits**:
1. `1cf29c5` - Added unified model: OracleNet
2. `9f88156` - Added training loop for one step, treating classifier & encoder as different entities
3. `a5912e9` - Added Stock Classifier from ESIM Paper
4. `bd8c330` - Added evaluation and inference logic
5. `048215a` - Added RESESIM

---

## Next Steps

### Immediate Fixes Required:
1. **Fix ESIMBlock bugs** (attribute name mismatches, method signatures)
2. **Fix import paths** (relative imports in res_esim_block.py)
3. **Fix typos** (stock_classifier import, LSTM parameter names)
4. **Standardize batch keys** (`'label'` vs `'labels'`)

### Future Work:
1. **Replace BERT with ELMo** embeddings
2. **Add data loading pipeline** for SNLI/MultiNLI
3. **Implement full training script** with checkpointing
4. **Hyperparameter tuning** (number of blocks, hidden dimension)
5. **Add model evaluation** on test sets

---

## Paper Reference

Li, Y., Wang, J., Lin, H., Zhang, S., & Yang, Z. (2019). *Residual Connected Enhanced Sequential Inference Model for Natural Language Inference*. IEEE.

**Key Contributions**:
- Residual connections prevent degradation in stacked ESIM blocks
- Enhanced contextual embeddings (BERT) improve performance
- Achieves competitive results on SNLI and MultiNLI datasets
