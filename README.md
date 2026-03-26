# OracleNet
Category B.

## Environment Setup

(Note: ELMo requires Python3.9 - use a different environment seperate to OracleNet for embedding computation)

Create a Python environment, and install the required packages:
```bash
pip install -r require310.txt
```

## Running OracleNet

From the root directory (inside the Oracle environment), run:
```bash
python -m res_esim.run.train
```

## Current Best Scores

Dev Set (Macro F1): 70.47%
Training Set (Macro F1): 74.7%

Model Name: **ab4fb87e**
