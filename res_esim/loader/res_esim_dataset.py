from pathlib import Path
import torch
from torch.utils.data import Dataset

class ResESIM_Dataset(Dataset):
    """
    Loads pre-computed full embedding tensors for OracleNet.
    
    Each .pt file contains embeddings pre-computed from:
      GloVe [300d] + ELMo [1024d] + CharCNN [100d] + POS [50d] + Negation [3d] = 1477d
    
    Args:
        pt_path : path to .pt file saved by NLU_Embeddings_v2.ipynb
                  (train_embeddings.pt or dev_embeddings.pt)
    """
    def __init__(self, pt_path: Path):
        data = torch.load(pt_path, map_location='cpu')
        self.prem_emb      = data['premise_emb']       # [N, 64, 1477]
        self.hyp_emb       = data['hypothesis_emb']    # [N, 64, 1477]
        self.prem_lengths  = data['premise_mask'].sum(dim=-1)    # [N]
        self.hyp_lengths   = data['hypothesis_mask'].sum(dim=-1) # [N]
        self.labels        = data['labels']             # [N]
        print(f'Loaded {pt_path}')
        print(f'  samples={len(self.labels)}, dim={self.prem_emb.shape[-1]}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'premise_embedding': self.prem_emb[idx],
            'hyp_embedding':     self.hyp_emb[idx],
            'premise_length':    self.prem_lengths[idx],
            'hyp_length':        self.hyp_lengths[idx],
            'label':             self.labels[idx],
        }