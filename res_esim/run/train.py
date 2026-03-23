import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import res_esim.trainer.training as res_esim_trainer
from res_esim.loader.res_esim_dataset import ResESIM_Dataset
from res_esim.model_layers.oracle_net import OracleNet


class HyperParameters:
    def __init__(self):
        self.INPUT_DIM = 1024 + 1  # +1 for negation flags.
        self.HIDDEN_DIM = 1024  # As per paper
        self.NUM_BLOCKS = 5  # Number of stacked ESIM blocks
        self.NUM_CLASSES = 3  # entailment, neutral, contradiction
        self.DROPOUT_RATE = 0.2  # As per paper (SNLI)

        # Number of attention heads in multi-head attention (ESIM)
        self.NUM_ATTN_HEADS = 8

        self.NUM_EPOCHS = 10

        self.BATCH_SIZE = 16  # Small batch for testing
        self.NUM_SAMPLES = 32  # Small dataset for testing

        self.LEARNING_RATE = 1e-4
        self.WARMUP_STEPS = 5
        self.TOTAL_STEPS = self.NUM_SAMPLES // self.BATCH_SIZE  # Steps per epoch


def train(
    model: OracleNet,
    device,
    hyperparameters: HyperParameters,
):

    # --- Setup Data Loaders ------------------------------
    # Paths
    PREM_NPY = Path(
        "/Users/vpremakantha/Documents/UOM/Y3/NLU/NLU-CW/output/elmo_train_prem.npy"
    )
    HYP_NPY = Path(
        "/Users/vpremakantha/Documents/UOM/Y3/NLU/NLU-CW/output/elmo_train_hyp.npy"
    )
    CSV_PATH = Path("/Users/vpremakantha/Documents/UOM/Y3/NLU/NLU-CW/data/train.csv")

    NEGATION_PATH = Path(
        "/Users/vpremakantha/Documents/UOM/Y3/NLU/NLU-CW/output/train_negation.pt"
    )

    # Dataset
    dataset = ResESIM_Dataset(PREM_NPY, HYP_NPY, CSV_PATH, negation_path=NEGATION_PATH)
    hyperparameters.NUM_SAMPLES = len(dataset)
    hyperparameters.TOTAL_STEPS = (
        hyperparameters.NUM_SAMPLES // hyperparameters.BATCH_SIZE
    )

    # Loader
    loader = DataLoader(
        dataset, batch_size=hyperparameters.BATCH_SIZE, shuffle=True, num_workers=0
    )

    # --- Setup Training Components -----------------------
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameters.LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Scheduler
    scheduler = res_esim_trainer.get_warmup_decay_scheduler(
        optimizer=optimizer,
        warmup_steps=hyperparameters.WARMUP_STEPS,
        total_steps=hyperparameters.TOTAL_STEPS,
    )

    # Loss
    criterion = nn.CrossEntropyLoss()

    # --- Output Directory ---------------------------------
    run_hash = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    out_dir = Path("output/training_runs") / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Training Loop ------------------------------------
    best_loss = float("inf")
    best_f1 = float("-inf")

    epoch_bar = tqdm(range(hyperparameters.NUM_EPOCHS), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        loss, accuracy, macro_f1 = res_esim_trainer.train_epoch(
            oracle=model,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        epoch_bar.set_postfix(
            loss=f"{loss:.4f}", acc=f"{accuracy:.4f}", f1=f"{macro_f1:.4f}"
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_loss = loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            with open(out_dir / "meta.json", "w") as f:
                json.dump(
                    {
                        "prem_npy": str(PREM_NPY),
                        "hyp_npy": str(HYP_NPY),
                        "trained_at": datetime.now().isoformat(),
                        "best_epoch": epoch,
                        "best_loss": loss,
                        "best_acc": accuracy,
                        "best_f1": macro_f1,
                        "hyperparameters": {
                            k: v for k, v in vars(hyperparameters).items()
                        },
                    },
                    f,
                    indent=2,
                )

    print(f"Best model saved to: {out_dir}")
    return model, best_loss, best_f1


def initialize_and_train():
    hyper_parameters = HyperParameters()
    model = OracleNet(
        input_dim=hyper_parameters.INPUT_DIM,
        hidden_dim=hyper_parameters.HIDDEN_DIM,
        num_blocks=hyper_parameters.NUM_BLOCKS,
        num_classes=hyper_parameters.NUM_CLASSES,
        dropout_rate=hyper_parameters.DROPOUT_RATE,
        num_heads=hyper_parameters.NUM_ATTN_HEADS,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")

    model.to(device)

    print("starting training....")
    model, best_loss, best_f1 = train(
        model=model, device=device, hyperparameters=hyper_parameters
    )

    print(f"Training Complete, Best Loss: {best_loss}, Best F1: {best_f1}")
    return


if __name__ == "__main__":
    initialize_and_train()
