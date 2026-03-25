import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import res_esim.trainer.evaluation as res_esim_eval
import res_esim.trainer.training as res_esim_trainer
from res_esim.loader.res_esim_dataset import ResESIM_Dataset
from res_esim.model_layers.oracle_net import OracleNet


class HyperParameters:
    def __init__(self):
        self.INPUT_DIM = 1477          # was 1024 + 1
        self.HIDDEN_DIM = 512
        self.NUM_BLOCKS = 3
        self.NUM_CLASSES = 2
        self.DROPOUT_RATE = 0.2
        self.NUM_ATTN_HEADS = 8
        self.NUM_EPOCHS = 15
        self.BATCH_SIZE = 32
        self.NUM_SAMPLES: int = 24432  # full train set
        self.LEARNING_RATE = 1e-4
        self.WARMUP_STEPS = 200        # was 5 — proper warmup
        self.TOTAL_STEPS: int = (self.NUM_SAMPLES // self.BATCH_SIZE) * self.NUM_EPOCHS
        


def train(
    model: OracleNet,
    device,
    hyperparameters: HyperParameters,
):

    # --- Setup Data Loaders ------------------------------
    TRAIN_PT = Path("output/train_embeddings.npz")
    DEV_PT   = Path("output/dev_embeddings.npz")     

    train_dataset = ResESIM_Dataset(TRAIN_PT)        
    dev_dataset   = ResESIM_Dataset(DEV_PT)

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparameters.BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=hyperparameters.BATCH_SIZE,
        shuffle=False, num_workers=0
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
        train_loss, train_acc, train_f1 = res_esim_trainer.train_epoch(
            oracle=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        dev_loss, dev_acc, dev_f1 = res_esim_eval.evaluate(
            oracle=model, loader=dev_loader, criterion=criterion, device=device
        )

        epoch_bar.set_postfix(t_f1=f"{train_f1:.4f}", d_f1=f"{dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_loss = dev_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            with open(out_dir / "meta.json", "w") as f:
                json.dump(
                    {
                        "train_pt": str(TRAIN_PT),
                        "dev_pt":   str(DEV_PT),
                        "trained_at": datetime.now().isoformat(),
                        "best_epoch": epoch,
                        "best_train_loss": train_loss,
                        "best_train_acc": train_acc,
                        "best_train_f1": train_f1,
                        "best_dev_loss": dev_loss,
                        "best_dev_acc": dev_acc,
                        "best_dev_f1": dev_f1,
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
