import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

import res_esim.trainer.training as res_esim_trainer
from res_esim.loader.res_esim_dataset import ResESIM_Dataset, collate_fn
from res_esim.model_layers.oracle_net import OracleNet
from embeddings import (
    Vocabulary,
    load_glove,
    build_char_vocab,
    build_pos_vocab,
    CharCNN,
    POSEmbedding,
    build_glove_layer,
    InputEmbeddingModule,
)
from util.tokenization import tokenise

# ELMo paths (download via downloader.py)
ELMO_OPTIONS = Path("bin/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json")
ELMO_WEIGHTS = Path("bin/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
GLOVE_PATH = Path("bin/glove/glove.6B.300d.txt")

# Data paths
TRAIN_CSV = Path("data/train.csv")
DEV_CSV = Path("data/dev.csv")


class HyperParameters:
    INPUT_DIM = 1477  # 300 (GloVe) + 1024 (ELMo) + 100 (CharCNN) + 50 (POS) + 3 (negation)
    HIDDEN_DIM = 512
    NUM_BLOCKS = 3
    NUM_CLASSES = 2
    DROPOUT_RATE = 0.2
    NUM_ATTN_HEADS = 8
    NUM_EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 200
    TOTAL_STEPS = (24432 // 32) * 15  # All epochs


def train(model, input_module, device, vocab, char2idx, pos2idx):
    """Train with real-time ELMo inference."""
    hyperparameters = HyperParameters()

    # Datasets (no pre-computed .npy files)
    train_dataset = ResESIM_Dataset(TRAIN_CSV, vocab, char2idx, pos2idx)
    dev_dataset = ResESIM_Dataset(DEV_CSV, vocab, char2idx, pos2idx)

    hyperparameters.TOTAL_STEPS = hyperparameters.NUM_EPOCHS * (
        len(train_dataset) // hyperparameters.BATCH_SIZE
    )

    # Loaders with custom collate_fn for raw tokens
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=hyperparameters.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Optimizer covers both input_module and oracle
    optimizer = torch.optim.Adam(
        list(input_module.parameters()) + list(model.parameters()),
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

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Output directory
    run_hash = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    out_dir = Path("output/training_runs") / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_f1 = float("-inf")

    epoch_bar = tqdm(range(hyperparameters.NUM_EPOCHS), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        # Train epoch
        train_loss, train_acc, train_f1 = _train_epoch(
            input_module=input_module,
            oracle=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        # Evaluate
        dev_loss, dev_acc, dev_f1 = _evaluate(
            input_module=input_module,
            oracle=model,
            loader=dev_loader,
            criterion=criterion,
            device=device,
        )

        epoch_bar.set_postfix(t_f1=f"{train_f1:.4f}", d_f1=f"{dev_f1:.4f}")

        # Save best model by dev F1
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(
                {
                    "input_module": input_module.state_dict(),
                    "oracle_net": model.state_dict(),
                    "vocab": vocab,
                    "char2idx": char2idx,
                    "pos2idx": pos2idx,
                },
                out_dir / "best_model.pt",
            )
            with open(out_dir / "meta.json", "w") as f:
                json.dump(
                    {
                        "trained_at": datetime.now().isoformat(),
                        "best_epoch": epoch,
                        "best_train_f1": train_f1,
                        "best_dev_loss": dev_loss,
                        "best_dev_acc": dev_acc,
                        "best_dev_f1": dev_f1,
                        "hyperparameters": {
                            "INPUT_DIM": hyperparameters.INPUT_DIM,
                            "HIDDEN_DIM": hyperparameters.HIDDEN_DIM,
                            "NUM_BLOCKS": hyperparameters.NUM_BLOCKS,
                            "NUM_CLASSES": hyperparameters.NUM_CLASSES,
                            "BATCH_SIZE": hyperparameters.BATCH_SIZE,
                            "LEARNING_RATE": hyperparameters.LEARNING_RATE,
                            "NUM_EPOCHS": hyperparameters.NUM_EPOCHS,
                        },
                    },
                    f,
                    indent=2,
                )

    print(f"Best model saved to: {out_dir}")
    return model, input_module, best_f1


def _train_epoch(
    input_module, oracle, loader, optimizer, scheduler, criterion, device, epoch=None
):
    """Train one epoch with real-time ELMo inference."""
    input_module.train()
    oracle.train()

    total_loss = 0.0
    all_preds, all_labels = [], []

    desc = f"Epoch {epoch}" if epoch is not None else "Batch"
    batch_bar = tqdm(loader, desc=desc, unit="batch", leave=False)
    for batch in batch_bar:
        # Move tensors to device
        prem_ids = batch["premise_ids"].to(device)
        hyp_ids = batch["hypothesis_ids"].to(device)
        prem_char = batch["premise_char"].to(device)
        hyp_char = batch["hypothesis_char"].to(device)
        prem_pos = batch["premise_pos"].to(device)
        hyp_pos = batch["hypothesis_pos"].to(device)
        prem_neg = batch["premise_neg"].to(device)
        hyp_neg = batch["hypothesis_neg"].to(device)
        prem_mask = batch["premise_mask"].to(device)
        hyp_mask = batch["hypothesis_mask"].to(device)
        labels = batch["label"].to(device)

        # Raw token lists (not tensors) for ELMo
        prem_raw = batch["premise_raw"]
        hyp_raw = batch["hypothesis_raw"]

        optimizer.zero_grad()

        # Forward pass through input embedding module with real-time ELMo
        prem_out = input_module(prem_ids, prem_char, prem_pos, prem_neg, prem_raw)
        hyp_out = input_module(hyp_ids, hyp_char, hyp_pos, hyp_neg, hyp_raw)

        prem_lens = prem_mask.sum(dim=1)
        hyp_lens = hyp_mask.sum(dim=1)

        # Forward pass through oracle
        logits = oracle(prem_out, hyp_out, prem_lens, hyp_lens)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(input_module.parameters()) + list(oracle.parameters()), max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, macro_f1


def _evaluate(input_module, oracle, loader, criterion, device):
    """Evaluate with real-time ELMo inference."""
    input_module.eval()
    oracle.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            # Move tensors to device
            prem_ids = batch["premise_ids"].to(device)
            hyp_ids = batch["hypothesis_ids"].to(device)
            prem_char = batch["premise_char"].to(device)
            hyp_char = batch["hypothesis_char"].to(device)
            prem_pos = batch["premise_pos"].to(device)
            hyp_pos = batch["hypothesis_pos"].to(device)
            prem_neg = batch["premise_neg"].to(device)
            hyp_neg = batch["hypothesis_neg"].to(device)
            prem_mask = batch["premise_mask"].to(device)
            hyp_mask = batch["hypothesis_mask"].to(device)
            labels = batch["label"].to(device)

            # Raw token lists for ELMo
            prem_raw = batch["premise_raw"]
            hyp_raw = batch["hypothesis_raw"]

            # Forward pass through input embedding module with real-time ELMo
            prem_out = input_module(prem_ids, prem_char, prem_pos, prem_neg, prem_raw)
            hyp_out = input_module(hyp_ids, hyp_char, hyp_pos, hyp_neg, hyp_raw)

            prem_lens = prem_mask.sum(dim=1)
            hyp_lens = hyp_mask.sum(dim=1)

            # Forward pass through oracle
            logits = oracle(prem_out, hyp_out, prem_lens, hyp_lens)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, macro_f1


def initialize_and_train():
    """Initialize and train with real-time ELMo inference."""
    print("=" * 80)
    print("Real-time ELMo Inference for ResESIM")
    print("=" * 80)

    # 1. Build vocabularies from training data only
    print("\n1. Building vocabularies...")
    df = pd.read_csv(TRAIN_CSV)

    prem_tokens = [tokenise(s) for s in df["premise"]]
    hyp_tokens = [tokenise(s) for s in df["hypothesis"]]
    all_train_tokens = prem_tokens + hyp_tokens

    vocab = Vocabulary(min_freq=2)
    vocab.build(all_train_tokens)
    print(f"   Vocabulary size: {len(vocab)}")

    char2idx = build_char_vocab(all_train_tokens)
    print(f"   Character vocabulary size: {len(char2idx)}")

    pos2idx = build_pos_vocab(all_train_tokens)
    print(f"   POS vocabulary size: {len(pos2idx)}")

    # 2. Build embedding components
    print("\n2. Building embedding layers...")
    glove_matrix = load_glove(GLOVE_PATH, vocab, embed_dim=300)
    print(f"   GloVe matrix shape: {glove_matrix.shape}")

    glove_layer = build_glove_layer(glove_matrix, freeze=True)
    char_cnn = CharCNN(char_vocab_size=len(char2idx), char_embed_dim=30)
    pos_embedding = POSEmbedding(pos_vocab_size=len(pos2idx), pos_embed_dim=50)

    # 3. InputEmbeddingModule with real-time ELMo
    print("\n3. Initializing InputEmbeddingModule with real-time ELMo...")
    input_module = InputEmbeddingModule(
        elmo_options=str(ELMO_OPTIONS),
        elmo_weights=str(ELMO_WEIGHTS),
        glove_layer=glove_layer,
        char_cnn=char_cnn,
        pos_embedding=pos_embedding,
        dropout_rate=0.5,
    )
    print(f"   InputEmbeddingModule output dim: {input_module.output_dim}")

    # 4. OracleNet
    print("\n4. Initializing OracleNet...")
    model = OracleNet(
        input_dim=HyperParameters.INPUT_DIM,
        hidden_dim=HyperParameters.HIDDEN_DIM,
        num_blocks=HyperParameters.NUM_BLOCKS,
        num_classes=HyperParameters.NUM_CLASSES,
        dropout_rate=HyperParameters.DROPOUT_RATE,
        num_heads=HyperParameters.NUM_ATTN_HEADS,
    )

    # 5. Device (MPS → CUDA → CPU)
    print("\n5. Setting up device...")
    if torch.backends.mps.is_available():
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"   Training on: {device}")

    model.to(device)
    input_module.to(device)

    # 6. Train
    print("\n6. Starting training...")
    print("=" * 80)
    model, input_module, best_f1 = train(
        model=model,
        input_module=input_module,
        device=device,
        vocab=vocab,
        char2idx=char2idx,
        pos2idx=pos2idx,
    )

    print("=" * 80)
    print(f"Training Complete. Best F1: {best_f1:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    initialize_and_train()
