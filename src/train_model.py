import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import json # For loading mapping files
import pandas as pd
import argparse # Added

from model import ArbitragePredictionModel

# --- Default Configuration (can be overridden by args or model_config.json) ---
DEFAULT_MODEL_SAVE_DIR = "models"
DEFAULT_BEST_MODEL_SAVE_PATH = os.path.join(DEFAULT_MODEL_SAVE_DIR, "best_arbitrage_model.pth")
DEFAULT_FINAL_MODEL_SAVE_PATH = os.path.join(DEFAULT_MODEL_SAVE_DIR, "final_arbitrage_model.pth")

# Hyperparameters (some might be moved to model_config or become args)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 # 0.0001
NUM_EPOCHS = 50 
VAL_SPLIT = 0.2
LAMBDA_MU = 1.0 # Weight for mu_loss in total_loss

# Embedding dimensions (will be loaded from model_config or set if not present)
# These are default if not in config, though model_config should ideally provide them
DEFAULT_ASSET_EMBED_DIM = 32
DEFAULT_TYPE_EMBED_DIM = 10 # Reduced from 16
DEFAULT_STATUS_EMBED_DIM = 8  # Reduced from 16
DEFAULT_ACTOR_TYPE_EMBED_DIM = 10 # For actor_type_id in sequence

# Transformer Model parameters (will be loaded from model_config or set if not present)
DEFAULT_D_MODEL = 128 # Smaller model for faster iteration
DEFAULT_NHEAD = 4
DEFAULT_NUM_ENCODER_LAYERS = 3
DEFAULT_DIM_FEEDFORWARD = 512
DEFAULT_DROPOUT = 0.1

# --- Dataset Class (Modified for separate cat and num features) ---
class TransactionSequenceDataset(Dataset):
    def __init__(self, X_cat_ids, X_num, y_p, y_mu):
        self.X_cat_ids = torch.tensor(X_cat_ids, dtype=torch.long) # Categorical IDs should be long
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y_p = torch.tensor(y_p, dtype=torch.float32) # One-hot, for CrossEntropy convert to indices later
        self.y_mu = torch.tensor(y_mu, dtype=torch.float32) # Float, potentially NaN

    def __len__(self):
        return len(self.X_cat_ids) # length based on first dimension of X_cat_ids

    def __getitem__(self, idx):
        return self.X_cat_ids[idx], self.X_num[idx], self.y_p[idx], self.y_mu[idx]

# --- Helper function to load vocab sizes ---
def load_vocab_size(mapping_file_path):
    try:
        with open(mapping_file_path, 'r') as f:
            mapping = json.load(f)
            return len(mapping)
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {mapping_file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {mapping_file_path}")
        raise

# --- Main Training Script ---
def main(args):
    os.makedirs(args.model_save_dir, exist_ok=True)

    # --- 1. Load Model Configuration ---
    print(f"Loading model configuration from {args.model_config_path}...")
    try:
        with open(args.model_config_path, 'r') as f_config:
            model_config = json.load(f_config)
    except FileNotFoundError:
        print(f"Error: Model configuration file not found at {args.model_config_path}.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.model_config_path}.")
        return

    # Extract parameters from model_config, using defaults if not present
    # Vocab sizes MUST be in model_config
    categorical_embedding_details = model_config.get('categorical_embedding_details')
    if not categorical_embedding_details:
        print("Error: 'categorical_embedding_details' not found in model_config.json")
        return
        
    # Example: preprocessor might save mapping file names as keys
    asset_vocab_size = categorical_embedding_details.get("asset_to_id.json") 
    type_vocab_size = categorical_embedding_details.get("type_to_id.json")
    status_vocab_size = categorical_embedding_details.get("status_to_id.json")
    actor_type_vocab_size = categorical_embedding_details.get("actor_type_to_id.json")

    if any(v is None for v in [asset_vocab_size, type_vocab_size, status_vocab_size, actor_type_vocab_size]):
        print(f"Error: One or more vocab sizes (asset, type, status, actor_type) not found in model_config.json under 'categorical_embedding_details'. Found: {categorical_embedding_details}")
        return
        
    num_numerical_features = model_config.get('num_numerical_features')
    if num_numerical_features is None:
        print("Error: 'num_numerical_features' not found in model_config.json")
        return
    
    sequence_length = model_config.get('sequence_length')
    if sequence_length is None:
        print("Error: 'sequence_length' not found in model_config.json")
        return

    p_target_classes = model_config.get('p_target_classes', actor_type_vocab_size) # Default to actor_type_vocab_size if not specified

    # Embedding dimensions and Transformer params (use defaults if not in config)
    asset_embed_dim = model_config.get('asset_embed_dim', DEFAULT_ASSET_EMBED_DIM)
    type_embed_dim = model_config.get('type_embed_dim', DEFAULT_TYPE_EMBED_DIM)
    status_embed_dim = model_config.get('status_embed_dim', DEFAULT_STATUS_EMBED_DIM)
    actor_type_embed_dim = model_config.get('actor_type_embed_dim', DEFAULT_ACTOR_TYPE_EMBED_DIM)
    d_model = model_config.get('d_model', DEFAULT_D_MODEL)
    nhead = model_config.get('nhead', DEFAULT_NHEAD)
    num_encoder_layers = model_config.get('num_encoder_layers', DEFAULT_NUM_ENCODER_LAYERS)
    dim_feedforward = model_config.get('dim_feedforward', DEFAULT_DIM_FEEDFORWARD)
    dropout_rate = model_config.get('dropout', DEFAULT_DROPOUT)
    
    print(f"Model config loaded. Asset Vocab: {asset_vocab_size}, Type Vocab: {type_vocab_size}, Status Vocab: {status_vocab_size}, ActorType Vocab: {actor_type_vocab_size}")
    print(f"Num Numerical Feats: {num_numerical_features}, Seq Len: {sequence_length}, P-Target Classes: {p_target_classes}")

    # --- 2. Load Training Data ---
    print(f"Loading training data from {args.input_npz}...")
    try:
        data = np.load(args.input_npz, allow_pickle=True)
        X_cat_ids_sequences = data['X_cat_ids_sequences']
        X_num_sequences = data['X_num_sequences']
        y_p_targets = data['y_p_targets']
        y_mu_targets = data['y_mu_targets']
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.input_npz}.")
        return
    except KeyError as e:
        print(f"Error: Missing expected key {e} in data file {args.input_npz}.")
        return

    print(f"Data loaded. Shapes: X_cat_ids={X_cat_ids_sequences.shape}, X_num={X_num_sequences.shape}, y_p={y_p_targets.shape}, y_mu={y_mu_targets.shape}")

    # --- BEGIN NAN/INF CHECK ---
    print("\nChecking for NaNs/Infs in loaded data...")
    if np.any(np.isnan(X_cat_ids_sequences)):
        print("WARNING: NaNs found in X_cat_ids_sequences!")
    if np.any(np.isinf(X_cat_ids_sequences)):
        print("WARNING: Infs found in X_cat_ids_sequences!")
    
    if np.any(np.isnan(X_num_sequences)):
        print("WARNING: NaNs found in X_num_sequences!")
        # Optional: print count or location
        print(f"  Total NaNs in X_num_sequences: {np.isnan(X_num_sequences).sum()}")
    if np.any(np.isinf(X_num_sequences)):
        print("WARNING: Infs found in X_num_sequences!")
        print(f"  Total Infs in X_num_sequences: {np.isinf(X_num_sequences).sum()}")
    
    if np.any(np.isnan(y_p_targets)):
        print("WARNING: NaNs found in y_p_targets! This should be one-hot encoded and not have NaNs.")
    # y_mu_targets can have NaNs, so we don't warn here for that.
    print("--- END NAN/INF CHECK ---\n")

    # Validate loaded data against model_config (sequence_length, num_features)
    if X_cat_ids_sequences.shape[1] != sequence_length or X_num_sequences.shape[1] != sequence_length:
        print(f"Error: Sequence length in data ({X_cat_ids_sequences.shape[1]}) does not match model_config ({sequence_length}).")
        return
    if X_num_sequences.shape[2] != num_numerical_features:
        print(f"Error: Number of numerical features in data ({X_num_sequences.shape[2]}) does not match model_config ({num_numerical_features}).")
        return
    # Add check for num_categorical_features if it's also in model_config and easily comparable

    # --- 3. Create Dataset and DataLoaders ---
    dataset = TransactionSequenceDataset(X_cat_ids_sequences, X_num_sequences, y_p_targets, y_mu_targets)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"Created DataLoaders: Train size={len(train_dataset)}, Val size={len(val_dataset)}")

    # --- 4. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 5. Instantiate Model ---
    model = ArbitragePredictionModel(
        asset_vocab_size=asset_vocab_size, 
        type_vocab_size=type_vocab_size, 
        status_vocab_size=status_vocab_size,
        actor_type_vocab_size=actor_type_vocab_size,
        asset_embed_dim=asset_embed_dim, 
        type_embed_dim=type_embed_dim, 
        status_embed_dim=status_embed_dim,
        actor_type_embed_dim=actor_type_embed_dim,
        num_numerical_features=num_numerical_features,
        d_model=d_model, 
        nhead=nhead, 
        num_encoder_layers=num_encoder_layers, 
        dim_feedforward=dim_feedforward,
        p_target_classes=p_target_classes,
        mu_target_dim=1,
        dropout=dropout_rate, 
        max_seq_len=sequence_length
    ).to(device)
    print("Model instantiated and moved to device.")

    # --- 6. Define Loss Functions and Optimizer ---
    p_loss_fn = nn.CrossEntropyLoss()
    mu_loss_fn = nn.MSELoss(reduction='mean') # Explicitly set reduction='mean'
    print(f"DEBUG_LOSS_FN: mu_loss_fn.reduction = {mu_loss_fn.reduction}")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ARB_SWAP ID for masking mu_loss (from model_config or a fixed known value)
    # Assuming actor_type_to_id.json defines ARB_SWAP as 0, USER_SWAP as 1 etc.
    # This should be robustly fetched if the mapping can change.
    # For now, let's assume model_config helps, or we have a standard.
    # If `actor_type_to_id.json` is standard, we could load it here, but `model_config` is preferred source for what model expects.
    # For this example, we rely on p_target_classes and assume ARB_SWAP is ID 0 if not explicitly stated otherwise in model_config.
    # A more robust way: model_config could store `{"ARB_SWAP": 0, ...}` directly or path to the mapping.
    # For now, we will assume ARB_SWAP is index 0 as per previous setup if actor_type_map is not in model_config.
    arb_swap_id = 0 # Default assumption: ARB_SWAP is class 0
    # Try to get it from config if available
    # arb_swap_id_from_config = model_config.get('actor_type_mappings', {}).get('ARB_SWAP')
    # if arb_swap_id_from_config is not None: arb_swap_id = arb_swap_id_from_config
    # else: print("Warning: ARB_SWAP ID not found in model_config, defaulting to 0.")
    # For now, stick to simple assumption. The preprocessing saves actor_type_to_id.json which has ARB_SWAP:0
    print(f"Using ARB_SWAP ID for mu_loss masking: {arb_swap_id} (Implicit: corresponds to first class if p_target_classes > 0)")
        
    print("Loss functions and optimizer defined.")

    # --- 7. Training Loop ---
    best_val_loss = float('inf')
    best_model_save_path = os.path.join(args.model_save_dir, "best_arbitrage_model.pth")
    final_model_save_path = os.path.join(args.model_save_dir, "final_arbitrage_model.pth")

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss, epoch_train_p_loss, epoch_train_mu_loss = 0.0, 0.0, 0.0
        
        for batch_idx, (X_cat_ids_batch, X_num_batch, y_p_batch, y_mu_batch) in enumerate(train_loader):
            X_cat_ids_batch = X_cat_ids_batch.to(device)
            X_num_batch = X_num_batch.to(device)
            y_p_batch = y_p_batch.to(device)
            y_mu_batch = y_mu_batch.to(device)

            optimizer.zero_grad()
            p_logits, mu_preds = model(X_cat_ids_batch, X_num_batch)
            y_p_indices = torch.argmax(y_p_batch, dim=1)
            loss_p = p_loss_fn(p_logits, y_p_indices)

            actual_next_actor_is_arb = (y_p_indices == arb_swap_id)
            y_mu_is_not_nan = ~torch.isnan(y_mu_batch.squeeze())
            mu_loss_calc_mask = actual_next_actor_is_arb & y_mu_is_not_nan
            active_mu_targets_for_loss_count = mu_loss_calc_mask.sum().item()

            if active_mu_targets_for_loss_count > 0:
                mu_preds_masked = mu_preds.squeeze()[mu_loss_calc_mask]
                y_mu_batch_masked = y_mu_batch.squeeze()[mu_loss_calc_mask]
                mu_actual_loss = mu_loss_fn(mu_preds_masked, y_mu_batch_masked)
            else:
                mu_actual_loss = torch.tensor(0.0, device=device)

            total_loss = loss_p + LAMBDA_MU * mu_actual_loss
            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_train_p_loss += loss_p.item()
            if active_mu_targets_for_loss_count > 0:
                 epoch_train_mu_loss += mu_actual_loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_p_loss = epoch_train_p_loss / len(train_loader)
        avg_train_mu_loss = epoch_train_mu_loss / len(train_loader) if len(train_loader) > 0 else 0 # Avoid division by zero with empty loader

        # Validation loop
        model.eval()
        epoch_val_loss, epoch_val_p_loss, epoch_val_mu_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for X_cat_ids_batch, X_num_batch, y_p_batch, y_mu_batch in val_loader:
                X_cat_ids_batch = X_cat_ids_batch.to(device)
                X_num_batch = X_num_batch.to(device)
                y_p_batch = y_p_batch.to(device)
                y_mu_batch = y_mu_batch.to(device)

                p_logits, mu_preds = model(X_cat_ids_batch, X_num_batch)
                y_p_indices = torch.argmax(y_p_batch, dim=1)
                loss_p = p_loss_fn(p_logits, y_p_indices)

                actual_next_actor_is_arb = (y_p_indices == arb_swap_id)
                y_mu_is_not_nan = ~torch.isnan(y_mu_batch.squeeze())
                mu_loss_calc_mask = actual_next_actor_is_arb & y_mu_is_not_nan
                active_mu_targets_for_loss_count = mu_loss_calc_mask.sum().item()

                if active_mu_targets_for_loss_count > 0:
                    mu_preds_masked = mu_preds.squeeze()[mu_loss_calc_mask]
                    y_mu_batch_masked = y_mu_batch.squeeze()[mu_loss_calc_mask]
                    mu_actual_loss = mu_loss_fn(mu_preds_masked, y_mu_batch_masked)
                else:
                    mu_actual_loss = torch.tensor(0.0, device=device)
                
                total_loss = loss_p + LAMBDA_MU * mu_actual_loss
                epoch_val_loss += total_loss.item()
                epoch_val_p_loss += loss_p.item()
                if active_mu_targets_for_loss_count > 0:
                    epoch_val_mu_loss += mu_actual_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_p_loss = epoch_val_p_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_mu_loss = epoch_val_mu_loss / len(val_loader) if len(val_loader) > 0 else 0

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
              f"Train Loss: {avg_train_loss:.4f} (P: {avg_train_p_loss:.4f}, Mu: {avg_train_mu_loss:.4f}), "
              f"Val Loss: {avg_val_loss:.4f} (P: {avg_val_p_loss:.4f}, Mu: {avg_val_mu_loss:.4f})")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_save_path)
            print(f"New best model saved to {best_model_save_path} (Val Loss: {best_val_loss:.4f})")

    torch.save(model.state_dict(), final_model_save_path)
    print(f"Final model saved to {final_model_save_path}")
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Arbitrage Prediction Model.")
    parser.add_argument("--input-npz", type=str, required=True,
                        help="Path to the input .npz file containing training sequences and targets.")
    parser.add_argument("--model-config-path", type=str, required=True,
                        help="Path to the model_config.json file.")
    parser.add_argument("--model-save-dir", type=str, default=DEFAULT_MODEL_SAVE_DIR,
                        help=f"Directory to save trained models. Default: {DEFAULT_MODEL_SAVE_DIR}")
    # Potentially add other hyperparameters as args later if needed (LR, Epochs, Batch Size etc.)
    
    cli_args = parser.parse_args()
    main(cli_args) 