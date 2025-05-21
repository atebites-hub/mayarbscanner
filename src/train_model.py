import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import json # For loading mapping files
import pandas as pd
import argparse # Added

from model import GenerativeTransactionModel

# --- Default Configuration (GENERATIVE MODEL - can be overridden by args) ---
DEFAULT_MODEL_SAVE_DIR = "models"
DEFAULT_GENERATIVE_MODEL_CONFIG_FILENAME = "model_config_generative_thorchain.json" # From preprocess
DEFAULT_INPUT_NPZ_GENERATIVE = "sequences_and_targets_generative_thorchain.npz" # From preprocess

DEFAULT_BEST_MODEL_GENERATIVE_SAVE_PATH = os.path.join(DEFAULT_MODEL_SAVE_DIR, "best_generative_model_thorchain.pth")
DEFAULT_FINAL_MODEL_GENERATIVE_SAVE_PATH = os.path.join(DEFAULT_MODEL_SAVE_DIR, "final_generative_model_thorchain.pth")

# Hyperparameters (some might be moved to model_config or become args)
BATCH_SIZE = 16 # Adjusted, generative model might be larger
LEARNING_RATE = 5e-5 # Adjusted
NUM_EPOCHS = 50 
VAL_SPLIT = 0.15 # Adjusted

# Embedding dimensions - these will be passed to the model
# The model itself will use its model_config to know which features get which embeddings
DEFAULT_EMBEDDING_DIM_CONFIG = {
    "default_cat_embed_dim": 32,  # Default for general categoricals not specified below
    "address_hash_embed_dim": 64, # For hashed addresses
    "asset_embed_dim": 48,        # For asset IDs (e.g., pool1_asset_id, in_coin1_asset_id)
    "action_type_embed_dim": 16,  # For action_type_id
    "action_status_embed_dim": 16 # For action_status_id
    # Add other specific feature_key prefixes if they need unique embedding dims
}

# Transformer Model parameters (can also be part of a more detailed model architecture config if needed)
DEFAULT_D_MODEL = 256
DEFAULT_NHEAD = 8
DEFAULT_NUM_ENCODER_LAYERS = 6
DEFAULT_DIM_FEEDFORWARD = 1024 # Typically 2x to 4x d_model
DEFAULT_DROPOUT = 0.1

# --- Dataset Class (for Generative Model) ---
class GenerativeTransactionDataset(Dataset):
    def __init__(self, X_sequences, Y_targets):
        # X_sequences: (num_samples, seq_len, num_features_total)
        # Y_targets:   (num_samples, num_features_total)
        self.X_sequences = torch.tensor(X_sequences, dtype=torch.float32) # Model will handle casting to long for relevant parts
        self.Y_targets = torch.tensor(Y_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X_sequences)

    def __getitem__(self, idx):
        return self.X_sequences[idx], self.Y_targets[idx]

# --- Composite Loss Function for Generative Model ---
def calculate_composite_loss(predictions, targets, feature_output_info_list, device):
    """
    Calculates a composite loss for the generative model.
    - CrossEntropyLoss for ID-mapped and Hashed categorical features.
    - BCEWithLogitsLoss for binary flag features.
    - MSELoss for scaled numerical features.

    Args:
        predictions (torch.Tensor): Model output (batch_size, total_output_dimension from model).
        targets (torch.Tensor): Ground truth labels (batch_size, num_input_features_ordered).
        feature_output_info_list (list): List of dictionaries from model.feature_output_info,
                                         detailing each feature's output slice and type.
        device (torch.device): Current device.

    Returns:
        torch.Tensor: The total combined loss.
        dict: Dictionary containing individual loss components (e.g., cat_loss, num_loss, flag_loss).
    """
    total_loss = torch.tensor(0.0, device=device)
    loss_components = {
        'categorical_loss': torch.tensor(0.0, device=device),
        'numerical_loss': torch.tensor(0.0, device=device),
        'flag_loss': torch.tensor(0.0, device=device)
    }
    num_cat_feats, num_num_feats, num_flag_feats = 0, 0, 0

    ce_loss_fn = nn.CrossEntropyLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    for feat_info in feature_output_info_list:
        pred_slice = predictions[:, feat_info['output_start_idx']:feat_info['output_end_idx']]
        # Targets are indexed by their original input order
        target_slice = targets[:, feat_info['original_input_index']]
        feat_type = feat_info['type']
        loss = torch.tensor(0.0, device=device)

        if feat_type == 'id_map' or feat_type == 'hash_cat':
            # pred_slice is (batch_size, vocab_size_for_this_feature)
            # target_slice is (batch_size,) and needs to be long type for CE
            loss = ce_loss_fn(pred_slice, target_slice.long())
            loss_components['categorical_loss'] += loss.item() # Store item for aggregation
            num_cat_feats += 1
        
        elif feat_type == 'binary_flag':
            # pred_slice is (batch_size, 1) for binary logits
            # target_slice is (batch_size,)
            loss = bce_loss_fn(pred_slice.squeeze(-1), target_slice.float())
            loss_components['flag_loss'] += loss.item()
            num_flag_feats += 1
        
        elif feat_type == 'numerical_scaled':
            # pred_slice is (batch_size, 1) for numerical predictions
            # target_slice is (batch_size,)
            loss = mse_loss_fn(pred_slice.squeeze(-1), target_slice.float())
            loss_components['numerical_loss'] += loss.item()
            num_num_feats += 1
        
        total_loss += loss # Accumulate actual loss tensor for backward pass
    
    # Average the component losses for logging if features of that type existed
    # Note: loss_components store sum of .item(), so divide by count for average
    if num_cat_feats > 0: loss_components['categorical_loss'] /= num_cat_feats
    if num_num_feats > 0: loss_components['numerical_loss'] /= num_num_feats
    if num_flag_feats > 0: loss_components['flag_loss'] /= num_flag_feats

    # Optionally average total_loss if desired, or sum is fine for backward.
    # If averaging total loss, divide by the total number of features considered in the loss:
    num_total_loss_terms = num_cat_feats + num_num_feats + num_flag_feats
    if num_total_loss_terms > 0:
        total_loss /= num_total_loss_terms
    
    return total_loss, loss_components

# --- Main Training Script ---
def main(args):
    os.makedirs(args.model_save_dir, exist_ok=True)

    # --- 1. Load Model Configuration from Preprocessing ---
    # The model_config_path will point to e.g. data/processed_ai_data_generative_test/thorchain_artifacts_v1/model_config_generative_thorchain.json
    print(f"Loading generative model configuration from {args.generative_model_config_path}...")
    with open(args.generative_model_config_path, 'r') as f_config:
        model_config_from_preprocessor = json.load(f_config)

    # --- Load all_id_mappings from its separate file ---
    all_mappings_filename = "all_id_mappings_generative_mayachain_s25.json" # Must match what preprocess saved
    artifacts_dir = os.path.dirname(args.generative_model_config_path) # Infer artifacts_dir
    all_mappings_path = os.path.join(artifacts_dir, all_mappings_filename)
    all_id_mappings_loaded = {}
    try:
        with open(all_mappings_path, 'r') as f_map_load:
            all_id_mappings_loaded = json.load(f_map_load)
        print(f"Successfully loaded all_id_mappings from {all_mappings_path}")
    except Exception as e_map_load:
        print(f"ERROR loading all_id_mappings from {all_mappings_path}: {e_map_load}. This may cause issues.")
    # --- End separate load ---

    # --- 2. Load Training Data (Generative Format) ---
    # The input_npz_generative path will point to e.g. data/processed_ai_data_generative_test/sequences_and_targets_generative_thorchain.npz
    print(f"Loading generative training data from {args.input_npz_generative}...")
    try:
        data = np.load(args.input_npz_generative, allow_pickle=True)
        X_sequences = data['X_sequences']
        Y_targets = data['Y_targets']
        # feature_columns_ordered_from_data = data['feature_columns_ordered'] # For validation against model_config
    except FileNotFoundError:
        print(f"Error: Generative data file not found at {args.input_npz_generative}.")
        return
    except KeyError as e:
        print(f"Error: Missing expected key {e} in generative data file {args.input_npz_generative}.")
        return

    print(f"Generative data loaded. Shapes: X_sequences={X_sequences.shape}, Y_targets={Y_targets.shape}")

    # --- Data Validation (Optional but good practice) ---
    if X_sequences.shape[2] != model_config_from_preprocessor['num_features_total']:
        print(f"CRITICAL ERROR: num_features_total in data ({X_sequences.shape[2]}) does not match model_config ({model_config_from_preprocessor['num_features_total']}).")
        return
    if X_sequences.shape[0] != Y_targets.shape[0]:
        print(f"CRITICAL ERROR: Number of samples in X_sequences ({X_sequences.shape[0]}) does not match Y_targets ({Y_targets.shape[0]}).")
        return
    if Y_targets.shape[1] != model_config_from_preprocessor['num_features_total']:
        print(f"CRITICAL ERROR: num_features_total in Y_targets ({Y_targets.shape[1]}) does not match model_config ({model_config_from_preprocessor['num_features_total']}).")
        return

    # --- 3. Create Dataset and DataLoaders ---
    dataset = GenerativeTransactionDataset(X_sequences, Y_targets)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"Created Generative DataLoaders: Train size={len(train_dataset)}, Val size={len(val_dataset)}")

    # --- 4. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 5. Instantiate Generative Model ---
    # Embedding dim config can be an arg or loaded from a file if complex
    embedding_config_for_model = json.loads(args.embedding_dim_config_json) if args.embedding_dim_config_json else DEFAULT_EMBEDDING_DIM_CONFIG
    print(f"Using embedding_dim_config: {embedding_config_for_model}")

    model = GenerativeTransactionModel(
        model_config=model_config_from_preprocessor,
        embedding_dim_config=embedding_config_for_model,
        d_model=args.d_model, 
        nhead=args.nhead, 
        num_encoder_layers=args.num_encoder_layers, 
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, 
        max_seq_len=model_config_from_preprocessor.get('sequence_length', 10) # Get from config
    ).to(device)
    print("GenerativeTransactionModel instantiated and moved to device.")

    # --- Calculate and Print Model Parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable model parameters: {trainable_params:,}")

    # --- Add model.feature_output_info to the config and re-save it ---
    # This makes the model's output structure available for inference/decoding scripts.
    if hasattr(model, 'feature_output_info') and model.feature_output_info:
        model_config_from_preprocessor['feature_output_info_model'] = model.feature_output_info
        # Also save other model-specific parameters that were used for instantiation
        model_config_from_preprocessor['d_model'] = args.d_model
        model_config_from_preprocessor['nhead'] = args.nhead
        model_config_from_preprocessor['num_encoder_layers'] = args.num_encoder_layers
        model_config_from_preprocessor['dim_feedforward'] = args.dim_feedforward
        model_config_from_preprocessor['dropout'] = args.dropout
        model_config_from_preprocessor['embedding_dim_config_used'] = embedding_config_for_model
        # sequence_length is already in the config from preprocessing

        # --- Define a new path for the augmented config ---
        base_dir = os.path.dirname(args.generative_model_config_path)
        original_filename = os.path.basename(args.generative_model_config_path)
        name, ext = os.path.splitext(original_filename)
        augmented_config_filename = f"{name}_AUGMENTED{ext}"
        augmented_config_path = os.path.join(base_dir, augmented_config_filename)

        try:
            print("DEBUG: Keys in model_config_from_preprocessor BEFORE saving:", list(model_config_from_preprocessor.keys()))
            # print("DEBUG: Content of 'all_id_mappings' BEFORE saving (first 500 chars):", str(model_config_from_preprocessor.get('all_id_mappings'))[:500]) # This key is no longer present
            print("DEBUG: Content of 'embedding_dim_config_used' BEFORE saving:", model_config_from_preprocessor.get('embedding_dim_config_used'))
            
            with open(augmented_config_path, 'w') as f_config_out: # Save to new path
                json.dump(model_config_from_preprocessor, f_config_out, indent=4)
            print(f"Updated and re-saved model configuration to {augmented_config_path} with model's feature_output_info.") # Log new path
        except Exception as e:
            print(f"Error re-saving model configuration: {e}. Inference scripts might have issues.")
    else:
        print("Warning: model.feature_output_info not found or empty. Model config not updated with it.")

    # --- 6. Define Loss Functions and Optimizer (Composite Loss will be a separate function) ---
    # The composite loss will be called inside the training loop.
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) # Using AdamW
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=1e-7)
    print("Optimizer and LR Scheduler defined.")

    # --- 7. Training Loop ---
    best_val_loss = float('inf')
    print(f"DEBUG: Best model save path: {args.best_model_save_path}")
    print(f"DEBUG: Final model save path: {args.final_model_save_path}")


    print("\nStarting training for Generative Model...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_total_loss = 0.0
        epoch_train_cat_loss, epoch_train_num_loss, epoch_train_flag_loss = 0.0, 0.0, 0.0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            
            # Pass model.feature_output_info and model_config_from_preprocessor to loss function
            total_loss, loss_components_batch = calculate_composite_loss(
                predictions, y_batch, model.feature_output_info, device # model_config not needed here
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()

            epoch_train_total_loss += total_loss.item()
            epoch_train_cat_loss += loss_components_batch['categorical_loss'] # Already itemized and averaged if applicable
            epoch_train_num_loss += loss_components_batch['numerical_loss']
            epoch_train_flag_loss += loss_components_batch['flag_loss']

            if batch_idx % 50 == 0: # Log every 50 batches
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Train Batch Loss: {total_loss.item():.4f}")
        
        avg_train_loss = epoch_train_total_loss / len(train_loader)
        avg_train_cat_loss = epoch_train_cat_loss / len(train_loader) # These are sums of potentially pre-averaged items.
        avg_train_num_loss = epoch_train_num_loss / len(train_loader) # Need to be careful here.
        avg_train_flag_loss = epoch_train_flag_loss / len(train_loader) # The per-batch loss_components are already averaged if multiple feats of same type.

        # Validation loop
        model.eval()
        epoch_val_total_loss = 0.0
        epoch_val_cat_loss, epoch_val_num_loss, epoch_val_flag_loss = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for x_batch_val, y_batch_val in val_loader:
                x_batch_val = x_batch_val.to(device)
                y_batch_val = y_batch_val.to(device)
                
                predictions_val = model(x_batch_val)
                total_loss_val, loss_components_val_batch = calculate_composite_loss(
                    predictions_val, y_batch_val, model.feature_output_info, device # model_config not needed here
                )
                
                epoch_val_total_loss += total_loss_val.item()
                epoch_val_cat_loss += loss_components_val_batch['categorical_loss']
                epoch_val_num_loss += loss_components_val_batch['numerical_loss']
                epoch_val_flag_loss += loss_components_val_batch['flag_loss']

        avg_val_loss = epoch_val_total_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_cat_loss = epoch_val_cat_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_num_loss = epoch_val_num_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_flag_loss = epoch_val_flag_loss / len(val_loader) if len(val_loader) > 0 else 0

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
              f"Train Loss: {avg_train_loss:.4f} (Cat: {avg_train_cat_loss:.4f}, Num: {avg_train_num_loss:.4f}, Flag: {avg_train_flag_loss:.4f}), "
              f"Val Loss: {avg_val_loss:.4f} (Cat: {avg_val_cat_loss:.4f}, Num: {avg_val_num_loss:.4f}, Flag: {avg_val_flag_loss:.4f})")

        scheduler.step(avg_val_loss) # Step the scheduler on validation loss

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.best_model_save_path)
            print(f"New best model saved to {args.best_model_save_path} (Val Loss: {best_val_loss:.4f})")
    
    torch.save(model.state_dict(), args.final_model_save_path)
    print(f"Final model saved to {args.final_model_save_path}")
    print("Training complete for Generative Model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Generative Transaction Prediction Model.")
    
    # Data and Config Paths
    parser.add_argument("--input-npz-generative", type=str, default=os.path.join("data", "processed_ai_data_generative_test", DEFAULT_INPUT_NPZ_GENERATIVE),
                        help="Path to the input .npz file with generative sequences and targets.")
    parser.add_argument("--generative-model-config-path", type=str, 
                        default=os.path.join("data", "processed_ai_data_generative_test", "thorchain_artifacts_v1", DEFAULT_GENERATIVE_MODEL_CONFIG_FILENAME),
                        help="Path to the model_config_generative.json file from preprocessing.")
    
    # Model Save Paths
    parser.add_argument("--model-save-dir", type=str, default=DEFAULT_MODEL_SAVE_DIR,
                        help="Directory to save trained models.")
    parser.add_argument("--best-model-save-path", type=str, default=DEFAULT_BEST_MODEL_GENERATIVE_SAVE_PATH,
                        help="Path to save the best model (overwrite if exists).")
    parser.add_argument("--final-model-save-path", type=str, default=DEFAULT_FINAL_MODEL_GENERATIVE_SAVE_PATH,
                        help="Path to save the final model (overwrite if exists).")

    # Model Hyperparameters (allow overriding defaults)
    parser.add_argument("--embedding-dim-config-json", type=str, default=json.dumps(DEFAULT_EMBEDDING_DIM_CONFIG), 
                        help='JSON string for embedding_dim_config. E.g., \'\'\'{"default_cat_embed_dim":16, ...}\'\'\'.')
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--nhead", type=int, default=DEFAULT_NHEAD)
    parser.add_argument("--num_encoder_layers", type=int, default=DEFAULT_NUM_ENCODER_LAYERS)
    parser.add_argument("--dim_feedforward", type=int, default=DEFAULT_DIM_FEEDFORWARD)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)

    cli_args = parser.parse_args()
    
    # Update global vars from CLI args if needed (or pass cli_args to main)
    BATCH_SIZE = cli_args.batch_size
    LEARNING_RATE = cli_args.learning_rate
    NUM_EPOCHS = cli_args.num_epochs
    VAL_SPLIT = cli_args.val_split
    
    main(cli_args) 