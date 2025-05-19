import torch
import torch.nn as nn
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse # Added

from model import ArbitragePredictionModel
from train_model import TransactionSequenceDataset # Re-use dataset class

# --- Default Configuration (can be overridden by args) ---
DEFAULT_OUTPUT_DIR = "models" # For saving plots
BATCH_SIZE_EVAL = 32

# Default model hyperparameter values (used if not found in model_config.json)
# These should ideally align with train_model.py defaults
DEFAULT_ASSET_EMBED_DIM = 32
DEFAULT_TYPE_EMBED_DIM = 10
DEFAULT_STATUS_EMBED_DIM = 8
DEFAULT_ACTOR_TYPE_EMBED_DIM = 10
DEFAULT_D_MODEL = 128
DEFAULT_NHEAD = 4
DEFAULT_NUM_ENCODER_LAYERS = 3
DEFAULT_DIM_FEEDFORWARD = 512
DEFAULT_DROPOUT = 0.1

def evaluate_model(args):
    print(f"--- Starting Model Evaluation ---")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Model Configuration ---
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

    # Extract parameters from model_config
    categorical_embedding_details = model_config.get('categorical_embedding_details')
    if not categorical_embedding_details:
        print("Error: 'categorical_embedding_details' not found in model_config.json")
        return

    asset_vocab_size = categorical_embedding_details.get("asset_to_id.json")
    type_vocab_size = categorical_embedding_details.get("type_to_id.json")
    status_vocab_size = categorical_embedding_details.get("status_to_id.json")
    actor_type_vocab_size = categorical_embedding_details.get("actor_type_to_id.json")
    if any(v is None for v in [asset_vocab_size, type_vocab_size, status_vocab_size, actor_type_vocab_size]):
        print("Error: One or more vocab sizes not found in model_config.json.")
        return

    num_numerical_features = model_config.get('num_numerical_features')
    sequence_length = model_config.get('sequence_length')
    if num_numerical_features is None or sequence_length is None:
        print("Error: 'num_numerical_features' or 'sequence_length' not in model_config.json.")
        return
    p_target_classes = model_config.get('p_target_classes', actor_type_vocab_size)

    asset_embed_dim = model_config.get('asset_embed_dim', DEFAULT_ASSET_EMBED_DIM)
    type_embed_dim = model_config.get('type_embed_dim', DEFAULT_TYPE_EMBED_DIM)
    status_embed_dim = model_config.get('status_embed_dim', DEFAULT_STATUS_EMBED_DIM)
    actor_type_embed_dim = model_config.get('actor_type_embed_dim', DEFAULT_ACTOR_TYPE_EMBED_DIM)
    d_model = model_config.get('d_model', DEFAULT_D_MODEL)
    nhead = model_config.get('nhead', DEFAULT_NHEAD)
    num_encoder_layers = model_config.get('num_encoder_layers', DEFAULT_NUM_ENCODER_LAYERS)
    dim_feedforward = model_config.get('dim_feedforward', DEFAULT_DIM_FEEDFORWARD)
    dropout_rate = model_config.get('dropout', DEFAULT_DROPOUT)
    print("Model configuration loaded.")

    # --- 3. Load Test Data ---
    print(f"Loading test data from {args.input_npz}...")
    try:
        data = np.load(args.input_npz, allow_pickle=True)
        X_cat_ids_sequences = data['X_cat_ids_sequences']
        X_num_sequences = data['X_num_sequences']
        y_p_targets_one_hot = data['y_p_targets']
        y_mu_targets = data['y_mu_targets']
    except FileNotFoundError:
        print(f"Error: Test data file not found at {args.input_npz}.")
        return
    except KeyError as e:
        print(f"Error: Missing key {e} in test data file {args.input_npz}.")
        return
    print(f"Test data loaded. Shapes: X_cat_ids={X_cat_ids_sequences.shape}, X_num={X_num_sequences.shape}, y_p={y_p_targets_one_hot.shape}")
    
    # Validate data against model_config
    if X_cat_ids_sequences.shape[1] != sequence_length or X_num_sequences.shape[1] != sequence_length:
        print(f"Error: Seq length in data ({X_cat_ids_sequences.shape[1]}) != model_config ({sequence_length}).")
        return
    if X_num_sequences.shape[2] != num_numerical_features:
        print(f"Error: Num numerical features in data ({X_num_sequences.shape[2]}) != model_config ({num_numerical_features}).")
        return

    eval_dataset = TransactionSequenceDataset(X_cat_ids_sequences, X_num_sequences, y_p_targets_one_hot, y_mu_targets)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    print(f"Created DataLoader for evaluation with {len(eval_dataset)} samples.")

    # --- 4. Instantiate and Load Model Weights ---
    model = ArbitragePredictionModel(
        asset_vocab_size=asset_vocab_size, type_vocab_size=type_vocab_size, status_vocab_size=status_vocab_size,
        actor_type_vocab_size=actor_type_vocab_size, asset_embed_dim=asset_embed_dim, type_embed_dim=type_embed_dim,
        status_embed_dim=status_embed_dim, actor_type_embed_dim=actor_type_embed_dim,
        num_numerical_features=num_numerical_features, d_model=d_model, nhead=nhead,
        num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
        p_target_classes=p_target_classes, mu_target_dim=1, dropout=dropout_rate, max_seq_len=sequence_length
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.model_weights_path, map_location=device))
        print(f"Model weights loaded successfully from {args.model_weights_path}")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {args.model_weights_path}.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model.eval()

    # --- 5. Perform Inference and Collect Predictions/Targets ---
    all_p_preds_indices, all_p_targets_indices = [], []
    all_mu_preds_for_arb, all_mu_targets_for_arb = [], []

    # ARB_SWAP ID - Assuming 0 from typical mapping, but could be made configurable
    arb_swap_id = 0 
    # For class labels in confusion matrix - use actor_type_vocab_size and assume 0..N-1
    # A more robust way would be to load actor_type_to_id.json (via model_config if path stored there)
    # For now, if model_config provided `p_target_classes`, we use that for label count.
    num_actor_classes = p_target_classes
    # Construct basic labels if full mapping isn't loaded: [Class 0, Class 1, ...]
    # This is a simplification. Ideally, model_config.json would point to the actor_type mapping file,
    # or preprocessor.py would save the id_to_label mapping in model_config.json.
    id_to_actor_type_map = {i: f"Class_{i}" for i in range(num_actor_classes)}
    # If actor_type_to_id.json is consistently named and available relative to model_config_path, could load it:
    actor_type_mapping_path_cand = os.path.join(os.path.dirname(args.model_config_path), "actor_type_to_id.json")
    if os.path.exists(actor_type_mapping_path_cand):
        try:
            with open(actor_type_mapping_path_cand, 'r') as f_map_act:
                loaded_actor_map = json.load(f_map_act)
                id_to_actor_type_map = {v: k for k,v in loaded_actor_map.items()} # Ensure v is int if keys are str
                arb_swap_id = loaded_actor_map.get('ARB_SWAP')
                if arb_swap_id is None: arb_swap_id = 0; print("Warning: ARB_SWAP not in loaded map, defaulting to 0 for mu_loss.")
                print(f"Loaded actor type mapping for labels. ARB_SWAP ID: {arb_swap_id}")
        except Exception as e_map_load:
            print(f"Warning: Could not load actor_type_to_id.json for class labels ({e_map_load}). Using generic labels.")
    else:
        print(f"Warning: actor_type_to_id.json not found at {actor_type_mapping_path_cand}. Using generic class labels.")

    with torch.no_grad():
        for X_cat_ids_batch, X_num_batch, y_p_batch_one_hot, y_mu_batch in eval_loader:
            X_cat_ids_batch, X_num_batch, y_p_batch_one_hot, y_mu_batch = \
                X_cat_ids_batch.to(device), X_num_batch.to(device), y_p_batch_one_hot.to(device), y_mu_batch.to(device)

            p_logits, mu_preds_batch = model(X_cat_ids_batch, X_num_batch)
            p_preds_indices_batch = torch.argmax(p_logits, dim=1).cpu().numpy()
            p_targets_indices_batch = torch.argmax(y_p_batch_one_hot, dim=1).cpu().numpy()
            all_p_preds_indices.extend(p_preds_indices_batch)
            all_p_targets_indices.extend(p_targets_indices_batch)

            mu_preds_batch_squeezed = mu_preds_batch.squeeze().cpu().numpy()
            y_mu_batch_squeezed = y_mu_batch.squeeze().cpu().numpy()
            
            for i in range(len(p_targets_indices_batch)):
                if p_targets_indices_batch[i] == arb_swap_id and not np.isnan(y_mu_batch_squeezed[i]):
                    all_mu_targets_for_arb.append(y_mu_batch_squeezed[i])
                    all_mu_preds_for_arb.append(mu_preds_batch_squeezed[i])
    
    all_p_preds_indices = np.array(all_p_preds_indices)
    all_p_targets_indices = np.array(all_p_targets_indices)
    all_mu_preds_for_arb = np.array(all_mu_preds_for_arb)
    all_mu_targets_for_arb = np.array(all_mu_targets_for_arb)

    # --- 6. Calculate and Print Metrics ---
    print("\n--- Actor Type Prediction (P Target) Metrics ---")
    accuracy = accuracy_score(all_p_targets_indices, all_p_preds_indices)
    print(f"Overall Accuracy: {accuracy:.4f}")

    class_labels_sorted = [id_to_actor_type_map[i] for i in sorted(id_to_actor_type_map.keys()) if i < num_actor_classes]
    unique_target_ids = sorted(np.unique(all_p_targets_indices).tolist())
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_p_targets_indices, all_p_preds_indices, 
        labels=unique_target_ids, # Use only labels present in targets
        zero_division=0
    )
    
    print("\nPrecision, Recall, F1-Score per class (present in targets):")
    for i, class_id in enumerate(unique_target_ids):
        label_name = id_to_actor_type_map.get(class_id, f"UnknownClass_{class_id}")
        print(f"  Class {label_name} (ID {class_id}): P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}")

    cm = confusion_matrix(all_p_targets_indices, all_p_preds_indices, labels=unique_target_ids)
    cm_df = pd.DataFrame(cm, index=[id_to_actor_type_map.get(i,f"Actual_{i}") for i in unique_target_ids], 
                         columns=[id_to_actor_type_map.get(i,f"Pred_{i}") for i in unique_target_ids])
    print("\nConfusion Matrix (only classes present in targets):")
    print(cm_df)
    
    plt.figure(figsize=(max(8, len(unique_target_ids)*2), max(6, len(unique_target_ids)*1.5)))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Actor Type Prediction')
    plt.ylabel('Actual Class'); plt.xlabel('Predicted Class')
    cm_save_path = os.path.join(args.output_dir, "confusion_matrix_actor_type.png")
    plt.savefig(cm_save_path); plt.close()
    print(f"Confusion matrix plot saved to {cm_save_path}")

    print("\n--- Profit Prediction (Mu Target) Metrics for ARB_SWAP cases ---")
    if len(all_mu_targets_for_arb) > 0:
        mu_mse = mean_squared_error(all_mu_targets_for_arb, all_mu_preds_for_arb)
        mu_rmse = np.sqrt(mu_mse); mu_mae = mean_absolute_error(all_mu_targets_for_arb, all_mu_preds_for_arb)
        print(f"Mu Eval Samples: {len(all_mu_targets_for_arb)}, MSE: {mu_mse:.4f}, RMSE: {mu_rmse:.4f}, MAE: {mu_mae:.4f}")
        plt.figure(figsize=(8, 6))
        plt.scatter(all_mu_targets_for_arb, all_mu_preds_for_arb, alpha=0.5)
        min_val = min(all_mu_targets_for_arb.min(), all_mu_preds_for_arb.min())
        max_val = max(all_mu_targets_for_arb.max(), all_mu_preds_for_arb.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title('Profit Prediction (Mu) - Actual vs. Predicted (ARB_SWAP cases)'); plt.xlabel('Actual Mu'); plt.ylabel('Predicted Mu')
        plt.grid(True)
        scatter_save_path = os.path.join(args.output_dir, "scatter_plot_mu_profit.png")
        plt.savefig(scatter_save_path); plt.close()
        print(f"Mu profit scatter plot saved to {scatter_save_path}")
    else:
        print("No valid ARB_SWAP samples for Mu evaluation.")

    print("\n--- Evaluation Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Arbitrage Prediction Model.")
    parser.add_argument("--input-npz", type=str, required=True,
                        help="Path to the input .npz file containing test sequences and targets.")
    parser.add_argument("--model-config-path", type=str, required=True,
                        help="Path to the model_config.json file.")
    parser.add_argument("--model-weights-path", type=str, required=True,
                        help="Path to the trained model weights (.pth file).")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save evaluation plots. Default: {DEFAULT_OUTPUT_DIR}")
    
    cli_args = parser.parse_args()
    evaluate_model(cli_args) 