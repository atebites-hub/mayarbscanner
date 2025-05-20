import torch
import numpy as np
import os
import json
import argparse
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from model import GenerativeTransactionModel 
# For accessing train_model defaults in argparse, and potentially calculate_composite_loss
import train_model 

# --- Default Configuration ---
DEFAULT_MODEL_DIR = "models"
DEFAULT_BEST_MODEL_FILENAME = "best_generative_model_thorchain.pth" 
DEFAULT_MODEL_CONFIG_FILENAME = "model_config_generative_thorchain.json"
DEFAULT_TEST_DATA_NPZ = "sequences_and_targets_generative_thorchain.npz" 
DEFAULT_ARTIFACTS_DIR = os.path.join("data", "processed_ai_data_generative_test", "thorchain_artifacts_v1")
DEFAULT_EVALUATION_OUTPUT_DIR = "evaluation_results_generative"

# Helper function to plot confusion matrix (can be expanded)
def plot_confusion_matrix_func(cm, class_names, title='Confusion Matrix', output_path='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def main(args):
    os.makedirs(args.evaluation_output_dir, exist_ok=True)

    # --- 1. Load Model Configuration ---
    model_config_path = os.path.join(args.artifacts_dir, args.model_config_filename)
    print(f"Loading generative model configuration from {model_config_path}...")
    try:
        with open(model_config_path, 'r') as f_config:
            model_config_from_preprocessor = json.load(f_config)
    except FileNotFoundError:
        print(f"Error: Generative model configuration file not found at {model_config_path}.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {model_config_path}.")
        return

    # --- 2. Load Test Data ---
    test_data_path = os.path.join(args.test_data_dir, args.test_data_npz)
    print(f"Loading test data from {test_data_path}...")
    try:
        data = np.load(test_data_path, allow_pickle=True)
        X_test_sequences = data['X_sequences']
        Y_test_targets = data['Y_targets']
        # feature_columns_ordered_from_data = data['feature_columns_ordered'] # For validation
    except FileNotFoundError:
        print(f"Error: Test data file not found at {test_data_path}.")
        return
    except KeyError as e:
        print(f"Error: Missing expected key {e} in test data file {test_data_path}.")
        return
    print(f"Test data loaded. Shapes: X_sequences={X_test_sequences.shape}, Y_targets={Y_test_targets.shape}")

    # --- 3. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 4. Instantiate Model ---
    embedding_config_for_model = json.loads(args.embedding_dim_config_json) 
    model = GenerativeTransactionModel(
        model_config=model_config_from_preprocessor,
        embedding_dim_config=embedding_config_for_model,
        d_model=args.d_model, 
        nhead=args.nhead, 
        num_encoder_layers=args.num_encoder_layers, 
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, 
        max_seq_len=model_config_from_preprocessor.get('sequence_length', 10)
    ).to(device)
    print("GenerativeTransactionModel instantiated.")
    
    # --- 5. Load Trained Model Weights ---
    model_weights_path = os.path.join(args.model_dir, args.model_filename)
    print(f"Loading trained model weights from {model_weights_path}...")
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_weights_path}.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval()
    print("Model weights loaded and model set to eval mode.")

    # --- 7. Generate Predictions ---
    all_predictions_list = []
    # For evaluation, we can process in batches if test set is large
    test_dataset = train_model.GenerativeTransactionDataset(X_test_sequences, Y_test_targets) # Use from train_model
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Generating predictions...")
    with torch.no_grad():
        for x_batch, _ in test_loader: # We only need x_batch for predictions
            x_batch = x_batch.to(device)
            predictions_batch = model(x_batch)
            all_predictions_list.append(predictions_batch.cpu().numpy())
    
    if not all_predictions_list:
        print("No predictions were generated. Check test data or model.")
        return

    all_predictions = np.concatenate(all_predictions_list, axis=0)
    all_targets = Y_test_targets # Ground truth next transactions
    print(f"Predictions generated. Shape: {all_predictions.shape}")

    # --- 8. Calculate Metrics ---
    results = {}
    feature_output_info_list = model.feature_output_info
    
    print("\n--- Evaluation Metrics ---")
    for feat_info in feature_output_info_list:
        feat_name = feat_info['name']
        feat_type = feat_info['type']
        output_start_idx = feat_info['output_start_idx']
        output_end_idx = feat_info['output_end_idx']
        original_input_idx = feat_info['original_input_index']

        pred_slice_all = all_predictions[:, output_start_idx:output_end_idx]
        target_slice_all = all_targets[:, original_input_idx]
        
        print(f"\nFeature: {feat_name} (Type: {feat_type})")
        
        if feat_type == 'id_cat' or feat_type == 'hash_cat':
            pred_ids = np.argmax(pred_slice_all, axis=1)
            target_ids = target_slice_all.astype(int)
            
            accuracy = accuracy_score(target_ids, pred_ids)
            f1_w = f1_score(target_ids, pred_ids, average='weighted', zero_division=0)
            print(f"  Accuracy: {accuracy:.4f}, F1 (weighted): {f1_w:.4f}")
            results[feat_name] = {'accuracy': accuracy, 'f1_weighted': f1_w}
            
            # Optional: Confusion Matrix for selected features
            if feat_name in ['action_type_id'] and feat_info.get('vocab_size', 0) > 1:
                # Get class names from mapping if available
                class_names = [None] * feat_info['vocab_size'] # Default
                mapping_details = model_config_from_preprocessor.get('categorical_id_mapping_details', {})
                found_map_name = None
                if 'asset' in feat_name: 
                    found_map_name = next((k for k in mapping_details if 'asset_to_id' in k), None)
                else:
                    search_prefix = feat_name.replace('_id', '').split('_')[-1]
                    if feat_name.replace('_id', '').endswith("memo_status"): search_prefix = "memo_status"
                    found_map_name = next((k for k in mapping_details if k.startswith(search_prefix + '_to_id')), None)
                
                if found_map_name and found_map_name in mapping_details:
                    inv_map = {v: k for k, v in mapping_details[found_map_name].items()}
                    class_names = [inv_map.get(i, str(i)) for i in range(feat_info['vocab_size'])]
                
                cm_path = os.path.join(args.evaluation_output_dir, f"cm_{feat_name}.png")
                labels_for_cm = np.arange(feat_info['vocab_size'])
                cm = confusion_matrix(target_ids, pred_ids, labels=labels_for_cm)
                plot_confusion_matrix_func(cm, class_names=class_names, title=f'CM for {feat_name}', output_path=cm_path)

        elif feat_type == 'binary':
            pred_binary = (pred_slice_all.squeeze(axis=-1) > 0).astype(int) if pred_slice_all.ndim > 1 else (pred_slice_all > 0).astype(int)
            target_binary = target_slice_all.astype(int)
            
            accuracy = accuracy_score(target_binary, pred_binary)
            f1 = f1_score(target_binary, pred_binary, zero_division=0)
            print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            results[feat_name] = {'accuracy': accuracy, 'f1': f1}

        elif feat_type == 'numerical':
            pred_values = pred_slice_all.squeeze(axis=-1) if pred_slice_all.ndim > 1 else pred_slice_all
            target_values = target_slice_all 
            
            mse = mean_squared_error(target_values, pred_values)
            mae = mean_absolute_error(target_values, pred_values)
            # r2 = r2_score(target_values, pred_values) # R2 can be misleading if model is poor
            print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")
            results[feat_name] = {'mse': mse, 'mae': mae}
            
            # Optional: Scatter plot for selected numericals
            if feat_name in ['action_date_unix_scaled']: # Example
                plt.figure(figsize=(8, 6))
                plt.scatter(target_values, pred_values, alpha=0.5)
                plt.plot([target_values.min(), target_values.max()], [target_values.min(), target_values.max()], 'r--')
                plt.title(f'Scatter Plot for {feat_name}')
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.savefig(os.path.join(args.evaluation_output_dir, f"scatter_{feat_name}.png"))
                plt.close()

    # --- 9. Save Results ---
    results_path = os.path.join(args.evaluation_output_dir, "evaluation_metrics.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation metrics saved to {results_path}")

    # Calculate overall test loss using the imported function
    try:
        overall_test_loss, loss_components = train_model.calculate_composite_loss(
            torch.tensor(all_predictions).to(device),
            torch.tensor(all_targets).to(device),
            model.feature_output_info,
            device
        )
        print(f"\nOverall Test Set Loss (calculated post-hoc): {overall_test_loss.item():.4f}")
        print(f"  Loss Components: {loss_components}")
        results['_overall_test_loss'] = overall_test_loss.item()
        results['_loss_components'] = {k: v.item() if hasattr(v, 'item') else v for k, v in loss_components.items()} # Convert tensors to numbers
        with open(results_path, 'w') as f: # Re-save with loss
            json.dump(results, f, indent=4)

    except Exception as e:
        print(f"Error calculating overall test loss: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Generative Transaction Prediction Model.")
    
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-filename", type=str, default=DEFAULT_BEST_MODEL_FILENAME)
    parser.add_argument("--artifacts-dir", type=str, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--model-config-filename", type=str, default=DEFAULT_MODEL_CONFIG_FILENAME)
    
    parser.add_argument("--test-data-dir", type=str, default=os.path.join("data", "processed_ai_data_generative_test"))
    parser.add_argument("--test-data-npz", type=str, default=DEFAULT_TEST_DATA_NPZ)
    
    parser.add_argument("--evaluation-output-dir", type=str, default=DEFAULT_EVALUATION_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32) 

    parser.add_argument("--embedding-dim-config-json", type=str, default=json.dumps(train_model.DEFAULT_EMBEDDING_DIM_CONFIG),
                        help='JSON string for embedding_dim_config for model instantiation.')
    parser.add_argument("--d_model", type=int, default=train_model.DEFAULT_D_MODEL)
    parser.add_argument("--nhead", type=int, default=train_model.DEFAULT_NHEAD)
    parser.add_argument("--num_encoder_layers", type=int, default=train_model.DEFAULT_NUM_ENCODER_LAYERS)
    parser.add_argument("--dim_feedforward", type=int, default=train_model.DEFAULT_DIM_FEEDFORWARD)
    parser.add_argument("--dropout", type=float, default=train_model.DEFAULT_DROPOUT)

    cli_args = parser.parse_args()
    main(cli_args) 