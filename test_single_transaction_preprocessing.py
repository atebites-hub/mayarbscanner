import json
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add parent directory to sys.path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess_ai_data import (
    ASSET_PRECISIONS,
    DEFAULT_ASSET_MAPPING_FILENAME_GENERATIVE,
    DEFAULT_MEMO_STATUS_MAPPING_FILENAME_GENERATIVE,
    DEFAULT_STATUS_MAPPING_FILENAME_GENERATIVE,
    DEFAULT_TYPE_MAPPING_FILENAME_GENERATIVE,
    HASH_SEED,
    HASH_VOCAB_SIZE_ADDRESS,
    HASH_VOCAB_SIZE_AFFILIATE_ADDRESS,
    NO_ADDRESS_STR,
    NO_ASSET_STR,
    NO_POOL_ASSET_STR,
    PAD_TOKEN_STR,
    UNKNOWN_TOKEN_STR,
    get_asset_precision,
    get_or_create_mapping_generative,
    preprocess_actions_for_generative_model,
    process_categorical_features_generative,
    process_numerical_features_generative,
)

def run_single_transaction_test():
    print("--- Starting Single Transaction Preprocessing Test ---")

    # --- Configuration ---
    base_artifacts_dir = "data/processed_ai_data_generative_maya_s25_l6_atomic/mayachain_artifacts_v1_atomic"
    raw_data_file = "data/transactions_data.json"
    
    model_config_filename = "model_config_generative_mayachain_s25_l6_atomic.json"
    scaler_filename = "scaler_generative_thorchain.pkl"

    model_config_path = os.path.join(base_artifacts_dir, model_config_filename)
    scaler_path = os.path.join(base_artifacts_dir, scaler_filename)

    # --- 1. Load Artifacts ---
    print(f"\n--- Loading Artifacts from {base_artifacts_dir} ---")
    if not os.path.exists(model_config_path):
        print(f"ERROR: Model config not found at {model_config_path}")
        return
    if not os.path.exists(scaler_path):
        print(f"ERROR: Scaler not found at {scaler_path}")
        return

    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    print("  Loaded model_config.json")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("  Loaded scaler.pkl")

    # Load mappings (using 'test' mode)
    # The mapping filenames are usually part of model_config or known defaults
    # For this test, we'll use the defaults from preprocess_ai_data.py, assuming they match the artifacts
    
    # It's safer to get mapping file names from feature_processing_details in model_config if available
    # However, for this test, we'll assume the default names were used for artifact generation.
    
    all_mappings = {}
    mapping_files_to_load = {
        "asset_to_id_generative_thorchain.json": True, # is_asset_mapping = True
        "type_to_id_generative_thorchain.json": False,
        "status_to_id_generative_thorchain.json": False,
        "memo_status_to_id_generative_thorchain.json": False,
    }

    for map_file, is_asset_map_flag in mapping_files_to_load.items():
        try:
            full_map_path = os.path.join(base_artifacts_dir, map_file)
            if os.path.exists(full_map_path):
                all_mappings[map_file] = get_or_create_mapping_generative(
                    full_map_path, None, mode='test', is_asset_mapping=is_asset_map_flag
                )
                print(f"  Loaded mapping: {map_file}")
            else:
                print(f"  WARNING: Mapping file {full_map_path} not found. Skipping.")
        except Exception as e:
            print(f"  ERROR loading mapping {map_file}: {e}")
            return


    # --- 2. Load Sample Transaction ---
    print(f"\n--- Loading Sample Transaction from {raw_data_file} ---")
    if not os.path.exists(raw_data_file):
        print(f"ERROR: Raw data file not found at {raw_data_file}")
        return
    
    with open(raw_data_file, 'r') as f:
        raw_data = json.load(f)
    
    if not raw_data.get('actions') or not isinstance(raw_data['actions'], list) or len(raw_data['actions']) == 0:
        print("ERROR: No actions found in raw data file.")
        return
        
    sample_action_raw = raw_data['actions'][0]
    print("  Loaded first transaction as sample.")
    print("\nOriginal Sample Transaction (selected fields):")
    print(f"  Type: {sample_action_raw.get('type')}")
    print(f"  Status: {sample_action_raw.get('status')}")
    print(f"  Date: {sample_action_raw.get('date')}")
    in_txs = sample_action_raw.get('in', [{}])[0].get('coins', [{}])
    if in_txs:
      print(f"  In Coin 1 Asset: {in_txs[0].get('asset')}, Amount: {in_txs[0].get('amount')}")
    out_txs = sample_action_raw.get('out', [{}])[0].get('coins', [{}])
    if out_txs:
      print(f"  Out Coin 1 Asset: {out_txs[0].get('asset')}, Amount: {out_txs[0].get('amount')}")


    # --- 3. Preprocess Single Transaction ---
    print("\n--- Preprocessing Single Transaction ---")
    
    # Wrap the single action in a list
    actions_list_for_preprocessing = [sample_action_raw]

    # Step 3.1: Initial flattening
    df_flattened, _ = preprocess_actions_for_generative_model(
        actions_list_for_preprocessing,
        base_artifacts_dir, # artifacts_dir used for consistency, though not for writing in this part
        mode='test' 
    )
    if df_flattened.empty:
        print("ERROR: Flattening returned empty DataFrame.")
        return
    print("  Step 3.1: Initial flattening completed.")

    # Step 3.2: Atomic Amount Conversion (copied from main in preprocess_ai_data.py)
    print("  Step 3.2: Converting amounts to numerical (atomic units).")
    atomic_conversion_map = {
        'in_coin1_amount_raw': ('in_coin1_asset_raw', 'in_coin1_amount_atomic'),
        'out_coin1_amount_raw': ('out_coin1_asset_raw', 'out_coin1_amount_atomic'),
        'meta_swap_liquidityFee_raw': ('pool1_asset_raw', 'meta_swap_liquidityFee_atomic'),
        'meta_swap_affiliateFee_raw': ('pool1_asset_raw', 'meta_swap_affiliateFee_atomic'),
        'meta_swap_networkFee1_amount_raw': ('meta_swap_networkFee1_asset_raw', 'meta_swap_networkFee1_amount_atomic'),
        'meta_swap_streaming_quantity_raw': ('in_coin1_asset_raw', 'meta_swap_streaming_quantity_atomic'),
        'meta_withdraw_imp_loss_protection_raw': ('MAYA.CACAO', 'meta_withdraw_imp_loss_protection_atomic')
    }
    for raw_col, (asset_col_name, atomic_col_name) in atomic_conversion_map.items():
        if raw_col in df_flattened.columns:
            def convert_raw_atomic_to_numerical(row): # Use 'row' as it's applied row-wise
                raw_amount_str = row.get(raw_col)
                if pd.isna(raw_amount_str) or raw_amount_str == "":
                    return np.nan
                try:
                    return float(raw_amount_str) # Already atomic, just convert to float
                except (ValueError, TypeError):
                    return np.nan
            df_flattened[atomic_col_name] = df_flattened.apply(convert_raw_atomic_to_numerical, axis=1)
    print("    Atomic amount conversion applied.")

    # Step 3.3: Categorical Processing
    df_categoricals_processed, processed_cat_col_names, _, _ = process_categorical_features_generative(
        df_flattened,
        base_artifacts_dir, # Critical for loading mappings
        mode='test'
    )
    print("  Step 3.3: Categorical processing completed.")
    
    # Step 3.4: Numerical Processing
    df_numericals_processed, processed_num_col_names_scaled, _ = process_numerical_features_generative(
        df_categoricals_processed, # Pass the DataFrame with categoricals processed
        scaler_path,
        mode='test'
    )
    print("  Step 3.4: Numerical processing completed.")

    # --- 4. Output and Compare ---
    print("\n--- Processed Features for Sample Transaction ---")
    
    final_processed_df = df_numericals_processed
    
    if final_processed_df.empty:
        print("ERROR: Final processed DataFrame is empty.")
        return

    final_feature_vector = final_processed_df.iloc[0] # Get the first (and only) row

    print("\nKey Amount-Related Features:")
    amount_features_to_check = [
        'in_coin1_amount_raw', 'in_coin1_amount_atomic', 'in_coin1_amount_norm_scaled',
        'out_coin1_amount_raw', 'out_coin1_amount_atomic', 'out_coin1_amount_norm_scaled',
        'meta_swap_liquidityFee_raw', 'meta_swap_liquidityFee_atomic', 'meta_swap_liquidityFee_norm_scaled'
    ]
    for f_name in amount_features_to_check:
        if f_name in final_feature_vector:
            print(f"  {f_name}: {final_feature_vector[f_name]}")
        elif f_name.replace('_raw', '_atomic') in final_feature_vector: # Check if atomic exists
             print(f"  {f_name} (raw not in final, atomic is): {final_feature_vector.get(f_name.replace('_raw', '_atomic'))}")
        elif f_name.replace('_raw', '_norm_scaled') in final_feature_vector: # Check if scaled exists
             print(f"  {f_name} (raw not in final, scaled is): {final_feature_vector.get(f_name.replace('_raw', '_norm_scaled'))}")
        else:
            print(f"  {f_name}: Not found in processed features")

    print("\nSelected Other Processed Features:")
    other_features_to_show = [
        'action_type_id', 'action_status_id', 
        'in_address_hash_id', 'pool1_asset_id',
        'action_date_unix_scaled', 'action_height_val_scaled',
        'meta_is_swap_flag'
    ]
    for f_name in other_features_to_show:
        if f_name in final_feature_vector:
            print(f"  {f_name}: {final_feature_vector[f_name]}")
        else:
            print(f"  {f_name}: Not found in processed features")
            
    print("\n--- Test Complete. Review the output above. ---")

if __name__ == "__main__":
    run_single_transaction_test() 