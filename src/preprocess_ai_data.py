print("EXECUTOR_AGENT_SANITY_CHECK_PRINT_TOP_OF_FILE_V3_GENERATIVE")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os
import pickle
import time
from datetime import datetime
import argparse
import mmh3

# --- Directory Constants (can be overridden by args where applicable) ---
DEFAULT_DATA_DIR = "data"
DEFAULT_PROCESSED_DATA_DIR_GENERATIVE = os.path.join(DEFAULT_DATA_DIR, "processed_ai_data_generative_test")

# --- Default File Names (GENERATIVE MODEL - can be overridden by args) ---
DEFAULT_INPUT_JSON = "transactions_data.json"
DEFAULT_OUTPUT_NPZ_GENERATIVE = "sequences_and_targets_generative_thorchain.npz"
DEFAULT_SCALER_FILENAME_GENERATIVE = "scaler_generative_thorchain.pkl"
DEFAULT_MODEL_CONFIG_FILENAME_GENERATIVE = "model_config_generative_thorchain.json"

# NEW mapping file names for generative model (will be dataset specific, e.g. thorchain)
DEFAULT_ASSET_MAPPING_FILENAME_GENERATIVE = "asset_to_id_generative_thorchain.json"
DEFAULT_TYPE_MAPPING_FILENAME_GENERATIVE = "type_to_id_generative_thorchain.json"
DEFAULT_STATUS_MAPPING_FILENAME_GENERATIVE = "status_to_id_generative_thorchain.json"
DEFAULT_MEMO_STATUS_MAPPING_FILENAME_GENERATIVE = "memo_status_to_id_generative_thorchain.json"

# Sequence length for model input
SEQUENCE_LENGTH = 10

# --- Feature Hashing Constants ---
HASH_SEED = 42
HASH_VOCAB_SIZE_ADDRESS = 20000
HASH_VOCAB_SIZE_AFFILIATE_ADDRESS = 5000
HASH_VOCAB_SIZE_TX_ID = 10000

# --- Padding and Unknown ID Constants (examples, will be refined) ---
PAD_TOKEN_STR = "PAD"
UNKNOWN_TOKEN_STR = "UNKNOWN"
NO_ASSET_STR = "NO_ASSET"
NO_POOL_ASSET_STR = "NO_POOL"
NO_ADDRESS_STR = "NO_ADDRESS"

# --- NEW: Canonical Feature Order for the Generative Model ---
# This order MUST be strictly maintained for model input/output consistency.
# It should include all features produced by the preprocessing steps.
CANONICAL_FEATURE_ORDER = [
    # Global Action ID-mapped Categoricals
    'action_status_id',
    'action_type_id',
    'pool1_asset_id',
    'pool2_asset_id',
    # Inbound Transaction ID-mapped/Hashed Categoricals & Flags
    'in_tx_id_present_flag',
    'in_address_present_flag',
    'in_address_hash_id',
    'in_memo_status_id',
    'in_coin1_present_flag',
    'in_coin1_asset_id',
    # Outbound Transaction ID-mapped/Hashed Categoricals & Flags
    'out_tx_id_present_flag',
    'out_address_present_flag',
    'out_address_hash_id',
    'out_coin1_present_flag',
    'out_coin1_asset_id',
    # Metadata Binary Flags (action type indicators)
    'meta_is_swap_flag',
    'meta_is_addLiquidity_flag',
    'meta_is_withdraw_flag',
    'meta_is_refund_flag',
    'meta_is_thorname_flag', # Assuming it might be added
    # Swap Specific Metadata (ID-mapped, Hashed, Flags)
    'meta_swap_networkFee1_asset_id',
    'meta_swap_target_asset_id',
    'meta_swap_affiliate_address_present_flag',
    'meta_swap_affiliate_address_hash_id',
    'meta_swap_memo_status_id',
    'meta_swap_is_streaming_flag',
    # AddLiquidity Metadata (none beyond flag for V1 based on current schema)
    # Withdraw Metadata (none beyond flag for V1 based on current schema)
    # Refund Metadata (ID-mapped)
    # 'meta_refund_reason_id', # If a refund reason mapping is added
    
    # Scaled Numerical Features (Order them logically)
    # Global Action Numerics
    'action_date_unix_scaled',
    'action_height_val_scaled',
    # Inbound Numerics
    'in_coin1_amount_norm_scaled',
    # Outbound Numerics
    'out_coin1_amount_norm_scaled',
    # Swap Specific Numerics
    'meta_swap_liquidityFee_norm_scaled',
    'meta_swap_swapSlip_bps_val_scaled',
    'meta_swap_networkFee1_amount_norm_scaled',
    'meta_swap_affiliateFee_norm_scaled',
    'meta_swap_streaming_count_val_scaled',
    'meta_swap_streaming_quantity_norm_scaled',
    # AddLiquidity Numerics
    'meta_addLiquidity_units_val_scaled',
    # Withdraw Numerics
    'meta_withdraw_units_val_scaled',
    'meta_withdraw_basis_points_val_scaled',
    'meta_withdraw_asymmetry_val_scaled',
    'meta_withdraw_imp_loss_protection_norm_scaled',
    # (No specific numerics for refund or thorname in current V1 schema beyond flags/basic IDs)
]

def get_or_create_mapping_generative(mapping_file_path, series=None, mode='train', pad_token=PAD_TOKEN_STR, unknown_token=UNKNOWN_TOKEN_STR, is_asset_mapping=False):
    """
    Creates or loads a categorical feature mapping for the generative model.
    Ensures PAD_TOKEN_STR maps to ID 0, and UNKNOWN_TOKEN_STR is present.
    If is_asset_mapping is True, CACAO (if present) will not be forced to ID 0, PAD_TOKEN_STR takes precedence for ID 0.
    """
    print(f"Managing mapping for: {mapping_file_path} (Mode: {mode})")
    if mode == 'train':
        if os.path.exists(mapping_file_path):
            print(f"  Train mode: Found existing mapping file. Will overwrite if structure is incorrect or if series forces new values.")
            # Even if it exists, we might rebuild it if series is provided and contains new values
            # or to ensure PAD/UNKNOWN structure.
        
        if series is None:
            # If no series, but file exists, try to load and verify. If not, create empty with PAD/UNKNOWN.
            if os.path.exists(mapping_file_path):
                try:
                    with open(mapping_file_path, 'r') as f:
                        mapping = json.load(f)
                    if pad_token in mapping and mapping[pad_token] == 0 and unknown_token in mapping:
                        print(f"  Loaded existing valid mapping from {mapping_file_path} (no series provided).")
                        return mapping
                    else:
                        print(f"  Existing mapping {mapping_file_path} is invalid/incomplete. Rebuilding.")
                except json.JSONDecodeError:
                    print(f"  Error reading existing mapping {mapping_file_path}. Rebuilding.")
            # Create a basic mapping if no series and file bad/missing
            print(f"  Creating new basic mapping (PAD/UNKNOWN only) for {mapping_file_path} as no series provided.")
            mapping = {pad_token: 0, unknown_token: 1}
            with open(mapping_file_path, 'w') as f:
                json.dump(mapping, f, indent=4)
            return mapping

        # If series is provided, build from scratch or update existing.
        print(f"  Train mode: Building/updating mapping from series for {mapping_file_path}.")
        unique_values = sorted(list(pd.Series(series).dropna().unique())) # Sort for consistent ID assignment
        
        mapping = {pad_token: 0} # PAD is always 0
        current_id = 1

        if unknown_token not in unique_values and unknown_token != pad_token:
            mapping[unknown_token] = current_id
            current_id += 1
        
        for val in unique_values:
            if val == pad_token: # Already handled
                continue
            if val == unknown_token and unknown_token in mapping: # Already handled
                continue
            if val not in mapping: # Add new values
                mapping[val] = current_id
                current_id += 1
        
        # Ensure unknown_token is in, if it wasn't in unique_values and wasn't pad_token
        if unknown_token not in mapping and unknown_token != pad_token:
             mapping[unknown_token] = len(mapping) # Assign it the next available ID

        with open(mapping_file_path, 'w') as f:
            json.dump(mapping, f, indent=4)
        print(f"  Saved new/updated mapping with {len(mapping)} entries to {mapping_file_path}")
        
    elif mode == 'test':
        if not os.path.exists(mapping_file_path):
            raise FileNotFoundError(f"ERROR (test mode): Mapping file {mapping_file_path} not found. Test mode requires pre-existing mappings from a training run.")
        print(f"  Loading mapping for test mode: {mapping_file_path}")
        with open(mapping_file_path, 'r') as f:
            mapping = json.load(f)
        # Basic validation for test mode (PAD should be 0)
        if pad_token not in mapping or mapping[pad_token] != 0:
            print(f"WARNING (test mode): Loaded mapping from {mapping_file_path} does not have '{pad_token}' as ID 0. This might cause issues.")
        if unknown_token not in mapping:
            print(f"WARNING (test mode): Loaded mapping from {mapping_file_path} does not contain '{unknown_token}'. This might cause issues.")
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'test'.")
    return mapping

def load_and_parse_raw_json(json_file_path):
    """Loads raw JSON data and extracts the list of actions."""
    print(f"Loading raw JSON data from: {json_file_path}")
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        if "actions" in data and isinstance(data["actions"], list):
            actions_list = data["actions"]
            print(f"Loaded {len(actions_list)} actions from JSON file.")
            return actions_list
        else:
            print("Error: JSON file does not contain an 'actions' list.")
            return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file_path}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading {json_file_path}: {e}")
        return []

def preprocess_actions_for_generative_model(actions_list, artifacts_dir, mode='train'):
    """
    Main function to process a list of action dictionaries into a flat feature DataFrame
    according to the generative model schema (hardcoded logic for V1 based on schema doc).
    This version focuses on flattening and initial feature extraction.
    Mappings, hashing, and scaling will be done in subsequent dedicated functions.
    """
    print(f"Starting preprocessing (flattening & initial extraction) for generative model. Mode: {mode}. Actions: {len(actions_list)}")
    
    processed_data_list = []
    for idx, action in enumerate(actions_list):
        flat_features = {}

        # --- Global Action Features ---
        flat_features['action_date_raw'] = action.get('date') # Keep as ns string for now
        flat_features['action_height_raw'] = action.get('height') # Keep as string for now
        flat_features['action_status_raw'] = action.get('status', UNKNOWN_TOKEN_STR) # Default to UNKNOWN
        flat_features['action_type_raw'] = action.get('type', UNKNOWN_TOKEN_STR) # Default to UNKNOWN
        
        pools = action.get('pools', [])
        flat_features['pool1_asset_raw'] = pools[0] if len(pools) > 0 else NO_POOL_ASSET_STR
        flat_features['pool2_asset_raw'] = pools[1] if len(pools) > 1 else NO_POOL_ASSET_STR

        # --- Inbound Transaction Features (First `in` entry) ---
        in_tx_list = action.get('in', [])
        in_tx = in_tx_list[0] if in_tx_list else {}
        
        flat_features['in_tx_id_present_flag'] = 1 if in_tx.get('txID') else 0
        # flat_features['in_tx_id_raw'] = in_tx.get('txID', '') # Not directly used for V1, presence flag is enough
        
        flat_features['in_address_present_flag'] = 1 if in_tx.get('address') else 0
        flat_features['in_address_raw'] = in_tx.get('address', NO_ADDRESS_STR) # For hashing later
        
        in_memo = in_tx.get('memo')
        if in_memo is None:
            flat_features['in_memo_status_raw'] = "NO_MEMO"
        elif in_memo == "":
            flat_features['in_memo_status_raw'] = "EMPTY_MEMO"
        else:
            flat_features['in_memo_status_raw'] = "NON_EMPTY_MEMO"

        in_coins_list = in_tx.get('coins', [])
        in_coin1 = in_coins_list[0] if in_coins_list else {}
        flat_features['in_coin1_present_flag'] = 1 if in_coin1 else 0
        flat_features['in_coin1_asset_raw'] = in_coin1.get('asset', NO_ASSET_STR)
        flat_features['in_coin1_amount_raw'] = in_coin1.get('amount', '0') # Keep as string for now
        # flat_features['in_coin1_affiliate_present_flag'] will be primarily from metadata.swap for V1

        # --- Outbound Transaction Features (First `out` entry) ---
        out_tx_list = action.get('out', [])
        out_tx = out_tx_list[0] if out_tx_list else {}
        
        flat_features['out_tx_id_present_flag'] = 1 if out_tx.get('txID') else 0
        # flat_features['out_tx_id_raw'] = out_tx.get('txID', '') # V1: presence flag

        flat_features['out_address_present_flag'] = 1 if out_tx.get('address') else 0
        flat_features['out_address_raw'] = out_tx.get('address', NO_ADDRESS_STR) # For hashing later

        out_coins_list = out_tx.get('coins', [])
        out_coin1 = out_coins_list[0] if out_coins_list else {}
        flat_features['out_coin1_present_flag'] = 1 if out_coin1 else 0
        flat_features['out_coin1_asset_raw'] = out_coin1.get('asset', NO_ASSET_STR)
        flat_features['out_coin1_amount_raw'] = out_coin1.get('amount', '0') # Keep as string

        # --- Metadata Features (Flattened, default to 0/PAD/NO_ASSET if not applicable action_type) ---
        metadata = action.get('metadata', {})
        action_type_upper = str(action.get('type', '')).upper()

        # Swap Metadata
        flat_features['meta_is_swap_flag'] = 1 if action_type_upper == 'SWAP' else 0
        swap_meta = metadata.get('swap', {})
        flat_features['meta_swap_liquidityFee_raw'] = swap_meta.get('liquidityFee', '0') if flat_features['meta_is_swap_flag'] else '0'
        flat_features['meta_swap_swapSlip_bps_raw'] = swap_meta.get('swapSlip', '0') if flat_features['meta_is_swap_flag'] else '0' # Schema uses 'swapSlip_bps', API might be 'swapSlip'
        
        # Network fees in swap metadata are often in an array. For V1, take first if present.
        swap_network_fees = swap_meta.get('networkFees', []) 
        swap_network_fee1 = swap_network_fees[0] if swap_network_fees else {}
        flat_features['meta_swap_networkFee1_asset_raw'] = swap_network_fee1.get('asset', NO_ASSET_STR) if flat_features['meta_is_swap_flag'] else NO_ASSET_STR
        flat_features['meta_swap_networkFee1_amount_raw'] = swap_network_fee1.get('amount', '0') if flat_features['meta_is_swap_flag'] else '0'
        
        flat_features['meta_swap_target_asset_raw'] = swap_meta.get('targetAsset', NO_ASSET_STR) if flat_features['meta_is_swap_flag'] else NO_ASSET_STR
        
        affiliate_address = swap_meta.get('affiliateAddress')
        flat_features['meta_swap_affiliate_address_present_flag'] = 1 if flat_features['meta_is_swap_flag'] and affiliate_address else 0
        flat_features['meta_swap_affiliate_address_raw'] = affiliate_address if flat_features['meta_swap_affiliate_address_present_flag'] else NO_ADDRESS_STR
        flat_features['meta_swap_affiliateFee_amount_raw'] = swap_meta.get('affiliateFee', '0') if flat_features['meta_swap_affiliate_address_present_flag'] else '0'

        swap_memo = swap_meta.get('memo')
        if not flat_features['meta_is_swap_flag'] or swap_memo is None:
            flat_features['meta_swap_memo_status_raw'] = "NO_MEMO"
        elif swap_memo == "":
            flat_features['meta_swap_memo_status_raw'] = "EMPTY_MEMO"
        else:
            flat_features['meta_swap_memo_status_raw'] = "NON_EMPTY_MEMO"

        flat_features['meta_swap_is_streaming_flag'] = 1 if flat_features['meta_is_swap_flag'] and swap_meta.get('isStreamingSwap') else 0
        flat_features['meta_swap_streaming_count_raw'] = str(swap_meta.get('streamingSwapCount', '0')) if flat_features['meta_swap_is_streaming_flag'] else '0' # API might be streamingSwap.count
        flat_features['meta_swap_streaming_quantity_raw'] = swap_meta.get('streamingSwapQuantity', '0') if flat_features['meta_swap_is_streaming_flag'] else '0'

        # AddLiquidity Metadata
        flat_features['meta_is_addLiquidity_flag'] = 1 if action_type_upper == 'ADDLIQUIDITY' else 0
        add_lp_meta = metadata.get('addLiquidity', {})
        flat_features['meta_addLiquidity_units_raw'] = add_lp_meta.get('liquidityUnits', '0') if flat_features['meta_is_addLiquidity_flag'] else '0'

        # Withdraw Metadata
        flat_features['meta_is_withdraw_flag'] = 1 if action_type_upper == 'WITHDRAW' or action_type_upper == 'WITHDRAWLIQUIDITY' else 0
        withdraw_meta = metadata.get('withdraw', {})
        flat_features['meta_withdraw_units_raw'] = withdraw_meta.get('liquidityUnits', '0') if flat_features['meta_is_withdraw_flag'] else '0'
        flat_features['meta_withdraw_basis_points_raw'] = withdraw_meta.get('basisPoints', '0') if flat_features['meta_is_withdraw_flag'] else '0' # Schema says basis_points, API might be basisPoints
        flat_features['meta_withdraw_asymmetry_raw'] = withdraw_meta.get('asymmetry', '0.0') if flat_features['meta_is_withdraw_flag'] else '0.0' # String float
        flat_features['meta_withdraw_imp_loss_protection_raw'] = withdraw_meta.get('ilProtection', '0') if flat_features['meta_is_withdraw_flag'] else '0' # Schema says imp_loss_protection, API might be ilProtection

        # Refund Metadata
        flat_features['meta_is_refund_flag'] = 1 if action_type_upper == 'REFUND' else 0
        refund_meta = metadata.get('refund', {})
        # Example: Schema indicates a `meta_refund_reason_status_raw` or similar. This would need parsing from API's refund details.
        # For V1, if detailed refund reason isn't consistently structured or easily categorizable, a simple presence flag or UNKNOWN for status.
        flat_features['meta_refund_reason_raw'] = refund_meta.get('reason', UNKNOWN_TOKEN_STR) if flat_features['meta_is_refund_flag'] else UNKNOWN_TOKEN_STR # Placeholder for now
        
        # THORName metadata (Example, if present)
        flat_features['meta_is_thorname_flag'] = 1 if action_type_upper == 'THORNAME' else 0
        # thorname_meta = metadata.get('thorname', {})
        # flat_features['meta_thorname_name_raw'] = thorname_meta.get('name', UNKNOWN_TOKEN_STR) if flat_features['meta_is_thorname_flag'] else UNKNOWN_TOKEN_STR
        # ... other thorname fields ...

        # Other action types (donate, switch, etc.) would have similar blocks if they have specific metadata
        # For now, their flags would be 0 and their specific metadata fields would get default '0'/NO_ASSET_STR/etc.

        processed_data_list.append(flat_features)
        
    df = pd.DataFrame(processed_data_list)
    print(f"Initial DataFrame created with {len(df)} rows and {len(df.columns)} columns from raw actions.")
    
    # Define the expected order of columns after this initial raw extraction step.
    # This helps ensure consistency before further processing and for debugging.
    # This list should align with the features defined in this flattening step.
    initial_raw_feature_columns = sorted(list(df.columns)) # For now, just sort them alphabetically
    df = df[initial_raw_feature_columns] # Reorder DF by these names
    
    print("Columns after initial flattening (alphabetical order for now):")
    for col_name in df.columns:
        print(f"  - {col_name}")

    # This function will eventually return more, like mappings and scaler, but for now, just the df.
    # The other parts (ID mapping, hashing, scaling) will be separate functions called in main().
    return df, initial_raw_feature_columns # Return df and the column order

def process_categorical_features_generative(df_input, artifacts_dir, mode='train'):
    """
    Processes categorical features in the DataFrame by:
    1. Applying ID mapping to low/mid cardinality categoricals (e.g., type, status, assets).
    2. Applying feature hashing to high cardinality categoricals (e.g., addresses).
    Returns the DataFrame with new processed feature columns and a list of these column names.
    """
    print(f"Processing categorical features for generative model. Mode: {mode}")
    df = df_input.copy()
    processed_categorical_column_names = []
    all_mappings_created = {}

    # --- 1. ID Mapping for Low/Mid Cardinality Categoricals ---
    # Define features to ID map and their corresponding raw column and mapping file suffix
    # The actual mapping filename will be like: args.dataset_name + mapping_file_suffix
    # For now, using the global DEFAULT_*_MAPPING_FILENAME_GENERATIVE constants
    
    id_map_configs = [
        {'raw_col': 'action_status_raw', 'id_col': 'action_status_id', 'map_file': DEFAULT_STATUS_MAPPING_FILENAME_GENERATIVE},
        {'raw_col': 'action_type_raw', 'id_col': 'action_type_id', 'map_file': DEFAULT_TYPE_MAPPING_FILENAME_GENERATIVE},
        {'raw_col': 'in_memo_status_raw', 'id_col': 'in_memo_status_id', 'map_file': DEFAULT_MEMO_STATUS_MAPPING_FILENAME_GENERATIVE},
        {'raw_col': 'meta_swap_memo_status_raw', 'id_col': 'meta_swap_memo_status_id', 'map_file': DEFAULT_MEMO_STATUS_MAPPING_FILENAME_GENERATIVE}, # Reuses memo status map
        # Asset columns will be handled together to create one asset_to_id map
        # {'raw_col': 'meta_refund_reason_raw', 'id_col': 'meta_refund_reason_id', 'map_file': "refund_reason_to_id_generative_thorchain.json"}, # Example if needed
    ]

    for config in id_map_configs:
        raw_col, id_col, map_file_name = config['raw_col'], config['id_col'], config['map_file']
        mapping_file_path = os.path.join(artifacts_dir, map_file_name)
        
        series_for_mapping = df[raw_col] if mode == 'train' else None
        current_mapping = get_or_create_mapping_generative(mapping_file_path, series_for_mapping, mode)
        all_mappings_created[map_file_name] = current_mapping
        
        # Use UNKNOWN_TOKEN_STR's ID for any unmappable values in test mode or if not in train series
        unknown_id = current_mapping.get(UNKNOWN_TOKEN_STR)
        if unknown_id is None: # Should not happen if get_or_create_mapping_generative works correctly
            print(f"WARNING: UNKNOWN_TOKEN_STR not found in mapping {map_file_name}. Using fallback ID 0 for {id_col}.")
            unknown_id = 0 
            
        df[id_col] = df[raw_col].map(current_mapping).fillna(unknown_id).astype(int)
        processed_categorical_column_names.append(id_col)
        print(f"  ID Mapped '{raw_col}' to '{id_col}'. Unique IDs: {df[id_col].nunique()}")

    # Special handling for ASSET ID mapping (consolidated)
    asset_raw_columns = [
        'pool1_asset_raw', 'pool2_asset_raw',
        'in_coin1_asset_raw', 'out_coin1_asset_raw',
        'meta_swap_networkFee1_asset_raw', 'meta_swap_target_asset_raw'
    ]
    asset_id_columns = [
        'pool1_asset_id', 'pool2_asset_id',
        'in_coin1_asset_id', 'out_coin1_asset_id',
        'meta_swap_networkFee1_asset_id', 'meta_swap_target_asset_id'
    ]
    
    asset_mapping_file_path = os.path.join(artifacts_dir, DEFAULT_ASSET_MAPPING_FILENAME_GENERATIVE)
    if mode == 'train':
        # Consolidate all unique asset strings from all relevant columns
        all_asset_values = pd.Series(dtype=str)
        for col in asset_raw_columns:
            if col in df:
                all_asset_values = pd.concat([all_asset_values, df[col][df[col] != NO_ASSET_STR]])
        unique_assets = all_asset_values.dropna().unique()
        asset_mapping = get_or_create_mapping_generative(asset_mapping_file_path, unique_assets, mode, is_asset_mapping=True)
    else: # Test mode
        asset_mapping = get_or_create_mapping_generative(asset_mapping_file_path, None, mode, is_asset_mapping=True)
    all_mappings_created[DEFAULT_ASSET_MAPPING_FILENAME_GENERATIVE] = asset_mapping
    
    pad_asset_id = asset_mapping.get(PAD_TOKEN_STR) # Should be 0
    unknown_asset_id = asset_mapping.get(UNKNOWN_TOKEN_STR)
    if unknown_asset_id is None: unknown_asset_id = pad_asset_id # Fallback

    for raw_col, id_col in zip(asset_raw_columns, asset_id_columns):
        if raw_col in df:
            # Map NO_ASSET_STR to pad_asset_id explicitly, others to their map or unknown_asset_id
            df[id_col] = df[raw_col].apply(lambda x: pad_asset_id if x == NO_ASSET_STR else asset_mapping.get(x, unknown_asset_id))
            df[id_col] = df[id_col].astype(int)
            processed_categorical_column_names.append(id_col)
            print(f"  ID Mapped Asset '{raw_col}' to '{id_col}'. Unique IDs: {df[id_col].nunique()}")
        else:
            print(f"Warning: Asset raw column '{raw_col}' not found in DataFrame for ID mapping.")

    # --- 2. Feature Hashing for High Cardinality Categoricals ---
    # Vocab size for hashing will be HASH_VOCAB_SIZE_ADDRESS, PAD_HASH_ID will be HASH_VOCAB_SIZE_ADDRESS (i.e. index N for vocab 0..N-1)
    PAD_ADDRESS_HASH_ID = HASH_VOCAB_SIZE_ADDRESS 
    PAD_AFFILIATE_HASH_ID = HASH_VOCAB_SIZE_AFFILIATE_ADDRESS

    hash_configs = [
        {'raw_col': 'in_address_raw', 'hash_col': 'in_address_hash_id', 'vocab_size': HASH_VOCAB_SIZE_ADDRESS, 'pad_id': PAD_ADDRESS_HASH_ID},
        {'raw_col': 'out_address_raw', 'hash_col': 'out_address_hash_id', 'vocab_size': HASH_VOCAB_SIZE_ADDRESS, 'pad_id': PAD_ADDRESS_HASH_ID},
        {'raw_col': 'meta_swap_affiliate_address_raw', 'hash_col': 'meta_swap_affiliate_address_hash_id', 'vocab_size': HASH_VOCAB_SIZE_AFFILIATE_ADDRESS, 'pad_id': PAD_AFFILIATE_HASH_ID},
    ]

    for config in hash_configs:
        raw_col, hash_col, vocab_size, pad_id_val = config['raw_col'], config['hash_col'], config['vocab_size'], config['pad_id']
        if raw_col in df:
            def hash_or_pad(value):
                if value == NO_ADDRESS_STR or pd.isna(value) or value == '':
                    return pad_id_val
                return mmh3.hash(str(value), seed=HASH_SEED) % vocab_size
            
            df[hash_col] = df[raw_col].apply(hash_or_pad).astype(int)
            processed_categorical_column_names.append(hash_col)
            print(f"  Feature Hashed '{raw_col}' to '{hash_col}'. Vocab Size: {vocab_size}. Unique IDs: {df[hash_col].nunique()}")
        else:
            print(f"Warning: Address raw column '{raw_col}' not found in DataFrame for hashing.")

    # --- 3. Collect Binary Flag Column Names (already created during flattening) ---
    binary_flag_columns = [col for col in df.columns if col.endswith('_flag') or col.endswith('_present_flag')]
    processed_categorical_column_names.extend(binary_flag_columns)
    print(f"  Collected {len(binary_flag_columns)} binary flag columns: {binary_flag_columns}")

    print(f"Total processed categorical/flag columns: {len(processed_categorical_column_names)}")
    return df, processed_categorical_column_names, all_mappings_created

def process_numerical_features_generative(df_input, scaler_path, mode='train'):
    """
    Processes numerical features by:
    1. Converting raw string numericals (dates, amounts, fees, etc.) to numeric types.
    2. Normalizing amounts by 1e8 where applicable.
    3. Scaling all processed numerical features using StandardScaler.
    Returns the DataFrame with new scaled numerical columns and a list of these column names.
    """
    print(f"Processing numerical features for generative model. Mode: {mode}")
    df = df_input.copy()
    processed_numerical_column_names = []
    
    # --- 1. Convert to Numeric & Normalize (where applicable) ---
    # Date/Height
    if 'action_date_raw' in df:
        df['action_date_unix'] = pd.to_numeric(df['action_date_raw'], errors='coerce') / 1_000_000_000 # ns to s
        processed_numerical_column_names.append('action_date_unix')
    if 'action_height_raw' in df:
        df['action_height_val'] = pd.to_numeric(df['action_height_raw'], errors='coerce')
        processed_numerical_column_names.append('action_height_val')

    # Columns needing 1e8 normalization (typically amounts, fees)
    # List all _raw columns that represent amounts that need 1e8 normalization
    amount_norm_cols_map = {
        'in_coin1_amount_raw': 'in_coin1_amount_norm',
        'out_coin1_amount_raw': 'out_coin1_amount_norm',
        'meta_swap_liquidityFee_raw': 'meta_swap_liquidityFee_norm',
        'meta_swap_networkFee1_amount_raw': 'meta_swap_networkFee1_amount_norm',
        'meta_swap_affiliateFee_amount_raw': 'meta_swap_affiliateFee_norm',
        'meta_swap_streaming_quantity_raw': 'meta_swap_streaming_quantity_norm',
        'meta_withdraw_imp_loss_protection_raw': 'meta_withdraw_imp_loss_protection_norm'
    }
    for raw_col, norm_col in amount_norm_cols_map.items():
        if raw_col in df:
            df[norm_col] = pd.to_numeric(df[raw_col], errors='coerce') / 1e8
            processed_numerical_column_names.append(norm_col)
        else:
            print(f"Warning: Amount column '{raw_col}' not found for 1e8 normalization.")

    # Other numericals (no 1e8 normalization, just convert to numeric)
    other_numeric_cols_map = {
        'meta_swap_swapSlip_bps_raw': 'meta_swap_swapSlip_bps_val',
        'meta_swap_streaming_count_raw': 'meta_swap_streaming_count_val',
        'meta_addLiquidity_units_raw': 'meta_addLiquidity_units_val',
        'meta_withdraw_units_raw': 'meta_withdraw_units_val',
        'meta_withdraw_basis_points_raw': 'meta_withdraw_basis_points_val',
        'meta_withdraw_asymmetry_raw': 'meta_withdraw_asymmetry_val' # Already float-like string
    }
    for raw_col, val_col in other_numeric_cols_map.items():
        if raw_col in df:
            df[val_col] = pd.to_numeric(df[raw_col], errors='coerce')
            processed_numerical_column_names.append(val_col)
        else:
            print(f"Warning: Numeric column '{raw_col}' not found for conversion.")
    
    # Ensure all listed processed_numerical_column_names actually exist, fill NaNs before scaling
    # This is important because if a raw col was missing, its processed version wouldn't be created.
    final_numerical_cols_to_scale = []
    for col_name in processed_numerical_column_names:
        if col_name in df:
            final_numerical_cols_to_scale.append(col_name)
        else:
            print(f"Warning: Column {col_name} was expected but not found in df before scaling. Skipping.")
    
    if not final_numerical_cols_to_scale:
        print("No numerical features to scale. Returning DataFrame as is from numerical processing.")
        return df, [] # Return empty list for scaled names

    print(f"Numerical columns to be scaled: {final_numerical_cols_to_scale}")

    # --- 2. Scale Numerical Features ---
    scaled_feature_names_result = [f + '_scaled' for f in final_numerical_cols_to_scale]

    if mode == 'train':
        scaler = StandardScaler()
        # Fill NaNs with 0 for fitting the scaler. Consider mean/median if more appropriate.
        df_to_scale = df[final_numerical_cols_to_scale].fillna(0).copy()
        scaler.fit(df_to_scale)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
        
        scaled_values = scaler.transform(df_to_scale) # Transform the same NaN-filled data
        # Assign back to new columns in the original DataFrame
        for i, feature_name in enumerate(final_numerical_cols_to_scale):
            df[feature_name + '_scaled'] = scaled_values[:, i]

    elif mode == 'test':
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"ERROR (test mode): Scaler file {scaler_path} not found.")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {scaler_path}")

        # Prepare DataFrame for transform, ensuring columns match scaler's expectations
        expected_features_from_scaler = getattr(scaler, 'feature_names_in_', None) or getattr(scaler, 'n_features_in_', None)
        if expected_features_from_scaler is None:
             print("Warning: Scaler does not have feature_names_in_ or n_features_in_. Cannot guarantee column order/count for transform.")
             # Fallback: use final_numerical_cols_to_scale, hoping it matches.
             df_to_transform = df[final_numerical_cols_to_scale].fillna(0).copy()
             if hasattr(scaler, 'n_features_in_') and df_to_transform.shape[1] != scaler.n_features_in_:
                 raise ValueError(f"Test data has {df_to_transform.shape[1]} features, but scaler was fit on {scaler.n_features_in_} features. Columns: {final_numerical_cols_to_scale}")
        else:
            if isinstance(expected_features_from_scaler, int): # If only n_features_in_ is available
                if len(final_numerical_cols_to_scale) != expected_features_from_scaler:
                    raise ValueError(f"Test data provides {len(final_numerical_cols_to_scale)} numerical features, but scaler expects {expected_features_from_scaler}.")
                # Assume final_numerical_cols_to_scale is in correct order
                df_to_transform = df[final_numerical_cols_to_scale].fillna(0).copy()
            else: # feature_names_in_ is available (list of names)
                df_to_transform = pd.DataFrame(columns=expected_features_from_scaler, index=df.index)
                for feature in expected_features_from_scaler:
                    if feature in df.columns:
                        df_to_transform[feature] = df[feature]
                    else:
                        print(f"Warning (test mode): Expected feature '{feature}' for scaler not found in test DataFrame. Filling with 0 for transform.")
                        df_to_transform[feature] = 0 
            df_to_transform = df_to_transform.fillna(0)

        scaled_values = scaler.transform(df_to_transform)

        # Assign back based on the order of columns used for transform
        cols_for_assignment = list(df_to_transform.columns)
        for i, feature_name in enumerate(cols_for_assignment):
            # Only create _scaled for originally intended columns (final_numerical_cols_to_scale)
            if feature_name in final_numerical_cols_to_scale: 
                 df[feature_name + '_scaled'] = scaled_values[:, i]
        
        # Ensure all expected scaled columns are present, even if with NaNs if source was missing from scaler
        for feature_name in final_numerical_cols_to_scale:
            if feature_name + '_scaled' not in df.columns:
                print(f"Warning: Scaled column {feature_name + '_scaled'} was not created during test mode. Adding as NaN.")
                df[feature_name + '_scaled'] = np.nan
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'test'.")

    # Print stats for verification
    for feature_name_orig in final_numerical_cols_to_scale:
        scaled_col_name = feature_name_orig + '_scaled'
        if scaled_col_name in df.columns:
            print(f"  Stats for '{scaled_col_name}': Min={df[scaled_col_name].min():.4f}, Max={df[scaled_col_name].max():.4f}, Mean={df[scaled_col_name].mean():.4f}, NaNs={df[scaled_col_name].isna().sum()}")
        else:
            print(f"  Scaled column '{scaled_col_name}' was not created (possibly due to missing original feature '{feature_name_orig}').")
            
    print(f"Numerical processing complete. {len(scaled_feature_names_result)} scaled features generated.")
    return df, scaled_feature_names_result

def generate_sequences_and_targets_generative(df_processed, seq_length, feature_columns_ordered):
    """
    Generates sequences and their corresponding targets for the generative model.
    The target for a sequence is the full feature vector of the next transaction.
    """
    print(f"Generating sequences of length {seq_length} for generative model...")
    X_sequences_list, Y_targets_list = [], []
    
    df_for_sequencing = df_processed[feature_columns_ordered].copy()
    
    num_samples = len(df_for_sequencing) - seq_length

    if num_samples < 0:
        print(f"Warning: Not enough data ({len(df_for_sequencing)} rows) to create sequences of length {seq_length}. Returning empty arrays.")
        return np.array([]), np.array([])

    for i in range(num_samples):
        X_seq = df_for_sequencing.iloc[i : i + seq_length].values
        Y_target_vec = df_for_sequencing.iloc[i + seq_length].values
        
        X_sequences_list.append(X_seq)
        Y_targets_list.append(Y_target_vec)

    X_sequences = np.array(X_sequences_list, dtype=np.float32)
    Y_targets = np.array(Y_targets_list, dtype=np.float32)
    
    print(f"Generated {len(X_sequences)} sequences.")
    print(f"X_sequences shape: {X_sequences.shape if X_sequences.size > 0 else 'empty'}")
    print(f"Y_targets shape: {Y_targets.shape if Y_targets.size > 0 else 'empty'}")
    return X_sequences, Y_targets

def main(args):
    print(f"--- Starting Data Preprocessing for GENERATIVE Model (Mode: {args.mode}) ---")

    # --- Path Setup ---
    input_json_path = os.path.join(args.data_dir, args.input_json)
    output_npz_path = os.path.join(args.processed_data_dir_generative, args.output_npz_generative)
    
    artifacts_dir = args.artifacts_dir_generative
    os.makedirs(artifacts_dir, exist_ok=True)

    scaler_path = os.path.join(artifacts_dir, args.scaler_filename_generative)
    model_config_path = os.path.join(artifacts_dir, args.model_config_filename_generative)
    
    os.makedirs(args.processed_data_dir_generative, exist_ok=True)

    if args.mode == 'train':
        if os.path.exists(output_npz_path):
            print(f"Removing old {output_npz_path} (train mode)")
            os.remove(output_npz_path)

    raw_actions_list = load_and_parse_raw_json(input_json_path)
    if not raw_actions_list:
        print("No actions loaded. Exiting.")
        return

    # Placeholder for the main new preprocessing function
    df_flattened, ordered_raw_feature_names_temp = preprocess_actions_for_generative_model(
        raw_actions_list, 
        artifacts_dir,
        args.mode
    )
    
    if df_flattened.empty:
        print("Preprocessing (flattening) returned an empty DataFrame. Cannot proceed.")
        return
        
    # For this intermediate step, the placeholder_feature_columns will be the ordered_raw_feature_names_temp
    # In the final version, this will be the list of *fully processed* (ID-mapped, hashed, scaled) feature names.
    placeholder_feature_columns_for_sequence_test = ordered_raw_feature_names_temp
    
    # --- TEMPORARY: Convert all selected raw columns to numeric for sequence generation test ---
    # This is a HACK. Real processing will create dedicated numerical columns.
    df_for_seq_test = df_flattened[placeholder_feature_columns_for_sequence_test].copy()
    for col in placeholder_feature_columns_for_sequence_test:
        # Attempt to convert to numeric, coercing errors. Then fill NaNs with 0.
        # This is very crude and only for testing the sequence generation flow.
        df_for_seq_test[col] = pd.to_numeric(df_for_seq_test[col], errors='coerce').fillna(0)
    # --- END TEMPORARY HACK ---

    if not placeholder_feature_columns_for_sequence_test:
        print("No features columns determined from flattening. Exiting.")
        return

    # ---> NEW: Call categorical processing function here
    df_categoricals_processed, processed_cat_col_names, all_mappings = process_categorical_features_generative(
        df_flattened, 
        artifacts_dir,
        args.mode
    )
    # For now, sequence generation test will still use the HACKED raw numeric columns.
    # Later, processed_cat_col_names will be combined with processed_num_col_names.
    # And df_categoricals_processed will be passed to numerical processing.

    # ---> NEW: Call numerical processing function here
    df_numericals_processed, processed_num_col_names_scaled = process_numerical_features_generative(
        df_categoricals_processed, # Pass the DataFrame that has categorical features processed
        scaler_path, # Pass the scaler path defined in main()
        args.mode
    )

    # ---> NEW: Combine all processed feature names for sequence generation
    # The order here MATTERS and must be consistent for training and inference.
    # Suggestion: categorical_ids, then hashed_ids, then binary_flags, then scaled_numericals.
    # processed_cat_col_names already contains ID-mapped, hashed, and flags in some order.
    # We need a defined final order based on schema for reproducibility.
    
    # For now, simple concatenation for testing, actual order should be based on schema.
    # This is still a placeholder for the true final ordered list of all features.
    final_ordered_feature_columns = processed_cat_col_names + processed_num_col_names_scaled
    if not final_ordered_feature_columns:
        print("No features available after categorical and numerical processing. Exiting.")
        return
    
    # Ensure all columns in final_ordered_feature_columns exist in df_numericals_processed
    # Fill any completely missing ones with 0s or NaNs (e.g. if a feature was never present in data)
    for col_final in final_ordered_feature_columns:
        if col_final not in df_numericals_processed.columns:
            print(f"Warning: Final expected feature '{col_final}' not in DataFrame. Adding as zeros.")
            df_numericals_processed[col_final] = 0 # Or np.nan, then fill before .values
            
    # Replace the old placeholder_feature_columns_for_sequence_test
    # HACK: Remove the old placeholder logic used for sequence testing
    # if not placeholder_feature_columns_for_sequence_test:
    #     print("No features columns determined from flattening. Exiting.")

    # 3. Generate Sequences and Targets
    #    Y_targets will be the full feature vector of the *next* transaction
    X_sequences, Y_targets = generate_sequences_and_targets_generative(
        df_numericals_processed, # Use the DataFrame that has all features processed
        SEQUENCE_LENGTH,
        final_ordered_feature_columns # This MUST be the final ordered list of all features
    )

    if X_sequences.size == 0 or Y_targets.size == 0:
        print("Sequence generation resulted in empty arrays. Nothing to save.")
    else:
        np.savez_compressed(
            output_npz_path,
            X_sequences=X_sequences,
            Y_targets=Y_targets,
            feature_columns_ordered=final_ordered_feature_columns # Save final ordered list
        )
        print(f"Sequences and targets saved to {output_npz_path}")

    if args.mode == 'train':
        print(f"Saving generative model configuration to {model_config_path}")
        model_config_generative = {
            'sequence_length': SEQUENCE_LENGTH,
            'num_features_total': len(final_ordered_feature_columns), # From actual processed features
            'feature_columns_ordered': final_ordered_feature_columns, # Actual ordered list
            'categorical_id_mapping_details': all_mappings, # Use the actual mappings
            'hashed_feature_details': { # Populate based on schema and HASH_VOCAB_SIZES
                'in_address_hash_id': HASH_VOCAB_SIZE_ADDRESS +1, # +1 to include PAD_ID
                'out_address_hash_id': HASH_VOCAB_SIZE_ADDRESS +1,
                'meta_swap_affiliate_address_hash_id': HASH_VOCAB_SIZE_AFFILIATE_ADDRESS +1
            },
            'scaler_path': scaler_path, # Path to the saved scaler
            'pad_token_str': PAD_TOKEN_STR,
            'unknown_token_str': UNKNOWN_TOKEN_STR,
            'no_asset_str': NO_ASSET_STR
        }

        with open(model_config_path, 'w') as f_config:
            json.dump(model_config_generative, f_config, indent=4)
        print("Generative model configuration saved (placeholder content).")

    print(f"--- Generative Preprocessing complete (Mode: {args.mode}). ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess transaction data for Generative AI model.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode of operation: 'train' to learn and save artifacts, 'test' to load pre-learned artifacts.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing the input JSON file.")
    parser.add_argument("--input-json", type=str, default=DEFAULT_INPUT_JSON,
                        help="Name of the input JSON file (e.g., transactions_data.json).")
    
    parser.add_argument("--processed-data-dir-generative", type=str, default=DEFAULT_PROCESSED_DATA_DIR_GENERATIVE,
                        help="Base directory to save output .npz file for generative model.")
    parser.add_argument("--output-npz-generative", type=str, default=DEFAULT_OUTPUT_NPZ_GENERATIVE,
                        help="Name of the output .npz file for generative model (e.g., sequences_generative_thorchain.npz).")
    
    parser.add_argument("--artifacts-dir-generative", type=str, default=os.path.join(DEFAULT_PROCESSED_DATA_DIR_GENERATIVE, "thorchain_artifacts_v1"),
                        help="Directory to save/load generative model artifacts (scaler, mappings, model_config_generative.json).")
    
    parser.add_argument("--scaler-filename-generative", type=str, default=DEFAULT_SCALER_FILENAME_GENERATIVE)
    parser.add_argument("--model-config-filename-generative", type=str, default=DEFAULT_MODEL_CONFIG_FILENAME_GENERATIVE)

    cli_args = parser.parse_args()
    main(cli_args) 