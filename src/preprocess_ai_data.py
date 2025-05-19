print("EXECUTOR_AGENT_SANITY_CHECK_PRINT_TOP_OF_FILE_V3")
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import os
import pickle
from collections import Counter
import requests
import time
from datetime import datetime
from pycoingecko import CoinGeckoAPI
import argparse

# --- Directory Constants (can be overridden by args where applicable) ---
DEFAULT_DATA_DIR = "data"
DEFAULT_PROCESSED_DATA_DIR = os.path.join(DEFAULT_DATA_DIR, "processed_ai_data")
DEFAULT_MODELS_DIR = "models" # For model_config.json

# --- Default File Names (can be overridden by args) ---
DEFAULT_INPUT_CSV = "historical_24hr_maya_transactions.csv"
DEFAULT_OUTPUT_NPZ = "sequences_and_targets.npz"
DEFAULT_SCALER_FILENAME = "scaler.pkl"
DEFAULT_MODEL_CONFIG_FILENAME = "model_config.json"
DEFAULT_ASSET_MAPPING_FILENAME = "asset_to_id.json"
DEFAULT_TYPE_MAPPING_FILENAME = "type_to_id.json"
DEFAULT_STATUS_MAPPING_FILENAME = "status_to_id.json"
DEFAULT_ACTOR_TYPE_MAPPING_FILENAME = "actor_type_to_id.json"
DEFAULT_CG_PRICE_CACHE_FILENAME = "coingecko_price_cache.json"

# --- Stablecoin Maya Asset Names for CACAO Price Derivation ---
RECOGNIZED_STABLECOIN_MAYA_ASSETS = {
    'ETH.USDC-0XA0B86991C6218B36C1D19D4A2E9EB0CE3606EB48',
    'ETH.USDT-0XDAC17F958D2EE523A2206206994597C13D831EC7',
    'USDC.USDC',
    'USDT.USDT'
}

# --- CoinGecko Configuration ---
CG_CLIENT = CoinGeckoAPI()
MAYA_TO_COINGECKO_MAP = {
    'ETH.ETH': 'ethereum',
    'BTC.BTC': 'bitcoin',
    'KUJI.KUJI': 'kujira',
    'USDC.USDC': 'usd-coin',
    'USDT.USDT': 'tether',
    'THOR.RUNE': 'thorchain',
    'BTC/BTC': 'bitcoin',
    'ETH/ETH': 'ethereum',
    'DASH.DASH': 'dash',
    'DASH/DASH': 'dash',
    'ETH.USDC-0XA0B86991C6218B36C1D19D4A2E9EB0CE3606EB48': 'usd-coin',
    'ETH/USDC-0XA0B86991C6218B36C1D19D4A2E9EB0CE3606EB48': 'usd-coin',
    'ETH.USDT-0XDAC17F958D2EE523A2206206994597C13D831EC7': 'tether',
    'ETH/USDT-0XDAC17F958D2EE523A2206206994597C13D831EC7': 'tether',
    'XRD.XRD': 'radix',
    'MAYA.CACAO': '29996',
}

# Sequence length for model input
SEQUENCE_LENGTH = 10

# Number of top arbitrageurs to track (remains for now, though not directly used in seq gen)
TOP_K_ARBS = 10

SELECTED_FEATURES = [
    "date", "status", "type", "pools", "in_asset", "in_amount", 
    "out_asset", "out_amount", "swap_liquidity_fee", "swap_slip_bps", 
    "swap_network_fee_asset", "swap_network_fee_amount", "transaction_id",
    "memo_str", "affiliate_id"
]

NUMERICAL_FEATURES_TO_SCALE = [
    "in_amount_norm", "out_amount_norm", "swap_liquidity_fee_norm", 
    "swap_slip_bps",
    "swap_network_fee_amount_norm", 
    "maya_price_P_m",
    "coingecko_price_P_u",
    "coingecko_price_P_u_out_asset"
]

# Will be dynamically determined based on mappings dir and args
# CATEGORICAL_FEATURES_TO_MAP = {
# "status": STATUS_MAPPING_FILE,
# "type": TYPE_MAPPING_FILE,
# "in_asset": ASSET_MAPPING_FILE, 
# "out_asset": ASSET_MAPPING_FILE,
# "swap_network_fee_asset": ASSET_MAPPING_FILE,
# "actor_type_id": None 
# }

UNK_ACTOR_ID = 3 # Corresponds to UNK_ACTOR in actor_type_to_id.json

def get_or_create_mapping(mapping_file_path, series=None, is_asset_mapping=False, default_value='UNKNOWN', mode='train'):
    if mode == 'train':
        if os.path.exists(mapping_file_path):
            print(f"Loading existing mapping: {mapping_file_path}")
            with open(mapping_file_path, 'r') as f:
                mapping = json.load(f)
        else:
            if series is None:
                raise ValueError(f"Mapping file {mapping_file_path} not found and no series provided to create it in train mode.")
            print(f"Creating new mapping: {mapping_file_path}")
            unique_values = series.unique()
            mapping = {val: i for i, val in enumerate(unique_values)}
            if default_value not in mapping:
                mapping[default_value] = len(mapping)
            # Asset mapping specific logic (CACAO ID 0) remains, ensure it's called correctly
            if is_asset_mapping and 'CACAO' not in mapping:
                temp_mapping = {'CACAO': 0}
                current_id = 1
                for val, id_val in mapping.items():
                    if val == 'CACAO': continue # Should not happen if CACAO not in mapping yet
                    temp_mapping[val] = current_id
                    current_id += 1
                mapping = temp_mapping
            elif is_asset_mapping and mapping.get('CACAO') != 0:
                # If CACAO exists but not ID 0, re-arrange
                cacao_id_old = mapping.pop('CACAO')
                temp_mapping = {'CACAO': 0}
                current_id = 1
                # Assign new IDs, ensuring CACAO's old ID is reused if something was at 0
                item_at_zero = None
                for k,v in mapping.items():
                    if v == 0:
                        item_at_zero = k
                        break
                
                for val, id_val in mapping.items():
                    if val == item_at_zero: # This item was at ID 0
                        temp_mapping[val] = cacao_id_old
                    else:
                        temp_mapping[val] = current_id
                        current_id +=1
                        if current_id == cacao_id_old and item_at_zero is not None: # Skip CACAO's old ID if it's being used
                            current_id +=1 
                mapping = temp_mapping

            with open(mapping_file_path, 'w') as f:
                json.dump(mapping, f, indent=4)
    elif mode == 'test':
        if not os.path.exists(mapping_file_path):
            raise FileNotFoundError(f"ERROR (test mode): Mapping file {mapping_file_path} not found. Test mode requires pre-existing mappings from a training run.")
        print(f"Loading mapping for test mode: {mapping_file_path}")
        with open(mapping_file_path, 'r') as f:
            mapping = json.load(f)
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'test'.")
    return mapping

def map_categorical_features(df, mappings_dir, mode='train'):
    print("Mapping categorical features...")
    
    # Define base categorical features and their respective mapping file names
    # actor_type_id is handled specially as it derives from actor_label
    categorical_features_config = {
        "status": DEFAULT_STATUS_MAPPING_FILENAME,
        "type": DEFAULT_TYPE_MAPPING_FILENAME,
        "in_asset": DEFAULT_ASSET_MAPPING_FILENAME,
        "out_asset": DEFAULT_ASSET_MAPPING_FILENAME,
        "swap_network_fee_asset": DEFAULT_ASSET_MAPPING_FILENAME
    }
    
    all_mapped_id_feature_names = []
    # Ensure asset mapping is loaded/created first if in train mode, or just loaded in test
    asset_mapping_path = os.path.join(mappings_dir, DEFAULT_ASSET_MAPPING_FILENAME)
    asset_mapping = get_or_create_mapping(asset_mapping_path, 
                                          pd.concat([df['in_asset'], df['out_asset'], df['swap_network_fee_asset']]).dropna().unique() if mode == 'train' else None, 
                                          is_asset_mapping=True, mode=mode)

    for feature, mapping_filename_template in categorical_features_config.items():
        mapping_file_path = os.path.join(mappings_dir, mapping_filename_template)
        current_mapping = asset_mapping if "asset" in feature else get_or_create_mapping(mapping_file_path, df[feature] if mode == 'train' else None, mode=mode)
        
        default_id = current_mapping.get('UNKNOWN')
        if default_id is None: 
            default_id = max(current_mapping.values()) + 1 if current_mapping else 0
            if mode == 'train': 
                print(f"Adding 'UNKNOWN' with ID {default_id} to mapping {mapping_file_path}")
                current_mapping['UNKNOWN'] = default_id
                with open(mapping_file_path, 'w') as f: json.dump(current_mapping, f, indent=4)
        
        df[feature + '_mapped'] = df[feature].map(current_mapping).fillna(default_id).astype(int)
        all_mapped_id_feature_names.append(feature + '_mapped')
        print(f"  Mapped '{feature}' to '{feature}_mapped'. Unique IDs: {df[feature + '_mapped'].nunique()}")

    # Handle actor_type_id separately
    actor_type_mapping_path = os.path.join(mappings_dir, DEFAULT_ACTOR_TYPE_MAPPING_FILENAME)
    # This mapping is static, so mode doesn't change creation, only ensures it exists or is correct
    actor_type_map = get_or_create_actor_type_mapping(actor_type_mapping_path) 
    df['actor_type_id_mapped'] = df['actor_type_id'].map(actor_type_map).fillna(actor_type_map.get('UNK_ACTOR')).astype(int)
    all_mapped_id_feature_names.append('actor_type_id_mapped')
    print(f"  Mapped 'actor_type_id' to 'actor_type_id_mapped'. Unique IDs: {df['actor_type_id_mapped'].nunique()}")
    
    return df, all_mapped_id_feature_names, asset_mapping # Return asset_mapping for vocab size

def get_or_create_actor_type_mapping(mapping_file_path):
    static_mapping = {'ARB_SWAP': 0, 'USER_SWAP': 1, 'NON_SWAP': 2, 'UNK_ACTOR': UNK_ACTOR_ID}
    if not os.path.exists(mapping_file_path):
        print(f"Creating static actor type mapping: {mapping_file_path}")
        with open(mapping_file_path, 'w') as f:
            json.dump(static_mapping, f, indent=4)
    else:
        with open(mapping_file_path, 'r') as f:
            mapping = json.load(f)
            # Ensure string keys from JSON are correctly compared if they were int-like
            mapping_corrected_keys = {str(k) if isinstance(k, int) else k: v for k,v in mapping.items()}
            if mapping_corrected_keys != static_mapping:
                print(f"Warning: Existing actor type mapping {mapping_file_path} ({mapping_corrected_keys}) differs from static definition ({static_mapping}). Overwriting.")
                with open(mapping_file_path, 'w') as f:
                    json.dump(static_mapping, f, indent=4)
                return static_mapping
        print(f"Loaded actor type mapping: {mapping_file_path}")
    return static_mapping

def determine_actor_type(row):
    tx_type = row.get('type')
    affiliate_id = row.get('affiliate_id') 

    if tx_type and isinstance(tx_type, str) and tx_type.upper() == 'SWAP':
        if pd.isna(affiliate_id) or str(affiliate_id).strip() == "":
            return 'ARB_SWAP'
        else:
            return 'USER_SWAP'
    elif tx_type:
        return 'NON_SWAP'
    else:
        return 'UNK_ACTOR'

def preprocess_features(df, cg_price_cache_path, mappings_dir):
    print("--- Entering preprocess_features ---")
    df['coingecko_price_P_u'] = np.nan
    df['coingecko_price_P_u_out_asset'] = np.nan
    
    price_cache = {}
    if os.path.exists(cg_price_cache_path):
        try:
            with open(cg_price_cache_path, 'r') as f_cache:
                loaded_cache_str_keys = json.load(f_cache)
            processed_cache = {}
            if isinstance(loaded_cache_str_keys, dict):
                for str_key, value in loaded_cache_str_keys.items():
                    try:
                        actual_key_list = json.loads(str_key)
                        if isinstance(actual_key_list, list) and len(actual_key_list) == 2:
                            processed_cache[tuple(actual_key_list)] = value
                        else:
                            print(f"Warning: Parsed cache key '{str_key}' is not a list of 2 elements. Skipping.")
                    except (json.JSONDecodeError, TypeError, ValueError) as e_parse:
                        print(f"Warning: Error parsing cache key '{str_key}'. Error: {e_parse}. Skipping.")
                price_cache = processed_cache
                print(f"Loaded and processed {len(price_cache)} items from CoinGecko cache: {cg_price_cache_path}")
            else:
                print(f"Warning: Content from {cg_price_cache_path} not a dict. Empty cache.")
        except json.JSONDecodeError:
            print(f"Warning: CoinGecko cache {cg_price_cache_path} corrupted. Empty cache.")
        except Exception as e:
            print(f"Warning: Could not load CoinGecko cache {cg_price_cache_path}. Error: {e}. Empty cache.")
    else:
        print(f"CoinGecko cache not found ({cg_price_cache_path}). Empty cache.")

    df['timestamp_raw'] = df['date'] 
    df['actor_label'] = df.apply(determine_actor_type, axis=1)
    actor_type_map = get_or_create_actor_type_mapping(os.path.join(mappings_dir, DEFAULT_ACTOR_TYPE_MAPPING_FILENAME))
    df['actor_type_id'] = df['actor_label'].map(actor_type_map).fillna(actor_type_map.get('UNK_ACTOR')).astype(int)
    print("Actor type distribution:")
    print(df['actor_label'].value_counts(normalize=True) * 100)
    
    df['maya_price_P_m'] = np.nan
    swap_filter = df['type'].str.upper() == 'SWAP'
    valid_swaps = swap_filter & df['in_amount_norm'].notna() & (df['in_amount_norm'] != 0) & df['out_amount_norm'].notna()
    df.loc[valid_swaps, 'maya_price_P_m'] = df.loc[valid_swaps, 'out_amount_norm'] / df.loc[valid_swaps, 'in_amount_norm']
    df['maya_price_P_m'].replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"DEBUG_PREPROC: Value counts of df['type'] BEFORE CoinGecko loop:\n{df['type'].value_counts()}")

    unique_asset_date_pairs = set()
    for index, row in df.iterrows():
        if pd.notna(row['in_asset']):
            # Convert nanosecond timestamp to seconds for datetime.fromtimestamp
            timestamp_in_seconds = row['timestamp_raw'] / 1_000_000_000
            unique_asset_date_pairs.add((row['in_asset'], datetime.fromtimestamp(timestamp_in_seconds).strftime('%d-%m-%Y')))
        if pd.notna(row['out_asset']):
            timestamp_in_seconds = row['timestamp_raw'] / 1_000_000_000
            unique_asset_date_pairs.add((row['out_asset'], datetime.fromtimestamp(timestamp_in_seconds).strftime('%d-%m-%Y')))

    cg_api_call_count = 0
    for maya_asset, date_str in unique_asset_date_pairs:
        cache_key = (maya_asset, date_str)
        if cache_key in price_cache:
            price = price_cache[cache_key]
        else:
            coingecko_id = MAYA_TO_COINGECKO_MAP.get(maya_asset)
            if coingecko_id:
                print(f"DEBUG_COINGECKO_LOOP: Attempt 1 for {coingecko_id} on {date_str}")
                price = fetch_coingecko_price_at_timestamp(coingecko_id, date_str) # Pass date_str directly
                cg_api_call_count += 1
                if price is not None:
                    price_cache[cache_key] = price
                    print(f"DEBUG_COINGECKO: Found price for {coingecko_id} ({maya_asset}) on {date_str}: ${price}")
                else:
                    print(f"DEBUG_COINGECKO: No price found for {coingecko_id} ({maya_asset}) on {date_str} after retries.")
            elif maya_asset == 'MAYA.CACAO': # Special handling for CACAO derivation if direct fetch fails or not mapped
                print(f"DEBUG_CACAO_PRICE: Attempting to derive price for {maya_asset} on {date_str} as it's not directly fetched or mapped for direct fetch.")
                # CACAO price derivation logic (simplified for this example, actual would be complex)
                # This is a placeholder, actual CACAO derivation via stablecoin pairs needs careful implementation if required
                # For now, if MAYA.CACAO ID '29996' fetch failed, we have no simple fallback here.
                # price = derive_cacao_price_via_stablecoins(df, date_str, price_cache, MAYA_TO_COINGECKO_MAP, CG_CLIENT)
                price = None # Placeholder
                if price is not None:
                    price_cache[cache_key] = price
                else:
                    print(f"DEBUG_CACAO_PRICE: Could not derive CACAO price for {date_str}.")
            else:
                price = None
                print(f"DEBUG_COINGECKO: No CoinGecko ID for Maya asset {maya_asset}")
        
        if price is not None:
            # Convert nanosecond timestamp to seconds for comparison
            mask_in = (df['in_asset'] == maya_asset) & (df['timestamp_raw'].apply(lambda ts_ns: datetime.fromtimestamp(ts_ns / 1_000_000_000).strftime('%d-%m-%Y')) == date_str)
            df.loc[mask_in, 'coingecko_price_P_u'] = price
            mask_out = (df['out_asset'] == maya_asset) & (df['timestamp_raw'].apply(lambda ts_ns: datetime.fromtimestamp(ts_ns / 1_000_000_000).strftime('%d-%m-%Y')) == date_str)
            df.loc[mask_out, 'coingecko_price_P_u_out_asset'] = price
    
    print(f"Made {cg_api_call_count} calls to CoinGecko API in this run.")
    try:
        cache_to_save = {json.dumps(list(k)): v for k, v in price_cache.items()} if price_cache else {}
        with open(cg_price_cache_path, 'w') as f_cache:
            json.dump(cache_to_save, f_cache, indent=4)
        print(f"Saved updated CoinGecko price cache ({len(cache_to_save)} items) to: {cg_price_cache_path}")
    except Exception as e:
        print(f"Warning: Could not save CoinGecko price cache to {cg_price_cache_path}. Error: {e}")

    print("--- P_m (Maya Price) Stats (End of preprocess_features) ---")
    print(df['maya_price_P_m'].describe())
    print("--- P_u (CoinGecko Price IN_ASSET) Stats ---")
    print(df['coingecko_price_P_u'].describe())
    print("--- P_u (CoinGecko Price OUT_ASSET) Stats ---")
    print(df['coingecko_price_P_u_out_asset'].describe())
    return df

def normalize_amounts(df, amount_cols=['in_amount', 'out_amount', 'swap_liquidity_fee', 'swap_network_fee_amount']):
    print("Normalizing amounts (dividing by 1e8)...")
    for col in amount_cols:
        if col in df.columns:
            df[col + '_norm'] = pd.to_numeric(df[col], errors='coerce') / 1e8
        else:
            print(f"Warning: Column {col} not found for normalization.")
            df[col + '_norm'] = np.nan # Ensure the column exists if it was expected
    return df

def scale_numerical_features(df, features_to_scale, scaler_path, mode='train'):
    print(f"Scaling numerical features: {features_to_scale} (Mode: {mode})")
    
    scaled_feature_names_result = [f + '_scaled' for f in features_to_scale]

    if mode == 'train':
        scaler = StandardScaler()
        existing_features_to_scale = [f for f in features_to_scale if f in df.columns]
        missing_features = [f for f in features_to_scale if f not in df.columns]
        if missing_features:
            print(f"Warning: Numerical features to scale missing from DataFrame: {missing_features}. They will be ignored for fitting scaler.")

        if not existing_features_to_scale:
            print("Error: No numerical features found to fit the scaler. Creating scaled columns with NaNs.")
            for feature in features_to_scale:
                df[feature + '_scaled'] = np.nan
            return df, [f + '_scaled' for f in features_to_scale]

        # Fill NaNs ONLY in the columns that will be used for fitting and transforming
        df_to_scale = df[existing_features_to_scale].fillna(0).copy() 
        
        scaler.fit(df_to_scale)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
        
        # Transform the same columns that were fitted
        scaled_values = scaler.transform(df_to_scale)
        # Assign back to new columns in the original DataFrame
        for i, feature_name in enumerate(existing_features_to_scale):
            df[feature_name + '_scaled'] = scaled_values[:, i]
        
        # For features that were originally requested but missing, create _scaled columns with NaN
        for feature_name in missing_features:
             df[feature_name + '_scaled'] = np.nan

    elif mode == 'test':
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"ERROR (test mode): Scaler file {scaler_path} not found.")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {scaler_path}")

        # In test mode, scaler.feature_names_in_ might tell us what it was trained on
        # We need to ensure the columns passed to transform match those
        # For simplicity, assume features_to_scale are the ones to use and they should exist
        # or have been handled by preprocessing if some are optional
        
        existing_features_to_transform = [f for f in features_to_scale if f in df.columns]
        # Create a DataFrame with the same columns the scaler was fit on, in the correct order.
        # scaler.feature_names_in_ holds the names from the fit stage.
        if not hasattr(scaler, 'feature_names_in_'):
             print("Warning: Scaler does not have feature_names_in_. Cannot guarantee column order for transform if features_to_scale is different from training.")
             # Fallback: use existing_features_to_transform, hoping order matches.
             # This could be risky if the set of columns in test data is different from train in an unexpected way.
             df_to_transform = df[existing_features_to_transform].fillna(0).copy()
             if df_to_transform.shape[1] != scaler.n_features_in_:
                 raise ValueError(f"Test data has {df_to_transform.shape[1]} features, but scaler was fit on {scaler.n_features_in_} features. Columns to transform: {existing_features_to_transform}")
        else:
            expected_features = scaler.feature_names_in_
            df_to_transform = pd.DataFrame(columns=expected_features, index=df.index)
            for feature in expected_features:
                if feature in df.columns:
                    df_to_transform[feature] = df[feature]
                else:
                    print(f"Warning (test mode): Expected feature '{feature}' for scaler not found in test DataFrame. Filling with 0 for transform.")
                    df_to_transform[feature] = 0 # Fill with 0 or handle as error
            df_to_transform = df_to_transform.fillna(0)

        scaled_values = scaler.transform(df_to_transform)

        # Assign back based on the order of expected_features (which should match scaler.feature_names_in_)
        # or existing_features_to_transform if feature_names_in_ was not available.
        cols_used_for_assignment = list(expected_features) if hasattr(scaler, 'feature_names_in_') else existing_features_to_transform
        for i, feature_name in enumerate(cols_used_for_assignment):
            if feature_name in features_to_scale: # Only create _scaled for originally requested ones
                 df[feature_name + '_scaled'] = scaled_values[:, i]
        
        # For any features in features_to_scale not covered (e.g. if not in scaler.feature_names_in_ but expected), fill with NaN
        for feature_name in features_to_scale:
            if feature_name + '_scaled' not in df.columns:
                df[feature_name + '_scaled'] = np.nan
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'test'.")

    # Print stats for verification
    for feature_name_orig in features_to_scale:
        scaled_col_name = feature_name_orig + '_scaled'
        if scaled_col_name in df.columns:
            print(f"  Stats for '{scaled_col_name}': Min={df[scaled_col_name].min():.4f}, Max={df[scaled_col_name].max():.4f}, Mean={df[scaled_col_name].mean():.4f}, NaNs={df[scaled_col_name].isna().sum()}")
        else:
            print(f"  Scaled column '{scaled_col_name}' was not created (possibly due to missing original feature).")
            
    return df, scaled_feature_names_result

def fetch_coingecko_price_at_timestamp(maya_asset_symbol, timestamp_seconds, retries=3, delay=5):
    """
    Fetches the historical price of a Maya asset from CoinGecko for a given timestamp.
    maya_asset_symbol: e.g., 'ETH.ETH', 'BTC.BTC'
    timestamp_seconds: Unix timestamp in seconds.
    Returns: Price in USD, or None if not found or error.
    """
    coingecko_asset_id = MAYA_TO_COINGECKO_MAP.get(maya_asset_symbol)
    if not coingecko_asset_id:
        print(f"DEBUG_COINGECKO: No CoinGecko ID mapping for Maya asset {maya_asset_symbol}")
        return None

    # # --- TEMPORARY DIAGNOSTIC: Skip thorchain --- 
    # if coingecko_asset_id == 'thorchain':
    #     print(f"DEBUG_COINGECKO_SKIP: Temporarily skipping API call for {coingecko_asset_id} ({maya_asset_symbol})")
    #     return None
    # # --- END TEMPORARY DIAGNOSTIC ---

    timestamp_in_seconds_cg = timestamp_seconds # Default if not needing conversion
    try:
        # Assuming timestamp_seconds might be in nanoseconds, try converting
        if timestamp_seconds > 10**12: # Heuristic: if it's a very large number, assume ns
            timestamp_in_seconds_cg = timestamp_seconds / 10**9
        
        date_obj = datetime.fromtimestamp(timestamp_in_seconds_cg)
        date_str = date_obj.strftime('%d-%m-%Y')
    except Exception as e:
        # print(f"DEBUG_COINGECKO: Error converting timestamp {timestamp_seconds} (raw) or {timestamp_in_seconds_cg} (converted) for {maya_asset_symbol}: {e}") # Silenced this specific error
        return None

    for attempt in range(retries):
        try:
            print(f"DEBUG_COINGECKO_LOOP: Attempt {attempt+1} for {coingecko_asset_id} on {date_str}")
            print(f"DEBUG_COINGECKO_PRE_CALL: Calling get_coin_history_by_id for {coingecko_asset_id} on {date_str}")
            historical_data = CG_CLIENT.get_coin_history_by_id(id=coingecko_asset_id, date=date_str, localization='false')
            print(f"DEBUG_COINGECKO_POST_CALL: Received response for {coingecko_asset_id}. Type: {type(historical_data)}. Data (first 100 chars): {str(historical_data)[:100]}")
            
            if (historical_data and 
                'market_data' in historical_data and 
                'current_price' in historical_data['market_data'] and 
                'usd' in historical_data['market_data']['current_price']):
                price_usd = historical_data['market_data']['current_price']['usd']
                print(f"DEBUG_COINGECKO: Found price for {coingecko_asset_id} ({maya_asset_symbol}) on {date_str}: ${price_usd}")
                if price_usd is None or price_usd <= 0:
                    print(f"DEBUG_COINGECKO: Price for {coingecko_asset_id} is null or zero: {price_usd}. Returning None.")
                    return None
                return float(price_usd)
            else:
                print(f"DEBUG_COINGECKO: Price data not found in expected format for {coingecko_asset_id} on {date_str}. Response: {historical_data}")
                if isinstance(historical_data, dict) and historical_data.get('status', {}).get('error_code') == 429:
                     print(f"DEBUG_COINGECKO: Rate limited. Waiting {delay * (attempt + 1)}s before retry...")
                     time.sleep(delay * (attempt + 1))
                elif attempt < retries - 1:
                    print(f"DEBUG_COINGECKO: Retrying after {delay}s...")
                    time.sleep(delay)
                else:
                    return None                    
        except Exception as e:
            print(f"DEBUG_COINGECKO: Error fetching price for {coingecko_asset_id} ({maya_asset_symbol}) on {date_str} (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    else:
                return None
    return None

def generate_targets(processed_df, actor_type_map):
    print("Starting target generation...")
    num_transactions = len(processed_df)
    
    # Target p: Actor type of the NEXT transaction
    # One-hot encoded: [ARB_SWAP, USER_SWAP, NON_SWAP, UNK_ACTOR]
    num_actor_classes_p = len(actor_type_map)
    processed_df['target_p'] = [list(np.zeros(num_actor_classes_p, dtype=int)) for _ in range(num_transactions)]
    
    # Target mu: Profitability of the NEXT transaction IF it's an ARB_SWAP
    # Profit = P_m(next_tx) - P_u(next_tx_in_asset_vs_USD)
    # This is a single float value, not one-hot. It's only valid if next tx is ARB_SWAP.
    processed_df['target_mu_profit'] = np.nan 

    arb_swap_id = actor_type_map.get('ARB_SWAP')
    if arb_swap_id is None: # Should not happen with static map
        raise ValueError("ARB_SWAP not found in actor_type_map during target generation.")

    for i in range(num_transactions - 1): # Iterate up to the second to last transaction
        current_tx_id_debug = processed_df.at[i, 'transaction_id']
        next_tx_actor_type_id = processed_df.at[i + 1, 'actor_type_id']
        next_tx_id_debug = processed_df.at[i+1, 'transaction_id']

        # --- P_target (actor type of next transaction) ---
        p_vector = np.zeros(num_actor_classes_p, dtype=int)
        if 0 <= next_tx_actor_type_id < num_actor_classes_p:
            p_vector[next_tx_actor_type_id] = 1
        else: # Should not happen if actor_type_id is mapped correctly
            p_vector[actor_type_map.get('UNK_ACTOR')] = 1 
        processed_df.at[i, 'target_p'] = list(p_vector)

        # --- Mu_target (profit of next transaction if it's an ARB_SWAP) ---
        if next_tx_actor_type_id == arb_swap_id:
            # Profit = P_m(next_tx) - P_u(next_tx_in_asset_vs_USD) for the *next* transaction
            # P_m(next_tx) is out_amount/in_amount for the *next* transaction's swap
            # P_u(next_tx) is CoinGecko price of *next* transaction's *in_asset* in USD
            
            # These are already pre-calculated and stored on the *next* row (i+1)
            p_m_next = processed_df.at[i + 1, 'maya_price_P_m']
            p_u_in_next = processed_df.at[i + 1, 'coingecko_price_P_u'] 
            p_u_out_next = processed_df.at[i + 1, 'coingecko_price_P_u_out_asset']

            # Add a threshold for p_m_next to avoid division by very small numbers or instability
            MIN_PM_THRESHOLD = 1e-9 

            if not pd.isna(p_m_next) and not pd.isna(p_u_in_next) and not pd.isna(p_u_out_next):
                # Ensure all components for the ratio are positive
                if p_u_in_next > 0 and p_m_next > MIN_PM_THRESHOLD and p_u_out_next > 0:
                    # Ratio = External_USD_Price_of_IN_Asset / Maya_Implied_USD_Price_of_IN_Asset
                    # Maya_Implied_USD_Price_of_IN_Asset = (1 / p_m_next) * p_u_out_next
                    # Ratio = p_u_in_next / ( (1 / p_m_next) * p_u_out_next )
                    # Ratio = (p_u_in_next * p_m_next) / p_u_out_next
                    
                    ratio_val = (p_u_in_next * p_m_next) / p_u_out_next
                    
                    if ratio_val > 0: # Argument of log must be positive
                        log_ratio_profit = np.log(ratio_val)
                        processed_df.at[i, 'target_mu_profit'] = log_ratio_profit
                        # print(f"DEBUG_LOG_RATIO_PROFIT (TX {current_tx_id_debug} for next {next_tx_id_debug}): "
                        #       f"p_u_in_next: {p_u_in_next:.4f}, p_m_next: {p_m_next:.4f}, p_u_out_next: {p_u_out_next:.4f}, "
                        #       f"Ratio: {ratio_val:.4f}, LogRatio: {log_ratio_profit:.4f}")
                    else:
                        processed_df.at[i, 'target_mu_profit'] = np.nan # Ratio non-positive, log undefined
                        # print(f"DEBUG_LOG_RATIO_PROFIT_SKIPPED_NON_POSITIVE_RATIO (TX {current_tx_id_debug} for next {next_tx_id_debug}): Ratio ({ratio_val}) non-positive.")
                else:
                    processed_df.at[i, 'target_mu_profit'] = np.nan # One of the price components is not positive
                    # print(f"DEBUG_LOG_RATIO_PROFIT_SKIPPED_NON_POSITIVE_PRICE (TX {current_tx_id_debug} for next {next_tx_id_debug}): "
                    #       f"p_u_in_next: {p_u_in_next}, p_m_next: {p_m_next}, p_u_out_next: {p_u_out_next}")
            # else: (This case means one of the base prices was NaN already)
                # target_mu_profit remains NaN by default
                # print(f"DEBUG_LOG_RATIO_PROFIT_SKIPPED_NAN_PRICE (TX {current_tx_id_debug} for next {next_tx_id_debug})")

    # For the last transaction, target_p is 'UNK_ACTOR' and target_mu is NaN
    if num_transactions > 0:
        p_vector_last = np.zeros(num_actor_classes_p, dtype=int)
        p_vector_last[actor_type_map.get('UNK_ACTOR')] = 1
        processed_df.at[num_transactions - 1, 'target_p'] = list(p_vector_last)

    print("Target generation completed (using log-ratio).")
    print("Stats for target_mu_profit (log-ratio):")
    print(processed_df['target_mu_profit'].describe())

    # REMOVED CLIPPING LOGIC as log-transform should stabilize the scale
    # if not processed_df['target_mu_profit'].isnull().all():
    #     lower_bound = processed_df['target_mu_profit'].quantile(0.05)
    #     upper_bound = processed_df['target_mu_profit'].quantile(0.95)
    #     print(f"Clipping target_mu_profit to range (5th-95th pctl): [{lower_bound:.4f}, {upper_bound:.4f}]")
    #     processed_df['target_mu_profit'] = processed_df['target_mu_profit'].clip(lower=lower_bound, upper=upper_bound)
    #     print("Stats for target_mu_profit AFTER clipping:")
    #     print(processed_df['target_mu_profit'].describe())
    # else:
    #     print("target_mu_profit is all NaN, skipping clipping.")

    return processed_df

def generate_sequences_separated(df, cat_id_feature_cols, num_feature_cols, target_p_col, target_mu_col, seq_length):
    print(f"Generating separated sequences of length {seq_length}...")
    X_cat_ids_list, X_num_list, y_p_targets_list, y_mu_targets_list = [], [], [], []
    num_samples = len(df) - seq_length

    if num_samples < 0:
        print(f"Warning: Not enough data ({len(df)} rows) to create sequences of length {seq_length}. Returning empty lists.")
        return np.array(X_cat_ids_list), np.array(X_num_list), np.array(y_p_targets_list), np.array(y_mu_targets_list)

    for i in range(num_samples):
        sequence_end_idx = i + seq_length - 1
        # target_idx = i + seq_length # Not directly used for selecting target data here, targets are from sequence_end_idx

        X_cat_seq_df = df.iloc[i:i + seq_length][cat_id_feature_cols]
        X_num_seq_df = df.iloc[i:i + seq_length][num_feature_cols]
        
        X_cat_ids_list.append(X_cat_seq_df.values.astype(np.int32)) # Ensure integer type for IDs
        X_num_list.append(X_num_seq_df.values.astype(np.float32))
        
        y_p_targets_list.append(df.at[sequence_end_idx, target_p_col])
        y_mu_targets_list.append(df.at[sequence_end_idx, target_mu_col])

    print(f"Generated {len(X_cat_ids_list)} sequences.")
    return np.array(X_cat_ids_list), np.array(X_num_list), np.array(y_p_targets_list), np.array(y_mu_targets_list)

def main(args):
    print(f"--- Starting Data Preprocessing for AI Model (Mode: {args.mode}) ---")

    # --- Path Setup ---
    input_csv_path = os.path.join(args.data_dir, args.input_csv)
    output_npz_path = os.path.join(args.processed_data_dir, args.output_npz)
    scaler_path = os.path.join(args.artifacts_dir, args.scaler_filename)
    mappings_dir = args.artifacts_dir # Mappings will be saved here
    model_config_path = os.path.join(args.artifacts_dir, args.model_config_filename)
    cg_price_cache_path = os.path.join(args.processed_data_dir, DEFAULT_CG_PRICE_CACHE_FILENAME)

    # Ensure directories exist
    os.makedirs(args.processed_data_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    if args.mode == 'train':
        if os.path.exists(output_npz_path):
            print(f"Removing old {output_npz_path} (train mode)")
            os.remove(output_npz_path)
        # In train mode, we might also remove old scaler/mappings if a full retrain is implied
        # For now, get_or_create_mapping and scale_numerical_features handle overwrite/creation logic
    
    print(f"Loading data from: {input_csv_path}")
    raw_df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(raw_df)} transactions from {input_csv_path}.")

    # Initial feature selection and basic normalization
    raw_df = raw_df[SELECTED_FEATURES].copy()
    df_normalized = normalize_amounts(raw_df.copy()) # Pass a copy

    # Feature engineering (P_m, P_u, actor type, etc.)
    df_features = preprocess_features(df_normalized.copy(), cg_price_cache_path, mappings_dir) # Pass a copy and paths

    # Map categorical features to IDs
    # The asset_mapping is returned to help build model_config later
    df_mapped, categorical_id_feature_columns, asset_mapping_loaded = map_categorical_features(df_features.copy(), mappings_dir, args.mode)

    # Scale numerical features
    df_scaled, numerical_feature_columns_scaled = scale_numerical_features(
        df_mapped.copy(), 
        NUMERICAL_FEATURES_TO_SCALE, 
        scaler_path, 
        args.mode
    )

    # Generate targets (p_target for actor type, mu_target for profit)
    actor_type_map_for_targets = get_or_create_actor_type_mapping(os.path.join(mappings_dir, DEFAULT_ACTOR_TYPE_MAPPING_FILENAME))
    df_targets = generate_targets(df_scaled.copy(), actor_type_map_for_targets)

    print(f"Number of categorical ID feature columns for sequence: {len(categorical_id_feature_columns)}")
    print(f"Categorical ID feature columns: {categorical_id_feature_columns}")
    print(f"Number of numerical feature columns for sequence: {len(numerical_feature_columns_scaled)}")
    print(f"Numerical feature columns: {numerical_feature_columns_scaled}")

    # Generate sequences
    X_cat_ids_sequences, X_num_sequences, y_p_targets, y_mu_targets = generate_sequences_separated(
        df_targets, 
        categorical_id_feature_columns, 
        numerical_feature_columns_scaled, 
        'target_p', 
        'target_mu_profit', 
        SEQUENCE_LENGTH
    )

    print(f"Generated {len(X_cat_ids_sequences)} sequences.")
    print(f"X_cat_ids_sequences shape: {X_cat_ids_sequences.shape}")
    print(f"X_num_sequences shape: {X_num_sequences.shape}")
    print(f"y_p_targets shape: {y_p_targets.shape}")
    print(f"y_mu_targets shape: {y_mu_targets.shape}")

    np.savez_compressed(
        output_npz_path,
        X_cat_ids_sequences=X_cat_ids_sequences,
        X_num_sequences=X_num_sequences,
        y_p_targets=y_p_targets,
        y_mu_targets=y_mu_targets,
        categorical_feature_columns=categorical_id_feature_columns, # Save for reference
        numerical_feature_columns=numerical_feature_columns_scaled # Save for reference
    )
    print(f"Sequences and targets saved to {output_npz_path}")

    if args.mode == 'train':
        print(f"Saving model configuration to {model_config_path}")
        model_config = {
            'categorical_embedding_details': {},
            'num_numerical_features': len(numerical_feature_columns_scaled),
            'sequence_length': SEQUENCE_LENGTH,
            # Add other model architecture params if they become dynamic
            'p_target_classes': y_p_targets.shape[1] # Number of classes for p_target
        }
        # Get vocab sizes from mappings
        # Asset mapping is already loaded as asset_mapping_loaded
        model_config['categorical_embedding_details'][DEFAULT_ASSET_MAPPING_FILENAME] = len(asset_mapping_loaded)
        
        other_mappings_configs = {
            "status": DEFAULT_STATUS_MAPPING_FILENAME,
            "type": DEFAULT_TYPE_MAPPING_FILENAME,
            "actor_type_id": DEFAULT_ACTOR_TYPE_MAPPING_FILENAME # actor_type_id_mapped uses this
        }
        for map_key, map_file_name in other_mappings_configs.items():
            map_path = os.path.join(mappings_dir, map_file_name)
            with open(map_path, 'r') as f_map:
                loaded_map = json.load(f_map)
            model_config['categorical_embedding_details'][map_file_name] = len(loaded_map)

        with open(model_config_path, 'w') as f_config:
            json.dump(model_config, f_config, indent=4)
        print("Model configuration saved.")

    print(f"--- Preprocessing and sequence generation complete (Mode: {args.mode}). ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Maya Protocol transaction data for AI model.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode of operation: 'train' to learn and save artifacts, 'test' to load pre-learned artifacts.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing the input CSV file.")
    parser.add_argument("--input-csv", type=str, default=DEFAULT_INPUT_CSV,
                        help="Name of the input CSV file (e.g., training_transactions.csv).")
    parser.add_argument("--processed-data-dir", type=str, default=DEFAULT_PROCESSED_DATA_DIR,
                        help="Directory to save output .npz file and CoinGecko cache.")
    parser.add_argument("--output-npz", type=str, default=DEFAULT_OUTPUT_NPZ,
                        help="Name of the output .npz file (e.g., training_sequences.npz).")
    parser.add_argument("--artifacts-dir", type=str, default=DEFAULT_PROCESSED_DATA_DIR, # Default to processed_data_dir for simplicity
                        help="Directory to save/load scaler, mappings, and model_config.json.")
    # Specific artifact names can be overridden if needed, but keeping them fixed for now
    parser.add_argument("--scaler-filename", type=str, default=DEFAULT_SCALER_FILENAME, help="Scaler filename.")
    parser.add_argument("--model-config-filename", type=str, default=DEFAULT_MODEL_CONFIG_FILENAME, help="Model config filename.")

    cli_args = parser.parse_args()
    main(cli_args) 