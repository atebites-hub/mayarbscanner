import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import os
import pickle # Added for saving scaler

# Define the path for the data and for saving mappings/scalers
DATA_DIR = "data"
OUTPUT_DIR = "data/processed_ai_data"
HISTORICAL_DATA_FILE = os.path.join(DATA_DIR, "historical_24hr_maya_transactions.csv")
ASSET_MAPPING_FILE = os.path.join(OUTPUT_DIR, "asset_to_id.json")
TYPE_MAPPING_FILE = os.path.join(OUTPUT_DIR, "type_to_id.json")
STATUS_MAPPING_FILE = os.path.join(OUTPUT_DIR, "status_to_id.json")
SCALER_FILE = os.path.join(OUTPUT_DIR, "scaler.pkl") # Using pickle for scaler

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Selected features based on Task 1.6.1
SELECTED_FEATURES = [
    "date", "status", "type", "pools", "in_asset", "in_amount", 
    "out_asset", "out_amount", "swap_liquidity_fee", "swap_slip_bps", 
    "swap_network_fee_asset", "swap_network_fee_amount", "transaction_id"
]

# Features to be numerically scaled
NUMERICAL_FEATURES_TO_SCALE = [
    "in_amount_norm", "out_amount_norm", "swap_liquidity_fee_norm", 
    "swap_slip_bps_norm", "swap_network_fee_amount_norm", "timestamp_norm",
    "time_delta_norm" # Time difference between consecutive transactions
]


def load_data(file_path):
    """Loads historical transaction data."""
    try:
        df = pd.read_csv(file_path)
        # Convert amounts and fees from string to numeric, coercing errors
        amount_cols = ['in_amount', 'out_amount', 'swap_liquidity_fee', 'swap_network_fee_amount']
        for col in amount_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Date is a nanosecond timestamp string, convert to numeric (seconds)
        if 'date' in df.columns:
            df['date'] = pd.to_numeric(df['date'], errors='coerce') / 1e9 # nanoseconds to seconds
            df.sort_values(by='date', inplace=True) # Ensure data is sorted by time for sequence generation

        # Keep only selected features
        # Note: 'pools' handling (splitting, mapping) might be complex and is deferred.
        # For now, 'transaction_id' is kept for potential debugging/tracing but not for model input vector.
        # 'swap_network_fee_asset' will be mapped like other assets.
        existing_selected_features = [f for f in SELECTED_FEATURES if f in df.columns]
        df = df[existing_selected_features]
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return pd.DataFrame()

def get_or_create_mapping(df, column_name, mapping_file, reserved_tokens=None):
    """
    Creates a mapping from unique values in a column to integer IDs.
    Saves the mapping to a file if it doesn't exist, or loads it.
    `reserved_tokens` is a dict like {"PAD": 0, "UNK": 1}
    """
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        return mapping

    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found in DataFrame for mapping.")
        return {}

    unique_values = df[column_name].dropna().unique().tolist()
    
    mapping = {}
    next_id = 0

    if reserved_tokens:
        for token, token_id in reserved_tokens.items():
            mapping[token] = token_id
        next_id = max(reserved_tokens.values()) + 1

    for value in unique_values:
        if value not in mapping:
            mapping[str(value)] = next_id # Ensure keys are strings for JSON
            next_id += 1
    
    # Add an UNK token if not already reserved, for values not seen during training
    if "UNK" not in mapping and (not reserved_tokens or "UNK" not in reserved_tokens):
        mapping["UNK"] = next_id 
        next_id +=1

    # Add a PAD token if not already reserved, for padding sequences
    if "PAD" not in mapping and (not reserved_tokens or "PAD" not in reserved_tokens):
         mapping["PAD"] = next_id

    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=4)
    return mapping

def map_column_to_ids(df, column_name, mapping):
    """Applies the mapping to a column, using UNK_ID for unseen values."""
    unk_id = mapping.get("UNK")
    if unk_id is None: # Should have UNK from get_or_create_mapping
        # Fallback if UNK is somehow missing, though this is defensive.
        # The last ID in mapping is typically PAD or UNK.
        # This ensures some valid integer is used.
        unk_id = max(mapping.values()) if mapping else 0 


    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found for ID mapping.")
        return pd.Series([unk_id] * len(df), name=f"{column_name}_id")

    return df[column_name].apply(lambda x: mapping.get(str(x), unk_id))

def preprocess_features(df):
    """
    Main function to orchestrate feature engineering.
    Handles categorical and numerical features.
    """
    if df.empty:
        print("DataFrame is empty. Skipping preprocessing.")
        return df, None # Return empty df and no scaler

    processed_df = df.copy()

    # 1. Handle Assets (in_asset, out_asset, swap_network_fee_asset)
    # Collect all unique assets from relevant columns to create a unified mapping
    all_assets = pd.Series(dtype=str)
    asset_cols = ['in_asset', 'out_asset', 'swap_network_fee_asset']
    for col in asset_cols:
        if col in processed_df.columns:
            all_assets = pd.concat([all_assets, processed_df[col].dropna()])
    
    unique_assets_df = pd.DataFrame({ 'asset': all_assets.unique()})

    # Define reserved tokens for padding and unknown assets
    asset_reserved_tokens = {"PAD_ASSET": 0, "UNK_ASSET": 1}
    asset_to_id = get_or_create_mapping(unique_assets_df, 'asset', ASSET_MAPPING_FILE, reserved_tokens=asset_reserved_tokens)

    for col in asset_cols:
        if col in processed_df.columns:
            processed_df[f'{col}_id'] = map_column_to_ids(processed_df, col, asset_to_id)

    # 2. Handle Transaction Type
    type_reserved_tokens = {"PAD_TYPE": 0, "UNK_TYPE": 1}
    type_to_id = get_or_create_mapping(processed_df, 'type', TYPE_MAPPING_FILE, reserved_tokens=type_reserved_tokens)
    if 'type' in processed_df.columns:
        processed_df['type_id'] = map_column_to_ids(processed_df, 'type', type_to_id)

    # 3. Handle Transaction Status
    status_reserved_tokens = {"PAD_STATUS": 0, "UNK_STATUS": 1}
    status_to_id = get_or_create_mapping(processed_df, 'status', STATUS_MAPPING_FILE, reserved_tokens=status_reserved_tokens)
    if 'status' in processed_df.columns:
        processed_df['status_id'] = map_column_to_ids(processed_df, 'status', status_to_id)

    # 4. Handle Timestamps ('date' is already seconds since epoch)
    # Create 'timestamp_norm' (will be scaled later)
    if 'date' in processed_df.columns:
        processed_df['timestamp_norm'] = processed_df['date']
        # Calculate time delta from previous transaction (in seconds)
        # Fill first NA with 0 (or a mean/median if preferred, but 0 is simple for first element)
        processed_df['time_delta_norm'] = processed_df['date'].diff().fillna(0)


    # 5. Prepare numerical features for scaling
    # These are amounts and fees. They were converted to numeric in load_data.
    # We'll create new columns for normalized versions to keep originals if needed.
    # Note: Coalesce with 0 for missing financial values before scaling.
    numerical_original_cols = ['in_amount', 'out_amount', 'swap_liquidity_fee', 'swap_slip_bps', 'swap_network_fee_amount']
    for col in numerical_original_cols:
        if col in processed_df.columns:
            processed_df[f'{col}_norm'] = processed_df[col].fillna(0) 
        else: # Ensure column exists even if all values were NaN initially or column missing
             processed_df[f'{col}_norm'] = 0


    # Columns to be scaled (ensure they exist, even if all zeros)
    cols_to_scale_final_check = []
    for f_name in NUMERICAL_FEATURES_TO_SCALE:
        if f_name not in processed_df.columns:
            processed_df[f_name] = 0.0 # Initialize if missing, e.g. if original date was all NaNs
        cols_to_scale_final_check.append(f_name)


    # Apply Scaling
    scaler = None
    if not processed_df.empty and cols_to_scale_final_check:
        # Filter out columns that might be all zeros or have no variance if that causes issues for scaler.
        # For now, proceed directly. MinMaxScaler is robust to zeros.
        data_to_scale = processed_df[cols_to_scale_final_check].values
        if data_to_scale.size > 0 : # Check if there's actually data to scale
            scaler = MinMaxScaler() # Or StandardScaler()
            scaled_values = scaler.fit_transform(data_to_scale)
            processed_df[cols_to_scale_final_check] = scaled_values
            
            # Save the scaler
            # import pickle # Moved to top
            # with open(SCALER_FILE, 'wb') as f:
            #     pickle.dump(scaler, f)
            # print(f"Scaler saved to {SCALER_FILE}")
        else:
            print("No data available to scale. Scaler not fitted.")
    else:
        print("DataFrame empty or no numerical features to scale. Scaler not created.")


    # Define final feature vector for sequences (excluding original non-ID/normalized columns)
    # Order of features matters for the model input.
    # Keep 'transaction_id' in processed_df for now for sequence generation mapping, but exclude from model_feature_columns
    
    # At this point, processed_df contains original cols, _id cols, and _norm cols.
    # We need to select which ones go into the model.

    return processed_df, scaler

def generate_sequences(df, sequence_length, feature_columns, pad_value=0):
    """
    Generates sequences of a fixed length from the DataFrame.
    Each sequence is a list of feature vectors.
    If a transaction is part of a sequence that would extend beyond the
    DataFrame boundaries, it is padded at the beginning with feature vectors
    consisting of `pad_value`.

    Args:
        df (pd.DataFrame): DataFrame containing processed and sorted transaction data.
                           Must include columns specified in `feature_columns`.
        sequence_length (int): The desired length of each sequence (M).
        feature_columns (list): List of column names to include in the feature vector 
                                for each transaction in a sequence.
        pad_value (float/int): Value to use for padding shorter sequences.

    Returns:
        np.ndarray: A 3D NumPy array of shape (num_transactions, sequence_length, num_features).
                    Returns an empty array if input df is empty or has too few rows.
        list: A list of transaction_ids corresponding to the primary transaction of each sequence.
    """
    if df.empty or len(df) == 0:
        print("DataFrame is empty, cannot generate sequences.")
        return np.array([]), []

    # Ensure all feature columns are present
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing feature columns for sequence generation: {missing_cols}")
        return np.array([]), []

    # Extract the relevant features and convert to NumPy array for efficiency
    feature_data = df[feature_columns].values
    num_transactions, num_features = feature_data.shape

    # Initialize an array for sequences, filled with pad_value
    # Each row in this array will correspond to a transaction in the original df
    # and will contain the sequence of `sequence_length` preceding transactions (including itself)
    sequences = np.full((num_transactions, sequence_length, num_features), pad_value, dtype=np.float32)

    for i in range(num_transactions):
        start_index = max(0, i - sequence_length + 1)
        end_index = i + 1
        
        actual_sequence_data = feature_data[start_index:end_index]
        actual_len = len(actual_sequence_data)
        
        # Place actual data at the end of the sequence window
        sequences[i, sequence_length - actual_len:, :] = actual_sequence_data

    transaction_ids = []
    if 'transaction_id' in df.columns:
        transaction_ids = df['transaction_id'].tolist()
    else:
        # If no transaction_id, generate placeholder IDs or raise error
        # For now, return empty list if not present, though it should be for traceability.
        print("Warning: 'transaction_id' column not found in DataFrame for sequence mapping.")

    return sequences, transaction_ids

# Placeholder for main execution
if __name__ == '__main__':
    raw_df = load_data(HISTORICAL_DATA_FILE)
    if not raw_df.empty:
        print(f"Loaded {len(raw_df)} transactions.")
        
        # Preprocess features
        processed_df, scaler = preprocess_features(raw_df)
        print("Feature preprocessing complete.")
        
        if scaler:
            print(f"Scaler information: Min={scaler.min_}, Scale={scaler.scale_}")

        # Define which columns from processed_df will form the feature vector for each step in a sequence
        # These should be the _id and _norm columns.
        # Order matters for consistency.
        model_feature_columns = [
            'in_asset_id', 'out_asset_id', 'swap_network_fee_asset_id', 
            'type_id', 'status_id',
            'in_amount_norm', 'out_amount_norm', 
            'swap_liquidity_fee_norm', 'swap_slip_bps_norm', 
            'swap_network_fee_amount_norm',
            'timestamp_norm', 'time_delta_norm'
        ]
        
        # Ensure all selected model feature columns exist in processed_df, fill with 0 if not (e.g. if a category had no values)
        for col in model_feature_columns:
            if col not in processed_df.columns:
                print(f"Warning: Model feature column '{col}' not found after preprocessing. Adding it as zeros.")
                processed_df[col] = 0 # Or a PAD_VALUE if more appropriate

        # Filter to only essential columns needed for sequence generation + model features + transaction_id
        cols_for_sequences = model_feature_columns + ['transaction_id'] # Keep tx_id if needed for context mapping
        
        # Check for missing columns again before slicing
        final_cols_for_sequences = [col for col in cols_for_sequences if col in processed_df.columns]
        if 'transaction_id' not in final_cols_for_sequences and 'transaction_id' in processed_df.columns:
             final_cols_for_sequences.append('transaction_id') # ensure tx_id is there if it exists at all
        
        sequencing_df = processed_df[final_cols_for_sequences].copy()

        print("\nProcessed DataFrame sample for sequence generation:")
        print(sequencing_df.head())
        print("\nData types of sequencing_df:")
        print(sequencing_df.dtypes)

        # Implement sequence generation (Task 1.6.3)
        M = 10 # Sequence length
        sequences, sequence_tx_ids = generate_sequences(sequencing_df, sequence_length=M, feature_columns=model_feature_columns)
        
        # Task 1.6.4: Prepare Data for PyTorch Models (NumPy array is good)
        # Task 1.6.5: Save the processed data
        if sequences.size > 0:
            print(f"\nGenerated {sequences.shape[0]} sequences, each of length {sequences.shape[1]} with {sequences.shape[2]} features.")
            
            # Save sequences
            sequences_output_path = os.path.join(OUTPUT_DIR, "sequences.npy")
            np.save(sequences_output_path, sequences)
            print(f"Sequences saved to {sequences_output_path}")

            # Save corresponding transaction_ids (if any)
            if sequence_tx_ids:
                tx_ids_output_path = os.path.join(OUTPUT_DIR, "sequence_transaction_ids.json")
                with open(tx_ids_output_path, 'w') as f:
                    json.dump(sequence_tx_ids, f, indent=4)
                print(f"Sequence transaction IDs saved to {tx_ids_output_path}")
            
            # Save the scaler if it was created and fitted
            if scaler:
                with open(SCALER_FILE, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"Scaler saved to {SCALER_FILE}")
            else:
                print("Scaler was not created or fitted, so not saved.")

            print("\nSample of the first sequence:")
            print(sequences[0])
            if sequence_tx_ids:
                print(f"Transaction ID for the first sequence: {sequence_tx_ids[0]}")

        else:
            print("No sequences were generated.")

    else:
        print("Could not load data. Exiting.") 