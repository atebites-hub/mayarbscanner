import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import os

# --- Configuration ---
INPUT_FILE = "data/simulated_historical_maya_transactions.csv"
SEQUENCE_LENGTH_M = 10  # Example sequence length

# --- Helper Functions ---
def create_sequences(data_df, sequence_length):
    """Creates sequences from the dataframe."""
    sequences = []
    labels = [] # Assuming we might want to predict the next step or a property of the sequence
    df_values = data_df.values
    
    if len(df_values) <= sequence_length:
        print(f"Data length ({len(df_values)}) is less than or equal to sequence length ({sequence_length}). Cannot create sequences.")
        return np.array([]), np.array([])

    for i in range(len(df_values) - sequence_length):
        sequences.append(df_values[i:i + sequence_length])
        # Example: predict something from the next step (e.g., delta_X of the trade following the sequence)
        # For now, let's just use a placeholder or the sequence itself if no clear label from problem desc.
        # labels.append(df_values[i + sequence_length]) 
    
    # For this task, the deliverable is "tensor sequences", so labels might not be strictly needed yet.
    # If labels are needed for a specific prediction task, this part should be adapted.
    return np.array(sequences)

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting data preprocessing for {INPUT_FILE}...")

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input data file not found at {INPUT_FILE}")
        print("Please run the historical_data.py script first to generate the data.")
        exit()

    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Successfully loaded data with {len(df)} records.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        exit()

    # Convert timestamp to something more usable for models if not already (e.g., numerical features)
    # For now, we'll drop it or convert to Unix timestamp if needed for sequence.
    # Let's convert to Unix timestamp then scale it.
    df['timestamp_unix'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 10**9
    df_processed = df.copy()

    # 2. Normalize Numerical Features
    numerical_features = ['amount_in', 'amount_out', 'delta_X', 'timestamp_unix']
    scaler = StandardScaler()
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    print("Normalized numerical features:", numerical_features)

    # 3. Embeddings for Categorical Features (Example for 'asset_in')
    # Note on arb_ID: Unique arb_IDs are not typically directly embedded for generalization.
    # Instead, features within the arb (like assets, pools) or properties derived from the ID (if structured)
    # would be embedded or used. For this simulation, we demonstrate embedding for 'asset_in'.
    
    all_assets = pd.concat([df_processed['asset_in'], df_processed['asset_out'], df_processed['source_pool'], df_processed['target_pool']]).unique()
    asset_to_idx = {asset: i for i, asset in enumerate(all_assets)}
    idx_to_asset = {i: asset for asset, i in asset_to_idx.items()}
    num_assets = len(all_assets)
    embedding_dim = 5 # Small embedding dimension for demonstration

    # Create an embedding layer (example)
    asset_embedding_layer = nn.Embedding(num_embeddings=num_assets, embedding_dim=embedding_dim)

    # Convert asset columns to indices
    df_processed['asset_in_idx'] = df_processed['asset_in'].map(asset_to_idx)
    df_processed['asset_out_idx'] = df_processed['asset_out'].map(asset_to_idx)
    # For simplicity, we'll use these indices directly. In a model, they'd feed into nn.Embedding.
    # To include embeddings directly in the DataFrame for now (as numeric values):
    # This is a conceptual step; in a PyTorch model, you'd pass indices to the embedding layer.
    # Here, we'll just use the indices as features, or one-hot encode them for simplicity without torch tensors in df.

    print(f"Created mappings for {num_assets} unique asset/pool identifiers.")
    print("Note: For a model, asset_in_idx, asset_out_idx etc. would be fed to an nn.Embedding layer.")
    print("The arb_ID column, being unique, is typically used as an identifier, not directly embedded for generalization in this manner.")

    # For sequence creation, select relevant features. Exclude original categoricals if indices are used.
    # Let's select numerical features and the new asset indices.
    # Dropping original timestamp, arb_ID, and original categorical asset/pool strings for the sequence data.
    features_for_sequence = numerical_features + ['asset_in_idx', 'asset_out_idx']
    # We might also want to map source_pool and target_pool to indices if they are used
    df_processed['source_pool_idx'] = df_processed['source_pool'].map(asset_to_idx).fillna(-1).astype(int) # fillna for safety
    df_processed['target_pool_idx'] = df_processed['target_pool'].map(asset_to_idx).fillna(-1).astype(int)
    features_for_sequence.extend(['source_pool_idx', 'target_pool_idx'])

    data_for_sequencing = df_processed[features_for_sequence].copy()
    # Handle any potential NaNs that might have arisen from mapping if new assets appear
    data_for_sequencing.fillna(0, inplace=True) # Simple imputation for demonstration

    print(f"Features selected for sequencing: {features_for_sequence}")
    print("Sample of data prepared for sequencing (first 5 rows):")
    print(data_for_sequencing.head())

    # 4. Build Sequences
    # The 'sequences' variable will hold the feature sequences.
    event_sequences = create_sequences(data_for_sequencing, SEQUENCE_LENGTH_M)

    if event_sequences.size > 0:
        # Convert to PyTorch tensors (as per deliverable: "outputting tensor sequences")
        tensor_sequences = torch.tensor(event_sequences, dtype=torch.float32)
        print(f"\nSuccessfully created sequences. Shape of tensor_sequences: {tensor_sequences.shape}")
        print("(Number of sequences, sequence length, number of features)")
        print("Sample of first sequence tensor:")
        print(tensor_sequences[0])
    else:
        print("\nNo sequences were generated.")

    print("\nData preprocessing finished.") 