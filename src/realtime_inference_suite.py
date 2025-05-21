import argparse
import json
import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn # Added for model definition
import math # Added for PositionalEncoding
import mmh3 # Added for hashing

# Assuming api_connections.py is accessible
from api_connections import fetch_recent_maya_actions

# --- Helper function for nested data extraction ---
def get_nested_value_from_df(row, path_str, default=None):
    """
    Safely extracts a value from a pandas Series (row) using a dot-separated path.
    Example path_str: 'in.0.coins.0.asset' would try to access row['in'][0]['coins'][0]['asset']
    """
    current_val = row
    try:
        parts = path_str.split('.')
        for part in parts:
            if isinstance(current_val, dict):
                current_val = current_val.get(part)
            elif isinstance(current_val, list):
                try:
                    idx = int(part)
                    if 0 <= idx < len(current_val):
                        current_val = current_val[idx]
                    else:
                        return default # Index out of bounds
                except ValueError:
                    return default # Part is not a valid list index
            else:
                return default # Cannot traverse further
            
            if current_val is None:
                return default # Path led to None
        return current_val
    except Exception:
        return default

# --- Global Constants (Placeholder - adjust as needed) ---
MODEL_CONTEXT_LENGTH = 25 # Default, will be overridden by loaded model_config if available
ACTIONS_PER_PAGE = 50 # Midgard API limit for /actions
API_DELAY_SECONDS = 0.5 # Delay between Midgard polling attempts
PREDICTION_POLL_INTERVAL_SECONDS = 60 # How often to check for a new actual transaction

# --- ASSET PRECISION HANDLING (Copied and updated from preprocess_ai_data.py) ---
# This map is CRUCIAL for correctly decoding model outputs (atomic amounts) back to float strings.
ASSET_PRECISIONS = {
    "ARB.DAI-0XDA10009CBD5D07DD0CECC66161FC93D7C9000DA1": 18, 
    "ARB.ETH": 18,
    "ARB.GLD-0XAFD091F140C21770F4E5D53D26B2859AE97555AA": 8, 
    "ARB.LEO-0X93864D81175095DD93360FFA2A529B8642F76A6E": 8, 
    "ARB.LINK-0XF97F4DF75117A78C1A5A0DBB814AF92458539FB4": 18,
    "ARB.PEPE-0X25D887CE7A35172C62FEBFD67A1856F20FAEBB00": 18,
    "ARB.TGT-0X429FED88F10285E61B12BDF00848315FBDFCC341": 8, 
    "ARB.USDC-0XAF88D065E77C8CC2239327C5EDB3A432268E5831": 6, 
    "ARB.USDT-0XFD086BC7CD5C481DCC9C85EBE478A1C0B69FCBB9": 6, 
    "ARB.WBTC-0X2F2A2543B76A4166549F7AAB2E75BEF0AEFC5B0F": 8, 
    "ARB.WSTETH-0X5979D7B546E38E414F7E9822514BE443A4800529": 18,
    "ARB.YUM-0X9F41B34F42058A7B74672055A5FAE22C4B113FD1": 8, 
    "BTC.BTC": 8, 
    "CACAO": 10, # For raw CACAO asset string if it appears, MAYA.CACAO is primary
    "DASH.DASH": 8, 
    "ETH.ETH": 18, 
    "ETH.MOCA-0X53312F85BBA24C8CB99CFFC13BF82420157230D3": 18,
    "ETH.PEPE-0X6982508145454CE325DDBE47A25D4EC3D2311933": 18,
    "ETH.USDC-0XA0B86991C6218B36C1D19D4A2E9EB0CE3606EB48": 6, 
    "ETH.USDT-0XDAC17F958D2EE523A2206206994597C13D831EC7": 6, 
    "KUJI.KUJI": 6,
    "KUJI.USK": 6, 
    "MAYA.CACAO": 10,
    "MAYA.MAYA": 4, 
    "THOR.RUNE": 8, 
    "XRD.XRD": 8,
}
DEFAULT_PRECISION = 8 # Default for any other assets encountered

def get_asset_precision(asset_string):
    """
    Returns the precision for a given asset string.
    Defaults to DEFAULT_PRECISION if the asset is not in ASSET_PRECISIONS.
    Handles NO_ASSET, PAD, UNKNOWN by returning 0 precision.
    """
    if not isinstance(asset_string, str):
        return DEFAULT_PRECISION
    # Handle special known non-asset strings first
    if asset_string.upper() in ["NO_ASSET", "PAD", "UNKNOWN"]:
        return 0 # These should not undergo 10**precision scaling
    return ASSET_PRECISIONS.get(asset_string.upper(), DEFAULT_PRECISION)
# --- END ASSET PRECISION HANDLING ---

# --- Model Definitions (Copied from src/model.py) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class GenerativeTransactionModel(nn.Module):
    def __init__(self,
                 model_config: dict, # This is the full model_config from JSON
                 embedding_dim_config: dict, # Separate config for embedding dimensions
                 d_model=256, # Default, can be overridden by model_config
                 nhead=8, # Default, can be overridden by model_config
                 num_encoder_layers=6, # Default, can be overridden by model_config
                 dim_feedforward=1024, # Default, can be overridden by model_config
                 dropout=0.1, # Default, can be overridden by model_config
                 max_seq_len=25): # Default, can be overridden by model_config
        super(GenerativeTransactionModel, self).__init__()

        # Override defaults with values from model_config if they exist
        self.d_model = model_config.get('d_model', d_model)
        self.nhead = model_config.get('nhead', nhead)
        self.num_encoder_layers = model_config.get('num_encoder_layers', num_encoder_layers)
        self.dim_feedforward = model_config.get('dim_feedforward', dim_feedforward)
        self.dropout = model_config.get('dropout', dropout)
        self.max_seq_len = model_config.get('sequence_length', max_seq_len) # Use sequence_length from config

        self.model_config_full = model_config # Store the full config
        self.embedding_dim_config = embedding_dim_config

        self.feature_columns_ordered = model_config['feature_columns_ordered']
        self.feature_processing_details = model_config['feature_processing_details'] # New source of truth

        self.embedders = nn.ModuleDict()
        self.feature_info_input = [] # Stores info for processing input sequence
        current_concat_dim_for_input_projection = 0

        # Extract specific dimensions from the provided embedding_dim_config
        default_cat_embed_dim = embedding_dim_config.get('default_cat_embed_dim', 16)
        asset_embed_dim = embedding_dim_config.get('asset_embed_dim', default_cat_embed_dim) 
        address_hash_embed_dim = embedding_dim_config.get('address_hash_embed_dim', default_cat_embed_dim)
        # action_type_embed_dim and action_status_embed_dim will be looked up directly if present, else default.
        
        self.total_output_dimension = 0
        self.feature_output_info = [] # Stores info for interpreting model output

        for idx, feature_name in enumerate(self.feature_columns_ordered):
            if feature_name not in self.feature_processing_details:
                raise ValueError(f"Feature '{feature_name}' not found in feature_processing_details of model_config.")
            
            f_details = self.feature_processing_details[feature_name]
            f_type = f_details['type']

            input_info = {'name': feature_name, 'index': idx, 'type': f_type}
            output_info_current_feature = {
                'name': feature_name,
                'original_input_index': idx,
                'type': f_type,
                'output_start_idx': self.total_output_dimension
            }

            if f_type == 'id_map':
                vocab_size = f_details['vocab_size']
                
                # Start with the general default, which itself comes from embedding_dim_config or a hardcoded 16
                embed_dim = default_cat_embed_dim 

                # Asset features override
                if 'asset' in feature_name.lower():
                    embed_dim = asset_embed_dim
                # Specific overrides based on keys present in embedding_dim_config
                # This logic should now accurately reflect src/model.py's behavior given the standard training config
                elif feature_name == 'action_type_id' and 'action_type_embed_dim' in embedding_dim_config:
                    embed_dim = embedding_dim_config['action_type_embed_dim']
                elif 'status' in feature_name.lower() and 'action_status_embed_dim' in embedding_dim_config:
                    # This covers 'action_status_id', 'in_memo_status_id', 'meta_swap_memo_status_id'
                    embed_dim = embedding_dim_config['action_status_embed_dim']
                # Note: If a feature like 'in_memo_status_id' does not trigger the 'status' clause above 
                # (e.g. if 'action_status_embed_dim' was missing from config) it would retain 'default_cat_embed_dim'.
                # However, 'action_status_embed_dim' IS in our standard training config.

                embed_key = f"embed_input_{feature_name}"
                self.embedders[embed_key] = nn.Embedding(vocab_size, embed_dim)
                input_info['embed_key'] = embed_key
                current_concat_dim_for_input_projection += embed_dim
                self.total_output_dimension += vocab_size
                output_info_current_feature['vocab_size'] = vocab_size

            elif f_type == 'hash_cat':
                vocab_size = f_details['hash_bins'] + 1 
                embed_key = f"embed_input_{feature_name}"
                current_hash_embed_dim = default_cat_embed_dim # Default for hashes not specified
                if 'address_hash' in feature_name.lower(): # Check if it's an address hash
                    current_hash_embed_dim = address_hash_embed_dim
                # Add elif for other specific hash types if they exist and have unique dims
                
                self.embedders[embed_key] = nn.Embedding(vocab_size, current_hash_embed_dim)
                input_info['embed_key'] = embed_key
                current_concat_dim_for_input_projection += current_hash_embed_dim
                self.total_output_dimension += vocab_size
                output_info_current_feature['vocab_size'] = vocab_size
            
            elif f_type == 'binary_flag':
                current_concat_dim_for_input_projection += 1 # Input is 0 or 1
                self.total_output_dimension += 1 # Output is a single logit
            
            elif f_type == 'numerical_scaled':
                current_concat_dim_for_input_projection += 1 # Input is a scaled float
                self.total_output_dimension += 1 # Output is a single predicted scaled float
            
            else:
                raise ValueError(f"Unknown feature type '{f_type}' for feature '{feature_name}' in feature_processing_details.")

            self.feature_info_input.append(input_info)
            output_info_current_feature['output_end_idx'] = self.total_output_dimension
            self.feature_output_info.append(output_info_current_feature)

        self.input_projection = nn.Linear(current_concat_dim_for_input_projection, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_len, dropout=self.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                   dim_feedforward=self.dim_feedforward, dropout=self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        self.output_projection = nn.Linear(self.d_model, self.total_output_dimension)

    def forward(self, x_sequence):
        batch_size, seq_len, _ = x_sequence.shape
        processed_features_for_input_concat = []

        for info in self.feature_info_input:
            feature_idx = info['index']
            feature_slice = x_sequence[:, :, feature_idx]

            # Corrected logic: id_map and hash_cat use embeddings
            if info['type'] == 'id_map' or info['type'] == 'hash_cat':
                embedded_slice = self.embedders[info['embed_key']](feature_slice.long())
                processed_features_for_input_concat.append(embedded_slice)
            # Corrected logic: binary_flag and numerical_scaled are passed as floats
            elif info['type'] == 'binary_flag' or info['type'] == 'numerical_scaled':
                processed_features_for_input_concat.append(feature_slice.float().unsqueeze(-1))
            else:
                 raise ValueError(f"Unknown feature type '{info['type']}' for feature '{info['name']}' in forward pass input processing.")
        
        concatenated_input_features = torch.cat(processed_features_for_input_concat, dim=-1)
        
        projected_input = self.input_projection(concatenated_input_features)
        encoded_input = self.pos_encoder(projected_input)
        transformer_output = self.transformer_encoder(encoded_input)
        
        last_step_output = transformer_output[:, -1, :]
        predicted_full_output_vector = self.output_projection(last_step_output)

        return predicted_full_output_vector
# --- End Model Definitions ---


def load_model_and_artifacts(model_path, artifacts_dir, model_config_filename):
    """
    Loads the trained model, its configuration, and processing artifacts.
    """
    print(f"Loading model and artifacts...")
    print(f"  Model path: {model_path}")
    print(f"  Artifacts directory: {artifacts_dir}")
    print(f"  Model config filename: {model_config_filename}")

    model_config_path = os.path.join(artifacts_dir, model_config_filename)
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")
    with open(model_config_path, 'r') as f:
        model_config_from_preprocessor = json.load(f)
    print(f"  Loaded model config from: {model_config_path}")

    # --- Load all_id_mappings from its separate file ---
    # Ensure artifacts_dir is correctly determined (it's an arg to this function)
    all_mappings_filename = "all_id_mappings_generative_mayachain_s25.json" # Must match what preprocess saved
    all_mappings_path = os.path.join(artifacts_dir, all_mappings_filename)
    all_mappings_loaded = {}
    try:
        with open(all_mappings_path, 'r') as f_map_load:
            all_mappings_loaded = json.load(f_map_load)
        print(f"  Successfully loaded all_id_mappings from {all_mappings_path}")
    except FileNotFoundError:
        print(f"  CRITICAL ERROR: All mappings file not found at {all_mappings_path}. Cannot proceed.")
        raise # Re-raise the exception to stop execution
    except Exception as e_map_load:
        print(f"  CRITICAL ERROR loading all_id_mappings from {all_mappings_path}: {e_map_load}. Cannot proceed.")
        raise # Re-raise to stop execution
    # --- End separate load ---

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Embedding dim config: Prioritize the one saved during training
    if 'embedding_dim_config_used' in model_config_from_preprocessor:
        embedding_config_for_model = model_config_from_preprocessor['embedding_dim_config_used']
        print(f"  Using embedding_dim_config_used from loaded model_config: {embedding_config_for_model}")
    else:
        # Fallback if 'embedding_dim_config_used' is somehow missing (should not happen with up-to-date training script)
        embedding_config_for_model = {
            'default_cat_embed_dim': 16, # A very basic default, might need adjustment
            'asset_embed_dim': 48,      # Updated based on error messages
            'address_hash_embed_dim': 64 # Updated based on error messages for address hashes
            # 'hash_embed_dim' was a more generic key, model uses 'address_hash_embed_dim' specifically if present
        }
        print(f"  Warning: 'embedding_dim_config_used' not found in model_config. Using an updated fallback based on error messages: {embedding_config_for_model}")
    # print(f"  Using embedding_dim_config: {embedding_config_for_model}") # Redundant, already printed specific source


    # Instantiate the model using the loaded config
    # The GenerativeTransactionModel __init__ will now use values from model_config_from_preprocessor
    # for d_model, nhead, num_encoder_layers, etc., or its own defaults if not found in config.
    model = GenerativeTransactionModel(
        model_config=model_config_from_preprocessor,
        embedding_dim_config=embedding_config_for_model
        # d_model, nhead, etc. are now primarily sourced from model_config_from_preprocessor within the class
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Model loaded successfully and set to evaluation mode on {device}.")

    # Load scaler
    scaler_path = model_config_from_preprocessor.get('scaler_path')
    if not scaler_path or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at path specified in model_config: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  Scaler loaded from: {scaler_path}")

    # MODEL_CONTEXT_LENGTH global can be updated from model_config
    global MODEL_CONTEXT_LENGTH
    MODEL_CONTEXT_LENGTH = model_config_from_preprocessor.get('sequence_length', MODEL_CONTEXT_LENGTH)
    print(f"  Global MODEL_CONTEXT_LENGTH updated to: {MODEL_CONTEXT_LENGTH}")

    return model, scaler, all_mappings_loaded, model_config_from_preprocessor, device


def preprocess_single_action_for_inference(raw_action_json, scaler, all_mappings, model_config, is_simulation_feedback=False):
    """
    Preprocesses a single raw transaction JSON into a 1D feature vector suitable for the model.
    This function mirrors the logic in src/preprocess_ai_data.py for a single data point.
    It uses model_config['feature_processing_details'] to guide the processing.
    """
    processed_features = {}
    feature_details = model_config['feature_processing_details']
    ordered_feature_names = model_config['feature_columns_ordered']
    
    # Create a pandas Series from the raw_action_json for easier value extraction via get_nested_value_from_df
    # However, get_nested_value_from_df expects a path like 'in.0.coins.0.asset'
    # The raw_json_path in feature_details is like "['in', 0, 'coins', 0, 'asset']"
    # We need a consistent way to extract values.

    # For now, let's assume raw_action_json is a dict and we'll use direct access based on raw_json_path
    # This part needs to be robust.

    temp_df_data = {} # To build a temporary single-row DataFrame for scaling

    for feature_name in ordered_feature_names:
        if feature_name not in feature_details:
            raise ValueError(f"Feature '{feature_name}' from ordered_columns not in feature_processing_details.")
        
        f_detail = feature_details[feature_name]
        raw_json_path_str = f_detail.get('raw_json_path', '') # Path like "['metadata','swap','targetAsset']"
        
        # Convert raw_json_path string to an actual list of path components
        # Example: "['in',0,'coins',0,'asset']" -> ['in', 0, 'coins', 0, 'asset']
        # This is a bit tricky due to mixed types (str, int).
        # A simplified get_value_by_path_list might be needed here if get_nested_value_from_df is not suitable
        
        # ---- Robust Value Extraction Logic (Placeholder - to be refined) ----
        # For now, this is highly simplified and relies on the exact structure of raw_json_path_str
        # and that raw_action_json can be navigated this way.
        current_val = raw_action_json
        extracted_raw_value = None
        is_present = True # Assume present unless extraction fails or logic dictates otherwise

        try:
            # Attempt to parse the path string like "['metadata','swap','targetAsset']"
            # This is a simplified parser, might not cover all edge cases of raw_json_path_str
            if is_simulation_feedback: # In simulation feedback, raw_action_json is already flat
                extracted_raw_value = raw_action_json.get(feature_name) # Get pre-filled value
                # For presence flags in simulation, if the source field (e.g. in_tx_id_raw) is part of the flat dict and not None, it implies presence.
                # However, the flags themselves (e.g. in_tx_id_present_flag) should be directly provided in raw_action_json.
                # So, is_present logic might rely on the flag value itself if provided.
                # If it's a 'present_flag', its value should already be in raw_action_json.
                if 'present_flag' in feature_name:
                    is_present = bool(extracted_raw_value) # if flag is 1, it's present
                else: # For other features, if value is None, consider it not present for logic flags
                    is_present = extracted_raw_value is not None
                current_val = None # Value is already in extracted_raw_value

            elif raw_json_path_str.startswith('[') and raw_json_path_str.endswith(']'):
                # Correctly parsing the list-like string path
                # Using json.loads for robust parsing of list-like strings
                try:
                    path_list = json.loads(raw_json_path_str.replace("'", "\"")) # Replace single with double quotes for JSON
                    for key_or_idx in path_list:
                        if isinstance(key_or_idx, str):
                            if isinstance(current_val, dict):
                                current_val = current_val.get(key_or_idx)
                            else:
                                is_present = False; break
                        elif isinstance(key_or_idx, int):
                            if isinstance(current_val, list) and 0 <= key_or_idx < len(current_val):
                                current_val = current_val[key_or_idx]
                            else:
                                is_present = False; break
                        else: # Should not happen with json.loads if path is well-formed
                            is_present = False; break
                        if current_val is None:
                            is_present = False; break 
                except json.JSONDecodeError:
                    # Fallback for non-JSON-parsable logic strings like "action.type == 'swap'"
                    if 'action.type' in raw_json_path_str: 
                        action_type_for_flag = raw_action_json.get('type', '').lower()
                        # Simplified: exact match on raw_column_name from f_detail to infer flag type
                        raw_col_name_for_flag_logic = f_detail.get('raw_column_name', '').lower()
                        if 'meta_is_swap_flag' in raw_col_name_for_flag_logic:
                            extracted_raw_value = 1 if action_type_for_flag == 'swap' else 0
                        elif 'meta_is_addliquidity_flag' in raw_col_name_for_flag_logic:
                            extracted_raw_value = 1 if action_type_for_flag == 'addliquidity' else 0
                        elif 'meta_is_withdraw_flag' in raw_col_name_for_flag_logic:
                            extracted_raw_value = 1 if action_type_for_flag == 'withdraw' else 0
                        elif 'meta_is_refund_flag' in raw_col_name_for_flag_logic:
                            extracted_raw_value = 1 if action_type_for_flag == 'refund' else 0
                        # Add more specific logic based on f_detail if needed
                        else:
                             is_present = False # Could not determine logic flag
                        current_val = None # Prevent falling into next if, value is extracted_raw_value
                    else:
                        is_present = False # Path string not understood
            
            elif 'action.type' in raw_json_path_str: # Direct handling for simple logic strings not in list format
                action_type_for_flag = raw_action_json.get('type', '').lower()
                raw_col_name_for_flag_logic = f_detail.get('raw_column_name', '').lower()
                if 'meta_is_swap_flag' in raw_col_name_for_flag_logic:
                    extracted_raw_value = 1 if action_type_for_flag == 'swap' else 0
                # ... add other similar elif blocks for other logic flags ...
                else:
                    is_present = False
                current_val = None
            else:
                 is_present = False
            
            if is_present and extracted_raw_value is None: # if not already set by logic flag
                extracted_raw_value = current_val
            elif not is_present and extracted_raw_value is None: # if not present and not set by logic flag
                 extracted_raw_value = None

        except Exception as e: # Added except block
            print(f"Warning: Error extracting raw value for '{feature_name}' using path '{raw_json_path_str}': {e}")
            extracted_raw_value = None
            is_present = False
        # ---- End Robust Value Extraction Logic ----

        f_type = f_detail['type']

        if f_type == 'id_map':
            mapping_file_name = f_detail['mapping_file']
            if mapping_file_name not in all_mappings:
                raise ValueError(f"Mapping file '{mapping_file_name}' for feature '{feature_name}' not in all_mappings.")
            current_mapping = all_mappings[mapping_file_name]
            
            unknown_token = model_config.get('unknown_token_str', 'UNKNOWN')
            pad_token = model_config.get('pad_token_str', 'PAD') # For assets, NO_ASSET_STR might map to PAD_ID
            no_asset_token = model_config.get('no_asset_str', 'NO_ASSET')

            val_to_map = str(extracted_raw_value) if extracted_raw_value is not None else unknown_token

            if 'asset' in feature_name.lower(): # Special handling for assets
                if extracted_raw_value is None or str(extracted_raw_value) == no_asset_token:
                    val_to_map = pad_token # Map NO_ASSET or None for assets to PAD_TOKEN
            
            processed_features[feature_name] = current_mapping.get(val_to_map, current_mapping.get(unknown_token))
            if processed_features[feature_name] is None: # Should not happen if UNKNOWN is in map
                 print(f"Warning: Could not map value '{val_to_map}' for '{feature_name}', UNKNOWN ID also missing. Defaulting to 0.")
                 processed_features[feature_name] = 0


        elif f_type == 'hash_cat':
            hash_seed = f_detail['hash_seed']
            hash_bins = f_detail['hash_bins']
            pad_id = f_detail['pad_id']
            no_address_token = model_config.get('no_address_str', 'NO_ADDRESS') # From preprocess

            if extracted_raw_value is None or str(extracted_raw_value) == '' or str(extracted_raw_value) == no_address_token:
                processed_features[feature_name] = pad_id
            else:
                if is_simulation_feedback: # Value is already the predicted hash ID
                    processed_features[feature_name] = int(extracted_raw_value)
                else: # Normal mode: hash the string value
                    hashed_value = mmh3.hash(str(extracted_raw_value), seed=hash_seed) % hash_bins
                    processed_features[feature_name] = hashed_value
        
        elif f_type == 'binary_flag':
            # For 'present_flag' types, is_present (derived from successful extraction) determines the value
            # For 'logic_flag' types (e.g. meta_is_swap_flag), extracted_raw_value holds the 0 or 1
            if 'present_flag' in feature_name:
                 processed_features[feature_name] = 1 if is_present and extracted_raw_value is not None else 0
            elif extracted_raw_value is not None: # Assumes logic-based flags already computed 0/1 into extracted_raw_value
                 processed_features[feature_name] = int(extracted_raw_value)
            else: # Fallback if logic not fully captured or unexpected state
                 processed_features[feature_name] = 0


        elif f_type == 'numerical_scaled':
            intermediate_col_name = f_detail['intermediate_column_name'] # e.g. action_date_unix
            norm_factor_str = f_detail.get('normalization_factor', "None")
            
            current_numerical_val = None
            if extracted_raw_value is not None:
                try:
                    current_numerical_val = float(extracted_raw_value)
                    if norm_factor_str != "None":
                        norm_factor = float(norm_factor_str)
                        current_numerical_val /= norm_factor
                except ValueError:
                    current_numerical_val = np.nan # Let scaler handle NaN if it was trained with them
            else:
                current_numerical_val = np.nan

            temp_df_data[intermediate_col_name] = current_numerical_val
            # Actual scaling happens after collecting all numericals for the row
            processed_features[feature_name] = np.nan # Placeholder, will be filled by scaler
            
        else:
            raise ValueError(f"Unknown feature type '{f_type}' for '{feature_name}' during single action preprocessing.")

    # --- Scale numerical features collected in temp_df_data ---
    if temp_df_data:
        df_single_row_numerical = pd.DataFrame([temp_df_data])
        
        # Ensure columns are in the order the scaler expects
        # scaler.feature_names_in_ might be a numpy array, handle appropriately
        scaler_feature_names = []
        if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
            scaler_feature_names = list(scaler.feature_names_in_)
        elif hasattr(scaler, 'n_features_in_'): # Fallback for older sklearn
             # This case is problematic as we don't have names, only count.
             # preprocess_ai_data.py saves scaler_features_names_in_order in config, but that is not used here yet.
             # For now, assume df_single_row_numerical columns are already in correct order if feature_names_in_ is missing.
             # This requires that temp_df_data keys (intermediate_col_name) were added in the scaler's training order.
            pass # Assume order is correct if feature_names_in_ is missing.
        
        # Reorder and select columns for scaling if scaler_feature_names is available
        if scaler_feature_names:
            # Ensure all expected scaler features are present, fill with NaN if missing
            for sf_name in scaler_feature_names:
                if sf_name not in df_single_row_numerical.columns:
                    df_single_row_numerical[sf_name] = np.nan
            df_to_scale_single_row = df_single_row_numerical[scaler_feature_names]
        else: # No feature_names_in_ from scaler, proceed with what we have (less robust)
            df_to_scale_single_row = df_single_row_numerical

        # Fill NaNs with 0 before scaling, consistent with preprocessing (or use mean if scaler was fit that way)
        scaled_values_row = scaler.transform(df_to_scale_single_row.fillna(0)) 
        
        scaled_df_row = pd.DataFrame(scaled_values_row, columns=df_to_scale_single_row.columns)

        for feature_name in ordered_feature_names:
            if feature_name in feature_details and feature_details[feature_name]['type'] == 'numerical_scaled':
                intermediate_name = feature_details[feature_name]['intermediate_column_name']
                if intermediate_name in scaled_df_row.columns:
                    processed_features[feature_name] = scaled_df_row[intermediate_name].iloc[0]
                else:
                    print(f"Warning: Scaled value for {intermediate_name} (-> {feature_name}) not found. Defaulting to 0.")
                    processed_features[feature_name] = 0.0 # Default if scaling failed or col mismatch
    
    # Convert processed_features dict to a 1D numpy array in the correct order
    feature_vector = np.array([processed_features[name] for name in ordered_feature_names], dtype=np.float32)
    return feature_vector


def preprocess_sequence_for_inference(raw_actions_json_list, scaler, all_mappings, model_config):
    """
    Preprocesses a list of raw transaction JSONs (a sequence) into a 2D numpy array.
    """
    global MODEL_CONTEXT_LENGTH # Use the possibly updated global
    
    # If raw_actions_json_list is shorter than context length, we need padding.
    # Padding should be done with feature vectors that represent "neutral" or "empty" steps.
    # These padding vectors should be consistent with how PAD_TOKENs were handled during training.
    # For simplicity, we can create a "pad_action_vector"
    
    # Create a dummy pad action JSON (mostly empty or default values)
    # This needs to match how padding would affect features during training (e.g. PAD_IDs)
    pad_action_raw_json = {} # Simplistic pad
    # A more robust pad_action_raw_json would set default "pad" values for keys expected by `preprocess_single_action_for_inference`
    # e.g. {'type': model_config.get('pad_token_str', 'PAD'), 'status': model_config.get('pad_token_str', 'PAD'), ...}
    # This is crucial for asset IDs mapping to PAD_ASSET_ID etc.
    # For now, `preprocess_single_action_for_inference` handles missing raw values by mapping to UNKNOWN or PAD for some features.

    pad_feature_vector = preprocess_single_action_for_inference(
        pad_action_raw_json, scaler, all_mappings, model_config
    )
    
    sequence_vectors = []
    
    num_actions_to_take = min(len(raw_actions_json_list), MODEL_CONTEXT_LENGTH)
    
    for i in range(num_actions_to_take):
        # Process from the end of the list to get the most recent actions
        action_json = raw_actions_json_list[-(num_actions_to_take - i)] 
        feature_vector = preprocess_single_action_for_inference(
            action_json, scaler, all_mappings, model_config
        )
        sequence_vectors.append(feature_vector)
        
    # Add padding if sequence is shorter than MODEL_CONTEXT_LENGTH
    num_padding_vectors = MODEL_CONTEXT_LENGTH - len(sequence_vectors)
    for _ in range(num_padding_vectors):
        sequence_vectors.insert(0, pad_feature_vector) # Prepend padding

    return np.array(sequence_vectors, dtype=np.float32)


def decode_prediction(raw_model_output_vector, scaler, all_mappings, model_config, model=None):
    """
    Decodes the raw output vector from the model into a human-readable dictionary.
    raw_model_output_vector: A 1D numpy array from model.predict_single_step (already on CPU).
    """
    print("Decoding raw model output...")
    if raw_model_output_vector.ndim > 1:
        raw_model_output_vector = raw_model_output_vector.squeeze(0) # Remove batch dim if present

    decoded_transaction = {}
    
    # Prioritize model.feature_output_info if model is provided
    feature_output_info_list = []
    if model and hasattr(model, 'feature_output_info') and model.feature_output_info:
        feature_output_info_list = model.feature_output_info
        print("  Using model.feature_output_info for decoding.")
    else:
        feature_output_info_list = model_config.get('feature_output_info_model', [])
        if not feature_output_info_list:
            raise ValueError("'feature_output_info_model' not found in model_config and model instance not provided or missing attribute. This is required for decoding.")
        else:
            print("  Using 'feature_output_info_model' from model_config for decoding.")


    feature_processing_details = model_config['feature_processing_details']

    # Prepare for inverse scaling: collect all scaled numerical predictions first
    # We need to know the order of numerical features as expected by the scaler
    scaler_feature_names = []
    if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
        scaler_feature_names = list(scaler.feature_names_in_)
    elif hasattr(scaler, 'n_features_in_') and hasattr(scaler, 'get_feature_names_out'):
        # Try to get names if possible for older sklearn
        try: scaler_feature_names = list(scaler.get_feature_names_out())
        except: pass
    
    if not scaler_feature_names and any(f['type'] == 'numerical_scaled' for f_name, f in feature_processing_details.items()):
        print("Warning: Scaler feature names not found. Numerical decoding might be inaccurate or fail.")
        # Attempt to get them from our config if scaler itself doesn't provide
        # This assumes feature_processing_details intermediate_column_name are the scaler names
        scaler_feature_names = [ 
            f_detail['intermediate_column_name'] 
            for f_name, f_detail in feature_processing_details.items() 
            if f_detail['type'] == 'numerical_scaled' and 'intermediate_column_name' in f_detail
        ]
        # Ensure this list is unique and ordered consistently with how scaler was fit - CANONICAL_FEATURE_ORDER might be needed for this part
        # For now, we hope the order from iterating feature_processing_details is somewhat correct or matches.
        # A more robust way is to save scaler_feature_names in model_config during preprocessing.
        temp_ordered_scaler_names = []
        for fn_ordered in model_config['feature_columns_ordered']:
            if fn_ordered in feature_processing_details and feature_processing_details[fn_ordered]['type'] == 'numerical_scaled':
                temp_ordered_scaler_names.append(feature_processing_details[fn_ordered]['intermediate_column_name'])
        if temp_ordered_scaler_names:
            scaler_feature_names = temp_ordered_scaler_names
            print(f"Reconstructed scaler_feature_names from config order: {scaler_feature_names}")
        else:
            print("Could not reconstruct scaler feature names from config.")

    # Create a dummy row for inverse_transform, ordered by scaler_feature_names
    # Initialize with zeros or NaNs
    dummy_scaled_numerical_row_dict = {name: 0.0 for name in scaler_feature_names}
    predicted_scaled_values_map = {} # Store predictions for numerical features before collective inverse scaling

    for f_out_info in feature_output_info_list:
        feature_name = f_out_info['name']
        f_detail = feature_processing_details.get(feature_name)
        if not f_detail:
            print(f"Warning: No processing details for feature '{feature_name}' in config. Skipping decoding for it.")
            decoded_transaction[feature_name] = "ERROR_NO_PROCESSING_DETAILS"
            continue

        f_type = f_out_info['type'] # Type from feature_output_info (same as f_detail['type'])
        start_idx = f_out_info['output_start_idx']
        end_idx = f_out_info['output_end_idx']
        feature_slice = raw_model_output_vector[start_idx:end_idx]

        if feature_name == 'in_coin1_amount_norm_scaled': # DEBUG PRINT
            print(f"  [DEBUG InCoin1Amount] Raw model output slice for {feature_name}: {feature_slice.item()}")

        # DEBUG PRINTS FOR LIQUIDITY FEE
        if feature_name == 'meta_swap_liquidityFee_atomic_scaled':
            print(f"  [DEBUG LiqFee] Raw model output slice for {feature_name}: {feature_slice.item()}")
        
        # DEBUG PRINTS FOR SWAP SLIP
        if feature_name == 'meta_swap_swapSlip_bps_val_scaled':
            print(f"  [DEBUG SwapSlip] Raw model output slice for {feature_name}: {feature_slice.item()}")

        if f_type == 'id_map':
            predicted_id = np.argmax(torch.softmax(torch.tensor(feature_slice), dim=-1).numpy())
            mapping_file_name = f_detail['mapping_file']
            if mapping_file_name in all_mappings:
                inverse_map = {v: k for k, v in all_mappings[mapping_file_name].items()}
                decoded_transaction[feature_name] = inverse_map.get(predicted_id, f"UNKNOWN_ID_{predicted_id}")
            else:
                # If mapping file not found, store the ID itself, prefixed to indicate the issue
                decoded_transaction[feature_name] = f"MAPPING_FILE_NOT_FOUND_ID_{predicted_id}"
        
        elif f_type == 'hash_cat':
            predicted_hashed_id = np.argmax(torch.softmax(torch.tensor(feature_slice), dim=-1).numpy())
            decoded_transaction[feature_name] = int(predicted_hashed_id) # Store the ID itself
        
        elif f_type == 'binary_flag':
            predicted_value = torch.sigmoid(torch.tensor(feature_slice)).item()
            decoded_transaction[feature_name] = 1 if predicted_value > 0.5 else 0
        
        elif f_type == 'numerical_scaled':
            # Store the predicted scaled value. We'll inverse transform all numericals together later.
            intermediate_name = f_detail['intermediate_column_name'] # This is the name in the scaler
            if intermediate_name in dummy_scaled_numerical_row_dict:
                dummy_scaled_numerical_row_dict[intermediate_name] = feature_slice.item() # single value
                predicted_scaled_values_map[feature_name] = feature_slice.item() # Keep track by final name too

                if feature_name == 'in_coin1_amount_norm_scaled' and scaler_feature_names: # DEBUG PRINT
                    try:
                        idx_in_scaler = scaler_feature_names.index(intermediate_name)
                        print(f"  [DEBUG InCoin1Amount] Scaler mean for {intermediate_name}: {scaler.mean_[idx_in_scaler]}")
                        print(f"  [DEBUG InCoin1Amount] Scaler scale (std) for {intermediate_name}: {scaler.scale_[idx_in_scaler]}")
                        norm_factor_debug = f_detail.get('normalization_factor', "None")
                        print(f"  [DEBUG InCoin1Amount] Normalization factor for {feature_name}: {norm_factor_debug}")
                    except ValueError:
                        print(f"  [DEBUG InCoin1Amount] Could not find {intermediate_name} in scaler_feature_names for debug.")
                    except AttributeError:
                        print(f"  [DEBUG InCoin1Amount] Scaler does not have mean_/scale_ attributes or they are not arrays.")
                
                # DEBUG PRINTS FOR LIQUIDITY FEE SCALER PARAMS
                if feature_name == 'meta_swap_liquidityFee_atomic_scaled' and scaler_feature_names:
                    try:
                        idx_in_scaler = scaler_feature_names.index(intermediate_name)
                        print(f"  [DEBUG LiqFee] Scaler mean for {intermediate_name}: {scaler.mean_[idx_in_scaler]}")
                        print(f"  [DEBUG LiqFee] Scaler scale (std) for {intermediate_name}: {scaler.scale_[idx_in_scaler]}")
                        # Liquidity fee uses 'ALREADY_ATOMIC', no specific norm_factor like 1e8 to print here for this stage
                    except ValueError:
                        print(f"  [DEBUG LiqFee] Could not find {intermediate_name} in scaler_feature_names for debug.")
                    except AttributeError:
                        print(f"  [DEBUG LiqFee] Scaler does not have mean_/scale_ attributes or they are not arrays.")

                # DEBUG PRINTS FOR SWAP SLIP SCALER PARAMS
                if feature_name == 'meta_swap_swapSlip_bps_val_scaled' and scaler_feature_names:
                    try:
                        idx_in_scaler = scaler_feature_names.index(intermediate_name)
                        print(f"  [DEBUG SwapSlip] Scaler mean for {intermediate_name}: {scaler.mean_[idx_in_scaler]}")
                        print(f"  [DEBUG SwapSlip] Scaler scale (std) for {intermediate_name}: {scaler.scale_[idx_in_scaler]}")
                        # Swap slip has norm_factor "None"
                    except ValueError:
                        print(f"  [DEBUG SwapSlip] Could not find {intermediate_name} in scaler_feature_names for debug.")
                    except AttributeError:
                        print(f"  [DEBUG SwapSlip] Scaler does not have mean_/scale_ attributes or they are not arrays.")

            else:
                print(f"Warning: Intermediate name '{intermediate_name}' for '{feature_name}' not in scaler feature names. Cannot decode.")
                decoded_transaction[feature_name] = "ERROR_SCALER_MISMATCH"
                predicted_scaled_values_map[feature_name] = feature_slice.item() # Store it anyway, might be useful for debug

        else:
            decoded_transaction[feature_name] = f"UNKNOWN_FEATURE_TYPE_IN_DECODE: {f_type}"

    # Perform inverse scaling for all numerical features together
    if scaler_feature_names and any(f_detail['type'] == 'numerical_scaled' for f_name, f_detail in feature_processing_details.items()):
        # Create the row vector in the correct order for the scaler
        ordered_scaled_values_for_scaler = [dummy_scaled_numerical_row_dict[name] for name in scaler_feature_names]
        
        try:
            # Reshape to 2D array as scaler expects (1, num_scaler_features)
            unscaled_values_row = scaler.inverse_transform(np.array(ordered_scaled_values_for_scaler).reshape(1, -1))
            unscaled_values_dict = {name: val for name, val in zip(scaler_feature_names, unscaled_values_row[0])}

            # Now populate the decoded_transaction with the unscaled and (potentially) un-normalized/un-atomized values
            
            # Define a map from atomic amount intermediate column names to their corresponding asset ID feature names (from decoded_transaction)
            # Keys are 'intermediate_column_name' for atomic amounts (e.g., 'in_coin1_amount_atomic')
            # Values are feature names of the *decoded asset string* (e.g., 'in_coin1_asset_id')
            atomic_to_asset_feature_map = {
                'in_coin1_amount_atomic': 'in_coin1_asset_id',
                'out_coin1_amount_atomic': 'out_coin1_asset_id',
                'meta_swap_liquidityFee_atomic': 'pool1_asset_id', # Assumes fee is in terms of the first pool asset
                'meta_swap_networkFee1_amount_atomic': 'meta_swap_networkFee1_asset_id',
                # For affiliate fee and ILP, if they are always in a fixed asset (e.g., MAYA.CACAO),
                # we can directly use that asset string instead of looking up a predicted one.
                'meta_swap_affiliateFee_atomic': 'MAYA.CACAO', # Fixed asset string
                'meta_swap_streaming_quantity_atomic': 'in_coin1_asset_id', # Assumes streaming quantity is of the input asset
                'meta_withdraw_imp_loss_protection_atomic': 'MAYA.CACAO' # Fixed asset string
            }

            for feature_name, f_detail in feature_processing_details.items():
                if f_detail['type'] == 'numerical_scaled':
                    intermediate_name = f_detail['intermediate_column_name'] # e.g., 'in_coin1_amount_atomic' or 'action_date_unix'
                    
                    if intermediate_name in unscaled_values_dict:
                        final_value = unscaled_values_dict[intermediate_name]
                        
                        # Check if this is an atomic amount that needs conversion back to float string
                        if f_detail.get('original_dtype_desc') == 'integer_atomic_units' or f_detail.get('normalization_factor') == "ALREADY_ATOMIC":
                            asset_string_or_feature_name = atomic_to_asset_feature_map.get(intermediate_name)
                            predicted_asset_str = "MAYA.CACAO" # Default asset if lookup fails

                            if asset_string_or_feature_name:
                                if '.' in asset_string_or_feature_name: # It's a direct asset string (e.g., "MAYA.CACAO")
                                    predicted_asset_str = asset_string_or_feature_name
                                elif asset_string_or_feature_name in decoded_transaction: # It's a feature name for the asset ID
                                    predicted_asset_str = decoded_transaction[asset_string_or_feature_name]
                                    # Handle cases where predicted asset string might be PAD/UNKNOWN from previous decoding step
                                    if predicted_asset_str in [model_config.get('pad_token_str', 'PAD'), model_config.get('unknown_token_str', 'UNKNOWN'), model_config.get('no_asset_str', 'NO_ASSET')]:
                                        # If asset is PAD/UNKNOWN/NO_ASSET, precision is 0, amount effectively stays as is or becomes 0.
                                        # get_asset_precision handles these by returning 0.
                                        pass # Let get_asset_precision handle it.
                                else:
                                    print(f"Warning: Asset ID feature '{asset_string_or_feature_name}' for amount '{intermediate_name}' not found in decoded_transaction. Defaulting to MAYA.CACAO for precision.")
                            else:
                                print(f"Warning: No asset mapping for atomic amount '{intermediate_name}'. Defaulting to MAYA.CACAO for precision.")

                            precision = get_asset_precision(predicted_asset_str)
                            
                            if precision > 0:
                                try:
                                    # Ensure atomic_value is treated as an integer before division
                                    atomic_int_value = int(round(final_value)) # Model might output float close to int
                                    float_value = atomic_int_value / (10**precision)
                                    # Format to a string with appropriate decimal places, avoiding scientific notation for typical amounts
                                    # Using f-string with a general format specifier for float, or a fixed number of decimals
                                    if precision <= 8: # For common crypto amounts
                                        final_value = f"{float_value:.{precision}f}" 
                                    else: # For higher precision like ETH, more decimals might be needed or general format
                                        final_value = f"{float_value:.18f}".rstrip('0').rstrip('.') # Max 18, strip trailing zeros
                                except ValueError:
                                    print(f"Warning: Could not convert atomic value {final_value} to int for {intermediate_name}. Keeping as is.")
                                    final_value = str(final_value) # Keep as string if conversion fails
                            else: # Precision is 0 (e.g. for NO_ASSET or if asset is PAD/UNKNOWN)
                                final_value = str(int(round(final_value))) # Should be integer string

                            # Store this final string amount using the original feature name (e.g. 'in_coin1_amount_norm_scaled')
                            # or a more generic name if reconstruct_to_midgard expects it differently.
                            # For now, let's update decoded_transaction[feature_name]
                            decoded_transaction[feature_name] = final_value
                            if feature_name == 'in_coin1_amount_norm_scaled': # DEBUG PRINT
                                print(f"  [DEBUG InCoin1Amount] Post-atomic conversion for {feature_name}: {final_value} (Asset: {predicted_asset_str}, Precision: {precision})")
                            # DEBUG PRINTS FOR LIQUIDITY FEE POST CONVERSION
                            if feature_name == 'meta_swap_liquidityFee_atomic_scaled':
                                print(f"  [DEBUG LiqFee] Post-atomic conversion for {feature_name}: {final_value} (Asset for precision: {predicted_asset_str}, Precision: {precision})")

                        else: # Not an atomic amount, but other numerical (e.g., date, slip_bps)
                            # Perform any non-atomic-specific un-normalization if needed (e.g. date from unixtime)
                            # The original 'normalization_factor' logic for non-atomics:
                            norm_factor_str = f_detail.get('normalization_factor', "None")
                            if norm_factor_str != "None" and norm_factor_str != "ALREADY_ATOMIC":
                                try: 
                                    final_value *= float(norm_factor_str)
                                except (ValueError, TypeError): 
                                    print(f"Warning: Could not apply norm_factor '{norm_factor_str}' to {feature_name}")
                            
                            # Specific handling for date (convert to int string) or other formats if needed
                            if 'date_unix' in intermediate_name:
                                final_value = str(int(round(final_value))) # Date is int string
                            elif 'height_val' in intermediate_name:
                                final_value = str(int(round(final_value))) # Height is int string
                            # For other values like slip, count, basis_points, they might need to be int strings
                            elif any(s in intermediate_name for s in ['Slip_bps', '_count', '_units', '_points']):
                                final_value = str(int(round(final_value)))
                                # DEBUG PRINTS FOR SWAP SLIP POST CONVERSION
                                if feature_name == 'meta_swap_swapSlip_bps_val_scaled':
                                    print(f"  [DEBUG SwapSlip] Final formatted value for {feature_name}: {final_value}") 
                            # Asymmetry might be float string
                            elif 'asymmetry' in intermediate_name:
                                 final_value = f"{final_value:.4f}" # Example formatting

                        decoded_transaction[feature_name] = final_value
                    
                    elif feature_name not in decoded_transaction: # if no error was set before by the main loop
                        decoded_transaction[feature_name] = predicted_scaled_values_map.get(feature_name, "ERROR_POST_INV_SCALE_MISSING")
        except Exception as e:
            print(f"Error during inverse_transform: {e}. Numerical values will be raw scaled outputs.")
            # If inverse transform fails, populate with the raw scaled values we collected
            for feature_name, scaled_val in predicted_scaled_values_map.items():
                decoded_transaction[feature_name] = f"SCALED_VALUE_ERROR_INV_TRANSFORM: {scaled_val}"
    
    print(f"  Decoded transaction: {json.dumps(decoded_transaction, indent=2)}")
    return decoded_transaction, predicted_scaled_values_map

def predict_single_step(model, processed_input_sequence, device):
    """
    Performs a single prediction step.
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(processed_input_sequence, dtype=torch.float32).to(device)
        if input_tensor.dim() == 2: # If (seq_len, features), add batch dim
            input_tensor = input_tensor.unsqueeze(0)
        
        raw_prediction = model(input_tensor) # Expected shape (batch_size, full_output_dim)
    return raw_prediction.cpu().numpy() # Return as numpy array

def run_next_transaction_prediction(args, model, scaler, all_mappings, model_config, device):
    """
    Runs the live next transaction prediction mode.
    Fetches the latest N transactions, predicts N+1, waits for actual N+1, compares, and repeats.
    """
    print("\n--- Starting Next Transaction Prediction Mode ---")
    
    current_model_context_length = model_config.get('sequence_length', MODEL_CONTEXT_LENGTH)
    print(f"Using model context length (sequence length): {current_model_context_length}")

    print(f"Fetching initial context of {current_model_context_length} transactions...")
    live_context_raw_actions = []
    try:
        initial_fetch_count = current_model_context_length + 10 
        raw_response = fetch_recent_maya_actions(limit=initial_fetch_count, offset=0)
        
        if not raw_response or 'actions' not in raw_response or not raw_response['actions']:
            print("Error: Could not fetch initial transaction context from MayaChain or no actions found. Exiting.")
            return

        recent_actions_list = raw_response['actions']
        recent_actions_list.sort(key=lambda x: int(x.get('date', 0))) 
        
        live_context_raw_actions = recent_actions_list[-current_model_context_length:]
        
        if len(live_context_raw_actions) < current_model_context_length:
            print(f"Warning: Could only fetch {len(live_context_raw_actions)} actions for initial context, "
                  f"needed {current_model_context_length}. Predictions might be less accurate.")
        else:
            print(f"Initial context of {len(live_context_raw_actions)} transactions loaded successfully.")
        if live_context_raw_actions:
            print(f"  Oldest in context: Date {live_context_raw_actions[0].get('date')} Type {live_context_raw_actions[0].get('type')}")
            print(f"  Newest in context: Date {live_context_raw_actions[-1].get('date')} Type {live_context_raw_actions[-1].get('type')}")

    except Exception as e:
        print(f"Error during initial context fetch: {e}")
        return

    for i in range(args.num_predictions_to_make):
        print(f"\n--- Prediction Iteration {i+1}/{args.num_predictions_to_make} ---")
        if not live_context_raw_actions or len(live_context_raw_actions) < current_model_context_length:
            print("Error: Not enough actions in context to make a prediction. Attempting to refetch context...")
            try:
                refetch_count = current_model_context_length + 5 # Fetch a bit more to be safe
                raw_response_refetch = fetch_recent_maya_actions(limit=refetch_count, offset=0)
                if raw_response_refetch and 'actions' in raw_response_refetch and raw_response_refetch['actions']:
                    recent_actions_list_refetch = raw_response_refetch['actions']
                    recent_actions_list_refetch.sort(key=lambda x: int(x.get('date', 0)))
                    live_context_raw_actions = recent_actions_list_refetch[-current_model_context_length:]
                    print(f"Re-fetched context. {len(live_context_raw_actions)} actions loaded.")
                    if len(live_context_raw_actions) < current_model_context_length:
                        print("Still not enough actions after refetch. Skipping iteration.")
                        time.sleep(args.next_tx_poll_interval) # Wait before trying again or next cycle
                        continue 
                else:
                    print("Failed to re-fetch context during iteration. Skipping iteration.")
                    time.sleep(args.next_tx_poll_interval)
                    continue
            except Exception as e_refetch:
                print(f"Error re-fetching context: {e_refetch}. Skipping iteration.")
                time.sleep(args.next_tx_poll_interval)
                continue

        print(f"Preprocessing sequence of {len(live_context_raw_actions)} actions for prediction...")
        try:
            processed_sequence = preprocess_sequence_for_inference(
                live_context_raw_actions, scaler, all_mappings, model_config
            )
        except Exception as e_preprocess:
            print(f"Error during preprocessing: {e_preprocess}. Skipping prediction for this iteration.")
            # Log problematic context for debugging if needed
            # print("Problematic context:", json.dumps(live_context_raw_actions, indent=2))
            # Attempt to recover by removing the oldest action and trying in the next iteration
            if live_context_raw_actions:
                live_context_raw_actions.pop(0)
                print("Removed oldest action from context due to preprocessing error. Will try again next iteration.")
            time.sleep(args.next_tx_poll_interval)
            continue

        print("Making prediction for the next transaction...")
        raw_model_output = predict_single_step(model, processed_sequence, device)

        print("Decoding prediction...")
        predicted_transaction_decoded, _ = decode_prediction(
            raw_model_output, scaler, all_mappings, model_config, model=model
        )
        
        print("Reconstructing predicted transaction to Midgard-like format...")
        reconstructed_predicted_tx = reconstruct_to_midgard_format(
            predicted_transaction_decoded, model_config, all_mappings, scaler
        )

        print("\nPredicted Next Transaction (Reconstructed Midgard-like format):")
        print(json.dumps(reconstructed_predicted_tx, indent=2, default=str))

        last_known_tx_timestamp = 0
        if live_context_raw_actions: # Ensure context is not empty
            last_known_tx_timestamp = int(live_context_raw_actions[-1].get('date', 0))
            print(f"\nWaiting for the actual next transaction (after timestamp {last_known_tx_timestamp})...")
        
        actual_next_transaction = None
        poll_attempts = 0
        # Ensure max_poll_attempts_next_tx is defined in args or use a default
        max_polls = getattr(args, 'max_poll_attempts_next_tx', 12) 

        while not actual_next_transaction and poll_attempts < max_polls:
            poll_attempts += 1
            print(f"  Polling attempt {poll_attempts}/{max_polls}...")
            try:
                candidate_limit = 10 # Fetch a few candidates
                raw_candidate_response = fetch_recent_maya_actions(limit=candidate_limit, offset=0) 
                if raw_candidate_response and 'actions' in raw_candidate_response and raw_candidate_response['actions']:
                    candidate_actions_list = raw_candidate_response['actions']
                    candidate_actions_list.sort(key=lambda x: int(x.get('date', 0))) 
                    
                    for action in candidate_actions_list:
                        action_ts = int(action.get('date', 0))
                        if action_ts > last_known_tx_timestamp:
                            is_new = True
                            # Robust check using txID if available
                            action_txid = None
                            in_txs_action = action.get('in', [])
                            out_txs_action = action.get('out', [])

                            if in_txs_action and isinstance(in_txs_action, list) and len(in_txs_action) > 0 and isinstance(in_txs_action[0], dict):
                                action_txid = in_txs_action[0].get('txID')
                            if not action_txid and out_txs_action and isinstance(out_txs_action, list) and len(out_txs_action) > 0 and isinstance(out_txs_action[0], dict):
                                action_txid = out_txs_action[0].get('txID')
                            if not action_txid: # Fallback if no txID found in 'in' or 'out'
                                action_txid = str(action_ts)

                            for seen_action in live_context_raw_actions:
                                seen_txid = None
                                in_txs_seen = seen_action.get('in', [])
                                out_txs_seen = seen_action.get('out', [])

                                if in_txs_seen and isinstance(in_txs_seen, list) and len(in_txs_seen) > 0 and isinstance(in_txs_seen[0], dict):
                                    seen_txid = in_txs_seen[0].get('txID')
                                if not seen_txid and out_txs_seen and isinstance(out_txs_seen, list) and len(out_txs_seen) > 0 and isinstance(out_txs_seen[0], dict):
                                    seen_txid = out_txs_seen[0].get('txID')
                                if not seen_txid: # Fallback for seen_action
                                    seen_txid = str(seen_action.get('date',0))
                                
                                if action_txid and seen_txid and action_txid == seen_txid:
                                    is_new = False
                                    break
                                # Fallback if no txID, compare timestamp and type (less robust)
                                if not action_txid and not seen_txid and int(action.get('date',0)) == int(seen_action.get('date',0)) and action.get('type') == seen_action.get('type'): 
                                    is_new = False
                                    break
                            
                            if is_new:
                                actual_next_transaction = action
                                print(f"  Actual next transaction found! Date: {actual_next_transaction.get('date')}, Type: {actual_next_transaction.get('type')}")
                                break 
                
                if not actual_next_transaction:
                    print(f"  No new transaction found yet. Waiting {args.next_tx_poll_interval}s...")
                    time.sleep(args.next_tx_poll_interval)

            except Exception as e_poll:
                print(f"Error during MayaChain polling: {e_poll}")
                time.sleep(args.next_tx_poll_interval) 
        
        if not actual_next_transaction:
            print(f"Failed to fetch the actual next transaction after {max_polls} attempts. Skipping comparison for this iteration.")
            # Optional: Try to advance context or wait longer before next major prediction cycle
            time.sleep(args.next_tx_poll_interval * 2) # Wait a bit longer if we missed one
            continue

        print("\nActual Next Transaction (Raw Midgard format):")
        print(json.dumps(actual_next_transaction, indent=2, default=str))

        # --- Comparison Logic (Placeholder for now, to be detailed) ---
        print("\n--- Comparison (Predicted vs Actual) ---")
        # Detailed comparison will involve comparing fields from reconstructed_predicted_tx
        # with actual_next_transaction. This might require some normalization or selective field comparison.
        # For example, compare 'type', 'status', key coins and amounts, key metadata fields.
        
        # Example: Compare type and status
        predicted_type = reconstructed_predicted_tx.get('type')
        actual_type = actual_next_transaction.get('type')
        print(f"  Type: Predicted='{predicted_type}', Actual='{actual_type}' ({'MATCH' if predicted_type == actual_type else 'MISMATCH'})")

        predicted_status = reconstructed_predicted_tx.get('status')
        actual_status = actual_next_transaction.get('status')
        print(f"  Status: Predicted='{predicted_status}', Actual='{actual_status}' ({'MATCH' if predicted_status == actual_status else 'MISMATCH'})")

        # Compare primary in-coin asset and amount (if present)
        # This needs careful handling of list structures and potential missing fields
        try:
            pred_in_coin_asset = reconstructed_predicted_tx.get('in', [{}])[0].get('coins', [{}])[0].get('asset')
            act_in_coin_asset = actual_next_transaction.get('in', [{}])[0].get('coins', [{}])[0].get('asset')
            print(f"  In-Coin Asset: Predicted='{pred_in_coin_asset}', Actual='{act_in_coin_asset}' ({'MATCH' if pred_in_coin_asset == act_in_coin_asset else 'MISMATCH'})")
            
            pred_in_coin_amount = int(reconstructed_predicted_tx.get('in', [{}])[0].get('coins', [{}])[0].get('amount', '0'))
            act_in_coin_amount = int(actual_next_transaction.get('in', [{}])[0].get('coins', [{}])[0].get('amount', '0'))
            amount_diff = abs(pred_in_coin_amount - act_in_coin_amount)
            print(f"  In-Coin Amount: Predicted='{pred_in_coin_amount}', Actual='{act_in_coin_amount}' (Diff: {amount_diff})")
        except (IndexError, TypeError, ValueError) as e_comp_in:
            print(f"  Could not compare in-coin details: {e_comp_in}")

        # Compare primary out-coin asset and amount (if present)
        try:
            pred_out_coin_asset = reconstructed_predicted_tx.get('out', [{}])[0].get('coins', [{}])[0].get('asset')
            act_out_coin_asset = actual_next_transaction.get('out', [{}])[0].get('coins', [{}])[0].get('asset')
            print(f"  Out-Coin Asset: Predicted='{pred_out_coin_asset}', Actual='{act_out_coin_asset}' ({'MATCH' if pred_out_coin_asset == act_out_coin_asset else 'MISMATCH'})")

            pred_out_coin_amount = int(reconstructed_predicted_tx.get('out', [{}])[0].get('coins', [{}])[0].get('amount', '0'))
            act_out_coin_amount = int(actual_next_transaction.get('out', [{}])[0].get('coins', [{}])[0].get('amount', '0'))
            amount_diff_out = abs(pred_out_coin_amount - act_out_coin_amount)
            print(f"  Out-Coin Amount: Predicted='{pred_out_coin_amount}', Actual='{act_out_coin_amount}' (Diff: {amount_diff_out})")
        except (IndexError, TypeError, ValueError) as e_comp_out:
            print(f"  Could not compare out-coin details: {e_comp_out}")

        # Add more detailed comparison for metadata based on type, etc.
        if predicted_type == 'swap' and actual_type == 'swap':
            pred_meta_swap = reconstructed_predicted_tx.get('metadata', {}).get('swap', {})
            act_meta_swap = actual_next_transaction.get('metadata', {}).get('swap', {})
            
            pred_slip = pred_meta_swap.get('swapSlip')
            act_slip = act_meta_swap.get('swapSlip') # Midgard uses BPS string, convert to int for comparison if needed
            try: # Convert to int for comparison, handle potential errors if None or not string int
                pred_slip_int = int(pred_slip) if pred_slip is not None else None
                act_slip_int = int(act_slip) if act_slip is not None else None
                slip_match = (pred_slip_int == act_slip_int)
                print(f"  Swap Slip (BPS): Predicted='{pred_slip_int}', Actual='{act_slip_int}' ({'MATCH' if slip_match else 'MISMATCH'})")
            except (ValueError, TypeError):
                print(f"  Swap Slip (BPS): Predicted='{pred_slip}', Actual='{act_slip}' (Could not convert to int for direct comparison)")

            pred_liq_fee = pred_meta_swap.get('liquidityFee')
            act_liq_fee = act_meta_swap.get('liquidityFee')
            try:
                pred_liq_fee_int = int(pred_liq_fee) if pred_liq_fee is not None else None
                act_liq_fee_int = int(act_liq_fee) if act_liq_fee is not None else None
                liq_fee_diff = abs(pred_liq_fee_int - act_liq_fee_int) if pred_liq_fee_int is not None and act_liq_fee_int is not None else 'N/A'
                print(f"  Swap Liquidity Fee: Predicted='{pred_liq_fee_int}', Actual='{act_liq_fee_int}' (Diff: {liq_fee_diff})")
            except (ValueError, TypeError):
                print(f"  Swap Liquidity Fee: Predicted='{pred_liq_fee}', Actual='{act_liq_fee}' (Could not convert to int for direct comparison)")

        live_context_raw_actions.append(actual_next_transaction)
        if len(live_context_raw_actions) > current_model_context_length:
            live_context_raw_actions.pop(0)
        
        print(f"Context updated. Newest action: Date {live_context_raw_actions[-1].get('date')}, Type: {live_context_raw_actions[-1].get('type')}")

        if i < args.num_predictions_to_make - 1:
            # Ensure inter_prediction_delay_seconds is defined in args or use a default
            delay_seconds = getattr(args, 'inter_prediction_delay_seconds', 5)
            print(f"Waiting {delay_seconds}s before next prediction cycle...")
            time.sleep(delay_seconds)

    print("\n--- Next Transaction Prediction Mode Finished ---")


def reconstruct_to_midgard_format(flat_decoded_tx, model_config, all_mappings, scaler):
    """ 
    Attempts to reconstruct a Midgard-like JSON structure from a flat decoded transaction.
    This is a complex process and will be an approximation.
    flat_decoded_tx: The dictionary output from decode_prediction.
    model_config, all_mappings, scaler: Loaded artifacts.
    """
    reconstructed_tx = {
        "date": None, # Placeholder
        "height": None, # Placeholder
        "status": None, # Placeholder
        "type": None, # Placeholder
        "in": [],
        "out": [],
        "pools": [],
        "metadata": {}
    }

    # Helper to get inverse mapping
    def get_inverse_map(mapping_file_name_key):
        if mapping_file_name_key in all_mappings:
            return {v: k for k, v in all_mappings[mapping_file_name_key].items()}
        return {}

    feature_processing_details = model_config.get('feature_processing_details', {})
    
    # --- Top-level fields ---
    # Type
    if 'action_type_id' in flat_decoded_tx: # decode_prediction now provides the string value
        reconstructed_tx['type'] = flat_decoded_tx['action_type_id']
        if "UNKNOWN_ID" in reconstructed_tx['type'] or "MAPPING_FILE_NOT_FOUND_ID" in reconstructed_tx['type']:
             reconstructed_tx['type'] = "UNKNOWN" # Generic fallback if ID couldn't be mapped
    
    # Status
    if 'action_status_id' in flat_decoded_tx: # decode_prediction now provides the string value
        reconstructed_tx['status'] = flat_decoded_tx['action_status_id']
        if "UNKNOWN_ID" in reconstructed_tx['status'] or "MAPPING_FILE_NOT_FOUND_ID" in reconstructed_tx['status']:
            reconstructed_tx['status'] = "unknown" # Midgard uses lowercase 'unknown'

    # --- Process Numerical Features --- 
    # Values in flat_decoded_tx for numerical features are already unscaled and un-normalized
    # by the decode_prediction function. We directly use them.
    unscaled_numericals = {}
    for f_name, f_details in feature_processing_details.items():
        if f_details['type'] == 'numerical_scaled':
            # f_name is the final feature name (e.g., 'action_date_unix_scaled')
            # flat_decoded_tx contains these directly as keys with their final values.
            if f_name in flat_decoded_tx:
                unscaled_numericals[f_name] = flat_decoded_tx[f_name]
            # If a numerical feature was not in flat_decoded_tx (e.g., PAD or not predicted),
            # it won't be in unscaled_numericals. Subsequent .get() calls will handle this.
    # The original complex block for inverse scaling and un-normalizing is removed.

    # Now populate reconstructed_tx with these unscaled_numericals
    # Date and Height
    action_date_val = unscaled_numericals.get('action_date_unix_scaled')
    if action_date_val is not None:
        reconstructed_tx['date'] = str(int(action_date_val)) # Date is typically int string
    action_height_val = unscaled_numericals.get('action_height_val_scaled')
    if action_height_val is not None:
        reconstructed_tx['height'] = str(int(action_height_val))

    # --- Pools ---
    asset_map_file = model_config.get('feature_processing_details', {}).get('pool1_asset_id', {}).get('mapping_file') # Assuming same map for all assets
    inv_asset_map = get_inverse_map(asset_map_file) # This helper remains useful for other parts if needed
    
    # For pools, flat_decoded_tx should directly contain the string asset names from decode_prediction
    pool1_asset_str = flat_decoded_tx.get('pool1_asset_id')
    if pool1_asset_str and "UNKNOWN_ID" not in pool1_asset_str and "MAPPING_FILE_NOT_FOUND_ID" not in pool1_asset_str and \
       pool1_asset_str != model_config.get('pad_token_str', 'PAD') and \
       pool1_asset_str != model_config.get('no_pool_asset_str', 'NO_POOL'):
        reconstructed_tx['pools'].append(pool1_asset_str)

    pool2_asset_str = flat_decoded_tx.get('pool2_asset_id')
    if pool2_asset_str and "UNKNOWN_ID" not in pool2_asset_str and "MAPPING_FILE_NOT_FOUND_ID" not in pool2_asset_str and \
       pool2_asset_str != model_config.get('pad_token_str', 'PAD') and \
       pool2_asset_str != model_config.get('no_pool_asset_str', 'NO_POOL'):
        if not reconstructed_tx['pools'] and pool1_asset_str and (model_config.get('pad_token_str', 'PAD') in pool1_asset_str or model_config.get('no_pool_asset_str', 'NO_POOL') in pool1_asset_str) : # if pool1 was PAD/NO_POOL but pool2 is valid
            reconstructed_tx['pools'].append("UNKNOWN_OR_PAD_POOL1") # Placeholder
        if reconstructed_tx['pools'] or (not reconstructed_tx['pools'] and pool1_asset_str and (model_config.get('pad_token_str', 'PAD') not in pool1_asset_str and model_config.get('no_pool_asset_str', 'NO_POOL') not in pool1_asset_str)): # if pool1 was valid or we added a placeholder
            reconstructed_tx['pools'].append(pool2_asset_str)
        elif not reconstructed_tx['pools']: # Both pool1 and pool2 might have been PAD initially
             reconstructed_tx['pools'].extend([f"UNKNOWN_OR_PAD_POOL1_FROM_{pool1_asset_str}", pool2_asset_str])


    # --- In/Out Transactions (Simplified for now) ---
    # This needs to reconstruct coin arrays, handle amounts (unscale/unnormalize), addresses (placeholders for hashes)
    # Example for one in_coin:
    in_tx_obj = {"coins": [], "memo": None, "address": None, "txID": None}
    if flat_decoded_tx.get('in_coin1_present_flag') == 1 and 'in_coin1_asset_id' in flat_decoded_tx:
        # flat_decoded_tx from decode_prediction should have the string asset name for in_coin1_asset_id
        asset_str = flat_decoded_tx['in_coin1_asset_id']
        if asset_str and "UNKNOWN_ID" not in asset_str and "MAPPING_FILE_NOT_FOUND_ID" not in asset_str and \
           asset_str != model_config.get('pad_token_str', 'PAD') and \
           asset_str != model_config.get('no_asset_str', 'NO_ASSET'):
            # Get the unscaled, un-normalized amount
            amount_val = unscaled_numericals.get('in_coin1_amount_atomic_scaled') # Corrected key
            amount_str = str(amount_val) if amount_val is not None else "0"
            in_tx_obj["coins"].append({"asset": asset_str, "amount": amount_str})
    
    if 'in_address_hash_id' in flat_decoded_tx:
        in_addr_hash_id = flat_decoded_tx['in_address_hash_id']
        pad_hash_id = feature_processing_details.get('in_address_hash_id', {}).get('pad_id')
        if in_addr_hash_id != pad_hash_id:
            in_tx_obj["address"] = f"simulated_address_for_hash_id_{in_addr_hash_id}"
    
    # Memo status (simplified)
    if 'in_memo_status_id' in flat_decoded_tx:
        # flat_decoded_tx from decode_prediction should have the string value
        memo_status_str = flat_decoded_tx['in_memo_status_id']
        if memo_status_str == "NON_EMPTY_MEMO":
            in_tx_obj["memo"] = "SIMULATED_NON_EMPTY_MEMO"
        elif memo_status_str == "EMPTY_MEMO":
            in_tx_obj["memo"] = ""
        # else: NO_MEMO means memo is None (default)

    if in_tx_obj["coins"] or in_tx_obj["address"] or in_tx_obj["memo"] is not None: # Only add if there's something in it
        reconstructed_tx['in'].append(in_tx_obj)

    # ... Similar logic for out_tx_obj ...
    out_tx_obj = {"coins": [], "memo": None, "address": None, "txID": None} # No memo for out usually
    if flat_decoded_tx.get('out_coin1_present_flag') == 1 and 'out_coin1_asset_id' in flat_decoded_tx:
        asset_str = flat_decoded_tx['out_coin1_asset_id'] # String value from decode_prediction
        if asset_str and "UNKNOWN_ID" not in asset_str and "MAPPING_FILE_NOT_FOUND_ID" not in asset_str and \
           asset_str != model_config.get('pad_token_str', 'PAD') and \
           asset_str != model_config.get('no_asset_str', 'NO_ASSET'):
            amount_val = unscaled_numericals.get('out_coin1_amount_atomic_scaled') # Corrected key
            amount_str = str(amount_val) if amount_val is not None else "0"
            out_tx_obj["coins"].append({"asset": asset_str, "amount": amount_str})
    if 'out_address_hash_id' in flat_decoded_tx:
        out_addr_hash_id = flat_decoded_tx['out_address_hash_id']
        pad_hash_id = feature_processing_details.get('out_address_hash_id', {}).get('pad_id')
        if out_addr_hash_id != pad_hash_id:
            out_tx_obj["address"] = f"simulated_address_for_hash_id_{out_addr_hash_id}"
    if out_tx_obj["coins"] or out_tx_obj["address"]:
        reconstructed_tx['out'].append(out_tx_obj)

    # --- Metadata (very simplified for now) ---
    if flat_decoded_tx.get('meta_is_swap_flag') == 1:
        swap_meta = {}
        # LiquidityFee (numerical)
        lf_val = unscaled_numericals.get('meta_swap_liquidityFee_atomic_scaled') # Corrected key
        if lf_val is not None: swap_meta["liquidityFee"] = str(lf_val)
        
        # TargetAsset (id_map) - should be string from decode_prediction
        ta_str = flat_decoded_tx.get('meta_swap_target_asset_id')
        if ta_str and "UNKNOWN_ID" not in ta_str and "MAPPING_FILE_NOT_FOUND_ID" not in ta_str and \
            ta_str != model_config.get('pad_token_str', 'PAD') and \
            ta_str != model_config.get('no_asset_str', 'NO_ASSET'): # NO_ASSET might be used for target if not applicable
            swap_meta["targetAsset"] = ta_str
        elif ta_str: # if it was PAD or UNKNOWN, reflect that if needed or omit
            swap_meta["targetAsset"] = ta_str # or None or omit, depending on desired Midgard representation for PAD target

        # swapSlip (numerical, bps)
        slip_val = unscaled_numericals.get('meta_swap_swapSlip_bps_val_scaled')
        if slip_val is not None: swap_meta["swapSlip"] = str(slip_val) # Changed (Midgard expects string for BPS too)

        # networkFees (array of coin objects)
        # Simplified: assuming one network fee, meta_swap_networkFee1_asset_id and meta_swap_networkFee1_amount_norm_scaled
        nf_asset_str = flat_decoded_tx.get('meta_swap_networkFee1_asset_id')
        nf_amount_val = unscaled_numericals.get('meta_swap_networkFee1_amount_atomic_scaled') # Corrected key
        if nf_asset_str and "UNKNOWN_ID" not in nf_asset_str and "MAPPING_FILE_NOT_FOUND_ID" not in nf_asset_str and \
           nf_asset_str != model_config.get('pad_token_str', 'PAD') and \
           nf_asset_str != model_config.get('no_asset_str', 'NO_ASSET') and nf_amount_val is not None:
            swap_meta["networkFees"] = [{"asset": nf_asset_str, "amount": str(nf_amount_val)}]
        
        # Other swap fields like tradeTarget, affiliateAddress, affiliateFee would be added similarly
        # Check if model_config has a feature for affiliateFee amount, if not, it cannot be reconstructed.
        # Current model_config (atomic_AUGMENTED) does not have 'meta_swap_affiliateFee_atomic_scaled'.
        # aff_fee_val = unscaled_numericals.get('meta_swap_affiliateFee_atomic_scaled') # This feature does not exist in config
        # if aff_fee_val is not None and float(aff_fee_val) > 0:
        #      swap_meta["affiliateFee"] = str(aff_fee_val)
        aff_addr_hash_id = flat_decoded_tx.get('meta_swap_affiliate_address_hash_id')
        pad_hash_id_aff = feature_processing_details.get('meta_swap_affiliate_address_hash_id', {}).get('pad_id')
        if aff_addr_hash_id is not None and aff_addr_hash_id != pad_hash_id_aff:
            swap_meta["affiliateAddress"] = f"simulated_affiliate_address_hash_id_{aff_addr_hash_id}"
            # If affiliate fee were a feature, it would be added here only if address is present

        # Streaming parameters
        if flat_decoded_tx.get('meta_swap_is_streaming_flag') == 1:
            stream_quant_val = unscaled_numericals.get('meta_swap_streaming_quantity_atomic_scaled') # Corrected key
            if stream_quant_val is not None: swap_meta["streamingQuantity"] = str(stream_quant_val)
            stream_count_val = unscaled_numericals.get('meta_swap_streaming_count_val_scaled')
            if stream_count_val is not None: swap_meta["streamingCount"] = str(stream_count_val)

        if swap_meta: # only add if not empty
            reconstructed_tx['metadata']['swap'] = swap_meta
            
    # --- AddLiquidity Metadata --- 
    elif flat_decoded_tx.get('meta_is_addLiquidity_flag') == 1:
        add_meta = {}
        lu_val = unscaled_numericals.get('meta_addLiquidity_units_val_scaled')
        if lu_val is not None: add_meta["liquidityUnits"] = str(lu_val) # Changed
        # Add other addLiquidity specific fields if any (e.g. runeAddress, assetAddress if they are separate features)
        if add_meta:
            reconstructed_tx['metadata']['addLiquidity'] = add_meta

    # --- Withdraw Metadata --- 
    elif flat_decoded_tx.get('meta_is_withdraw_flag') == 1:
        withdraw_meta = {}
        lu_val_wd = unscaled_numericals.get('meta_withdraw_units_val_scaled')
        if lu_val_wd is not None: withdraw_meta["liquidityUnits"] = str(lu_val_wd) # Changed
        
        ilp_val = unscaled_numericals.get('meta_withdraw_imp_loss_protection_norm_scaled')
        if ilp_val is not None: withdraw_meta["ilProtection"] = str(ilp_val) # Changed
        
        asym_val = unscaled_numericals.get('meta_withdraw_asymmetry_val_scaled')
        # Asymmetry from decode_prediction is already a formatted string like "0.1234"
        if asym_val is not None: withdraw_meta["asymmetry"] = str(asym_val) 
        
        basis_pts_val = unscaled_numericals.get('meta_withdraw_basis_points_val_scaled')
        if basis_pts_val is not None: withdraw_meta["basisPoints"] = str(basis_pts_val) # Changed

        # Network fees for withdraw (similar to swap)
        # Assuming wd_networkFee1_asset_id, wd_networkFee1_amount_norm_scaled if these features exist
        # For now, this part is illustrative and depends on actual features trained for withdraw network fees.
        # Example: 
        # nf_wd_asset_str = flat_decoded_tx.get('meta_withdraw_networkFee1_asset_id') 
        # nf_wd_amount_val = unscaled_numericals.get('meta_withdraw_networkFee1_amount_norm_scaled')
        # if nf_wd_asset_str and ... and nf_wd_amount_val is not None:
        #     withdraw_meta["networkFees"] = [{"asset": nf_wd_asset_str, "amount": str(int(nf_wd_amount_val))}]

        if withdraw_meta:
            reconstructed_tx['metadata']['withdrawLiquidity'] = withdraw_meta # Midgard key is withdrawLiquidity

    # ... Add other metadata types based on flags (refund, thorname, etc.)
    # Example for refund (simplified):
    elif flat_decoded_tx.get('meta_is_refund_flag') == 1:
        reconstructed_tx['metadata']['refund'] = {
            "reason": flat_decoded_tx.get('meta_refund_reason_str', "SIMULATED_REFUND_REASON") # Assuming a feature for refund reason string
        }

    # Final cleanup: remove empty lists for in/out/pools if nothing was added
    if not reconstructed_tx['in']: reconstructed_tx.pop('in')
    if not reconstructed_tx['out']: reconstructed_tx.pop('out')
    if not reconstructed_tx['pools']: reconstructed_tx.pop('pools')
    if not reconstructed_tx['metadata']: reconstructed_tx.pop('metadata')

    return reconstructed_tx

def run_generative_simulation(args, model, scaler, all_mappings, model_config, device):
    """
    Runs the mode for generating a sequence of transactions.
    """
    print(f"--- Starting Generative Simulation Mode (simulating {args.num_simulation_steps} steps) ---")
    current_model_context_length = model_config.get('sequence_length', MODEL_CONTEXT_LENGTH)
    print(f"Using model context length: {current_model_context_length}")

    # 1. Initialize with a seed sequence of N real transactions
    print(f"Fetching initial seed sequence of {current_model_context_length} transactions...")
    # Fetch slightly more initially in case some are not suitable or to get a good starting point
    initial_fetch_count = current_model_context_length + 10
    raw_response_initial_seed = fetch_recent_maya_actions(limit=initial_fetch_count, offset=0) # Renamed var
    
    if not raw_response_initial_seed or 'actions' not in raw_response_initial_seed or not raw_response_initial_seed['actions']:
        print("Error: Could not fetch initial transaction context from MayaChain or no actions found for seed. Exiting simulation.")
        return
    
    initial_seed_actions = raw_response_initial_seed['actions']
    initial_seed_actions.sort(key=lambda x: int(x.get('date', 0))) # Sort oldest to newest
    current_raw_actions = initial_seed_actions[-current_model_context_length:] # Take the most recent N for context


    # The simulation needs to maintain a sequence of *processed* features
    # all_mappings is now passed as an argument to this function
    try:
        current_processed_sequence = preprocess_sequence_for_inference(
            current_raw_actions, scaler, all_mappings, model_config # Pass all_mappings here
        )
    except NotImplementedError:
        print("Exiting simulation due to unimplemented preprocessing.")
        return
    except Exception as e:
        print(f"Error during initial preprocessing for simulation: {e}. Exiting.")
        return

    generated_transactions_decoded = [] # To store the human-readable generated sequence
    generated_transactions_reconstructed = [] # New list for Midgard-like structures
    
    for i in range(args.num_simulation_steps):
        print(f"\nSimulation step {i+1}/{args.num_simulation_steps}")

        # 2. Predict the next transaction based on the current_processed_sequence
        print("  Generating next simulated transaction...")
        raw_model_output = predict_single_step(model, current_processed_sequence, device)

        # 3. Decode the prediction
        try:
            # Pass all_mappings to decode_prediction as well
            simulated_transaction_decoded, raw_prediction_details_simulation = decode_prediction(
                raw_model_output, scaler, all_mappings, model_config, model=model # Pass model
            )
        except NotImplementedError:
            print("  Exiting simulation due to unimplemented decoding.")
            return
        except Exception as e:
            print(f"  Error during decoding simulated transaction: {e}. Skipping this step.")
            # How to recover? Maybe try to generate again, or stop? For now, stop.
            return
            
        print(f"  Simulated transaction (decoded): {json.dumps(simulated_transaction_decoded, indent=2)}")
        generated_transactions_decoded.append(simulated_transaction_decoded)

        # 4. CRITICAL: Convert this decoded (human-readable/JSON-like) transaction *back* into processed features
        
        # Step 4a: Prepare `feedback_action_flat_dict`
        feedback_action_flat_dict = {}
        feature_processing_details_sim = model_config.get('feature_processing_details', {})
        ordered_feature_names_sim = model_config.get('feature_columns_ordered', [])

        for feature_name_sim in ordered_feature_names_sim:
            f_detail_sim = feature_processing_details_sim.get(feature_name_sim, {})
            predicted_value = simulated_transaction_decoded.get(feature_name_sim)
            
            # For most features, the decoded value can be used directly as the "raw" input
            # for preprocess_single_action_for_inference in feedback mode.
            # preprocess_single_action_for_inference will handle re-mapping string names for ID_map features,
            # using direct hash IDs for hash_cat, using 0/1 for binary, and re-normalizing/scaling numericals.
            feedback_action_flat_dict[feature_name_sim] = predicted_value

            # Specific handling for numerical values:
            # decode_prediction returns unscaled, un-normalized values.
            # preprocess_single_action_for_inference expects raw values it can then normalize and scale.
            # The `predicted_value` for numericals from `simulated_transaction_decoded` IS this "raw" value.
            # The current logic in preprocess_single_action_for_inference for simulation feedback:
            #   `extracted_raw_value = raw_action_json.get(feature_name)`
            # This means for numerical features, it will correctly pick up the unscaled, un-normalized value
            # from `feedback_action_flat_dict` and then proceed to normalize/scale it.
            
            # For hash_cat features, `simulated_transaction_decoded` contains the predicted hash bin ID.
            # `preprocess_single_action_for_inference` (in feedback mode) will use this ID directly.
            
            # For id_map features, `simulated_transaction_decoded` contains the string name (e.g., "SWAP").
            # `preprocess_single_action_for_inference` will map this string name back to its ID.

        # Step 4b: Call `preprocess_single_action_for_inference` for feedback
        print(f"  Preparing next input vector from decoded simulation: {json.dumps(feedback_action_flat_dict, indent=2, default=str)}")
        try:
            next_input_feature_vector = preprocess_single_action_for_inference(
                raw_action_json=feedback_action_flat_dict,
                scaler=scaler,
                all_mappings=all_mappings,
                model_config=model_config,
                is_simulation_feedback=True 
            )
        except Exception as e_feedback_preprocess:
            print(f"  Error during feedback preprocessing for simulation step {i+1}: {e_feedback_preprocess}")
            print(f"  Problematic feedback_action_flat_dict: {json.dumps(feedback_action_flat_dict, indent=2, default=str)}")
            print("  Stopping simulation due to feedback preprocessing error.")
            break # Stop simulation if feedback preprocessing fails

        # Step 4c: Update `current_processed_sequence`
        # Ensure current_processed_sequence is a list of lists/numpy arrays before manipulation
        if not isinstance(current_processed_sequence, list):
            current_processed_sequence = list(current_processed_sequence) # Convert from np.array if needed

        if len(current_processed_sequence) >= current_model_context_length:
            current_processed_sequence.pop(0) # Remove the oldest
        current_processed_sequence.append(next_input_feature_vector)
        
        # Convert back to numpy array for the next model input
        current_processed_sequence = np.array(current_processed_sequence, dtype=np.float32)
        
        if current_processed_sequence.shape[0] != current_model_context_length:
            print(f"  Error: current_processed_sequence length ({current_processed_sequence.shape[0]}) "
                  f"does not match model context length ({current_model_context_length}) after update. Stopping.")
            break
        
        print(f"  Simulated step {i+1} processed for feedback loop.")
        # The loop will continue with the updated current_processed_sequence

        # Reconstruct to Midgard format for output file
        reconstructed_midgard_tx = reconstruct_to_midgard_format(
            simulated_transaction_decoded, model_config, all_mappings, scaler
        )
        generated_transactions_reconstructed.append(reconstructed_midgard_tx)

    if args.output_simulation_file and generated_transactions_reconstructed:
        with open(args.output_simulation_file, 'w') as f:
            json.dump(generated_transactions_reconstructed, f, indent=2)
        print(f"Simulated transaction sequence (reconstructed) saved to {args.output_simulation_file}")
    elif args.output_simulation_file and not generated_transactions_decoded:
        print("No transactions were generated to save.")

    print("\n--- Generative Simulation Mode Finished ---")


def main():
    print("[DEBUG] >>> main() function started.") # DEBUG PRINT
    parser = argparse.ArgumentParser(description="Realtime Inference Suite for Generative Transaction Model")
    
    # Common arguments
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument("--artifacts-dir", type=str, required=True, help="Directory containing preprocessing artifacts (scaler, mappings, model_config).")
    parser.add_argument("--model-config-filename", type=str, default="model_config_generative_mayachain_s25.json", 
                        help="Filename of the model configuration JSON within the artifacts directory.")
    
    # Mode-specific arguments
    parser.add_argument("--mode", type=str, choices=['predict_next', 'simulate'], required=True, help="Operating mode.")
    parser.add_argument("--num_predictions_to_make", type=int, default=10, 
                        help="For 'predict_next' mode: how many live predictions to make.")
    parser.add_argument("--next_tx_poll_interval", type=int, default=PREDICTION_POLL_INTERVAL_SECONDS,
                        help="For 'predict_next' mode: seconds to wait between polling for the actual next transaction.")
    parser.add_argument("--max_poll_attempts_next_tx", type=int, default=12, # e.g., 12 * 10s = 2 minutes
                        help="For 'predict_next' mode: max attempts to find the actual next transaction.")
    parser.add_argument("--inter_prediction_delay_seconds", type=int, default=5,
                        help="For 'predict_next' mode: seconds to wait between completing one prediction/comparison cycle and starting the next.")
    parser.add_argument("--num-simulation-steps", type=int, default=50, 
                        help="Number of steps to run in generative simulation (for 'simulate' mode).") # Corrected hyphenation
    parser.add_argument("--output-simulation-file", type=str, default="simulated_transactions_mayachain.json", 
                        help="File to save the sequence of generated transactions (for 'simulate' mode).")

    args = parser.parse_args()
    print(f"[DEBUG] Arguments parsed: {args}") # DEBUG PRINT

    try:
        print("[DEBUG] Attempting to load model and artifacts...") # DEBUG PRINT
        # Renamed `all_mappings` parameter from load_model_and_artifacts to `loaded_all_mappings`
        # to avoid confusion with any local `all_mappings` variables if they existed.
        model, scaler, loaded_all_mappings, model_config, device = load_model_and_artifacts(
            args.model_path, 
            args.artifacts_dir, 
            args.model_config_filename
        )
        print("[DEBUG] Model and artifacts loaded successfully.") # DEBUG PRINT
    except Exception as e:
        print(f"Error during model loading: {e}")
        return # Exit if model loading fails

    # Mode dispatch
    if args.mode == "predict_next":
        run_next_transaction_prediction(args, model, scaler, loaded_all_mappings, model_config, device) # Pass loaded_all_mappings
    elif args.mode == "simulate":
        # Pass all_mappings to run_generative_simulation as it's needed by preprocess_sequence_for_inference called within it
        run_generative_simulation(args, model, scaler, loaded_all_mappings, model_config, device) 
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()