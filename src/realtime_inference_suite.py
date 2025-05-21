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
PREDICTION_POLL_INTERVAL_SECONDS = 10 # How often to check for a new actual transaction

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
            'default_cat_embed_dim': 16, # A very basic default
            'asset_embed_dim': 32, 
            'hash_embed_dim': 24 
        }
        print(f"  Warning: 'embedding_dim_config_used' not found in model_config. Using a basic fallback: {embedding_config_for_model}")
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


def decode_prediction(raw_model_output_vector, scaler, all_mappings, model_config):
    """
    Decodes the raw output vector from the model into a human-readable dictionary.
    raw_model_output_vector: A 1D numpy array from model.predict_single_step (already on CPU).
    """
    print("Decoding raw model output...")
    if raw_model_output_vector.ndim > 1:
        raw_model_output_vector = raw_model_output_vector.squeeze(0) # Remove batch dim if present

    decoded_transaction = {}
    feature_output_info_list = model_config.get('feature_output_info_model', [])
    # Fallback if not in config, but ideally it should be saved by train script or directly use model.feature_output_info if model instance is passed
    if not feature_output_info_list:
        # This is a critical piece of info. If the model object isn't passed to provide model.feature_output_info,
        # it MUST be in the model_config.json. For now, we assume it is called 'feature_output_info_model'.
        # As a last resort, we might try to reconstruct it if enough details are in feature_processing_details
        # but that's prone to errors. Let's make it a requirement for now.
        raise ValueError("'feature_output_info_model' not found in model_config. This is required for decoding.")

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

            # Now populate the decoded_transaction with the unscaled and un-normalized values
            for feature_name, f_detail in feature_processing_details.items():
                if f_detail['type'] == 'numerical_scaled':
                    intermediate_name = f_detail['intermediate_column_name']
                    if intermediate_name in unscaled_values_dict:
                        final_value = unscaled_values_dict[intermediate_name]
                        norm_factor_str = f_detail.get('normalization_factor', "None")
                        if norm_factor_str != "None":
                            try: final_value *= float(norm_factor_str)
                            except ValueError: pass # If norm_factor is not a float convertible string
                        decoded_transaction[feature_name] = final_value
                    # If not in unscaled_values_dict, it means it wasn't in scaler_feature_names or error occurred
                    # The placeholder ERROR_SCALER_MISMATCH or the raw scaled value would already be there
                    elif feature_name not in decoded_transaction: # if no error was set before
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

def run_next_transaction_prediction(args, model, scaler, model_config, device):
    """
    Runs the live next transaction prediction mode.
    Fetches the latest N transactions, predicts N+1, waits for actual N+1, compares, and repeats.
    """
    print("\\n--- Starting Next Transaction Prediction Mode ---")
    
    # Use sequence_length from model_config, fall back to global if not present
    current_model_context_length = model_config.get('sequence_length', MODEL_CONTEXT_LENGTH)
    print(f"Using model context length (sequence length): {current_model_context_length}")

    # Load all_id_mappings using the correct key from model_config
    # all_mappings = model_config.get('all_id_mappings', {}) # OLD WAY - REMOVED
    # if not all_mappings:
    #     print("Warning: 'all_id_mappings' not found in model_config. ID-based decoding/preprocessing might be limited.")

    # Use the all_mappings_loaded from the separate file, passed from load_model_and_artifacts
    # This variable is now passed directly to functions needing it, e.g. preprocess_sequence_for_inference

    # 1. Initialize context with the latest transactions
    print(f"Fetching initial context of {current_model_context_length} transactions...")
    live_context_raw_actions = []
    try:
        # Fetch slightly more initially in case some are not suitable or to get a good starting point
        initial_fetch_count = current_model_context_length + 10 
        raw_response = fetch_recent_maya_actions(limit=initial_fetch_count, offset=0)
        
        if not raw_response or 'actions' not in raw_response or not raw_response['actions']:
            print("Error: Could not fetch initial transaction context from MayaChain or no actions found. Exiting.")
            return

        recent_actions_list = raw_response['actions']
        # Sort by date (older to newer) to ensure correct sequence for model
        # Midgard API usually returns newer first, so we reverse if needed, or sort directly.
        # The API itself might sometimes return them sorted, but explicit sort is safer.
        recent_actions_list.sort(key=lambda x: int(x.get('date', 0))) 
        
        # Take the most recent N actions for the initial context
        live_context_raw_actions = recent_actions_list[-current_model_context_length:]
        
        if len(live_context_raw_actions) < current_model_context_length:
            print(f"Warning: Could only fetch {len(live_context_raw_actions)} actions for initial context, "
                  f"needed {current_model_context_length}. Predictions might be less accurate.")
        else:
            print(f"Initial context of {len(live_context_raw_actions)} transactions loaded successfully.")
            print(f"  Oldest in context: Date {live_context_raw_actions[0].get('date')} Type {live_context_raw_actions[0].get('type')}")
            print(f"  Newest in context: Date {live_context_raw_actions[-1].get('date')} Type {live_context_raw_actions[-1].get('type')}")

    except Exception as e:
        print(f"Error during initial context fetch: {e}")
        return

    # 2. Loop for N predictions
    for i in range(args.num_predictions_to_make):
        print(f"\\n--- Prediction Iteration {i+1}/{args.num_predictions_to_make} ---")
        if not live_context_raw_actions or len(live_context_raw_actions) < current_model_context_length:
            print("Error: Not enough actions in context to make a prediction. Skipping iteration.")
            time.sleep(args.next_tx_poll_interval) # Wait before trying to refetch
            # Attempt to re-fetch context
            try:
                raw_response_refetch = fetch_recent_maya_actions(limit=current_model_context_length, offset=0)
                if raw_response_refetch and 'actions' in raw_response_refetch and raw_response_refetch['actions']:
                    recent_actions_list_refetch = raw_response_refetch['actions']
                    recent_actions_list_refetch.sort(key=lambda x: int(x.get('date', 0)))
                    live_context_raw_actions = recent_actions_list_refetch[-current_model_context_length:]
                    print(f"Re-fetched context. {len(live_context_raw_actions)} actions loaded.")
                    if len(live_context_raw_actions) < current_model_context_length:
                        continue # Skip if still not enough
                else:
                    print("Failed to re-fetch context.")
                    continue
            except Exception as e_refetch:
                print(f"Error re-fetching context: {e_refetch}")
                continue


        # a. Preprocess current context
        print(f"Preprocessing sequence of {len(live_context_raw_actions)} actions for prediction...")
        try:
            processed_sequence = preprocess_sequence_for_inference(
                live_context_raw_actions, scaler, all_mappings, model_config
            )
        except Exception as e_preprocess:
            print(f"Error during preprocessing: {e_preprocess}. Skipping prediction.")
            # Potentially log the problematic context:
            # print("Problematic context:", json.dumps(live_context_raw_actions, indent=2))
            time.sleep(args.next_tx_poll_interval)
            live_context_raw_actions.pop(0) # Remove oldest to try to recover
            print("Removed oldest action from context due to preprocessing error.")
            continue


        # b. Get prediction from model
        print("Making prediction for the next transaction...")
        raw_model_output = predict_single_step(model, processed_sequence, device)

        # c. Decode prediction
        print("Decoding prediction...")
        predicted_transaction_decoded, raw_prediction_details = decode_prediction(
            raw_model_output, scaler, all_mappings, model_config
        )
        
        print("\\nPredicted Next Transaction:")
        # Basic print, can be enhanced
        for key, value in predicted_transaction_decoded.items():
            print(f"  {key}: {value}")

        # d. Wait and poll for the actual next transaction
        last_known_tx_timestamp = int(live_context_raw_actions[-1].get('date', 0))
        print(f"\\nWaiting for the actual next transaction (after timestamp {last_known_tx_timestamp})...")
        
        actual_next_transaction = None
        poll_attempts = 0
        max_poll_attempts = args.max_poll_attempts_next_tx # Add this arg later

        while not actual_next_transaction and poll_attempts < max_poll_attempts:
            poll_attempts += 1
            print(f"  Polling attempt {poll_attempts}/{max_poll_attempts}...")
            try:
                # Fetch a small number of recent actions, hoping the next one is there
                # Offset 0 always gets the very latest.
                raw_candidate_response = fetch_recent_maya_actions(limit=10, offset=0) 
                if raw_candidate_response and 'actions' in raw_candidate_response and raw_candidate_response['actions']:
                    candidate_actions_list = raw_candidate_response['actions']
                    candidate_actions_list.sort(key=lambda x: int(x.get('date', 0))) # Sort oldest to newest
                    for action in candidate_actions_list:
                        action_ts = int(action.get('date', 0))
                        # We need an action that is strictly newer than our last known
                        # And also ensure it's not one we already have (e.g. by checking txID if available)
                        # For simplicity now, just timestamp. A robust check would use txID.
                        # Midgard timestamps are in nanoseconds, Python time.time() is seconds. Be careful with direct comparison.
                        # Midgard action 'date' is unix timestamp * 1e9 (nanoseconds)
                        if action_ts > last_known_tx_timestamp:
                            # Check if we already processed this one by looking at its txID (if available)
                            # or a combination of date and type to avoid duplicates from polling.
                            # This basic check helps if the API returns the same "latest" for a bit.
                            is_new = True
                            action_txid = action.get('in', [{}])[0].get('txID', action.get('out', [{}])[0].get('txID', str(action_ts)))
                            for seen_action in live_context_raw_actions:
                                seen_txid = seen_action.get('in', [{}])[0].get('txID', seen_action.get('out', [{}])[0].get('txID', str(seen_action.get('date',0))))
                                if action_txid and seen_txid and action_txid == seen_txid:
                                    is_new = False
                                    break
                                if not action_txid and not seen_txid and int(action.get('date',0)) == int(seen_action.get('date',0)) and action.get('type') == seen_action.get('type'): # Fallback if no txID
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
                time.sleep(args.next_tx_poll_interval) # Wait before retrying
        
        if not actual_next_transaction:
            print(f"Failed to fetch the actual next transaction after {max_poll_attempts} attempts. Skipping comparison for this iteration.")
            # To avoid getting stuck, we might just try to predict again with the same context
            # or advance the context artificially if this happens too often. For now, just continue.
            time.sleep(args.next_tx_poll_interval) # Wait before next prediction cycle
            continue

        # e. Print actual and compare (placeholder for detailed comparison)
        print("\\nActual Next Transaction:")
        # Basic print
        # Convert actual_next_transaction to a more comparable format if needed,
        # e.g., by running it through a simplified version of preprocess_single_action_for_inference
        # to get its raw values aligned with what feature_processing_details expects.
        # For now, we'll extract directly or use a helper.
        
        actual_tx_simple_dict = {k: v for k, v in actual_next_transaction.items() if not isinstance(v, (dict, list))}
        actual_tx_simple_dict['metadata'] = actual_next_transaction.get('meta') # Add metadata separately for easier access
        actual_tx_simple_dict['in_data'] = actual_next_transaction.get('in')
        actual_tx_simple_dict['out_data'] = actual_next_transaction.get('out')

        print(json.dumps(actual_tx_simple_dict, indent=2, default=str))


        print("\\n--- Comparison (Predicted vs Actual) ---")
        
        feature_output_info_list = model_config.get('feature_output_info_model', [])
        feature_processing_details = model_config.get('feature_processing_details', {})

        if not feature_output_info_list or not feature_processing_details:
            print("  Cannot perform detailed comparison: model_config missing feature_output_info_model or feature_processing_details.")
        else:
            for f_info in feature_output_info_list:
                feat_name = f_info['name']
                feat_type = f_info['type']
                pred_val_decoded = predicted_transaction_decoded.get(feat_name)
                
                # Get actual value - this requires mapping feat_name to the raw JSON structure
                # We need to use feature_processing_details[feat_name]['raw_json_path'] etc.
                # This is similar to what preprocess_single_action_for_inference does to get raw values.
                
                actual_raw_val = None
                f_detail_for_actual = feature_processing_details.get(feat_name)
                
                if f_detail_for_actual:
                    raw_json_path_str = f_detail_for_actual.get('raw_json_path', '')
                    # Simplified extraction for actual value - mirrors parts of preprocess_single_action_for_inference
                    current_val_actual = actual_next_transaction
                    is_present_actual = True
                    try:
                        if raw_json_path_str.startswith('[') and raw_json_path_str.endswith(']'):
                            path_list_actual = json.loads(raw_json_path_str.replace("'", "\"")) # Corrected: no need for extra backslash here
                            for key_or_idx_actual in path_list_actual:
                                if isinstance(key_or_idx_actual, str):
                                    if isinstance(current_val_actual, dict): # Check if current_val_actual is a dict
                                        current_val_actual = current_val_actual.get(key_or_idx_actual)
                                    else: # If not a dict, path cannot be traversed further
                                        is_present_actual = False; break
                                elif isinstance(key_or_idx_actual, int):
                                    if isinstance(current_val_actual, list) and 0 <= key_or_idx_actual < len(current_val_actual):
                                        current_val_actual = current_val_actual[key_or_idx_actual]
                                    else: 
                                        is_present_actual = False; break
                                else: 
                                    is_present_actual = False; break
                                if current_val_actual is None: 
                                    is_present_actual = False; break
                            if is_present_actual: actual_raw_val = current_val_actual
                        
                        elif 'action.type' in raw_json_path_str: # Logic flags
                             action_type_actual = actual_next_transaction.get('type','').lower()
                             raw_col_name_actual = f_detail_for_actual.get('raw_column_name','').lower()
                             if 'meta_is_swap_flag' in raw_col_name_actual: actual_raw_val = 1 if action_type_actual == 'swap' else 0
                             elif 'meta_is_addliquidity_flag' in raw_col_name_actual: actual_raw_val = 1 if action_type_actual == 'addliquidity' else 0
                             # ... other logic flags for meta_is_withdraw_flag, meta_is_refund_flag etc.
                             elif 'meta_is_withdraw_flag' in raw_col_name_actual: actual_raw_val = 1 if action_type_actual == 'withdraw' else 0
                             elif 'meta_is_refund_flag' in raw_col_name_actual: actual_raw_val = 1 if action_type_actual == 'refund' else 0
                             elif 'meta_is_thorname_flag' in raw_col_name_actual: actual_raw_val = 1 if action_type_actual == 'thorname' else 0 # Example for thorname
                             else: actual_raw_val = None 
                        # else: actual_raw_val remains None if path string is not list-like or known logic string

                    except Exception as e_extract_actual: 
                        print(f"      Warning: Error extracting actual_raw_val for {feat_name} using path '{raw_json_path_str}': {e_extract_actual}")
                        actual_raw_val = None 
                                   
                print(f"  Feature: {feat_name} (Type: {feat_type})")
                
                if feat_type == 'id_map' or feat_type == 'hash_cat':
                    pred_id_str = str(pred_val_decoded) # Decoded already gives string for id_map, ID for hash_cat
                    
                    # For actual, we need to get the mapped ID from raw value
                    actual_id = None
                    if f_detail_for_actual:
                        if feat_type == 'id_map':
                            mapping_file = f_detail_for_actual['mapping_file']
                            current_mapping_actual = all_mappings.get(mapping_file, {})
                            unknown_token_actual = model_config.get('unknown_token_str', 'UNKNOWN')
                            no_asset_token_actual = model_config.get('no_asset_str', 'NO_ASSET')
                            pad_token_actual = model_config.get('pad_token_str', 'PAD')

                            val_to_map_actual = str(actual_raw_val) if actual_raw_val is not None else unknown_token_actual
                            if 'asset' in feat_name.lower() and (actual_raw_val is None or str(actual_raw_val) == no_asset_token_actual):
                                val_to_map_actual = pad_token_actual
                            actual_id = current_mapping_actual.get(val_to_map_actual, current_mapping_actual.get(unknown_token_actual))
                        
                        elif feat_type == 'hash_cat':
                            hash_seed_actual = f_detail_for_actual['hash_seed']
                            hash_bins_actual = f_detail_for_actual['hash_bins']
                            pad_id_actual = f_detail_for_actual['pad_id']
                            no_address_token = model_config.get('no_address_str', 'NO_ADDRESS')
                            if actual_raw_val is None or str(actual_raw_val) == '' or str(actual_raw_val) == no_address_token:
                                actual_id = pad_id_actual
                            else:
                                actual_id = mmh3.hash(str(actual_raw_val), seed=hash_seed_actual) % hash_bins_actual
                    
                    actual_id_str = str(actual_id) if actual_id is not None else "N/A (raw or map error)"
                    match = "MATCH" if pred_id_str == actual_id_str else "MISMATCH"
                    print(f"    Predicted: {pred_id_str}, Actual ID: {actual_id_str} ({match})")
                    
                    # Show probability for actual ID if available in raw_prediction_details
                    if feat_name in raw_prediction_details and isinstance(raw_prediction_details[feat_name], dict) and 'logits' in raw_prediction_details[feat_name] and actual_id is not None:
                        logits = raw_prediction_details[feat_name]['logits']
                        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
                        if 0 <= actual_id < len(probs):
                            print(f"      Probability for actual ID ({actual_id}): {probs[actual_id]:.4f}")
                        else:
                            print(f"      Actual ID ({actual_id}) out of range for probability calculation (max vocab index: {len(probs)-1}).")
                    elif feat_name in raw_prediction_details and actual_id is None:
                        print(f"      Cannot show probability for actual ID as actual_id could not be determined.")


                elif feat_type == 'binary_flag':
                    pred_flag = int(pred_val_decoded)
                    actual_flag = None
                    if actual_raw_val is not None: # Assumes actual_raw_val is already 0 or 1 for flags
                        actual_flag = int(actual_raw_val)
                    
                    actual_flag_str = str(actual_flag) if actual_flag is not None else "N/A"
                    match = "MATCH" if pred_flag == actual_flag else "MISMATCH"
                    print(f"    Predicted: {pred_flag}, Actual: {actual_flag_str} ({match})")

                elif feat_type == 'numerical_scaled':
                    pred_numerical_unscaled = pred_val_decoded # decode_prediction already unscaled this
                    
                    # For actual, we need to get raw, then normalize, then scale (or just get the final scaled value if we were to re-preprocess)
                    # This is tricky. For a simpler comparison here, let's just show the raw actual.
                    # A full comparison would re-run the exact preprocessing steps on actual_raw_val.
                    actual_val_for_comp = "N/A (raw extract error)"
                    if actual_raw_val is not None:
                        try:
                            actual_val_for_comp = float(actual_raw_val)
                            # Optional: if we want to compare unscaled pred vs un-normalized actual
                            norm_factor_str = f_detail_for_actual.get('normalization_factor', "None")
                            if norm_factor_str != "None":
                                # predicted is already un-normalized by decode_prediction
                                pass # pred_numerical_unscaled is already at this stage
                            
                            print(f"    Predicted (unscaled, un-normalized): {pred_numerical_unscaled:.4f}, Actual (raw): {actual_val_for_comp:.4f}")
                            if isinstance(pred_numerical_unscaled, (int, float)) and isinstance(actual_val_for_comp, (int, float)):
                                diff = abs(pred_numerical_unscaled - actual_val_for_comp)
                                print(f"      Absolute Difference (Pred_unscaled vs Actual_raw): {diff:.4f}")

                        except ValueError:
                             print(f"    Predicted (unscaled, un-normalized): {pred_numerical_unscaled:.4f}, Actual (raw): {actual_raw_val} (cannot convert actual to float for diff)")
                    else:
                        print(f"    Predicted (unscaled, un-normalized): {pred_numerical_unscaled:.4f}, Actual (raw): N/A")
                else:
                    print(f"    Predicted: {pred_val_decoded}, Actual: (comparison not implemented for this type)")
        
        # For numericals, using the 'raw_prediction_details' from decode_prediction might be useful
        # as it can contain raw logits or scaled values before final transformations.

        # Example of accessing a specific predicted numerical (scaled) vs actual (raw, needs preprocessing for direct comparison)
        # if 'meta_swap_swapSlip_bps_val_scaled' in predicted_transaction_decoded and 'meta' in actual_next_transaction and 'swap' in actual_next_transaction['meta']:
        #     predicted_slip_scaled = predicted_transaction_decoded['meta_swap_swapSlip_bps_val_scaled']
        #     actual_slip_raw = actual_next_transaction['meta']['swap'].get('swapSlip') # This is in BPS (0-10000)
        #     # To compare, actual_slip_raw would need to go through the same scaling as in preprocessing
        #     print(f"    Swap Slip (Example): Predicted (scaled): {predicted_slip_scaled}, Actual (raw BPS): {actual_slip_raw}")


        # f. Update context: Add actual, remove oldest
        live_context_raw_actions.append(actual_next_transaction)
        if len(live_context_raw_actions) > current_model_context_length:
            live_context_raw_actions.pop(0)
        
        print(f"Context updated. Newest action: Date {live_context_raw_actions[-1].get('date')}, Type: {live_context_raw_actions[-1].get('type')}")

        # g. Pause if not the last prediction
        if i < args.num_predictions_to_make - 1:
            print(f"Waiting {args.inter_prediction_delay_seconds}s before next prediction cycle...") # Add this arg
            time.sleep(args.inter_prediction_delay_seconds)

    print("\\n--- Next Transaction Prediction Mode Finished ---")


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
    numerical_values_to_unscale = {}
    numerical_feature_final_names_map = {} # Maps intermediate_name -> final_feature_name

    for f_name, f_details in feature_processing_details.items():
        if f_details['type'] == 'numerical_scaled':
            if f_name in flat_decoded_tx:
                intermediate_name = f_details['intermediate_column_name']
                numerical_values_to_unscale[intermediate_name] = flat_decoded_tx[f_name]
                numerical_feature_final_names_map[intermediate_name] = f_name 
            # else: The feature might not have been predicted or was PAD, handle later or rely on scaler default

    unscaled_numericals = {}
    if numerical_values_to_unscale:
        scaler_feature_names_in_order = []
        if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
            scaler_feature_names_in_order = list(scaler.feature_names_in_)
        elif model_config.get('scaler_features_names_in_order'): # Fallback to config if available
            scaler_feature_names_in_order = model_config['scaler_features_names_in_order']
        
        if not scaler_feature_names_in_order:
            print("Warning in reconstruct: Scaler feature names order not found. Numerical reconstruction might be incorrect.")
            # Attempt to use keys from numerical_values_to_unscale as a last resort, but order is not guaranteed
            scaler_feature_names_in_order = list(numerical_values_to_unscale.keys())

        # Prepare a row for inverse_transform, ensuring all expected scaler features are present and in order
        row_for_inverse_transform = []
        for s_name in scaler_feature_names_in_order:
            row_for_inverse_transform.append(numerical_values_to_unscale.get(s_name, 0.0)) # Default to 0.0 if missing for some reason
        
        try:
            unscaled_row = scaler.inverse_transform(np.array(row_for_inverse_transform).reshape(1, -1))[0]
            temp_unscaled_dict = {name: val for name, val in zip(scaler_feature_names_in_order, unscaled_row)}

            # Apply inverse normalization
            for intermediate_name, unscaled_val in temp_unscaled_dict.items():
                final_feature_name = numerical_feature_final_names_map.get(intermediate_name)
                if final_feature_name and final_feature_name in feature_processing_details:
                    f_detail_for_norm = feature_processing_details[final_feature_name]
                    norm_factor_str = f_detail_for_norm.get('normalization_factor', "None")
                    final_value = unscaled_val
                    if norm_factor_str != "None":
                        try: 
                            norm_factor = float(norm_factor_str)
                            if norm_factor != 0: # Avoid division by zero, though should not happen for factors
                                final_value *= norm_factor # Inverse of division is multiplication
                        except ValueError:
                            print(f"Warning: Could not parse normalization_factor '{norm_factor_str}' for {final_feature_name}")
                    unscaled_numericals[final_feature_name] = final_value
                else:
                    # This intermediate name from scaler might not map back to a final feature if config is misaligned
                    unscaled_numericals[intermediate_name] = unscaled_val # Store with intermediate name as fallback
        except Exception as e_unscale:
            print(f"Error during inverse_transform/un-normalization in reconstruct: {e_unscale}. Numerical values will be raw.")
            # Fallback: use the direct values from flat_decoded_tx for numericals if unscaling fails
            for f_name_iter, f_details_iter in feature_processing_details.items():
                if f_details_iter['type'] == 'numerical_scaled' and f_name_iter in flat_decoded_tx:
                    unscaled_numericals[f_name_iter] = flat_decoded_tx[f_name_iter] 

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
            amount_val = unscaled_numericals.get('in_coin1_amount_norm_scaled')
            amount_str = str(int(amount_val)) if amount_val is not None else "0"
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
            amount_val = unscaled_numericals.get('out_coin1_amount_norm_scaled')
            amount_str = str(int(amount_val)) if amount_val is not None else "0"
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
        lf_val = unscaled_numericals.get('meta_swap_liquidityFee_norm_scaled')
        if lf_val is not None: swap_meta["liquidityFee"] = str(int(lf_val))
        
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
        if slip_val is not None: swap_meta["swapSlip"] = str(int(slip_val)) # BPS are usually int

        # networkFees (array of coin objects)
        # Simplified: assuming one network fee, meta_swap_networkFee1_asset_id and meta_swap_networkFee1_amount_norm_scaled
        nf_asset_str = flat_decoded_tx.get('meta_swap_networkFee1_asset_id')
        nf_amount_val = unscaled_numericals.get('meta_swap_networkFee1_amount_norm_scaled')
        if nf_asset_str and "UNKNOWN_ID" not in nf_asset_str and "MAPPING_FILE_NOT_FOUND_ID" not in nf_asset_str and \
           nf_asset_str != model_config.get('pad_token_str', 'PAD') and \
           nf_asset_str != model_config.get('no_asset_str', 'NO_ASSET') and nf_amount_val is not None:
            swap_meta["networkFees"] = [{"asset": nf_asset_str, "amount": str(int(nf_amount_val))}]
        
        # Other swap fields like tradeTarget, affiliateAddress, affiliateFee would be added similarly
        aff_fee_val = unscaled_numericals.get('meta_swap_affiliateFee_norm_scaled')
        if aff_fee_val is not None and aff_fee_val > 0: # Only include if positive fee
             swap_meta["affiliateFee"] = str(int(aff_fee_val))
             aff_addr_hash_id = flat_decoded_tx.get('meta_swap_affiliate_address_hash_id')
             pad_hash_id_aff = feature_processing_details.get('meta_swap_affiliate_address_hash_id', {}).get('pad_id')
             if aff_addr_hash_id is not None and aff_addr_hash_id != pad_hash_id_aff:
                 swap_meta["affiliateAddress"] = f"simulated_affiliate_address_hash_id_{aff_addr_hash_id}"

        # Streaming parameters
        if flat_decoded_tx.get('meta_swap_is_streaming_flag') == 1:
            stream_quant_val = unscaled_numericals.get('meta_swap_streaming_quantity_norm_scaled')
            if stream_quant_val is not None: swap_meta["streamingQuantity"] = str(int(stream_quant_val))
            stream_count_val = unscaled_numericals.get('meta_swap_streaming_count_val_scaled')
            if stream_count_val is not None: swap_meta["streamingCount"] = str(int(stream_count_val))

        if swap_meta: # only add if not empty
            reconstructed_tx['metadata']['swap'] = swap_meta
            
    # --- AddLiquidity Metadata --- 
    elif flat_decoded_tx.get('meta_is_addLiquidity_flag') == 1:
        add_meta = {}
        lu_val = unscaled_numericals.get('meta_addLiquidity_units_val_scaled')
        if lu_val is not None: add_meta["liquidityUnits"] = str(int(lu_val))
        # Add other addLiquidity specific fields if any (e.g. runeAddress, assetAddress if they are separate features)
        if add_meta:
            reconstructed_tx['metadata']['addLiquidity'] = add_meta

    # --- Withdraw Metadata --- 
    elif flat_decoded_tx.get('meta_is_withdraw_flag') == 1:
        withdraw_meta = {}
        lu_val_wd = unscaled_numericals.get('meta_withdraw_units_val_scaled')
        if lu_val_wd is not None: withdraw_meta["liquidityUnits"] = str(int(lu_val_wd))
        
        ilp_val = unscaled_numericals.get('meta_withdraw_imp_loss_protection_norm_scaled')
        if ilp_val is not None: withdraw_meta["ilProtection"] = str(int(ilp_val)) # Impermanent Loss Protection
        
        asym_val = unscaled_numericals.get('meta_withdraw_asymmetry_val_scaled')
        if asym_val is not None: withdraw_meta["asymmetry"] = f"{asym_val:.4f}" # Asymmetry is usually a float 0-1
        
        basis_pts_val = unscaled_numericals.get('meta_withdraw_basis_points_val_scaled')
        if basis_pts_val is not None: withdraw_meta["basisPoints"] = str(int(basis_pts_val)) # Percentage of liquidity withdrawn in BPS

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
        print(f"\\nSimulation step {i+1}/{args.num_simulation_steps}")

        # 2. Predict the next transaction based on the current_processed_sequence
        print("  Generating next simulated transaction...")
        raw_model_output = predict_single_step(model, current_processed_sequence, device)

        # 3. Decode the prediction
        try:
            # Pass all_mappings to decode_prediction as well
            simulated_transaction_decoded, raw_prediction_details_simulation = decode_prediction(
                raw_model_output, scaler, all_mappings, model_config
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

    print("\\n--- Generative Simulation Mode Finished ---")


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
        run_next_transaction_prediction(args, model, scaler, model_config, device)
    elif args.mode == "simulate":
        # Pass all_mappings to run_generative_simulation as it's needed by preprocess_sequence_for_inference called within it
        run_generative_simulation(args, model, scaler, loaded_all_mappings, model_config, device) 
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()