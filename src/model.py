import torch
import torch.nn as nn
import math

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
                 model_config: dict, 
                 embedding_dim_config: dict, 
                 d_model=256, 
                 nhead=8, 
                 num_encoder_layers=6, 
                 dim_feedforward=1024,
                 dropout=0.1, 
                 max_seq_len=10):
        super(GenerativeTransactionModel, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.model_config = model_config
        self.embedding_dim_config = embedding_dim_config

        self.feature_columns_ordered = model_config['feature_columns_ordered']
        # self.num_features_total = model_config['num_features_total'] # This will be the number of *input* features, not output dimension
        
        self.embedders = nn.ModuleDict()
        # feature_info stores info about INPUT features for embedding and concatenation
        self.feature_info_input = [] 
        current_concat_dim_for_input_projection = 0

        default_cat_embed_dim = embedding_dim_config.get('default_cat_embed_dim', 16)
        asset_embed_dim = embedding_dim_config.get('asset_embed_dim', default_cat_embed_dim)
        hash_embed_dim = embedding_dim_config.get('hash_embed_dim', default_cat_embed_dim)

        # For calculating total output dimension and storing output slice info
        self.total_output_dimension = 0
        self.feature_output_info = [] # List of dicts: {'name', 'type', 'original_input_index', 'output_start_idx', 'output_end_idx', 'vocab_size' (if cat)}

        for idx, feature_name in enumerate(self.feature_columns_ordered):
            input_info = {'name': feature_name, 'index': idx}
            output_info_current_feature = {'name': feature_name, 'original_input_index': idx, 'output_start_idx': self.total_output_dimension}
            
            feature_type_determined = ""
            vocab_size = 0 # For categoricals

            if feature_name.endswith('_id') and not feature_name.endswith('_hash_id'): 
                input_info['type'] = 'id_cat'
                feature_type_determined = 'id_cat'
                current_vocab_size = None
                
                # New matching logic for vocab size from model_config['categorical_id_mapping_details']
                # This part needs to correctly associate feature_name (e.g., 'action_status_id', 'pool1_asset_id')
                # with the correct mapping dictionary stored in model_config['categorical_id_mapping_details']
                # and then take its length for the vocab size.
                
                target_feature_base = feature_name.replace('_id', '') # e.g., "action_status", "pool1_asset", "in_coin1_asset"

                if 'asset' in target_feature_base: 
                    # All asset-related features use the single 'asset_to_id_...json' mapping
                    asset_map_file_key = next((k for k in model_config['categorical_id_mapping_details'] if 'asset_to_id' in k), None)
                    if asset_map_file_key and asset_map_file_key in model_config['categorical_id_mapping_details']:
                        current_vocab_size = len(model_config['categorical_id_mapping_details'][asset_map_file_key])
                    else:
                        raise ValueError(f"Asset mapping file key not found or invalid in model_config for feature: {feature_name}")
                else:
                    # For non-asset IDs like 'action_status_id', 'in_memo_status_id', 'meta_swap_memo_status_id'
                    # The goal is to find the corresponding map, e.g., 'status_to_id_...', 'memo_status_to_id_...'
                    # target_feature_base examples: "action_status", "in_memo_status", "meta_swap_memo_status"
                    
                    # Extract the core part, e.g., "status" from "action_status", or "memo_status" from "in_memo_status"
                    # This is a bit heuristic; relies on consistent naming from preprocessing.
                    core_name_parts = target_feature_base.split('_')
                    if "memo" in core_name_parts and "status" in core_name_parts:
                        search_key_prefix = "memo_status"
                    elif "status" in core_name_parts:
                        search_key_prefix = "status"
                    elif "type" in core_name_parts: # for action_type_id
                        search_key_prefix = "type"
                    # Add other specific non-asset ID types if necessary, e.g. 'refund_reason'
                    else: 
                        # Fallback or raise error if pattern not recognized
                        search_key_prefix = core_name_parts[-1] # Takes 'status' from 'action_status', 'type' from 'action_type'

                    found_map_key = None
                    for map_key in model_config['categorical_id_mapping_details']:
                        if map_key.startswith(search_key_prefix + '_to_id'):
                            found_map_key = map_key
                            break
                    
                    if found_map_key and found_map_key in model_config['categorical_id_mapping_details']:
                        current_vocab_size = len(model_config['categorical_id_mapping_details'][found_map_key])
                    else:
                        # If a direct match like 'status_to_id' failed for 'action_status_id',
                        # and it's not an asset, this indicates an issue.
                        pass # current_vocab_size remains None, will be caught by the check below

                if current_vocab_size is None:
                    raise ValueError(f"Could not determine vocab size for ID feature: {feature_name}. target_feature_base: {target_feature_base}")
                
                vocab_size = current_vocab_size
                embed_dim = asset_embed_dim if 'asset' in feature_name else default_cat_embed_dim
                embed_key = f"embed_input_{feature_name}" # Distinguish from potential output embeddings if ever needed
                self.embedders[embed_key] = nn.Embedding(vocab_size, embed_dim)
                input_info['embed_key'] = embed_key
                current_concat_dim_for_input_projection += embed_dim
                self.total_output_dimension += vocab_size # Categorical features output logits for each class
                output_info_current_feature['vocab_size'] = vocab_size

            elif feature_name.endswith('_hash_id'): 
                input_info['type'] = 'hash_cat'
                feature_type_determined = 'hash_cat'
                current_vocab_size = model_config['hashed_feature_details'].get(feature_name)
                if current_vocab_size is None: raise ValueError(f"Could not determine vocab size for hashed feature: {feature_name}")
                
                vocab_size = current_vocab_size
                embed_key = f"embed_input_{feature_name}"
                self.embedders[embed_key] = nn.Embedding(vocab_size, hash_embed_dim)
                input_info['embed_key'] = embed_key
                current_concat_dim_for_input_projection += hash_embed_dim
                self.total_output_dimension += vocab_size # Hashed features also output logits for each class hash bucket
                output_info_current_feature['vocab_size'] = vocab_size
            
            elif feature_name.endswith('_flag') or feature_name.endswith('_present_flag'):
                input_info['type'] = 'binary'
                feature_type_determined = 'binary'
                current_concat_dim_for_input_projection += 1 
                self.total_output_dimension += 1 # Binary flags output a single logit

            elif feature_name.endswith('_scaled'): 
                input_info['type'] = 'numerical'
                feature_type_determined = 'numerical'
                current_concat_dim_for_input_projection += 1 
                self.total_output_dimension += 1 # Numericals output a single value
            else:
                raise ValueError(f"Unknown feature type for {feature_name} based on suffix.")
            
            self.feature_info_input.append(input_info)
            output_info_current_feature['type'] = feature_type_determined
            output_info_current_feature['output_end_idx'] = self.total_output_dimension
            self.feature_output_info.append(output_info_current_feature)

        self.input_projection = nn.Linear(current_concat_dim_for_input_projection, d_model)
        self.pos_encoder = PositionalEncoding(d_model, self.max_seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        print(f"DEBUG: Total calculated output dimension for model: {self.total_output_dimension}")
        self.output_projection = nn.Linear(d_model, self.total_output_dimension)

    def forward(self, x_sequence):
        # x_sequence shape: (batch_size, seq_len, num_input_features_ordered)
        batch_size, seq_len, _ = x_sequence.shape
        processed_features_for_input_concat = []

        for info in self.feature_info_input:
            feature_idx = info['index']
            feature_slice = x_sequence[:, :, feature_idx] 

            if info['type'] == 'id_cat' or info['type'] == 'hash_cat':
                embedded_slice = self.embedders[info['embed_key']](feature_slice.long())
                processed_features_for_input_concat.append(embedded_slice)
            elif info['type'] == 'binary' or info['type'] == 'numerical':
                processed_features_for_input_concat.append(feature_slice.float().unsqueeze(-1))
        
        concatenated_input_features = torch.cat(processed_features_for_input_concat, dim=-1)
        
        projected_input = self.input_projection(concatenated_input_features)
        encoded_input = self.pos_encoder(projected_input)
        transformer_output = self.transformer_encoder(encoded_input)
        
        last_step_output = transformer_output[:, -1, :] 
        predicted_full_output_vector = self.output_projection(last_step_output) 

        return predicted_full_output_vector # Shape: (batch_size, self.total_output_dimension)


# Example Usage (main block will need significant update for generative model)
if __name__ == '__main__':
    print("--- GenerativeTransactionModel Example Usage (Conceptual - Revised Output) ---")
    
    dummy_feature_columns = [
        'action_status_id', 'pool1_asset_id', 'in_address_hash_id', 
        'in_tx_id_present_flag', 'action_date_unix_scaled'
    ]
    # num_total_feats_input = len(dummy_feature_columns) # Input features

    dummy_model_config = {
        'feature_columns_ordered': dummy_feature_columns,
        'num_features_total': len(dummy_feature_columns), # Number of features in the input sequence from preprocessing
        'categorical_id_mapping_details': {
            'status_to_id_generative_thorchain.json': 5, 
            'asset_to_id_generative_thorchain.json': 150 
        },
        'hashed_feature_details': {
            'in_address_hash_id': 20001 
        },
        'sequence_length': 10
    }

    dummy_embedding_config = {
        'default_cat_embed_dim': 10,
        'asset_embed_dim': 16,
        'hash_embed_dim': 12
    }

    generative_model = GenerativeTransactionModel(
        model_config=dummy_model_config,
        embedding_dim_config=dummy_embedding_config,
        d_model=64, nhead=2, num_encoder_layers=1, dim_feedforward=128, # Smaller example
        max_seq_len=dummy_model_config['sequence_length']
    )

    batch_s = 4
    seq_l = dummy_model_config['sequence_length']
    num_input_features = dummy_model_config['num_features_total']
    dummy_x_sequence = torch.rand(batch_s, seq_l, num_input_features) 
    dummy_x_sequence[:, :, 0] = torch.randint(0, 5, (batch_s, seq_l)).float()    # action_status_id (vocab 5)
    dummy_x_sequence[:, :, 1] = torch.randint(0, 150, (batch_s, seq_l)).float() # pool1_asset_id (vocab 150)
    dummy_x_sequence[:, :, 2] = torch.randint(0, 20001, (batch_s, seq_l)).float()# in_address_hash_id (vocab 20001)
    dummy_x_sequence[:, :, 3] = torch.randint(0, 2, (batch_s, seq_l)).float()    # in_tx_id_present_flag (binary)
    # action_date_unix_scaled (numerical) remains random float

    print(f"Dummy input x_sequence shape: {dummy_x_sequence.shape}")
    predicted_vector = generative_model(dummy_x_sequence)
    # Expected output dimension: 5 (status) + 150 (asset) + 20001 (hash) + 1 (flag) + 1 (numerical) = 20158
    print(f"Predicted vector shape: {predicted_vector.shape}") 
    print(f"Expected output dimension (calculated by model): {generative_model.total_output_dimension}")
    
    # For debugging, print feature_output_info
    # print("\nFeature Output Info from Model:")
    # for info in generative_model.feature_output_info:
    #     print(info)

    print("Model Instantiation and Forward Pass Example Completed (Revised Output).")


# Remove or comment out the old ArbitragePredictionModel class if no longer needed.
# class ArbitragePredictionModel(nn.Module): ... 