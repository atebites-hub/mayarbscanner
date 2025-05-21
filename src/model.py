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
                 model_config: dict, # This is the full model_config from JSON
                 embedding_dim_config: dict, # Separate config for embedding dimensions
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 max_seq_len=25): # Default, can be overridden by model_config if not in preprocessor config
        super(GenerativeTransactionModel, self).__init__()

        # Model architectural parameters
        self.d_model = model_config.get('d_model', d_model)
        self.nhead = model_config.get('nhead', nhead)
        self.num_encoder_layers = model_config.get('num_encoder_layers', num_encoder_layers)
        self.dim_feedforward = model_config.get('dim_feedforward', dim_feedforward)
        self.dropout_rate = model_config.get('dropout', dropout) # Use a different name to avoid conflict with module
        self.max_seq_len = model_config.get('sequence_length', max_seq_len)

        self.model_config_full = model_config # Store the full config
        self.embedding_dim_config = embedding_dim_config

        self.feature_columns_ordered = model_config['feature_columns_ordered']
        self.feature_processing_details = model_config['feature_processing_details']

        self.embedders = nn.ModuleDict()
        self.feature_info_input = []
        current_concat_dim_for_input_projection = 0

        default_cat_embed_dim = embedding_dim_config.get('default_cat_embed_dim', 16)
        asset_embed_dim = embedding_dim_config.get('asset_embed_dim', default_cat_embed_dim)
        hash_embed_dim = embedding_dim_config.get('hash_embed_dim', default_cat_embed_dim)

        self.total_output_dimension = 0
        self.feature_output_info = []

        for idx, feature_name in enumerate(self.feature_columns_ordered):
            if feature_name not in self.feature_processing_details:
                raise ValueError(f"Feature '{feature_name}' not found in feature_processing_details of model_config.")
            
            f_details = self.feature_processing_details[feature_name]
            f_type = f_details['type']

            input_info = {'name': feature_name, 'index': idx, 'type': f_type}
            output_info_current_feature = {
                'name': feature_name,
                'original_input_index': idx,
                'type': f_type, # Use the determined f_type from feature_processing_details
                'output_start_idx': self.total_output_dimension
            }

            if f_type == 'id_map':
                vocab_size = f_details['vocab_size']
                # Determine embed_dim based on feature name characteristics if needed, or use a general one
                embed_dim = asset_embed_dim if 'asset' in feature_name.lower() else default_cat_embed_dim
                # Specific overrides from embedding_dim_config based on feature_name can be added here too
                if 'action_type' in feature_name.lower() and 'action_type_embed_dim' in embedding_dim_config:
                    embed_dim = embedding_dim_config['action_type_embed_dim']
                elif 'status' in feature_name.lower() and 'action_status_embed_dim' in embedding_dim_config: # Example for status
                    embed_dim = embedding_dim_config['action_status_embed_dim']
                
                embed_key = f"embed_input_{feature_name}"
                self.embedders[embed_key] = nn.Embedding(vocab_size, embed_dim)
                input_info['embed_key'] = embed_key
                current_concat_dim_for_input_projection += embed_dim
                self.total_output_dimension += vocab_size
                output_info_current_feature['vocab_size'] = vocab_size

            elif f_type == 'hash_cat':
                # Vocab size for hashed features includes the pad_id.
                # 'hash_bins' is the modulo value. Max ID can be hash_bins (if pad_id is hash_bins).
                # So, embedding layer size should be f_details['hash_bins'] + 1.
                vocab_size = f_details['hash_bins'] + 1 
                embed_dim = hash_embed_dim # Use specific hash_embed_dim
                if 'address_hash_embed_dim' in embedding_dim_config: # More specific override
                     embed_dim = embedding_dim_config['address_hash_embed_dim']

                embed_key = f"embed_input_{feature_name}"
                self.embedders[embed_key] = nn.Embedding(vocab_size, embed_dim)
                input_info['embed_key'] = embed_key
                current_concat_dim_for_input_projection += embed_dim
                self.total_output_dimension += vocab_size
                output_info_current_feature['vocab_size'] = vocab_size
            
            elif f_type == 'binary_flag':
                current_concat_dim_for_input_projection += 1
                self.total_output_dimension += 1
            
            elif f_type == 'numerical_scaled':
                current_concat_dim_for_input_projection += 1
                self.total_output_dimension += 1
            
            else:
                raise ValueError(f"Unknown feature type '{f_type}' for feature '{feature_name}' in feature_processing_details.")

            self.feature_info_input.append(input_info)
            output_info_current_feature['output_end_idx'] = self.total_output_dimension
            self.feature_output_info.append(output_info_current_feature)

        self.input_projection = nn.Linear(current_concat_dim_for_input_projection, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_len, dropout=self.dropout_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                   dim_feedforward=self.dim_feedforward, dropout=self.dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        
        print(f"DEBUG: Total calculated output dimension for model: {self.total_output_dimension}")
        self.output_projection = nn.Linear(self.d_model, self.total_output_dimension)

    def forward(self, x_sequence):
        # x_sequence shape: (batch_size, seq_len, num_input_features_ordered)
        batch_size, seq_len, _ = x_sequence.shape
        processed_features_for_input_concat = []

        for info in self.feature_info_input:
            feature_idx = info['index']
            feature_slice = x_sequence[:, :, feature_idx] 

            if info['type'] == 'id_map' or info['type'] == 'hash_cat':
                embedded_slice = self.embedders[info['embed_key']](feature_slice.long())
                processed_features_for_input_concat.append(embedded_slice)
            elif info['type'] == 'binary_flag' or info['type'] == 'numerical_scaled': # Match types from feature_processing_details
                processed_features_for_input_concat.append(feature_slice.float().unsqueeze(-1))
            else: # Should not be reached if __init__ validates types
                 raise ValueError(f"Unknown feature type '{info['type']}' for feature '{info['name']}' in forward pass input processing.")
        
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
    
    # Simplified dummy_feature_columns and model_config for the example
    # The actual model_config will come from preprocessing.
    dummy_feature_columns = [
        'action_type_id', 'pool1_asset_id', 'in_address_hash_id', 
        'meta_is_swap_flag', 'action_height_val_scaled'
    ]

    dummy_model_config = {
        'feature_columns_ordered': dummy_feature_columns,
        'feature_processing_details': {
            'action_type_id': {'type': 'id_map', 'vocab_size': 10},
            'pool1_asset_id': {'type': 'id_map', 'vocab_size': 50},
            'in_address_hash_id': {'type': 'hash_cat', 'hash_bins': 100}, # vocab will be 101
            'meta_is_swap_flag': {'type': 'binary_flag'},
            'action_height_val_scaled': {'type': 'numerical_scaled'}
        },
        'sequence_length': 5, # Shorter sequence for example
        'd_model': 64, # Passed directly or via model_config for real run
        'nhead': 2,
        'num_encoder_layers': 1,
        'dim_feedforward': 128,
        'dropout': 0.1
    }

    dummy_embedding_config = {
        'default_cat_embed_dim': 8,
        'asset_embed_dim': 10, # pool1_asset_id will use this if logic matches
        'hash_embed_dim': 6,   # in_address_hash_id will use this
        'action_type_embed_dim': 5 # action_type_id will use this
    }

    # Ensure num_features_total is consistent if used (though model derives from columns)
    dummy_model_config['num_features_total'] = len(dummy_feature_columns) 

    generative_model = GenerativeTransactionModel(
        model_config=dummy_model_config,
        embedding_dim_config=dummy_embedding_config
        # d_model etc. will be taken from dummy_model_config if present, or defaults
    )

    batch_s = 2
    seq_l = dummy_model_config['sequence_length']
    num_input_features = len(dummy_feature_columns)
    
    dummy_x_sequence = torch.rand(batch_s, seq_l, num_input_features) 
    # Populate with plausible IDs for categorical/hashed
    dummy_x_sequence[:, :, 0] = torch.randint(0, dummy_model_config['feature_processing_details']['action_type_id']['vocab_size'], (batch_s, seq_l)).float()
    dummy_x_sequence[:, :, 1] = torch.randint(0, dummy_model_config['feature_processing_details']['pool1_asset_id']['vocab_size'], (batch_s, seq_l)).float()
    dummy_x_sequence[:, :, 2] = torch.randint(0, dummy_model_config['feature_processing_details']['in_address_hash_id']['hash_bins'] + 1, (batch_s, seq_l)).float()
    dummy_x_sequence[:, :, 3] = torch.randint(0, 2, (batch_s, seq_l)).float() # binary flag

    print(f"Dummy input x_sequence shape: {dummy_x_sequence.shape}")
    predicted_vector = generative_model(dummy_x_sequence)
    # Expected output dimension based on dummy_model_config:
    # action_type_id (10) + pool1_asset_id (50) + in_address_hash_id (101) + meta_is_swap_flag (1) + action_height_val_scaled (1) = 163
    print(f"Predicted vector shape: {predicted_vector.shape}") 
    print(f"Expected output dimension (calculated by model): {generative_model.total_output_dimension}")
    
    print("\nFeature Output Info from Model:")
    for info in generative_model.feature_output_info:
        print(info)

    print("Model Instantiation and Forward Pass Example Completed (Revised Output).")


# Remove or comment out the old ArbitragePredictionModel class if no longer needed.
# class ArbitragePredictionModel(nn.Module): ... 