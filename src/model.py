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

class ArbitragePredictionModel(nn.Module):
    def __init__(self, 
                 # Vocabulary sizes for embeddings
                 asset_vocab_size, 
                 type_vocab_size, 
                 status_vocab_size,
                 actor_type_vocab_size, # For actor_type_id_mapped
                 # Embedding dimensions
                 asset_embed_dim, 
                 type_embed_dim, 
                 status_embed_dim,
                 actor_type_embed_dim,
                 # Numerical features
                 num_numerical_features, # Number of scaled numerical features
                 # Transformer parameters
                 d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024,
                 # Output heads
                 p_target_classes=4, # ARB_SWAP, USER_SWAP, NON_SWAP, UNK_ACTOR
                 mu_target_dim=1,    # Single float value for profit
                 dropout=0.1, max_seq_len=10):
        super(ArbitragePredictionModel, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.in_asset_embed = nn.Embedding(asset_vocab_size, asset_embed_dim)
        self.out_asset_embed = nn.Embedding(asset_vocab_size, asset_embed_dim)
        self.fee_asset_embed = nn.Embedding(asset_vocab_size, asset_embed_dim) # for swap_network_fee_asset_mapped
        self.type_embed = nn.Embedding(type_vocab_size, type_embed_dim)
        self.status_embed = nn.Embedding(status_vocab_size, status_embed_dim)
        self.actor_type_embed = nn.Embedding(actor_type_vocab_size, actor_type_embed_dim) # for actor_type_id_mapped

        total_embedded_dim = (asset_embed_dim * 3) + type_embed_dim + status_embed_dim + actor_type_embed_dim
        
        # Linear layer to project concatenated features (all embedded categoricals + numericals) to d_model
        self.input_projection = nn.Linear(total_embedded_dim + num_numerical_features, d_model)

        self.pos_encoder = PositionalEncoding(d_model, self.max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output heads
        self.p_head = nn.Linear(d_model, p_target_classes) # Predicts 4 classes for actor type
        self.mu_head = nn.Linear(d_model, mu_target_dim)   # Predicts 1 float for profit

        # Define the expected order and number of categorical (IDs) and numerical features
        # These indices must correspond to how X_sequences are constructed in preprocess_ai_data.py
        # when it saves mapped IDs instead of one-hot encoded features.
        # Example:
        self.categorical_feature_keys = [
            'status_mapped', 
            'type_mapped', 
            'in_asset_mapped', 
            'out_asset_mapped', 
            'swap_network_fee_asset_mapped',
            'actor_type_id_mapped' # This is the actor type of the current transaction in sequence
        ]
        # num_numerical_features will be passed, e.g., 7 (for the 7 scaled numericals)
        
        # Store embedders in a dictionary for easier access in forward pass
        self.embedders = {
            'status_mapped': self.status_embed,
            'type_mapped': self.type_embed,
            'in_asset_mapped': self.in_asset_embed,
            'out_asset_mapped': self.out_asset_embed,
            'swap_network_fee_asset_mapped': self.fee_asset_embed,
            'actor_type_id_mapped': self.actor_type_embed
        }
        self.num_categorical_ids = len(self.categorical_feature_keys)

    def forward(self, x_cat_ids, x_num):
        # x_cat_ids shape: (batch_size, seq_len, num_categorical_id_features)
        # x_num shape: (batch_size, seq_len, num_numerical_features)
        
        all_embeddings = []
        for i in range(self.num_categorical_ids):
            key = self.categorical_feature_keys[i]
            embedder = self.embedders[key]
            all_embeddings.append(embedder(x_cat_ids[:, :, i]))
            
        concatenated_cat_embeds = torch.cat(all_embeddings, dim=-1)
        
        combined_features = torch.cat([concatenated_cat_embeds, x_num], dim=-1)

        projected_input = self.input_projection(combined_features)
        
        encoded_input = self.pos_encoder(projected_input)

        transformer_output = self.transformer_encoder(encoded_input)
        
        last_step_output = transformer_output[:, -1, :]

        p_logits = self.p_head(last_step_output)
        mu_predictions = self.mu_head(last_step_output)

        return p_logits, mu_predictions

if __name__ == '__main__':
    # Example Usage (for testing the model structure)
    batch_size = 4
    seq_len = 10 # M from requirements
    
    # These should come from the actual data properties after preprocessing
    asset_vocab = 50  # Example: Max ID for any asset + 1
    type_vocab = 5    # Example
    status_vocab = 4  # Example
    actor_type_vocab = 4 # ARB_SWAP, USER_SWAP, NON_SWAP, UNK_ACTOR
    
    asset_dim = 32
    type_dim = 10
    status_dim = 8
    actor_type_dim = 10

    num_numerical_feats = 8 # Updated: was 7, now 8 based on preprocess_ai_data.py output
                            # in_amount_norm, out_amount_norm, swap_liquidity_fee_norm, 
                            # swap_slip_bps, swap_network_fee_amount_norm, 
                            # maya_price_P_m, coingecko_price_P_u, coingecko_price_P_u_out_asset

    model = ArbitragePredictionModel(
        asset_vocab_size=asset_vocab, asset_embed_dim=asset_dim,
        type_vocab_size=type_vocab, type_embed_dim=type_dim,
        status_vocab_size=status_vocab, status_embed_dim=status_dim,
        actor_type_vocab_size=actor_type_vocab, actor_type_embed_dim=actor_type_dim,
        num_numerical_features=num_numerical_feats,
        d_model=256, nhead=4, num_encoder_layers=3, dim_feedforward=512, # Smaller example
        p_target_classes=actor_type_vocab, # Should match actor_type_vocab_size for p_head
        mu_target_dim=1,
        dropout=0.1, max_seq_len=seq_len
    )

    # Create dummy input tensors
    # x_cat_ids: (batch_size, seq_len, num_categorical_id_features)
    # num_categorical_id_features = 6 
    # (status, type, in_asset, out_asset, fee_asset, actor_type_of_current_tx)
    dummy_x_cat_ids = torch.randint(0, status_vocab, (batch_size, seq_len, model.num_categorical_ids)) 
    # Correcting asset ID ranges for dummy data more carefully:
    dummy_x_cat_ids[:,:,2] = torch.randint(0, asset_vocab, (batch_size, seq_len)) # in_asset
    dummy_x_cat_ids[:,:,3] = torch.randint(0, asset_vocab, (batch_size, seq_len)) # out_asset
    dummy_x_cat_ids[:,:,4] = torch.randint(0, asset_vocab, (batch_size, seq_len)) # fee_asset
    dummy_x_cat_ids[:,:,0] = torch.randint(0, status_vocab, (batch_size, seq_len)) # status
    dummy_x_cat_ids[:,:,1] = torch.randint(0, type_vocab, (batch_size, seq_len))   # type
    dummy_x_cat_ids[:,:,5] = torch.randint(0, actor_type_vocab, (batch_size, seq_len)) # actor_type_id_mapped (of current tx)


    dummy_x_num = torch.rand(batch_size, seq_len, num_numerical_feats)

    print(f"Dummy x_cat_ids shape: {dummy_x_cat_ids.shape}")
    print(f"Dummy x_num shape: {dummy_x_num.shape}")

    # Forward pass
    p_logits_output, mu_preds_output = model(dummy_x_cat_ids, dummy_x_num)

    print(f"p_logits output shape: {p_logits_output.shape}")   # Expected: (batch_size, actor_type_vocab)
    print(f"mu_preds output shape: {mu_preds_output.shape}") # Expected: (batch_size, 1)

    # Test Positional Encoding standalone
    pe_test = PositionalEncoding(d_model=256, max_len=10, dropout=0.1)
    test_tensor = torch.rand(batch_size, seq_len, 256)
    pe_out = pe_test(test_tensor)
    print(f"PositionalEncoding output shape: {pe_out.shape}") 