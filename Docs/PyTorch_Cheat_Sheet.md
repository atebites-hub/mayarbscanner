# PyTorch Cheat Sheet for Transformer-Based Arbitrage Prediction

This document provides a quick reference for PyTorch components and functions relevant to building the transformer model for arbitrage prediction (Phase 2).

**Official Documentation:**
*   PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
*   `torch.nn.TransformerEncoder`: [https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
*   `torch.nn.TransformerEncoderLayer`: [https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
*   PyTorch on Mac (for M-series chips): [https://developer.apple.com/metal/pytorch/](https://developer.apple.com/metal/pytorch/)

## Core Modules & Classes

### 1. `torch.nn.Module`
Base class for all neural network modules. Your models should subclass `nn.Module`.
*   **Usage:**
    ```python
    import torch.nn as nn
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            # Define layers here
            self.linear = nn.Linear(10, 1)
        def forward(self, x):
            # Define forward pass
            return self.linear(x)
    ```

### 2. `torch.nn.TransformerEncoder`
A stack of N `TransformerEncoderLayer`s.
*   **Key Parameters:**
    *   `encoder_layer`: An instance of `nn.TransformerEncoderLayer`.
    *   `num_layers`: Number of sub-encoder-layers in the encoder (required).
    *   `norm`: Optional layer normalization component.
*   **Example:**
    ```python
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    # Input: (sequence_length, batch_size, embedding_dim)
    # Output: (sequence_length, batch_size, embedding_dim)
    ```

### 3. `torch.nn.TransformerEncoderLayer`
A single layer of a Transformer encoder.
*   **Key Parameters:**
    *   `d_model`: The number of expected features in the input (required).
    *   `nhead`: The number of heads in the multiheadattention models (required).
    *   `dim_feedforward`: The dimension of the feedforward network model (default=2048).
    *   `dropout`: The dropout value (default=0.1).
    *   `activation`: The activation function of the intermediate layer, can be a string ("relu" or "gelu") or a unary callable (default: "relu").
    *   `batch_first`: If `True`, then the input and output tensors are provided as (batch, seq, feature). Default: `False` (seq, batch, feature). **Project Requirements: Consider setting to `True` for easier data handling.**
*   **Example:**
    ```python
    layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, batch_first=True)
    # Input: (batch_size, sequence_length, embedding_dim) if batch_first=True
    # Output: (batch_size, sequence_length, embedding_dim) if batch_first=True
    ```

### 4. `torch.nn.Linear`
Applies a linear transformation to the incoming data: `y = xA^T + b`.
*   **Usage:** For MLP heads.
*   **Example:**
    ```python
    self.mlp_head_p = nn.Linear(input_features, num_arbs) # For p_j
    self.mlp_head_mu = nn.Linear(input_features, num_arbs) # For mu_j
    ```

### 5. `torch.nn.Embedding`
A simple lookup table that stores embeddings of a fixed dictionary and size.
*   **Usage:** For `arb_ID` embeddings.
*   **Example:**
    ```python
    # num_embeddings = number of unique arb_IDs
    # embedding_dim = dimension of the embedding vector
    self.arb_embedding = nn.Embedding(num_embeddings=100, embedding_dim=32)
    # Input: LongTensor of arbitrary shape containing the indices to extract
    # Output: (*, embedding_dim)
    ```

## Activation Functions
Located in `torch.nn`.
*   `nn.ReLU()`
*   `nn.Sigmoid()`: For probability outputs `p_j`.
*   `nn.GELU()`

## Loss Functions
Located in `torch.nn`.
*   `nn.BCEWithLogitsLoss()`: Numerically stable BCE loss. Input is logits (before sigmoid).
    *   **Usage:** For `p_j` prediction.
*   `nn.MSELoss()`: Mean Squared Error.
    *   **Usage:** For `μ_j` prediction (when `p_j=1`).

## Optimizers
Located in `torch.optim`.
*   `torch.optim.Adam`
    *   **Usage:**
        ```python
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ```

## Tensor Operations
*   **Creating Tensors:**
    *   `torch.tensor(data)`
    *   `torch.randn(shape)`
    *   `torch.zeros(shape)`
    *   `torch.ones(shape)`
*   **Reshaping:**
    *   `tensor.view(new_shape)`
    *   `tensor.reshape(new_shape)`
    *   `tensor.permute(dims)`: For transposing dimensions (e.g., batch_first handling).
*   **Concatenation:**
    *   `torch.cat((tensor1, tensor2), dim=0)`
*   **Stacking:**
    *   `torch.stack((tensor1, tensor2), dim=0)`

## GPU / Device Management
*   **Check for MPS (Apple Silicon) / CUDA:**
    ```python
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    ```
*   **Moving Model/Tensors to Device:**
    ```python
    model.to(device)
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    ```

## Saving and Loading Models
*   **Saving:**
    ```python
    # Save only model state_dict (recommended)
    torch.save(model.state_dict(), 'model_state.pth')
    # Save entire model
    torch.save(model, 'model.pth')
    ```
*   **Loading:**
    ```python
    # Load state_dict
    model = MyModel(*args, **kwargs) # Instantiate model first
    model.load_state_dict(torch.load('model_state.pth'))
    model.eval() # Set to evaluation mode

    # Load entire model
    model = torch.load('model.pth')
    model.eval() # Set to evaluation mode
    ```

## Training Loop Essentials
```python
# model = YourModel()
# criterion_p = nn.BCEWithLogitsLoss()
# criterion_mu = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters())
# model.to(device)

# model.train() # Set model to training mode
# for epoch in range(num_epochs):
#     for batch_data, batch_targets_p, batch_targets_mu in train_loader:
#         batch_data = batch_data.to(device)
#         batch_targets_p = batch_targets_p.to(device)
#         batch_targets_mu = batch_targets_mu.to(device)

#         optimizer.zero_grad()
#         pred_p_logits, pred_mu = model(batch_data)

#         loss_p = criterion_p(pred_p_logits, batch_targets_p)
#         # Mask mu loss for where target_p is 0
#         mask = batch_targets_p > 0.5 
#         loss_mu = criterion_mu(pred_mu[mask], batch_targets_mu[mask])
        
#         total_loss = loss_p + loss_mu # Adjust weighting as needed
#         total_loss.backward()
#         optimizer.step()
```

## Model Evaluation
```python
# model.eval() # Set model to evaluation mode
# with torch.no_grad(): # Disable gradient calculations
#     for batch_data, ... in val_loader:
#         # ... make predictions
#         # ... calculate metrics
```

## Float16 Precision (for M3 Max optimization - Phase 3)
```python
# model.half() # Converts model parameters and buffers to half precision (float16)
# input_tensor = input_tensor.half() # Convert input tensors as well
```
This is usually done during inference or if training with mixed precision.
Ensure your hardware/PyTorch version supports it well (MPS has good float16 support).

---
This cheat sheet covers the primary PyTorch functionalities needed for Phase 2. Refer to the official documentation for more advanced topics or specific details. 