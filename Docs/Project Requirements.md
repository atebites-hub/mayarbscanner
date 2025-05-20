# Product Requirements Document (PRD)

## Title
Maya Protocol Generative Transaction Prediction Model

## Problem Statement
Understanding and predicting transaction flow dynamics within the Maya Protocol is crucial for various applications, including identifying potential arbitrage opportunities, assessing network congestion, and optimizing trading strategies. This project aims to build a system that ingests Maya Protocol transaction data, preprocesses it into a comprehensive feature set, and uses an AI model to generatively predict the entire subsequent transaction in a sequence.

## Objectives
- Fetch historical and real-time raw transaction data from Maya Protocol's Midgard API (initially using provided THORChain JSON data for development).
- Preprocess raw transaction data into a rich feature set, encompassing all transaction types and their diverse attributes. This includes:
    - Defining and applying a comprehensive `GenerativeModel_Feature_Schema.md`.
    - Implementing robust ID mapping for known categorical features.
    - Employing feature hashing for high-cardinality categorical features (e.g., addresses).
    - Applying appropriate scaling and normalization to numerical features.
    - Handling missing or variable features using presence flags and placeholder IDs.
- Build an AI model (Transformer-based `GenerativeTransactionModel`) that learns to predict the full feature vector of the next transaction in a sequence, given a history of preceding transactions.
- Develop a composite loss function suitable for training a model with a high-dimensional, mixed-type (categorical, numerical, binary) output.
- Implement evaluation metrics to assess the model's performance on a per-feature basis (e.g., accuracy for categoricals, MSE for numericals) and potentially overall transaction similarity.
- Future: Develop a Flask web UI to display insights derived from the generative model.

## Functional Requirements

### Data Collection & Management
- Initial Development Data: Utilize a provided JSON file (`data/transactions_data.json`) containing THORChain transaction data (structurally similar to Maya Midgard `/actions` output) for initial model development and testing.
- Future Data Fetching: `src/fetch_realtime_transactions.py` will be enhanced to fetch raw, unfiltered JSON transaction data directly from the Maya Protocol Midgard API. This will include capabilities for fetching extended historical periods and handling API pagination if necessary.

### Data Preprocessing (`src/preprocess_ai_data.py` - Generative Model Focus)
- Load raw JSON transaction data.
- Flatten and process transactions according to `Docs/GenerativeModel_Feature_Schema.md`.
- For each feature defined in the schema:
    - **Categorical (Low-Cardinality):** Map to numerical IDs using learned `*_to_id_generative_*.json` mappings (e.g., `action_type_to_id`, `asset_to_id`). Handle unknown values with a dedicated `UNKNOWN` ID and padding with `PAD_ID`.
    - **Categorical (High-Cardinality, e.g., Addresses):** Apply feature hashing (e.g., `mmh3`) to a predefined vocabulary size (e.g., `address_hash_id`).
    - **Numerical:** Convert to appropriate numeric types, handle `NaN`s (e.g., with 0 or mean/median), normalize by fixed factors if applicable (e.g., 1e8 for crypto amounts), and scale using `StandardScaler` (scaler saved as `scaler_generative_*.pkl`).
    - **Binary/Flags:** Convert to 0/1. Handle missing by assuming 0 (absence) or a dedicated category if meaningful.
- Ensure a `CANONICAL_FEATURE_ORDER` is maintained for consistency between preprocessing, model input, and output interpretation.
- Generate sequences of `M` transactions as input features for the AI model.
- The target for each input sequence will be the complete, processed feature vector of the `M+1`-th transaction.
- Save preprocessed data as `sequences_and_targets_generative_*.npz` and all artifacts (mappings, scaler, model configuration) to a designated artifacts directory (e.g., `data/processed_ai_data_generative_test/thorchain_artifacts_v1/`).
- Create `model_config_generative_*.json` storing paths to artifacts, feature order, mapping details, and hash vocabulary sizes.

### AI Model (`src/model.py` - `GenerativeTransactionModel`)
- Use a Transformer Encoder architecture.
- **Input:** Sequences of `M` fully processed transaction vectors (concatenation of all features from the schema).
- **Embeddings:** Dynamically created `nn.Embedding` layers for ID-mapped and hashed categorical features, based on `model_config_generative_*.json`.
- **Input Projection:** Project the concatenated embedded categorical features and raw numerical/flag features to the model's `d_model` dimension.
- **Positional Encoding & Transformer Encoder:** Standard Transformer components.
- **Output Projection:** A final linear layer that maps the Transformer output to `total_output_dimension`. This dimension is the sum of:
    - `vocab_size` for each ID-mapped categorical feature (outputting logits).
    - `hash_vocab_size` for each feature-hashed categorical feature (outputting logits).
    - `1` for each numerical feature (outputting a single regression value).
    - `1` for each binary flag feature (outputting a single logit).
- The model stores `feature_output_info` (detailing type, start/end index in the output vector for each feature) to aid loss calculation and evaluation.

### Model Training (`src/train_model.py` - Generative Focus)
- Load preprocessed sequences and targets from `*_generative_*.npz`.
- Load `model_config_generative_*.json` to instantiate `GenerativeTransactionModel` correctly.
- Implement `calculate_composite_loss` function that:
    - Iterates through `model.feature_output_info`.
    - Slices the `predictions` and `targets` tensors according to each feature's output span.
    - Applies `nn.CrossEntropyLoss` for ID-mapped and hashed categorical features.
    - Applies `nn.BCEWithLogitsLoss` for binary flag features.
    - Applies `nn.MSELoss` for numerical features.
    - Aggregates these losses (e.g., sum or average).
- Use an optimizer (e.g., `torch.optim.AdamW`) and a learning rate scheduler (e.g., `ReduceLROnPlateau`).
- Save best and final model weights (`best_generative_model_*.pth`, `final_generative_model_*.pth`).

### Model Evaluation (`src/evaluate_model_generative.py`)
- Load a trained `GenerativeTransactionModel` and its corresponding `model_config_generative_*.json`.
- Load test data (sequences and targets).
- For each feature in `model.feature_output_info`:
    - Extract corresponding predictions and targets.
    - Calculate and report relevant metrics:
        - Categorical: Accuracy, F1-score (micro/macro/weighted), confusion matrix for key categoricals.
        - Numerical: MSE, MAE, R2 score (if applicable).
        - Binary Flags: Accuracy, F1-score, confusion matrix.
- Calculate and report the overall composite loss on the test set.
- Save detailed metrics to a JSON file (e.g., `evaluation_results_generative/evaluation_metrics.json`) and plots to the same directory.

### Web UI (`src/app.py` - Future for Generative Model)
- Future: Display analyses of sequences generated or predicted by the model.
- Future: Visualize attention mechanisms or feature importance if derivable.

## Non-Functional Requirements
- **Data Integrity:** Ensure correct parsing and processing of all transaction fields according to the schema.
- **Model Performance:** The model should demonstrate capability in predicting features of the next transaction with reasonable accuracy/error rates. Success will be measured by improvements in per-feature metrics and decreasing composite loss.
- **Reproducibility:** Ensure that preprocessing, training, and evaluation are reproducible given the same data and configuration.
- **Scalability (Future Data):** The system should be designed with consideration for future expansion to handle larger datasets from Maya Protocol, including efficient processing and storage.
- **Modularity:** Code for preprocessing, model definition, training, and evaluation should be well-separated and modular.

## Success Criteria
- `src/preprocess_ai_data.py` successfully processes raw JSON data into the specified NPZ format and generates all required artifacts as per `Docs/GenerativeModel_Feature_Schema.md`.
- The `GenerativeTransactionModel` trains without errors, and the composite loss (and its components) decreases on both training and validation sets.
- `src/evaluate_model_generative.py` produces per-feature metrics and an overall loss assessment for the test set, allowing for quantitative evaluation of the model's predictive capabilities.
- (Future) The system can successfully ingest and process large-scale raw transaction data fetched directly from Maya Protocol.
- (Future) Analysis of predicted transaction sequences yields meaningful insights or identifies potential patterns (e.g., emergent arbitrage).