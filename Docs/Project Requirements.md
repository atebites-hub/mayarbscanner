# Product Requirements Document (PRD)

## Title
Maya Protocol Generative Transaction Prediction Model

## Problem Statement
Understanding and predicting transaction flow and block composition dynamics within the Maya Protocol is crucial for various applications, including identifying potential arbitrage opportunities, assessing network congestion, and optimizing trading strategies. This project aims to build a system that ingests Maya Protocol block data, preprocesses it into a comprehensive feature set, and uses an AI model to generatively predict the entire subsequent block in a sequence.

## Objectives
- Fetch historical and real-time full block data from Maya Protocol's Mayanode API and Tendermint RPC endpoints.
- Implement robust Protobuf decoding for transaction data obtained from Tendermint RPC.
- Store all parsed block components (headers, transactions, messages, events, attributes) in a fully relational database.
- Preprocess parsed block data into a rich feature set, suitable for AI model training. This includes:
    - Defining and applying a comprehensive feature schema for blocks.
    - Implementing robust ID mapping for known categorical features.
    - Employing feature hashing for high-cardinality categorical features (e.g., addresses).
    - Applying appropriate scaling and normalization to numerical features.
    - Handling missing or variable features using presence flags and placeholder IDs.
- Build an AI model (Transformer-based `GenerativeBlockModel`) that learns to predict the full feature vector of the next block in a sequence, given a history of preceding blocks.
- Develop a composite loss function suitable for training a model with a high-dimensional, mixed-type (categorical, numerical, binary) output.
- Implement evaluation metrics to assess the model's performance on a per-feature basis (e.g., accuracy for categoricals, MSE for numericals) and potentially overall block similarity.
- Future: Develop a Flask web UI to display insights derived from the generative model.

## Functional Requirements

### Data Collection & Management
- Data Sources: Mayanode API (`https://mayanode.mayachain.info/mayachain/block`) and Tendermint RPC (`https://tendermint.mayachain.info/block`, `/unconfirmed_txs`).
- `src/api_connections.py`: Provides functions to connect and fetch data from these sources.
- `src/fetch_realtime_transactions.py`: Orchestrates data fetching, including historical catch-up (with `aiohttp` for concurrency), specific range/count/target fetching, and continuous polling for new blocks. Manages data insertion into the database.
- `src/database_utils.py`: Handles SQLite database operations, including schema creation, granular data insertion for all block components (fully relational), and reconstruction of Mayanode API-like JSON from the database.
- `src/common_utils.py`: Provides core parsing logic for block data from different sources, including transformation of decoded Tendermint transactions to a Mayanode API-like structure. Handles Protobuf decoding of `cosmos.tx.v1beta1.Tx` using `betterproto` stubs.

### Data Preprocessing (`src/preprocess_ai_data.py` - Generative Block Model Focus)
- Load parsed and relationally stored block data from the database.
- Flatten and process block components according to a defined block feature schema.
- For each feature defined in the schema:
    - **Categorical (Low-Cardinality):** Map to numerical IDs using learned `*_to_id_generative_*.json` mappings (e.g., `action_type_to_id`, `asset_to_id`). Handle unknown values with a dedicated `UNKNOWN` ID and padding with `PAD_ID`.
    - **Categorical (High-Cardinality, e.g., Addresses):** Apply feature hashing (e.g., `mmh3`) to a predefined vocabulary size (e.g., `address_hash_id`).
    - **Numerical:** Convert to appropriate numeric types, handle `NaN`s (e.g., with 0 or mean/median), normalize by fixed factors if applicable (e.g., 1e8 for crypto amounts), and scale using `StandardScaler` (scaler saved as `scaler_generative_*.pkl`).
    - **Binary/Flags:** Convert to 0/1. Handle missing by assuming 0 (absence) or a dedicated category if meaningful.
- Ensure a `CANONICAL_FEATURE_ORDER` is maintained for consistency between preprocessing, model input, and output interpretation.
- Generate sequences of `M` blocks as input features for the AI model.
- The target for each input sequence will be the complete, processed feature vector of the `M+1`-th block.
- Save preprocessed data as `sequences_and_targets_generative_*.npz` and all artifacts (mappings, scaler, model configuration) to a designated artifacts directory (e.g., `data/processed_ai_data_generative_test/thorchain_artifacts_v1/`).
- Create `model_config_generative_*.json` storing paths to artifacts, feature order, mapping details, and hash vocabulary sizes.

### AI Model (`src/model.py` - `GenerativeBlockModel`)
- Use a Transformer Encoder architecture (or similar sequence model).
- **Input:** Sequences of `M` fully processed block feature vectors.
- **Embeddings:** Dynamically created `nn.Embedding` layers for ID-mapped and hashed categorical features derived from block data, based on `model_config_generative_*.json`.
- **Input Projection:** Project the concatenated embedded categorical features and raw numerical/flag features to the model's `d_model` dimension.
- **Positional Encoding & Transformer Encoder:** Standard Transformer components.
- **Output Projection:** A final linear layer that maps the Transformer output to `total_output_dimension`. This dimension is the sum of:
    - `vocab_size` for each ID-mapped categorical feature (outputting logits).
    - `hash_vocab_size` for each feature-hashed categorical feature (outputting logits).
    - `1` for each numerical feature (outputting a single regression value).
    - `1` for each binary flag feature (outputting a single logit).
- The model stores `feature_output_info` (detailing type, start/end index in the output vector for each feature) to aid loss calculation and evaluation.

### Model Training (`src/train_model.py` - Generative Block Model Focus)
- Load preprocessed block sequences and targets from `*_generative_*.npz`.
- Load `model_config_generative_*.json` to instantiate `GenerativeBlockModel` correctly.
- Implement `calculate_composite_loss` function appropriate for block-level features (potentially hierarchical or multi-component).
- Use an optimizer (e.g., `torch.optim.AdamW`) and a learning rate scheduler (e.g., `ReduceLROnPlateau`).
- Save best and final model weights (`best_generative_block_model_*.pth`, `final_generative_block_model_*.pth`).

### Model Evaluation (`src/evaluate_model_generative_block.py` - New name or updated)
- Load a trained `GenerativeBlockModel` and its corresponding `model_config_generative_*.json`.
- Load test data (block sequences and targets).
- For each component/feature of the block:
    - Extract corresponding predictions and targets.
    - Calculate and report relevant metrics:
        - Categorical: Accuracy, F1-score (micro/macro/weighted), confusion matrix for key categoricals.
        - Numerical: MSE, MAE, R2 score (if applicable).
        - Binary Flags: Accuracy, F1-score, confusion matrix.
- Calculate and report the overall composite loss on the test set.
- Save detailed metrics to a JSON file (e.g., `evaluation_results_generative_block/evaluation_metrics.json`) and plots to the same directory.

### Web UI (`src/app.py` - For CACAO Dividend Viewer and Future Model Insights)
- Initial Scope: Develop a Flask application (Task 10.2.D) to display CACAO dividend payouts identified from stored Mayanode blocks. This serves as an end-to-end test of the data pipeline (fetch -> parse -> store -> query -> display).
- Future: Display analyses of sequences generated or predicted by the block model.
- Future: Visualize attention mechanisms or feature importance if derivable from the block model.

## Non-Functional Requirements
- **Data Integrity:** Ensure correct parsing and processing of all block fields according to the schema.
- **Model Performance:** The model should demonstrate capability in predicting features of the next block with reasonable accuracy/error rates. Success will be measured by improvements in per-feature metrics and decreasing composite loss.
- **Reproducibility:** Ensure that data fetching, parsing (including Protobuf decoding), preprocessing, training, and evaluation are reproducible given the same data and configuration.
- **Scalability (Future Data):** The system should be designed with consideration for future expansion to handle larger datasets from Maya Protocol, including efficient data storage, querying, and processing.
- **Modularity:** Code for data handling, preprocessing, model definition, training, and evaluation should be well-separated and modular.

## Success Criteria
- `src/fetch_realtime_transactions.py` successfully fetches and stores block data into the relational database, handling various modes (historical, polling) and providing clear console feedback.
- `src/common_utils.py` accurately parses block data from different sources, and `src/database_utils.py` correctly stores and can reconstruct this data.
- Protobuf decoding for Tendermint transactions is robust and integrated.
- `src/preprocess_ai_data.py` successfully processes stored block data into the specified NPZ format and generates all required artifacts for block prediction.
- The `GenerativeBlockModel` trains without errors, and the composite loss decreases on both training and validation sets.
- `src/evaluate_model_generative_block.py` (or equivalent) produces per-feature/component metrics and an overall loss assessment for the test set, allowing for quantitative evaluation of the block model's predictive capabilities.
- The CACAO Dividend Viewer Flask app (Task 10.2.D) successfully displays dividend information from the database, validating the data pipeline.
- (Future) The system can successfully ingest and process large-scale raw block data fetched directly from Maya Protocol.
- (Future) Analysis of predicted block sequences yields meaningful insights or identifies potential patterns (e.g., emergent arbitrage).