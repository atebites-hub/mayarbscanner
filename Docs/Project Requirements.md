# Product Requirements Document (PRD)

## Title
Maya Protocol Arbitrage Scanner with AI-Powered Prediction

## Problem Statement
Arbitrage opportunities exist within the Maya Protocol ecosystem and between Maya Protocol and external markets. Identifying these opportunities and predicting the behavior of other actors (arbitrageurs, regular users) can lead to profitable strategies. This project aims to build a system that fetches Maya Protocol transaction data, preprocesses it, and uses an AI model to predict future transaction types and potential arbitrage profits.

## Objectives
- Fetch historical and real-time transaction data from Maya Protocol's Midgard API.
- Preprocess data, including feature engineering and integration of external market prices (CoinGecko).
- **Crucially, distinguish between `ARB_SWAP` (SWAP txns *without* `affiliate_id`) and `USER_SWAP` (SWAP txns *with* `affiliate_id`).**
- Build an AI model (Transformer-based) to predict:
    - `p_target`: The type of the next actor/transaction (e.g., `ARB_SWAP`, `USER_SWAP`, `NON_SWAP`, `UNK_ACTOR`).
    - `mu_target`: The potential profit if the next transaction is an `ARB_SWAP`.
- Develop a Flask web UI to display fetched data.
- Ensure the AI model can be trained and eventually run for inference.

## Functional Requirements

### Data Collection & Streaming
- Fetch 24-hour historical transaction data from Maya Protocol's Midgard API (`src/fetch_realtime_transactions.py`).
    - Fields: All relevant fields from the `/actions` endpoint.
- Stream live confirmed and pending transactions from Midgard API (`src/realtime_stream_manager.py`).
- Parse transaction details, including memo strings to identify SWAP parameters and `affiliate_id` (`src/common_utils.py` - though parsing logic for `affiliate_id` is mainly handled directly in preprocessing for now based on its presence).
- **Future Enhancement:** Extend the capability of `src/fetch_realtime_transactions.py` to download a significantly larger historical dataset (e.g., ~100 days or 2400 hours) for more comprehensive model training. This will require handling API pagination and managing increased data processing times and storage.

### Data Preprocessing (`src/preprocess_ai_data.py`)
- Load historical transaction data (`data/historical_24hr_maya_transactions.csv`).
- Identify transaction types: `ARB_SWAP`, `USER_SWAP`, `NON_SWAP` based on transaction type and `affiliate_id` presence in SWAPs. Mapped to `actor_type_id`.
- Fetch external market prices (`P_u`) for relevant assets from CoinGecko API at transaction timestamps (daily granularity). Caching implemented.
- Comprehensive `MAYA_TO_COINGECKO_MAP` used to map various Maya asset string formats to CoinGecko IDs.
- Engineer features: Categorical mappings (e.g., `asset_id`, `actor_type_id`), one-hot encoding of categoricals, numerical normalization (for amounts) and scaling (for amounts, prices, bps).
- Calculate Maya internal prices (`P_m`) from transaction data.
- Calculate `target_mu_profit` for transactions preceding an `ARB_SWAP` based on `P_m_next` and `P_u_next_in_asset`.
- Generate sequences of `M` transactions for input to the AI model (input features are ~90 wide).
- Store preprocessed data (e.g., `data/processed_ai_data/intermediate_processed_data.csv`, `data/processed_ai_data/sequences_and_targets.npz`).

### AI Model (`src/model.py`)
- Use a Transformer Encoder architecture.
- **Input:** Sequences of `M` preprocessed transaction vectors (approx. 90 features per transaction).
- **Embeddings:** For one-hot encoded categorical features (derived from `actor_type_id` of past transactions, `asset_id`, `type_id`, `status_id`).
- **Output Heads:**
    - `p_target` head: Predicts probability distribution over next actor types (`ARB_SWAP`, `USER_SWAP`, `NON_SWAP`, `UNK_ACTOR` - 4 classes).
    - `mu_target` head: Predicts a single float value for profit if the next transaction is predicted to be an `ARB_SWAP`.

### Model Training (`src/train_model.py`)
- Load preprocessed sequences and targets from `sequences_and_targets.npz`.
- Instantiate the AI model.
- Implement loss functions:
    - `nn.CrossEntropyLoss` for `p_target`.
    - `nn.MSELoss` for `mu_target` (masked, applied only when actual next transaction is `ARB_SWAP`).
- Use an optimizer (e.g., `torch.optim.Adam`).

### Web UI (`src/app.py`)
- Display 24-hour historical transactions.
- Display live feed of confirmed transactions.
- Display live feed of pending transactions.

## Non-Functional Requirements
- **Data Accuracy:** Ensure correct parsing of transaction data and affiliate IDs. Price data relies on CoinGecko's daily historicals.
- **Price Accuracy:** Strive for accurate fetching of CoinGecko prices at correct timestamps. Current granularity is daily.
- **Model Performance:** The model should demonstrate predictive capability for actor types and arbitrage profit.
- **Usability:** The Flask UI should be clear and informative.
- **Scalability (Future Data):** The system should be designed with consideration for future expansion to handle larger datasets (e.g., 100+ days of transactions), including efficient processing and storage.

## Success Criteria
- `src/preprocess_ai_data.py` runs successfully, correctly identifying actor types and generating non-NaN `coingecko_price_P_u` and `target_mu_profit` values for a high percentage of relevant transactions. (Achieved)
- The AI model trains without errors, and loss decreases. (Next Phase)
- The Flask UI displays data as expected. (Achieved for Phase 1 features)
- (Future) Model evaluation metrics (e.g., accuracy for actor type, RMSE for profit) meet defined thresholds.