# Implementation Plan

The implementation is divided into four phases, with tasks detailed for coding agents to execute. The total timeline is 8 weeks, assuming a focused development effort.

## Phase 1: Data Collection, Initial UI, and Initial AI Preprocessing (Completed)

*   **Task 1.1: Setup API Connections and Fetch Historical Data**
    *   **Description:** Connect to Maya Protocol Midgard API to fetch 24-hour historical transaction data.
    *   **Deliverable:** `src/fetch_realtime_transactions.py` script; `data/historical_24hr_maya_transactions.csv`.
*   **Task 1.2: Implement Real-time Transaction Streaming**
    *   **Description:** Develop a system to stream live confirmed and pending transactions from Midgard API.
    *   **Deliverable:** `src/realtime_stream_manager.py`.
*   **Task 1.3: Develop Flask Web UI**
    *   **Description:** Create a web interface to display historical, live confirmed, and live pending transactions.
    *   **Deliverable:** `src/app.py`, HTML templates in `templates/`, static files in `static/`.
*   **Task 1.4: Initial AI Data Preprocessing (Uniswap-based)**
    *   **Description:** Preprocess historical data for an AI model, including feature engineering and sequence generation. Used Uniswap for external price data in this initial version.
    *   **Deliverable:** Initial version of `src/preprocess_ai_data.py`; `src/common_utils.py` for parsing.

## Phase 2: Model Development and Data Refinement (COMPLETED & SUPERSEDED)

*   **Task 2.0: Refactor `src/preprocess_ai_data.py` and Verify (COMPLETED)**
    *   **Description:** Updated the preprocessing script based on the new arbitrage identification logic (SWAPs without `affiliate_id` are `ARB_SWAP`) and switched external price source from Uniswap to CoinGecko. Iteratively debugged and refined asset mapping, scaling logic, and timestamp handling. Implemented independent train/test workflow.
    *   **Sub-tasks Completed:** (Summarized)
        *   Implemented new `actor_type_id` (`ARB_SWAP`, `USER_SWAP`, `NON_SWAP`).
        *   Integrated CoinGecko API with file-based caching.
        *   Expanded `MAYA_TO_COINGECKO_MAP`.
        *   Corrected numerical scaling.
        *   Updated `target_mu_profit` calculation.
        *   Refactored `fetch_realtime_transactions.py`, `preprocess_ai_data.py`, `train_model.py`, `evaluate_model.py` for robust train/test splits and artifact management.
    *   **Deliverable:** Verified and refactored data pipeline scripts, producing `sequences_and_targets.npz` and `model_config.json` for training, and enabling independent evaluation.

*   **Task 2.1 (COMPLETED): Update `src/model.py` - Transformer Encoder**
    *   **Description:** Verified the Transformer Encoder in `src/model.py` accepted new input features from the refactored preprocessing script.
    *   **Deliverable:** Verified `src/model.py`.

*   **Task 2.2 (COMPLETED): Update `src/model.py` - Prediction Heads**
    *   **Description:** Verified the prediction heads in `src/model.py` for `p_target` (actor type) and `mu_target` (profit metric).
    *   **Deliverable:** Verified `src/model.py`.

*   **Task 2.3 (COMPLETED): Update `src/train_model.py`**
    *   **Description:** Updated the training script to load the new data format, instantiate the revised model, and adjust loss functions.
    *   **Deliverable:** Updated `src/train_model.py` capable of training the Phase 2 model.

*   **Task 2.4 (COMPLETED): Model Evaluation**
    *   **Description:** Implemented and ran model evaluation metrics on an independent test set.
    *   **Deliverable:** Evaluation results and analysis for the Phase 2 model.

## Phase 3: Generative Transaction Prediction Model (CURRENT FOCUS)

**Overarching Goal:** Shift from predicting specific arbitrage signals to a more fundamental approach of predicting the *entire next transaction* in a sequence. This model will process all transaction types and learn the underlying dynamics of the Maya Protocol without relying on external price feeds. Arbitrage opportunities will be identified as an emergent property from the predicted transaction sequences.

**Key Technical Decisions:**
-   **Data Inclusion:** Utilize all transaction types (user swaps, system actions, etc.) from raw JSON data (initially `data/transactions_data.json` - THORChain data, future Maya Midgard).
-   **Feature Engineering:**
    -   Define a comprehensive schema for all possible transaction features (`Docs/GenerativeModel_Feature_Schema.md`).
    -   Handle variable/missing categorical features using special placeholder IDs and binary presence flags.
    -   Use feature hashing (`mmh3`) for high-cardinality categorical features like addresses.
-   **Model Architecture:** Transformer-based (`GenerativeTransactionModel`), designed to output a full predicted transaction.
-   **Target Variable:** The complete feature vector of the subsequent transaction.
-   **Loss Function:** A composite loss calculated across all predicted features of the next transaction.

**Task Breakdown:**

*   **Task 3.1: Define Comprehensive Transaction Feature Schema (COMPLETED)**
    *   **Description:** Analyzed historical transaction data (THORChain sample) and Maya Protocol API documentation to identify all potential fields. For each field, determined its type and defined a preprocessing strategy (ID mapping, feature hashing, scaling, placeholder IDs, presence flags).
    *   **Deliverable:** `Docs/GenerativeModel_Feature_Schema.md` detailing the schema.
    *   **Success Criteria:** Schema document is complete, covering all transaction fields with clear processing plans.

*   **Task 3.2: Refactor `src/preprocess_ai_data.py` for Generative Model (COMPLETED)**
    *   **Description:** Overhauled the script based on the new schema. Removed CoinGecko logic. Process raw JSON transactions. Implemented feature hashing, placeholder IDs, and presence flags. The target for an input sequence is the full feature vector of the next transaction. Updated artifact saving (`model_config_generative_thorchain.json`).
    *   **Deliverable:** Updated `src/preprocess_ai_data.py`; new `sequences_and_targets_generative_thorchain.npz` format; `model_config_generative_thorchain.json`.
    *   **Success Criteria:** Script produces the new NPZ format and artifacts aligned with the schema using `data/transactions_data.json`.

*   **Task 3.3: Design and Implement Generative Model (`src/model.py`) (COMPLETED)**
    *   **Description:** Adapted the Transformer model to be generative (`GenerativeTransactionModel`). The output layer produces predictions for each feature of the target transaction (logits for categorical, regression for numerical/flags). Implemented `feature_output_info` for loss calculation.
    *   **Deliverable:** Updated `src/model.py` with the new `GenerativeTransactionModel`.
    *   **Success Criteria:** Model instantiates and its forward pass produces outputs matching the target transaction feature structure.

*   **Task 3.4: Update `src/train_model.py` for Generative Model (COMPLETED)**
    *   **Description:** Adapted the training script for the new data format and generative model. Implemented a composite loss function (sum of cross-entropy for categoricals, BCE for flags, MSE for numericals). Trained for 50 epochs successfully.
    *   **Deliverable:** Updated `src/train_model.py`.
    *   **Success Criteria:** Training loop runs with the composite loss, model weights are saved, and validation loss shows improvement.

*   **Task 3.5: Develop `src/evaluate_model_generative.py` (IN PROGRESS)**
    *   **Description:** Created a new evaluation script. Calculates per-feature prediction metrics (accuracy, F1, MSE, MAE). Implemented methods to assess overall transaction similarity and to analyze predicted sequences. Script runs successfully for 50-epoch model.
    *   **Deliverable:** New `src/evaluate_model_generative.py`; evaluation reports/plots in `evaluation_results_generative/`.
    *   **Success Criteria:** Script produces meaningful metrics for generative quality and allows for analysis of predicted transaction sequences. Initial metrics generated.

## Phase 4: Advanced Generative Model Refinements & Application (Future - Post Generative Model V1)
- Topics for consideration:
    - Implementing robust train/validation/test splits for generative model data.
    - Fetching and integrating large-scale Maya Protocol raw JSON data.
    - Integrating mempool data into the generative model.
    - Developing strategies based on sequences predicted by the generative model (e.g., arbitrage detection, optimal routing).
    - Real-time inference pipeline for the generative model.
    - Monitoring and retraining strategies for the generative model.

## Timeline (Adjusted based on progress)
- **Phase 1:** Completed
- **Phase 2:** COMPLETED & SUPERSEDED
- **Phase 3 (Generative Model):** CURRENT FOCUS
- **Phase 4 (Advanced Generative Model Refinements & Application):** Future

## Phase 5: Data and Model Iteration (OLD - Absorbed into Generative Model Lifecycle)

*   ~~**Task 5.1: Expand Historical Data Collection**~~
    *   ~~**Description:** Modify `src/fetch_realtime_transactions.py` to retrieve approximately 100 days (2400 hours) of historical transaction data from the Midgard API. This will involve handling potential API pagination for large data requests and managing increased data volume.~~
    *   ~~**Code Guidance:** Update API call parameters for extended time ranges. Implement a loop with offset/limit parameters if Midgard uses pagination for large results.~~
    *   ~~**Deliverable:** Updated `src/fetch_realtime_transactions.py` capable of fetching extended historical data; strategy for managing and processing the larger dataset (e.g., updating `data/historical_transactions.csv` or using a new file structure).~~
    *   **(Note: This is now part of Phase 4, focused on raw JSON for the generative model.)**

## Risks and Mitigations
- **Risk:** Incomplete Maya blockchain data (Mitigated by using THORChain data for initial generative model dev; to be addressed by robust Maya Midgard fetching in Phase 4).
- **Risk:** Model too large for MacBook (To be assessed during model finalization).
  - **Mitigation:** Reduce layers or use model distillation.
- **Risk:** Slow inference (To be assessed post-training).
  - **Mitigation:** Shorten sequence length or optimize model.

## Next Steps
- Continue refining and evaluating the **Phase 3 Generative Transaction Prediction Model** (Task 3.5).
- Plan for Phase 4, including robust data fetching and train/test splitting for Maya data.