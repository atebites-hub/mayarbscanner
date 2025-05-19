# .cursor/scratchpad.md - Maya Protocol Arbitrage Scanner

## Background and Motivation

The project aims to build an arbitrage scanner for the Maya Protocol. This involves fetching historical and real-time transaction data, preprocessing it for an AI model, and using that model to predict arbitrage opportunities. A Flask web UI will display the data and insights.

Phase 1 (Data Fetching & Initial UI) is complete. We are currently in Phase 2 (Model Development).

A key clarification has been made:
-   **Arbitrage transactions (`ARB_SWAP`)** are SWAP transactions that *do not* have an `affiliate_id`.
-   **User swaps (`USER_SWAP`)** are SWAP transactions that *do* have an `affiliate_id`.

This new logic, along with a switch from Uniswap to CoinGecko for external price data, has led to a significant refactoring of `src/preprocess_ai_data.py`.

## Key Challenges and Analysis

-   **Data Preprocessing Complexity:** Ensuring `src/preprocess_ai_data.py` correctly implements the new arbitrage logic and CoinGecko integration is crucial. The calculation of `target_mu_profit` needs careful verification.
-   **CoinGecko API Usage:**
    -   **Rate Limiting:** The free tier of CoinGecko API has rate limits (10-30 calls/minute). In-memory caching per run was implemented.
    -   **File-based Caching:** Daily CoinGecko prices (asset, date -> price) are now cached to `data/processed_ai_data/coingecko_price_cache.json`. This significantly speeds up subsequent runs and reduces API calls for existing data. 
        -   *Future Consideration:* When implementing higher-granularity (e.g., minute-by-minute) price fetching with a paid API, this file-based caching will be even more critical for managing API call volume and costs over time, ensuring that previously fetched granular prices are reused.
    -   **Data Granularity & Potential Bias:** The free CoinGecko API (`get_coin_history_by_id`) provides daily historical prices. This is a known simplification. 
        -   *Implication*: For transactions very close to the time of the daily CoinGecko price snapshot (especially the most recent transactions in a 24-hour data pull), the internal Maya price (`P_m`) and the external daily CoinGecko price (`P_u`) might appear more aligned simply due to less time for divergence. This could potentially lead the model to underestimate arbitrage opportunities for these recent transactions compared to if a more contemporaneous external price was available. This underscores the importance of future improvements with higher-granularity data.
        -   *Future Improvement: With a paid CoinGecko subscription, investigate endpoints for more granular historical data.*
    -   **Timestamp Conversion:** Maya Protocol timestamps (nanoseconds) are converted to seconds for CoinGecko.
    -   **Asset Identifier Mapping:** Ensuring all relevant asset strings from Maya data (e.g., 'ETH.ETH', 'ETH/ETH', specific contract addresses for ERC20s) are correctly mapped to CoinGecko IDs in `MAYA_TO_COINGECKO_MAP` is vital for comprehensive price fetching. This has been significantly improved.
-   **Model Adaptation:** The AI model (`src/model.py`) and training script (`src/train_model.py`) will need substantial updates.

## High-level Task Breakdown (Phase 2 - Model Development)

1.  **Task 2.0: Refactor and Verify `src/preprocess_ai_data.py` (COMPLETED)**
    *   **Description:** The `src/preprocess_ai_data.py` script was updated for the new arbitrage logic, CoinGecko integration, refined profit calculation, separate categorical/numerical outputs, and robust file-based caching.
    *   **Key Outcomes:** Produces `sequences_and_targets.npz` with `X_cat_ids_sequences`, `X_num_sequences`, `y_p_targets`, `y_mu_targets`, and column name lists. Caching is verified.
    *   **Deliverable:** Verified `src/preprocess_ai_data.py`.

2.  **Task 2.1: Update `src/model.py` - Transformer Encoder (COMPLETED)**
    *   **Description:** Verify and ensure the Transformer Encoder in `src/model.py` correctly accepts the separate categorical ID sequences (`X_cat_ids_sequences`) and numerical sequences (`X_num_sequences`) from the refactored preprocessing script. This includes ensuring embedding layers are appropriately defined and the model's `forward` pass correctly processes and combines these inputs.
    *   **Key Outcomes:** 
        *   Confirmed `model.py` structure (`self.categorical_feature_keys`, `self.embedders`, `forward` method) correctly handles the separate categorical ID and numerical inputs.
        *   Order of categorical features in `model.py` matches `categorical_feature_columns` from `preprocess_ai_data.py`.
        *   Updated the example `if __name__ == '__main__':` block in `model.py` to reflect the correct number of numerical features (8).
    *   **Success Criteria:** Model structure is verified to be compatible with the new data format. Example usage updated.
    *   **Deliverable:** Verified `src/model.py` for encoder input compatibility.

3.  **Task 2.2: Update `src/model.py` - Prediction Heads (COMPLETED)**
    *   **Description:** Verify the prediction heads in `src/model.py` are correctly defined for the two target variables.
        *   `p_target` head: Predicts the next actor type (`ARB_SWAP`, `USER_SWAP`, `NON_SWAP`, `UNK_ACTOR`) - categorical output (4 classes).
        *   `mu_target` head: Predicts a single float value for profit if the next transaction is an `ARB_SWAP`.
    *   **Key Outcomes:**
        *   Confirmed `p_head` in `model.py` outputs `p_target_classes` (e.g., 4) logits.
        *   Confirmed `mu_head` in `model.py` outputs `mu_target_dim` (e.g., 1) value.
        *   These align with `y_p_targets` and `y_mu_targets` from `preprocess_ai_data.py`.
    *   **Success Criteria:** Model prediction heads are verified to match the shape and type of target variables.
    *   **Deliverable:** Verified `src/model.py` for prediction head compatibility.

4.  **Task 2.3 (REVISED): Update `src/train_model.py` (NEXT FOCUS)**
    *   **Description:** Update the training script to:
        *   Load the new data format from `sequences_and_targets.npz` (i.e., `X_cat_ids_sequences`, `X_num_sequences`, `y_p_targets`, `y_mu_targets`).
        *   Determine vocabulary sizes for each categorical feature by loading the corresponding `*.json` mapping files (e.g., `asset_to_id.json`, `type_to_id.json`, `status_to_id.json`, `actor_type_to_id.json`) from `data/processed_ai_data/` and using `len(mapping)`.
        *   Determine the number of numerical features from the shape of `X_num_sequences`.
        *   Instantiate the revised model from `src/model.py` with these dynamically determined vocabulary sizes and number of numerical features.
        *   Implement appropriate loss functions: `CrossEntropyLoss` for `p_target`, `MSELoss` for `mu_target` (masked to apply only when actual next is `ARB_SWAP`).
    *   **Success Criteria:** Training loop runs without errors using the new data and model structure; loss values decrease over initial epochs.

5.  **Task 2.4: Model Evaluation**
    *   **Description:** Implement and run model evaluation metrics (e.g., accuracy/F1 for actor type, RMSE/MAE for profit prediction).
    *   **Success Criteria:** Evaluation metrics are calculated and reported.

6.  **Task 2.5 (Validation): Perform Full Pipeline Test with Fresh Data and Document**
    *   **Description:** Execute the entire data pipeline (fetch, preprocess, train, evaluate) using newly fetched transaction data. This serves as a comprehensive test of the current system with fresh inputs and ensures reproducibility. Document this procedure in `README.md` for ongoing use. Maximize CoinGecko cache utilization.
    *   **Sub-steps:**
        1.  Fetch fresh 24hr transaction data using `src/fetch_realtime_transactions.py`.
        2.  Preprocess the data using `src/preprocess_ai_data.py`, ensuring CoinGecko cache is utilized.
        3.  Train the model using `src/train_model.py` on the newly preprocessed data.
        4.  Evaluate the newly trained model using `src/evaluate_model.py`.
        5.  Update `README.md` with instructions for this full test run.
        6.  Update this scratchpad task status.
    *   **Success Criteria:** All pipeline scripts execute successfully with fresh data. `README.md` is updated with clear instructions. Evaluation metrics from the fresh run are available.
    *   **Deliverable:** Successful pipeline execution with fresh data, updated `README.md`, and updated scratchpad.
    *   **Status:** SUPERSEDED by Task 2.6 due to miscommunication on testing methodology.

7.  **Task 2.6: Implement Independent Train/Test Evaluation Workflow**
    *   **Description:** Refactor the data fetching, preprocessing, training, and evaluation scripts to support a true train/test split for model evaluation. The model will be trained on one dataset and then evaluated on a separate, unseen dataset from an earlier time period.
    *   **Data Definition:**
        *   **Training/Validation Data:** Transactions from the most recent 24 hours.
        *   **Test Data:** Transactions from the 24-hour period immediately preceding the training data (i.e., 48 hours ago to 24 hours ago).
    *   **Key Script Modifications & Workflow:**
        1.  **`src/fetch_realtime_transactions.py`:**
            *   Add arguments: `--output-file` (e.g., `data/training_transactions.csv`, `data/test_transactions.csv`), `--hours-ago-start` (e.g., `24` for training, `48` for test), `--duration-hours` (e.g., `24`).
            *   **Deliverable:** Script can fetch distinct datasets for training and testing. (COMPLETED)
        2.  **`src/preprocess_ai_data.py`:**
            *   Add arguments: `--mode` (`train` or `test`), `--input-csv`, `--output-npz` (e.g., `training_sequences.npz`, `test_sequences.npz`), `--artifacts-dir` (for scaler, mappings, model_config).
            *   `train` mode: Learns and saves scaler, mappings, and model config (`model_config.json`). Saves to specified output NPZ.
            *   `test` mode: Loads scaler, mappings. Applies them. Saves to specified output NPZ. Does *not* save model config.
            *   **Deliverable:** Script can preprocess data in 'train' and 'test' modes, managing artifacts correctly. (COMPLETED)
        3.  **`src/train_model.py`:**
            *   Add arguments: `--input-npz` (for training sequences), `--model-config-path` (to load), `--model-save-dir`.
            *   Uses loaded config to instantiate the model. Saves trained model weights to the specified directory.
            *   **Deliverable:** Script trains model using specified sequences and pre-defined model configuration. (COMPLETED)
        4.  **`src/evaluate_model.py`:**
            *   Add arguments: `--input-npz` (for test sequences), `--model-config-path` (to load), `--model-weights-path`, `--output-dir` (for plots).
            *   Uses loaded config and weights for evaluation on the specified test data. Saves plots to output directory.
            *   **Deliverable:** Script evaluates a pre-trained model on specified test sequences using a pre-defined model configuration. (COMPLETED)
    *   **Overall Success Criteria:**
        *   All scripts are successfully refactored with new arguments and logic.
        *   The new workflow (fetch train data, fetch test data, preprocess train, train model, preprocess test, evaluate on test) executes without errors.
        *   Model evaluation metrics are generated based on an independent test set.
        *   `README.md` is updated with clear instructions for the new train/test workflow.
    *   **Deliverable:** Fully functional and documented independent train/test evaluation pipeline.

## Project Status Board

-   [x] Phase 1: Data Fetching, Parsing, Storage, Initial UI, Initial AI Preprocessing (Uniswap)
-   [x] Phase 2 - Task 2.0: Refactor and Verify `src/preprocess_ai_data.py` (CoinGecko, new profit, separate cat/num outputs, caching fixed)
-   [x] Phase 2 - Task 2.1: Update `src/model.py` - Transformer Encoder (Verified, example updated)
-   [x] Phase 2 - Task 2.2: Update `src/model.py` - Prediction Heads (Verified)
-   [x] Phase 2 - Task 2.3: Update `src/train_model.py` (COMPLETED - Training loop runs, losses decrease, NaN issues resolved)
-   [x] Phase 2 - Task 2.4: Model Evaluation (COMPLETED - Initial metrics and plots generated)
-   [x] Phase 2 - Task 2.5 (Validation): Perform Full Pipeline Test with Fresh Data and Document (SUPERSEDED)
-   [x] Phase 2 - Task 2.6: Implement Independent Train/Test Evaluation Workflow (COMPLETED)
    -   [x] Sub-Task 2.6.1: Modify `src/fetch_realtime_transactions.py` (COMPLETED)
    -   [x] Sub-Task 2.6.2: Modify `src/preprocess_ai_data.py` (COMPLETED)
    -   [x] Sub-Task 2.6.3: Modify `src/train_model.py` (COMPLETED)
    -   [x] Sub-Task 2.6.4: Modify `src/evaluate_model.py` (COMPLETED)
    -   [x] Sub-Task 2.6.5: Execute full train/test workflow (COMPLETED)
        -   [x] Fetch training data (most recent 24h, 1840 actions) to `data/training_transactions.csv` (COMPLETED)
        -   [x] Fetch test data (48h ago to 24h ago, 1128 actions) to `data/test_transactions.csv` (COMPLETED)
        -   [x] Preprocess training data (`data/training_transactions.csv` -> `data/processed_ai_data/training_sequences.npz`, artifacts to `data/processed_ai_data/`) (COMPLETED)
        -   [x] Train model (using `data/processed_ai_data/training_sequences.npz` and `data/processed_ai_data/model_config.json`, models saved to `models/best_arbitrage_model.pth` and `models/final_arbitrage_model.pth`) (COMPLETED)
        -   [x] Preprocess test data (`data/test_transactions.csv` -> `data/processed_ai_data/test_sequences.npz`, using existing artifacts, 1118 sequences generated) (COMPLETED)
        -   [x] Evaluate model on test data (COMPLETED - See Executor's Feedback)
    -   [x] Sub-Task 2.6.6: Update `README.md` with new workflow instructions (COMPLETED)
-   [ ] **CURRENT FOCUS: Discuss evaluation results and decide next steps (Model Iteration or Phase 3)**

## Executor's Feedback or Assistance Requests

-   **Task 2.6 (Independent Train/Test) - Data Fetching:**
    -   Successfully fetched training data (most recent 24h, 1840 actions) to `data/training_transactions.csv`.
    -   Successfully fetched test data (48h-24h ago, 1128 actions) to `data/test_transactions.csv`.

-   **Task 2.6 (Independent Train/Test) - Preprocess Training Data:**
    -   Successfully preprocessed `data/training_transactions.csv` in `train` mode.
    -   Artifacts created/updated in `data/processed_ai_data/`: `training_sequences.npz`, `scaler.pkl`, `model_config.json`, and mapping files.
    -   1830 sequences generated.

-   **Task 2.6 (Independent Train/Test) - Train Model:**
    -   Successfully trained model using `data/processed_ai_data/training_sequences.npz` and `data/processed_ai_data/model_config.json`.
    -   Models saved to `models/best_arbitrage_model.pth` and `models/final_arbitrage_model.pth`.

-   **Task 2.6 (Independent Train/Test) - Preprocess Test Data:**
    -   Successfully preprocessed `data/test_transactions.csv` in `test` mode.
    -   Used existing artifacts (scaler, mappings) from `data/processed_ai_data/`.
    -   Output saved to `data/processed_ai_data/test_sequences.npz`.
    -   1118 sequences generated.
    -   CoinGecko cache was utilized and updated (14 new API calls).

-   **Task 2.5 (Validation) - Sub-step 1 (Fetch Data):** Successfully fetched 1929 actions from the last 24 hours. Data saved to `data/historical_24hr_maya_transactions.csv`.

-   **Task 2.5 (Validation) - Sub-step 2 (Preprocess Data):** Successfully preprocessed the fresh data. `sequences_and_targets.npz` and other related files in `data/processed_ai_data/` have been updated. The CoinGecko cache was read (22 items) and updated (25 items), indicating new prices were fetched and cached as expected.

-   **Task 2.5 (Validation) - Sub-step 3 (Train Model):** Successfully trained the model using the newly preprocessed data. The training ran for 50 epochs. `models/best_arbitrage_model.pth` and `models/final_arbitrage_model.pth` have been updated.

-   **Task 2.5 (Validation) - Sub-step 4 (Evaluate Model):** Successfully evaluated the newly trained `best_arbitrage_model.pth`. Metrics and plots (`models/confusion_matrix_actor_type.png`, `models/scatter_plot_mu_profit.png`) have been updated. See new metrics below.

-   **Task 2.5 (Validation) - Sub-step 5 (Update README):** `README.md` has been updated with instructions for performing an end-to-end test with fresh data.

-   **Task 2.5 (Validation) - COMPLETED:** The full pipeline test with fresh data is complete, and documentation has been updated.

-   **Model Evaluation (Task 2.4) Complete.** Key results from `src/evaluate_model.py` on the validation set:
    -   **Actor Type Prediction:**
        -   Overall Accuracy: 90.31%
        -   `ARB_SWAP`: Precision=0.9058, Recall=0.9909, F1=0.9465 (Very good at finding ARB_SWAPs)
        -   `USER_SWAP`: All metrics 0.0 (Likely no USER_SWAPs in validation set or all misclassified)
        -   `NON_SWAP`: Precision=0.8571, Recall=0.3462, F1=0.4932 (Low recall, many NON_SWAPs misclassified as ARB_SWAP)
        -   `UNK_ACTOR`: All metrics 0.0
        -   Confusion Matrix saved to `models/confusion_matrix_actor_type.png`.
    -   **Profit (Mu) Prediction (for true ARB_SWAPs with non-NaN targets, N=318):**
        -   MSE: 134.5363
        -   RMSE: 11.5990
        -   MAE: 9.2995
        -   Scatter plot saved to `models/scatter_plot_mu_profit.png`.

-   **NEW Model Evaluation Results (Task 2.5 - Fresh Data Run):**
    -   **Actor Type Prediction:**
        -   Overall Accuracy: 91.38%
        -   `ARB_SWAP`: Precision=0.9122, Recall=1.0000, F1=0.9541
        -   `USER_SWAP`: All metrics 0.0
        -   `NON_SWAP`: Precision=1.0000, Recall=0.1750, F1=0.2979 (Low recall persists)
        -   `UNK_ACTOR`: All metrics 0.0
        -   Confusion Matrix updated: `models/confusion_matrix_actor_type.png`.
    -   **Profit (Mu) Prediction (for true ARB_SWAPs with non-NaN targets, N=328):**
        -   MSE: 125.3422
        -   RMSE: 11.1956
        -   MAE: 9.2270
        -   Scatter plot updated: `models/scatter_plot_mu_profit.png`.

-   **Next Steps & Considerations:**
    -   Review generated plots (`confusion_matrix_actor_type.png`, `scatter_plot_mu_profit.png`).
    -   Check class distribution of `actor_label` in `data/processed_ai_data/intermediate_processed_data.csv` to understand `USER_SWAP` prevalence.
    -   Discuss strategies for improving `NON_SWAP` recall and overall model performance if desired before moving to Phase 3.

## Lessons

-   When calculating profit metrics involving ratios or inverse prices (like `1/P_m`), be highly aware of potential numerical instability and division by small numbers leading to extreme outliers. 
-   Simple percentile clipping might not be enough if the underlying distribution is extremely skewed; the percentiles themselves can be outliers.
-   Log-transforming ratios (`log(A/B)`) can be an effective way to stabilize variance, make the distribution more symmetric, and handle scale issues for target variables in regression tasks, especially when A and B are prices or quantities.

## Future Enhancements (User Requested)

-   **Expand Historical Data Fetching:** Increase the historical data collection period from the current 24 hours to approximately 100 days (~2400 hours). This will require:
    -   Modifying `src/fetch_realtime_transactions.py` to handle requests for extended time ranges.
    -   Implementing logic to handle API pagination if Midgard returns large datasets in chunks.
    -   Consideration for increased data storage and processing times for both data fetching and subsequent model training.

## Lessons

-   **Arbitrage Definition:** `

-   **Task 2.6 (Independent Train/Test) - Evaluate Model on Test Data:**
    -   Model `models/best_arbitrage_model.pth` evaluated on `data/processed_ai_data/test_sequences.npz`.
    -   **Actor Type Prediction:**
        -   Overall Accuracy: 83.09%
        -   `ARB_SWAP`: Precision=0.8309, Recall=1.0000, F1=0.9077
        -   `NON_SWAP`: Precision=0.0000, Recall=0.0000, F1=0.0000 (All 189 NON_SWAP in test set misclassified as ARB_SWAP)
        -   Confusion Matrix saved to `models/confusion_matrix_actor_type.png`.
    -   **Profit (Mu) Prediction (for 287 true ARB_SWAPs with non-NaN targets):**
        -   MSE: 118.7236
        -   RMSE: 10.8960
        -   MAE: 9.3089
        -   Scatter plot saved to `models/scatter_plot_mu_profit.png`.
    -   *Key Issue: Model struggles significantly with NON_SWAP classification, labeling them as ARB_SWAP.*