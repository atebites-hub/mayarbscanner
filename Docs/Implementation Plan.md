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

## Phase 2: Model Development and Data Refinement (In Progress)

*   **Task 2.0: Refactor `src/preprocess_ai_data.py` and Verify (COMPLETED)**
    *   **Description:** Updated the preprocessing script based on the new arbitrage identification logic (SWAPs without `affiliate_id` are `ARB_SWAP`) and switched external price source from Uniswap to CoinGecko. Iteratively debugged and refined asset mapping, scaling logic, and timestamp handling.
    *   **Sub-tasks Completed:**
        *   Implemented new `actor_type_id` (`ARB_SWAP`, `USER_SWAP`, `NON_SWAP`).
        *   Integrated CoinGecko API (`pycoingecko`) with caching for fetching historical prices (`P_u`).
        *   Expanded `MAYA_TO_COINGECKO_MAP` to cover more asset string formats from data.
        *   Corrected `NUMERICAL_FEATURES_TO_SCALE` to resolve scaling warnings.
        *   Update `target_mu_profit` calculation using Maya internal price (`P_m`) and CoinGecko price (`P_u`).
    *   **Deliverable:** Refactored and verified `src/preprocess_ai_data.py`, now producing `sequences_and_targets.npz` with high data integrity for `P_u` and `target_mu_profit`.

*   **Task 2.1 (REVISED): Update `src/model.py` - Transformer Encoder (NEXT FOCUS)**
    *   **Description:** Modify the Transformer Encoder in `src/model.py` to accept new input features (approx. 90 features after one-hot encoding and scaling) from the refactored preprocessing script. Ensure embedding layers are correctly configured for the one-hot encoded categorical features.
    *   **Deliverable:** Updated `src/model.py` with the revised encoder structure.

*   **Task 2.2 (REVISED): Update `src/model.py` - Prediction Heads** (Pending Task 2.1 completion)
    *   **Description:** Update the prediction heads in `src/model.py`:
        *   `p_target` head: To predict the next actor type (`ARB_SWAP`, `USER_SWAP`, `NON_SWAP`, `UNK_ACTOR`).
        *   `mu_target` head: To predict a single float value for profit if the next transaction is an `ARB_SWAP`.
    *   **Deliverable:** Updated `src/model.py` with revised prediction heads.

*   **Task 2.3 (REVISED): Update `src/train_model.py`** (Pending Task 2.2 completion)
    *   **Description:** Update the training script to load the new data format from `sequences_and_targets.npz`, instantiate the revised model, and adjust loss functions (`CrossEntropyLoss` for `p_target`, `MSELoss` for `mu_target` with masking).
    *   **Deliverable:** Updated `src/train_model.py`.

*   **Task 2.4: Model Evaluation** (Pending successful training from Task 2.3)
    *   **Description:** Implement and run model evaluation metrics.
    *   **Deliverable:** Evaluation results and analysis.

## Phase 3: Advanced Model Features and UI Integration (Future)

## Phase 4: Real-Time Inference and Deployment (Future)
- **Task 4.1: Set Up Real-Time Pipeline**
  - **Description:** Automate fetching and encoding of the last `M` trades.
  - **Code Guidance:** Use a loop or event-driven script (e.g., with `asyncio`) to update data.
  - **Deliverable:** A live data pipeline.

- **Task 4.2: Integrate Mempool Data**
  - **Description:** Adjust predictions based on pending trades.
  - **Code Guidance:** Write logic to override `p_j=1` and `μ_j=pending_ΔX` for mempool-detected arbs.
  - **Deliverable:** Enhanced prediction script.

- **Task 4.3: Implement Arbitrage Strategy**
  - **Description:** Set `ΔX` based on predictions.
  - **Code Guidance:** Calculate `ΔX = max(predicted μ_j) + 0.0000001`, simulate profit.
  - **Deliverable:** A strategy execution module.

- **Task 4.4: Monitor and Retrain**
  - **Description:** Automate performance tracking and weekly retraining.
  - **Code Guidance:** Use a cron job or script to retrain with new data.
  - **Deliverable:** A monitoring and retraining system.

## Timeline (Adjusted based on progress)
- **Phase 1:** Completed
- **Phase 2:** In Progress (Task 2.3 in focus)
- **Phase 3 & 4:** Future

## Phase 5: Data and Model Iteration (Future)

*   **Task 5.1: Expand Historical Data Collection**
    *   **Description:** Modify `src/fetch_realtime_transactions.py` to retrieve approximately 100 days (2400 hours) of historical transaction data from the Midgard API. This will involve handling potential API pagination for large data requests and managing increased data volume.
    *   **Code Guidance:** Update API call parameters for extended time ranges. Implement a loop with offset/limit parameters if Midgard uses pagination for large results.
    *   **Deliverable:** Updated `src/fetch_realtime_transactions.py` capable of fetching extended historical data; strategy for managing and processing the larger dataset (e.g., updating `data/historical_transactions.csv` or using a new file structure).

## Risks and Mitigations
- **Risk:** Incomplete Maya blockchain data (Mitigated by successful historical fetch for now).
- **Risk:** Model too large for MacBook (To be assessed during model finalization).
  - **Mitigation:** Reduce layers or use model distillation.
- **Risk:** Slow inference (To be assessed post-training).
  - **Mitigation:** Shorten sequence length or optimize model.
- **Risk:** CoinGecko API limitations (Free tier daily granularity, rate limits - caching implemented).
  - **Mitigation:** Use paid tier for finer granularity if needed for model performance.

## Next Steps
- Proceed with **Task 2.1: Update `src/model.py` - Transformer Encoder**.