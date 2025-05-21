# Implementation Plan

The implementation is divided into five phases. The total timeline is flexible, adapting to development progress.

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

## Phase 3: Generative Transaction Prediction Model - Core Development (Largely Completed)

**Overarching Goal:** Shift to predicting the *entire next transaction* in a sequence using only internal protocol data.

**Key Technical Decisions & Outcomes:**
-   **Data Inclusion:** All transaction types from raw JSON (initially THORChain sample, future Maya Midgard).
-   **Feature Engineering:** Comprehensive schema (`Docs/GenerativeModel_Feature_Schema.md`), placeholder IDs, presence flags, feature hashing (`mmh3`) for addresses.
-   **Model Architecture:** Transformer-based (`GenerativeTransactionModel`) outputting a full predicted transaction vector.
-   **Target Variable:** Complete feature vector of the subsequent transaction.
-   **Loss Function:** Composite loss (cross-entropy for categoricals, BCE for flags, MSE for numericals).
-   **Artifacts:** `model_config.json` (augmented by training script) is central, storing feature processing details, model architecture, and output info. `all_id_mappings_generative_mayachain_s25.json` stores mappings.

**Task Breakdown & Status:**

*   **Task 3.1: Define Comprehensive Transaction Feature Schema (COMPLETED)**
    *   **Deliverable:** `Docs/GenerativeModel_Feature_Schema.md`.
*   **Task 3.2: Refactor `src/preprocess_ai_data.py` for Generative Model (COMPLETED)**
    *   **Deliverable:** Updated `src/preprocess_ai_data.py`; new NPZ format; `model_config_generative_thorchain_s25_l6.json` and `scaler_generative_thorchain_s25_l6.pkl` (example filenames for s25_l6 model).
*   **Task 3.3: Design and Implement Generative Model (`src/model.py`) (COMPLETED)**
    *   **Deliverable:** Updated `src/model.py` with `GenerativeTransactionModel`.
*   **Task 3.4: Update `src/train_model.py` for Generative Model (COMPLETED)**
    *   **Deliverable:** Updated `src/train_model.py`; trained models (e.g., `best_generative_model_thorchain_s25_l6.pth`). `model_config.json` is augmented with model details.
*   **Task 3.5: Develop `src/evaluate_model_generative.py` (COMPLETED)**
    *   **Deliverable:** New `src/evaluate_model_generative.py`; evaluation reports/plots (e.g., in `evaluation_results_generative_s25_l6/`). Includes weighted performance score using `feature_weights.json`.
*   **Task 3.6: Experiment with Model Architecture (COMPLETED)**
    *   **Description:** Trained and evaluated a model with sequence length 25 and 6 encoder layers (`s25_l6`).
    *   **Outcome:** Achieved an Overall Weighted Performance Score of **0.8316**.

## Phase 4: Realtime Inference Suite & Advanced Generative Model Refinements (CURRENT FOCUS)

**Overarching Goal:** Build and test `src/realtime_inference_suite.py` for running the generative model in simulation and live prediction modes. Refine the model and data pipeline for real-world Maya Protocol data.

**Task Breakdown & Status:**

*   **Task 4.1: Core Infrastructure for `realtime_inference_suite.py` (COMPLETED)**
    *   **Description:** Setup argument parsing, model loading (`load_model_and_artifacts`), and basic mode dispatch.
*   **Task 4.2: Implement Preprocessing for Inference (COMPLETED)**
    *   **Description:** Developed `preprocess_single_action_for_inference` and `preprocess_sequence_for_inference` to handle raw Midgard JSONs and feedback from simulations.
*   **Task 4.3: Implement Prediction Decoding (COMPLETED)**
    *   **Description:** Developed `decode_prediction` to convert raw model output vector into a flat, human-readable dictionary of unscaled/un-normalized feature values.
*   **Task 4.4: Implement Generative Simulation Mode (`run_generative_simulation`) (COMPLETED - Initial Version)**
    *   **Description:** Implemented simulation loop with seed sequence, prediction, decoding, and feedback preprocessing. Saves flat decoded transactions.
*   **Task 4.5: Implement Midgard-Format Reconstruction (`reconstruct_to_midgard_format`) (IN PROGRESS)**
    *   **Description:** Convert flat decoded transactions into a nested JSON structure mimicking Midgard's `/actions` output. This is key for usability and comparison.
    *   **Current Challenge:** Ensuring the `edit_file` tool correctly applies the latest version of this function, which must use numerical values directly from `flat_decoded_tx` (as they are pre-processed by `decode_prediction`).
    *   **Next Step:** Verify/correct the function in `src/realtime_inference_suite.py`, then test its output structure and values thoroughly.
*   **Task 4.6: Implement Live Next Transaction Prediction Mode (`run_next_transaction_prediction`) (FUTURE)**
    *   **Description:** Develop the mode to fetch live data, predict the next transaction, wait for the actual transaction, and compare.
*   **Task 4.7: Robust Testing and Refinement of Inference Suite (FUTURE)**
    *   **Description:** Thoroughly test both simulation and live prediction modes. Improve error handling, logging, and output formats.
*   **Task 4.8: Integrate Real Maya Protocol Data (FUTURE)**
    *   **Description:** Adapt `src/fetch_realtime_transactions.py` to fetch large-scale raw JSON from Maya Midgard. Implement robust train/validation/test splits using this data. Retrain and re-evaluate the generative model on actual Maya data.

## Phase 5: Deployment and Iteration (Future)
-   Deploying the refined system.
-   Developing strategies based on predicted sequences (arbitrage, optimal routing, etc.).
-   Establishing continuous monitoring, retraining, and iteration cycles.
-   Enhancing the (currently minimal) Flask Web UI to visualize insights from the generative model.

## Risks and Mitigations
- **Risk:** `edit_file` tool instability for complex changes (Mitigation: Simpler edits, full function replacement, careful verification by the next agent).
- **Risk:** Model performance on real Maya data vs. sample THORChain data (Mitigation: Task 4.8 focuses on retraining with Maya data).
- **Risk:** Model too large for desired inference speed/resources (Mitigation: Future optimization, distillation if needed).

## Next Steps
- **Immediate:** Resolve Task 4.5 by ensuring `reconstruct_to_midgard_format` in `src/realtime_inference_suite.py` is correct and verified.
- **Following:** Proceed with Task 4.6 (live prediction mode) and subsequent tasks in Phase 4 as outlined in the scratchpad.