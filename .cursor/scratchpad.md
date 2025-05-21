# .cursor/scratchpad.md - Maya Protocol Arbitrage Scanner

## Background and Motivation

The project aims to build an arbitrage scanner for the Maya Protocol. This involves fetching historical and real-time transaction data, preprocessing it for an AI model, and using that model to predict arbitrage opportunities.

We have pivoted towards a **Generative Transaction Prediction Model**. The goal is to predict the *entire next transaction* in a sequence. Arbitrage identification will be an *emergent property* derived from analyzing the sequence of predicted transactions.

## Key Challenges and Analysis

-   **Comprehensive Feature Engineering:** Defining a fixed schema for all Maya transaction types and handling missing values (special IDs, presence flags, hashing).
-   **High-Dimensional Target Prediction:** Predicting a large number of features for the next transaction.
-   **Composite Loss Function:** Appropriately weighting errors from different feature types.
-   **Evaluation Metrics:** Developing metrics for transaction quality (per-feature accuracy, overall similarity, weighted scores).
-   **Realtime Inference:** Implementing a robust suite for live prediction and simulation, including feedback loops and accurate reconstruction of Midgard-like JSON.
-   **Accurate Numerical Scaling:** Handling diverse scales of transaction amounts (e.g., BTC vs. Dogecoin) and fees effectively to prevent model prediction distortion.

## High-level Task Breakdown

**Phase 1: Data Fetching & Initial UI (COMPLETED)**

**Phase 2: Arbitrage-Specific Predictive Model (COMPLETED & SUPERSEDED by Generative Model)**

**Phase 3: Generative Transaction Prediction Model - Core Development (LARGELY COMPLETED)**
*   **Task 3.1: Define Comprehensive Transaction Feature Schema (COMPLETED)**
*   **Task 3.2: Refactor `src/preprocess_ai_data.py` for Generative Model (COMPLETED)**
    *   Includes `train` and `test` modes, artifact generation (scaler, mappings, comprehensive `model_config.json` with `feature_processing_details` and `all_id_mappings_generative_mayachain_s25.json`).
*   **Task 3.3: Design and Implement Generative Model (`src/model.py`) (COMPLETED)**
    *   Transformer-based model capable of handling mixed feature types and predicting a full transaction vector.
*   **Task 3.4: Update `src/train_model.py` for Generative Model (COMPLETED)**
    *   Trains the generative model, saves best/final weights, and augments `model_config.json` with model architecture details and output feature info.
*   **Task 3.5: Develop `src/evaluate_model_generative.py` (COMPLETED)**
    *   Evaluates the generative model on a test set.
    *   Calculates per-feature metrics (accuracy for categorical/binary, MAE/MSE for numerical).
    *   Implements an "Overall Weighted Performance Score" using `feature_weights.json`.
*   **Task 3.6: Experiment with Model Architecture & Context Length (COMPLETED)**
    *   Changed `SEQUENCE_LENGTH` from 10 to 25 in `src/preprocess_ai_data.py`.
    *   Trained and evaluated a new model (s25_l6) with 6 encoder layers, batch size 128, 20 epochs.
    *   Achieved "Overall Weighted Performance Score": **0.8316**.

**Phase 4: Realtime Inference Suite (`src/realtime_inference_suite.py`) (IN PROGRESS)**
*   **Task 4.1: Core Infrastructure & Model Loading (COMPLETED)**
*   **Task 4.2: Implement `preprocess_single_action_for_inference` and `preprocess_sequence_for_inference` (COMPLETED)**
*   **Task 4.3: Implement `decode_prediction` (COMPLETED)**
*   **Task 4.4: Implement `run_generative_simulation` (COMPLETED - Tested with `reconstruct_to_midgard_format`)**
*   **Task 4.5: Implement `reconstruct_to_midgard_format` (COMPLETED)**
    *   **Description:** Converts the flat decoded transaction dictionary (from `decode_prediction`) into a nested JSON structure closely resembling Midgard's `/actions` endpoint output.
    *   **Status:** Verified that the function correctly uses numerical values directly from `flat_decoded_tx` (as they are already unscaled/un-normalized by `decode_prediction`). Simulation run and output `simulated_transactions_reconstructed_v3_s25_l6.json` reviewed; `date`, numerical fields, and metadata structures appear correctly reconstructed based on direct model output.
*   **Task 4.6: Implement `run_next_transaction_prediction` (IN PROGRESS - Live testing with new model)**
    *   **Description:** Live mode: fetches latest transactions, predicts N+1, waits for actual N+1, compares.
    *   **Current Status:** Initial implementation complete, needs testing
*   **Task 4.7: Testing, Refinement, and Documentation (Phase 4 Finalization) (FUTURE)**
    *   Thoroughly test both simulation and live prediction modes. Refine outputs, logging, error handling.
*   **Task 4.8: Project Cleanup and Documentation Update (FUTURE)**
    *   Update `README.md` for Phase 3 and 4.
    *   Update `Docs/Implementation Plan.md`.
    *   General code cleanup.

## Project Status Board

-   [x] Phase 1: Data Fetching & Initial UI
-   [x] Phase 2: Arbitrage-Specific Predictive Model
-   [x] Phase 3: Generative Transaction Prediction Model - Core Development
-   [ ] **CURRENT FOCUS: Phase 4 - Realtime Inference Suite (`src/realtime_inference_suite.py`)**
    -   [x] Task 4.1: Core Infrastructure & Model Loading
    -   [x] Task 4.2: Implement `preprocess_single_action_for_inference` and `preprocess_sequence_for_inference`
    -   [x] Task 4.3: Implement `decode_prediction`
    -   [x] Task 4.4: Implement `run_generative_simulation` (Tested with `reconstruct_to_midgard_format`)
    -   [x] Task 4.5: Implement `reconstruct_to_midgard_format` (Verified, numerical handling corrected)
    -   [ ] Task 4.6: Implement `run_next_transaction_prediction` (IN PROGRESS - Live testing with new model)
    -   [ ] Task 4.7: Testing, Refinement, and Documentation (Phase 4 Finalization)
    -   [ ] Task 4.8: Project Cleanup and Documentation Update (Overall Project Finalization)
-   [ ] **Phase 5: Data Refinement & Model Retraining (NEW)**
    -   [x] **Task 5.1: Refine Amount Handling in `preprocess_ai_data.py` (COMPLETED)**
        -   [x] **Sub-Task 5.1.A: Implement Asset-Specific Precision Handling (COMPLETED)**
            -   Define `ASSET_PRECISIONS` map for MayaChain assets (updated with data-driven analysis).
            -   Implement `get_asset_precision` function.
            -   Modify `preprocess_actions_for_generative_model` (within `main`) to correctly handle amounts from `data/transactions_data.json` (confirmed to be already atomic integers) by converting them to numerical types without further multiplication by 10^precision. Store in `*_amount_atomic` columns.
            -   Modify `process_numerical_features_generative` to use these numerical atomic amounts for generating `*_norm_scaled` columns and update `feature_configs_numerical`.
        -   [x] **Sub-Task 5.1.B: Update `decode_prediction` in `realtime_inference_suite.py` (COMPLETED & VERIFIED)**
            -   Copy updated `ASSET_PRECISIONS` map and `get_asset_precision`.
            -   Ensure `decode_prediction` correctly uses this map to convert unscaled ATOMIC model outputs back to standard float string representations by DIVIDING by 10^precision.
        -   [x] **Sub-Task 5.1.C: Verify `reconstruct_to_midgard_format` in `realtime_inference_suite.py` (COMPLETED & VERIFIED)**
            -   Confirm that `reconstruct_to_midgard_format` correctly handles the float string amounts produced by the updated `decode_prediction`.
    -   [ ] **Task 5.2: Implement Advanced Amount Scaling (Per-Asset) (IN PROGRESS)**
        -   [ ] **Sub-Task 5.2.A: Modify `src/preprocess_ai_data.py` for Per-Asset Scaling (TODO)**
            -   Implement dictionary structure for `scalers_collection` (per-asset, global, fallback).
            -   Implement fitting logic for per-asset, global, and fallback scalers.
            -   Implement transform logic using appropriate scalers based on asset context.
            -   Update `model_config.json` to reflect the new scaling strategy for each numerical feature (global vs. per-asset, asset context column).
        -   [ ] **Sub-Task 5.2.B: Modify `src/realtime_inference_suite.py` for Per-Asset Scaling (TODO)**
            -   Update `load_model_and_artifacts` to correctly load and interpret the new `scalers_collection`.
            -   Refactor `decode_prediction` to use the correct per-asset (or global/fallback) scaler during `inverse_transform`, ensuring asset types are decoded before their dependent amounts.
    -   [ ] **Task 5.3: Retrain Generative Model (TODO - Depends on 5.2)**
        -   Run `src/preprocess_ai_data.py` in `train` mode with the corrected per-asset amount handling.
        -   Run `src/train_model.py` using the new data and artifacts.
    -   [ ] **Task 5.4: Evaluate Retrained Model (TODO - Depends on 5.3)**
        -   Run `src/evaluate_model_generative.py` on the new model.
    -   [ ] Task 5.5: Test Realtime Inference with Retrained Model (FUTURE - Depends on 5.3)
    -   [ ] Task 5.6: Implement Proper Train/Test Split & Re-evaluate Model (FUTURE)

## Executor's Feedback or Assistance Requests

-   **Phase 3 Model (s25_l6) Evaluation:** Completed successfully. Overall Weighted Performance Score: 0.8316.
-   **Phase 4 - `reconstruct_to_midgard_format`:** Correction for numerical handling (using direct values from `decode_prediction`) has been applied and verified via simulation output. The function appears to be working as intended.
-   **Phase 4 - `run_next_transaction_prediction`:** Initial implementation is complete. This mode now needs to be tested by running against the live MayaChain API to ensure it fetches data, predicts, polls, compares, and updates context correctly. The comparison logic is currently basic and can be expanded if needed after initial live tests.

## Lessons

-   `model_config.json` is the central configuration hub.
-   `decode_prediction` handles unscaling/un-normalization. `reconstruct_to_midgard_format` uses these processed values directly.
-   Simulation feedback loop requires `is_simulation_feedback=True` for `preprocess_single_action_for_inference`.
-   The `edit_file` tool can be unreliable for complex replacements; providing the full function body for replacement is a strategy, but may still require re-application or manual verification. (Confirmed: This was an issue, but direct application of corrected logic for numericals in `reconstruct_to_midgard_format` was successful).

---
*Older logs from Phase 2 and early Phase 3 have been condensed or removed for clarity.*