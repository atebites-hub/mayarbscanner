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
*   **Task 4.4: Implement `run_generative_simulation` (COMPLETED - Initial Version, saves flat JSON)**
*   **Task 4.5: Implement `reconstruct_to_midgard_format` (IN PROGRESS)**
    *   **Description:** Converts the flat decoded transaction dictionary (from `decode_prediction`) into a nested JSON structure closely resembling Midgard's `/actions` endpoint output.
    *   **Current Status:** The primary remaining issue is to ensure the `edit_file` tool correctly applies the latest version of this function, which uses numerical values directly from `flat_decoded_tx` (as they are already unscaled/un-normalized by `decode_prediction`). The logic for reconstructing nested structures for various transaction types also needs final verification after the function is correctly updated in the file.
    *   **Next Step for New Agent:** Verify that the `reconstruct_to_midgard_format` function in `src/realtime_inference_suite.py` matches the intended corrected version (provided in prior conversation history, specifically the one designed to replace the entire function body). If not, apply the correct version. Then, run the simulation to produce `simulated_transactions_reconstructed_v3_s25_l6.json` and review its structure and values, particularly for `date`, numerical fields, and metadata sections for swap, addLiquidity, and withdraw.
*   **Task 4.6: Implement `run_next_transaction_prediction` (FUTURE)**
    *   Live mode: fetches latest transactions, predicts N+1, waits for actual N+1, compares.
*   **Task 4.7: Testing, Refinement, and Documentation (Phase 4 Finalization - FUTURE)**
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
    -   [x] Task 4.4: Implement `run_generative_simulation` (Initial Version - saves flat decoded JSON)
    -   [ ] Task 4.5: Implement `reconstruct_to_midgard_format` (IN PROGRESS - Needs correct function application and verification)
    -   [ ] Task 4.6: Implement `run_next_transaction_prediction`
    -   [ ] Task 4.7: Testing, Refinement, and Documentation (Phase 4 Finalization)
    -   [ ] Task 4.8: Project Cleanup and Documentation Update (Overall Project Finalization)

## Executor's Feedback or Assistance Requests

-   **Phase 3 Model (s25_l6) Evaluation:** Completed successfully. Overall Weighted Performance Score: 0.8316.
-   **Phase 4 - `src/realtime_inference_suite.py` - `reconstruct_to_midgard_format`:** The `edit_file` tool has been consistently failing to correctly apply the latest intended changes to ensure it uses numerical values directly from `flat_decoded_tx`. The primary task for the next agent is to ensure the correct version of this function is in place and then test it.

## Lessons

-   `model_config.json` is the central configuration hub.
-   `decode_prediction` handles unscaling/un-normalization. `reconstruct_to_midgard_format` uses these processed values directly.
-   Simulation feedback loop requires `is_simulation_feedback=True` for `preprocess_single_action_for_inference`.
-   The `edit_file` tool can be unreliable for complex replacements; providing the full function body for replacement is a strategy, but may still require re-application or manual verification.

---
*Older logs from Phase 2 and early Phase 3 have been condensed or removed for clarity.*