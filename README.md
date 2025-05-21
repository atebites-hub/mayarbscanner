# Maya Protocol Arbitrage Scanner

This project aims to build an AI-driven system to analyze and understand transaction dynamics on the Maya Protocol. Initially focused on direct arbitrage prediction, the project has **evolved to focus on a generative model** capable of predicting subsequent transactions in a sequence, using only internal Maya Protocol transaction data.

## Project Phases

-   **Phase 1: Data Infrastructure (Completed)** - Focused on fetching, parsing, and storing Maya Protocol transaction data. Initial web UI setup.
-   **Phase 2: Predictive Arbitrage Model (Completed & Superseded)** - Focused on data preprocessing for AI, building a Transformer-based model to predict specific arbitrage signals. *This phase is complete and has been superseded by the more fundamental approach in Phase 3.*
-   **Phase 3: Generative Transaction Prediction Model - Core Development (Largely Completed)** - Developing a Transformer-based model to predict the *entire next transaction* in a sequence. This involved:
    -   Utilizing a comprehensive set of features from all transaction types.
    -   Employing techniques like feature hashing and robust ID mapping.
    -   Training a model to predict the complete feature vector of the subsequent transaction.
    -   Developing evaluation scripts with per-feature metrics and a weighted performance score.
    -   Successfully trained and evaluated a model (`s25_l6`) with sequence length 25 and 6 encoder layers.
-   **Phase 4: Realtime Inference Suite & Advanced Generative Model Refinements (Current Focus)** - Building a suite (`src/realtime_inference_suite.py`) for running the generative model in simulation and live prediction modes. This includes reconstructing Midgard-like JSON outputs. Future work in this phase will involve robust testing, integrating actual Maya Protocol data, and potentially mempool data.
-   **Phase 5: Deployment and Iteration (Future)** - Deploying the refined system and establishing continuous iteration cycles.

## Current Status

The project has largely completed **Phase 3 (Generative Transaction Prediction Model)** and is now actively developing **Phase 4 (Realtime Inference Suite)**.
Key achievements in Phase 3 include:
-   Defining a comprehensive feature schema.
-   Refactoring `src/preprocess_ai_data.py` for generative modeling (raw JSON to features, artifact generation).
-   Implementing the `GenerativeTransactionModel` in `src/model.py`.
-   Updating `src/train_model.py` with a composite loss and model config augmentation.
-   Developing `src/evaluate_model_generative.py` and achieving a weighted performance score of **0.8316** with the `s25_l6` model.

Work on `src/realtime_inference_suite.py` (Phase 4) has progressed to:
-   Loading models and artifacts.
-   Preprocessing single actions and sequences for inference.
-   Decoding model predictions into a flat feature dictionary.
-   Initial implementation of a simulation mode (`run_generative_simulation`) with a feedback loop.
-   Ongoing refinement of `reconstruct_to_midgard_format` to convert decoded predictions into well-structured Midgard-like JSON.

## Getting Started: Step-by-Step Instructions (Phase 3 & Initial Phase 4)

Follow these steps to set up the project, preprocess data, train the generative model, evaluate it, and run the initial simulation mode of the inference suite.

### 1. Setup

   a. **Clone the Repository:**
      ```bash
      git clone <repository_url> # Replace <repository_url> with the actual URL
      cd mayarbscanner 
      ```

   b. **Create and Activate a Python Virtual Environment:**
      It's highly recommended to use a virtual environment.
      ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      ```
      (On Windows, activation is `.venv\Scripts\activate`)

   c. **Install Dependencies:**
      Install all required Python packages from `requirements.txt`.
      ```bash
      pip install -r requirements.txt
      ```

### 2. Data

   -   For the current development of the Phase 3 generative model, a sample JSON file named `data/transactions_data.json` (containing THORChain transaction data, structurally similar to Maya Midgard `/actions` output) is used.
   -   Ensure this file is present in the `data/` directory.
   -   *Future Work:* `src/fetch_realtime_transactions.py` will be updated to fetch raw, unfiltered JSON transaction data directly from the Maya Protocol Midgard API to create larger and more diverse datasets.

### 3. Data Preprocessing (Generative Model - Training Data)

   This script processes the raw JSON transaction data (e.g., `data/transactions_data.json`) in `train` mode. It flattens transactions according to a defined schema, performs ID mapping for categorical features, uses feature hashing for high-cardinality features like addresses, scales numerical features, and generates sequences where the target is the full feature vector of the next transaction. Artifacts (mappings, scaler, model configuration) are saved to a specified directory (e.g., `data/processed_ai_data_generative_train_s25_l6/`). The processed sequences are saved as an NPZ file (e.g., `data/processed_ai_data_generative_train_s25_l6/sequences_and_targets_generative_thorchain_s25_l6.npz`).

   ```bash
   python src/preprocess_ai_data.py --mode train \
       --data-dir data \
       --input-json transactions_data.json \
       --processed-data-dir-generative data/processed_ai_data_generative_train_s25_l6 \
       --output-npz-generative sequences_and_targets_generative_thorchain_s25_l6.npz \
       --artifacts-dir-generative data/processed_ai_data_generative_train_s25_l6/thorchain_artifacts_v1 \
       --scaler-filename-generative scaler_generative_thorchain_s25_l6.pkl \
       --model-config-filename-generative model_config_generative_thorchain_s25_l6.json
   ```
   *Expected Outcome:* Files like `sequences_and_targets_generative_thorchain_s25_l6.npz`, `scaler_generative_thorchain_s25_l6.pkl`, `model_config_generative_thorchain_s25_l6.json`, and various `*_to_id_generative_thorchain.json` mapping files are created in the specified artifacts directory.

### 4. Model Training (Generative Model - s25_l6)

   This script loads the preprocessed generative sequences and the model configuration. It trains the `GenerativeTransactionModel` using a composite loss function and saves the best performing model (based on validation loss) and the final model to the `models/` directory.

   ```bash
   python src/train_model.py \
       --input-npz-generative data/processed_ai_data_generative_train_s25_l6/sequences_and_targets_generative_thorchain_s25_l6.npz \
       --generative-model-config-path data/processed_ai_data_generative_train_s25_l6/thorchain_artifacts_v1/model_config_generative_thorchain_s25_l6.json \
       --model-save-dir models \
       --best-model-save-path models/best_generative_model_thorchain_s25_l6.pth \
       --final-model-save-path models/final_generative_model_thorchain_s25_l6.pth \
       --num_encoder_layers 6 \
       --num_epochs 20 \
       --batch_size 128
   ```
   *Expected Outcome:*
     - Console output showing training progress (overall loss and components like categorical, numerical, flag losses).
     - Model files `models/best_generative_model_thorchain_s25_l6.pth` and `models/final_generative_model_thorchain_s25_l6.pth` created/updated.

### 5. Data Preprocessing (Generative Model - Test Data)

   This script processes the raw JSON transaction data (e.g., `data/test_data.json`) in `test` mode. It uses the artifacts from the training preprocessing to generate features for the generative model. The processed sequences are saved as an NPZ file (e.g., `data/processed_ai_data_generative_test_s25_l6/sequences_and_targets_generative_thorchain_s25_l6.npz`).

   ```bash
   python src/preprocess_ai_data.py --mode test \
       --input-json test_data.json \
       --data-dir data \
       --artifacts-dir-generative data/processed_ai_data_generative_train_s25_l6/thorchain_artifacts_v1 \
       --processed-data-dir-generative data/processed_ai_data_generative_test_s25_l6 \
       --output-npz-generative sequences_and_targets_generative_thorchain_s25_l6.npz \
       --scaler-filename-generative scaler_generative_thorchain_s25_l6.pkl \
       --model-config-filename-generative model_config_generative_thorchain_s25_l6_AUGMENTED.json
   ```
   *Expected Outcome:* Processed test data NPZ in `data/processed_ai_data_generative_test_s25_l6/`. Note the output model config filename is augmented.

### 6. Model Evaluation (Generative Model - s25_l6)

   This script loads a trained generative model (e.g., `models/best_generative_model_thorchain_s25_l6.pth`), its configuration, and test data (currently using the same NPZ as training/validation for initial testing). It evaluates the model's performance on predicting each feature of the next transaction, prints per-feature metrics, and saves results to the `evaluation_results_generative_s25_l6/` directory.

   ```bash
   python src/evaluate_model_generative.py \
       --model-dir models \
       --model-filename best_generative_model_thorchain_s25_l6.pth \
       --artifacts-dir data/processed_ai_data_generative_train_s25_l6/thorchain_artifacts_v1 \
       --model-config-filename model_config_generative_thorchain_s25_l6_AUGMENTED.json \
       --test-data-dir data/processed_ai_data_generative_test_s25_l6 \
       --test-data-npz sequences_and_targets_generative_thorchain_s25_l6.npz \
       --evaluation-output-dir evaluation_results_generative_s25_l6 \
       --feature-weights-json feature_weights.json
   ```
   *Expected Outcome:*
     - Console output showing per-feature metrics (Accuracy, F1-score, MSE, MAE).
     - Overall test set loss and its components.
     - JSON file with detailed metrics (`evaluation_results_generative_s25_l6/evaluation_metrics.json`).
     - Plots, such as confusion matrices for selected categorical features (e.g., `evaluation_results_generative_s25_l6/cm_action_type_id.png`).

### 7. Realtime Inference Suite - Simulation Mode (Phase 4 - Initial Test)

   Runs the generative simulation using the trained `s25_l6` model.

   ```bash
   python src/realtime_inference_suite.py --mode simulate \
       --model-path models/best_generative_model_thorchain_s25_l6.pth \
       --artifacts-dir data/processed_ai_data_generative_train_s25_l6/thorchain_artifacts_v1 \
       --model-config-filename model_config_generative_thorchain_s25_l6_AUGMENTED.json \
       --num-simulation-steps 10 \
       --output-simulation-file simulated_transactions_reconstructed_s25_l6.json
   ```
   *Expected Outcome:* Console output showing simulation steps and a file `simulated_transactions_reconstructed_s25_l6.json` containing the generated transactions in a Midgard-like format.

## Project Structure

-   `src/`: Contains all Python source code.
    -   `fetch_realtime_transactions.py`: Fetches historical data (primarily for Phase 2, to be adapted for Phase 4 raw JSON fetching).
    -   `preprocess_ai_data.py`: Preprocesses data for the generative model.
    -   `model.py`: Defines the `GenerativeTransactionModel`.
    -   `train_model.py`: Script for training the generative model.
    -   `evaluate_model_generative.py`: Script for evaluating the generative model.
    -   `realtime_inference_suite.py`: Script for running the model in simulation or live prediction modes.
    -   `app.py`: Flask application for the web UI (early stages, future development).
    -   `static/`, `templates/`: For the Flask web UI.
-   `data/`: Stores raw and processed data.
    -   `transactions_data.json`, `test_data.json`: Sample raw transaction data.
    -   `feature_weights.json`: Weights for generative model evaluation.
    -   `processed_ai_data_generative_train_s25_l6/`, `processed_ai_data_generative_test_s25_l6/`: Outputs from Phase 3/4 preprocessing.
-   `models/`: Stores trained model weights (`*.pth`).
-   `evaluation_results_generative_s25_l6/`: Stores evaluation outputs for the `s25_l6` model.
-   `Docs/`: Project documentation (Requirements, Implementation Plan, Feature Schema, etc.).
-   `requirements.txt`: Python package dependencies.
-   `README.md`: This file.

## Key Technologies

-   Python 3
-   PyTorch (for the AI model)
-   Pandas, NumPy (for data manipulation)
-   Scikit-learn (for evaluation metrics)
-   Matplotlib, Seaborn (for plotting)
-   `mmh3` (for feature hashing)
-   Flask (for the web UI - future development)
-   Maya Protocol Midgard API (target data source)

## Future Enhancements (Post Initial Phase 4)

Once the initial Generative Transaction Prediction Model (Phase 3) is robustly established and evaluated:

-   **Dedicated Test Set Creation:** Implement a proper train/validation/test split strategy for the generative model, ensuring the test set is truly unseen.
-   **Expand Maya Data Fetching:** Fully implement fetching of large-scale, raw JSON transaction data from the Maya Protocol Midgard API.
-   **Advanced Generative Model Features:** Explore more complex architectures, attention mechanisms, or conditioning variables.
-   **Mempool Data Integration:** Investigate incorporating real-time mempool data to potentially refine short-term transaction predictions.
-   **Strategy Development from Predictions:** Design and test strategies based on the sequences generated by the model (e.g., for arbitrage, optimal routing, or other DeFi applications).
-   **Real-time Inference Pipeline:** Build a robust pipeline for live data ingestion and real-time transaction sequence prediction.
-   **Web UI Development:** Enhance the Flask UI to visualize predicted sequences and potential opportunities identified by the generative model.

## Contributing

(Placeholder for contribution guidelines if the project becomes collaborative.)
