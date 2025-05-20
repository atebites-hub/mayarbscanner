# Maya Protocol Arbitrage Scanner

This project aims to build an AI-driven system to analyze and understand transaction dynamics on the Maya Protocol. Initially focused on direct arbitrage prediction, the project has **evolved to focus on a generative model** capable of predicting subsequent transactions in a sequence, using only internal Maya Protocol transaction data.

## Project Phases

-   **Phase 1: Data Infrastructure (Completed)** - Focused on fetching, parsing, and storing Maya Protocol transaction data. Initial web UI setup.
-   **Phase 2: Predictive Arbitrage Model (Completed & Superseded)** - Focused on data preprocessing for AI, building a Transformer-based model to predict specific arbitrage signals (actor type and a profit metric), training, and evaluation using external price data (CoinGecko). *This phase is complete and has been superseded by the more fundamental approach in Phase 3.*
-   **Phase 3: Generative Transaction Prediction Model (Current Focus)** - Developing a Transformer-based model to predict the *entire next transaction* in a sequence. This involves:
    -   Utilizing a comprehensive set of features from all transaction types.
    -   Employing techniques like feature hashing for high-cardinality categorical data (e.g., addresses) and robust ID mapping for other categoricals.
    -   Training a model whose output layer predicts the complete feature vector of the subsequent transaction.
    -   Arbitrage identification and other insights are expected to be emergent properties derived from analyzing sequences of predicted transactions, rather than direct prediction targets.
-   **Phase 4: Advanced Generative Model Refinements & Application (Future)** - Iterating on the generative model, potentially integrating mempool data, developing prediction-based strategies, and building a real-time pipeline.
-   **Phase 5: Deployment and Iteration (Future)** - Deploying the refined system and establishing continuous iteration cycles.

## Current Status

The project has completed initial work on the **Phase 3: Generative Transaction Prediction Model**. This includes defining a comprehensive feature schema, refactoring the data preprocessing pipeline (`src/preprocess_ai_data.py`) to handle raw JSON transaction data and output features for the generative model, designing and implementing the `GenerativeTransactionModel` in `src/model.py`, and updating the training script (`src/train_model.py`) with a composite loss function. An initial evaluation script (`src/evaluate_model_generative.py`) has also been developed and tested.

## Getting Started: Step-by-Step Instructions (Phase 3 Generative Model)

Follow these steps to set up the project, preprocess data, train the generative model, and evaluate its performance.

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

### 3. Data Preprocessing (Generative Model)

   This script processes the raw JSON transaction data (e.g., `data/transactions_data.json`) in `train` mode. It flattens transactions according to a defined schema, performs ID mapping for categorical features, uses feature hashing for high-cardinality features like addresses, scales numerical features, and generates sequences where the target is the full feature vector of the next transaction. Artifacts (mappings, scaler, model configuration) are saved to a specified directory (e.g., `data/processed_ai_data_generative_test/thorchain_artifacts_v1/`). The processed sequences are saved as an NPZ file (e.g., `data/processed_ai_data_generative_test/sequences_and_targets_generative_thorchain.npz`).

   ```bash
   python src/preprocess_ai_data.py --mode train \
       --data-dir data \
       --input-json transactions_data.json \
       --processed-data-dir-generative data/processed_ai_data_generative_test \
       --output-npz-generative sequences_and_targets_generative_thorchain.npz \
       --artifacts-dir-generative data/processed_ai_data_generative_test/thorchain_artifacts_v1
   ```
   *Expected Outcome:* Files like `sequences_and_targets_generative_thorchain.npz`, `scaler_generative_thorchain.pkl`, `model_config_generative_thorchain.json`, and various `*_to_id_generative_thorchain.json` mapping files are created in the specified artifacts directory.

### 4. Model Training (Generative Model)

   This script loads the preprocessed generative sequences and the model configuration. It trains the `GenerativeTransactionModel` using a composite loss function and saves the best performing model (based on validation loss) and the final model to the `models/` directory.

   ```bash
   python src/train_model.py \
       --input-npz-generative data/processed_ai_data_generative_test/sequences_and_targets_generative_thorchain.npz \
       --generative-model-config-path data/processed_ai_data_generative_test/thorchain_artifacts_v1/model_config_generative_thorchain.json \
       --model-save-dir models \
       --num_epochs 50 # Or your desired number of epochs
   ```
   *Expected Outcome:*
     - Console output showing training progress (overall loss and components like categorical, numerical, flag losses).
     - Model files `models/best_generative_model_thorchain.pth` and `models/final_generative_model_thorchain.pth` created/updated.

### 5. Model Evaluation (Generative Model)

   This script loads a trained generative model (e.g., `models/best_generative_model_thorchain.pth`), its configuration, and test data (currently using the same NPZ as training/validation for initial testing). It evaluates the model's performance on predicting each feature of the next transaction, prints per-feature metrics, and saves results to the `evaluation_results_generative/` directory.

   ```bash
   python src/evaluate_model_generative.py \
       --model-dir models \
       --model-filename best_generative_model_thorchain.pth \
       --artifacts-dir data/processed_ai_data_generative_test/thorchain_artifacts_v1 \
       --model-config-filename model_config_generative_thorchain.json \
       --test-data-dir data/processed_ai_data_generative_test \
       --test-data-npz sequences_and_targets_generative_thorchain.npz \
       --evaluation-output-dir evaluation_results_generative
   ```
   *Expected Outcome:*
     - Console output showing per-feature metrics (Accuracy, F1-score, MSE, MAE).
     - Overall test set loss and its components.
     - JSON file with detailed metrics (`evaluation_results_generative/evaluation_metrics.json`).
     - Plots, such as confusion matrices for selected categorical features (e.g., `evaluation_results_generative/cm_action_type_id.png`).

## Project Structure

-   `src/`: Contains all Python source code.
    -   `fetch_realtime_transactions.py`: Fetches historical data (primarily for Phase 2, to be adapted for Phase 3 raw JSON fetching).
    -   `preprocess_ai_data.py`: Preprocesses data for the AI model (now tailored for the generative model).
    -   `model.py`: Defines the PyTorch Transformer model architecture (now `GenerativeTransactionModel`).
    -   `train_model.py`: Script for training the model (adapted for the generative model).
    -   `evaluate_model.py`: Script for evaluating the Phase 2 predictive arbitrage model.
    -   `evaluate_model_generative.py`: Script for evaluating the Phase 3 generative model.
    -   `common_utils.py`: (Currently minimal use) Utility functions.
    -   `app.py`: Flask application for the web UI (early stages, future development).
    -   `static/`, `templates/`: For the Flask web UI.
-   `data/`: Stores raw and processed data.
    -   `transactions_data.json`: Sample raw THORChain transaction data used for Phase 3 development.
    -   `processed_ai_data/`: Output from Phase 2 preprocessing.
    -   `processed_ai_data_generative_test/`: Output from Phase 3 preprocessing (sequences, mappings, scaler, config for the generative model).
-   `models/`: Stores trained model weights (`*.pth`) and some evaluation plots from Phase 2.
-   `evaluation_results_generative/`: Stores evaluation outputs (metrics JSON, plots) for the Phase 3 generative model.
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

## Future Enhancements (Post-Phase 3 Generative Model V1)

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
