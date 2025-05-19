# Maya Protocol Arbitrage Scanner

This project aims to build an arbitrage scanner for the Maya Protocol. It involves fetching transaction data, preprocessing it, training an AI model to predict arbitrage opportunities, and evaluating the model's performance.

## Project Phases

-   **Phase 1: Data Infrastructure (Completed)** - Focused on fetching, parsing, and storing Maya Protocol transaction data. Initial web UI setup.
-   **Phase 2: AI Model Development (Completed)** - Focused on data preprocessing for AI, building a Transformer-based model, training, and evaluation.
-   **Phase 3: Real-time Pipeline Development (Future)** - Develop a system for real-time data ingestion, prediction, and opportunity alerting.
-   **Phase 4: Deployment and Iteration (Future)** - Deploy the system and continuously iterate on the model and strategies.

## Current Status

The project has completed Phase 2. We have a functional data preprocessing pipeline, a trainable AI model, and an initial set of evaluation metrics based on an independent test set.

## Getting Started: Step-by-Step Instructions (Train/Test Workflow)

Follow these steps to set up the project, fetch distinct training and testing datasets, preprocess them, train the model, and evaluate its performance on unseen test data.

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

### 2. Data Fetching for Training and Testing

   a. **Fetch Training Data (e.g., most recent 24 hours):**
      This command fetches transaction data from the last 24 hours and saves it as `data/training_transactions.csv`.
      ```bash
      python src/fetch_realtime_transactions.py --output-file training_transactions.csv --hours-ago-start 24 --duration-hours 24
      ```
      *Expected Outcome:* `data/training_transactions.csv` is created.

   b. **Fetch Test Data (e.g., 48 to 24 hours ago):**
      This command fetches transaction data from the 24-hour period immediately preceding the training data and saves it as `data/test_transactions.csv`.
      ```bash
      python src/fetch_realtime_transactions.py --output-file test_transactions.csv --hours-ago-start 48 --duration-hours 24
      ```
      *Expected Outcome:* `data/test_transactions.csv` is created.

### 3. Data Preprocessing

   a. **Preprocess Training Data:**
      This script processes `data/training_transactions.csv` in `train` mode. It learns and saves data transformations (scaler, mappings for categorical features) and a model configuration file to `data/processed_ai_data/`. The processed sequences are saved as `data/processed_ai_data/training_sequences.npz`.
      ```bash
      python src/preprocess_ai_data.py --mode train --input-csv training_transactions.csv --output-npz training_sequences.npz --artifacts-dir data/processed_ai_data --data-dir data --processed-data-dir data/processed_ai_data
      ```
      *Expected Outcome:* Files like `training_sequences.npz`, `scaler.pkl`, `model_config.json`, and various `*.json` mapping files are created in `data/processed_ai_data/`.

   b. **Preprocess Test Data:**
      This script processes `data/test_transactions.csv` in `test` mode. It loads and applies the transformations (scaler, mappings) learned during training data preprocessing. The processed sequences are saved as `data/processed_ai_data/test_sequences.npz`.
      ```bash
      python src/preprocess_ai_data.py --mode test --input-csv test_transactions.csv --output-npz test_sequences.npz --artifacts-dir data/processed_ai_data --data-dir data --processed-data-dir data/processed_ai_data
      ```
      *Expected Outcome:* `data/processed_ai_data/test_sequences.npz` is created.

### 4. Model Training

   This script loads the preprocessed training sequences (`training_sequences.npz`) and the `model_config.json`. It trains the AI model and saves the best performing model (based on validation loss) and the final model to the `models/` directory.
   ```bash
   python src/train_model.py --input-npz data/processed_ai_data/training_sequences.npz --model-config-path data/processed_ai_data/model_config.json --model-save-dir models
   ```
   *Expected Outcome:*
     - Console output showing training progress.
     - Model files `models/best_arbitrage_model.pth` and `models/final_arbitrage_model.pth` created/updated.

### 5. Model Evaluation on Test Data

   This script loads the trained model (e.g., `models/best_arbitrage_model.pth`), the `model_config.json`, and the preprocessed test sequences (`test_sequences.npz`). It evaluates the model's performance on this unseen test data, prints metrics, and saves plots to the `models/` directory.
   ```bash
   python src/evaluate_model.py --input-npz data/processed_ai_data/test_sequences.npz --model-config-path data/processed_ai_data/model_config.json --model-weights-path models/best_arbitrage_model.pth --output-dir models
   ```
   *Expected Outcome:*
     - Console output showing accuracy, precision, recall, F1-scores, MSE, RMSE, MAE for the test set.
     - Updated confusion matrix plot (`models/confusion_matrix_actor_type.png`).
     - Updated scatter plot for profit prediction (`models/scatter_plot_mu_profit.png`).

## Project Structure

-   `src/`: Contains all Python source code.
    -   `fetch_realtime_transactions.py`: Fetches historical data.
    -   `preprocess_ai_data.py`: Preprocesses data for the AI model.
    -   `model.py`: Defines the PyTorch Transformer model architecture.
    -   `train_model.py`: Script for training the model.
    -   `evaluate_model.py`: Script for evaluating the trained model.
    -   `common_utils.py`: (Currently minimal use) Utility functions.
    -   `app.py`: Flask application for the web UI (early stages).
    -   `static/`, `templates/`: For the Flask web UI.
-   `data/`: Stores raw and processed data.
    -   `historical_24hr_maya_transactions.csv`: Raw data from Midgard.
    -   `processed_ai_data/`: Output from `preprocess_ai_data.py` (sequences, mappings, scaler).
-   `models/`: Stores trained model weights and evaluation plots.
-   `Docs/`: Project documentation (Requirements, Implementation Plan, etc.).
-   `requirements.txt`: Python package dependencies.
-   `README.md`: This file.

## Key Technologies

-   Python 3
-   PyTorch (for the AI model)
-   Pandas, NumPy (for data manipulation)
-   Scikit-learn (for evaluation metrics)
-   Matplotlib, Seaborn (for plotting)
-   Flask (for the web UI - future development)
-   CoinGecko API (via `pycoingecko` for external price data)
-   Maya Protocol Midgard API

## Future Enhancements (from `.cursor/scratchpad.md`)

-   **Expand Historical Data Fetching:** Increase the historical data collection period from the current 24 hours to approximately 100 days (~2400 hours).
-   **Advanced Model Features:** Explore more complex model architectures or feature engineering techniques.
-   **Real-time Inference and Strategy Execution:** Develop the pipeline for live arbitrage detection and potentially automated execution (Phase 3 & 4).
-   **Web UI Development:** Enhance the Flask UI for better data visualization and interaction.

## Contributing

(Placeholder for contribution guidelines if the project becomes collaborative.)
