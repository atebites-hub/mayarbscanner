# mayarbscanner
Next generation transactions predictor transformer model for the Maya blockchain.

## Project Overview

This project aims to build a sophisticated scanner for identifying potential arbitrage opportunities on the Maya Protocol. It involves several key components:

1.  **Data Collection**: Fetching historical and real-time transaction data from the Maya Protocol's Midgard API.
2.  **Data Streaming & Web UI**: Continuously streaming new transactions (confirmed and pending) and displaying this data, along with historical transactions, in a Flask-based web application.
3.  **AI Data Preprocessing**: Transforming raw transaction data into a format suitable for machine learning models, including feature engineering, normalization, and sequence generation.
4.  **Arbitrage Prediction Model**: Developing a model (e.g., using Transformers) to predict potential arbitrage opportunities based on the processed transaction sequences.
5.  **Reporting & Simulation**: Implementing mechanisms to report identified opportunities and potentially simulate trading strategies.

## Current Status

**Phase 1: Core Data Engine, Initial Display, and AI Preprocessing - COMPLETE**

We have successfully completed the first phase of the project, which includes:

*   **Historical Data Fetching**: Implemented scripts to fetch all Maya Protocol actions from the last 24 hours and store them.
*   **Real-time Data Streaming**: Developed a `RealtimeStreamManager` to continuously poll for new confirmed and pending transactions from the Midgard API.
*   **Pending Block Construction**: Logic to manage a conceptual "pending block" of transactions.
*   **Flask Web Application**: A web interface (`app.py`) that displays:
    *   24-hour historical transaction data.
    *   A live feed of confirmed transactions.
    *   A live feed of pending transactions.
*   **AI Data Preprocessing**: Created a pipeline (`src/preprocess_ai_data.py`) that:
    *   Loads the 24-hour historical data.
    *   Performs feature engineering (handles categorical data, normalizes numerical data, creates time-based features).
    *   Generates sequences of transactions (length M=10) suitable for time-series modeling.
    *   Saves the processed data (sequences, mappings, scaler) for use in model training.
*   **Testing**: Initial test suites for data fetching, streaming, and Flask endpoints have been implemented.

**Next Steps: Phase 2 - Arbitrage Identification (Model Development)**

Phase 2 will focus on building the core machine learning model to identify arbitrage opportunities. Key tasks will include:

*   Defining the Transformer encoder architecture.
*   Implementing the prediction mechanism (e.g., predicting profitability or likelihood of an arbitrage).
*   Developing the training loop and custom loss functions.
*   Training the model on the preprocessed data from Phase 1.

## Project Structure

*   `src/`: Contains the main source code for data fetching, preprocessing, streaming, and the Flask web application.
*   `data/`: Intended for storing raw and processed data. 
    *   `data/historical_24hr_maya_transactions.csv`: Stores raw transaction data fetched from Midgard.
    *   `data/processed_ai_data/`: Contains AI-ready data, including `sequences.npy`, feature mappings, and the data scaler.
*   `templates/`: HTML templates for the Flask application.
*   `static/`: Static assets (CSS, JavaScript) for the Flask application.
*   `Docs/`: Project documentation, including requirements and implementation plans.
*   `.venv/`: Python virtual environment.
*   `README.md`: This file.

## Setup and Running

(Instructions to be added once the application is more mature and ready for easier standalone execution, including dependency installation and running the web server and other components.)
