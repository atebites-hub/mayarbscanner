# Product Requirements Document (PRD)

## Title
Mayanode Real-time Monitor & Generative Block Prediction System

## Problem Statement
Understanding and predicting transaction flow and block composition dynamics within the Maya Protocol is crucial for various applications, including identifying potential arbitrage opportunities, assessing network congestion, and optimizing trading strategies. This project aims to build a system that:
1.  Efficiently ingests and stores Maya Protocol block data in a highly performant relational database.
2.  Provides a real-time web interface for monitoring current blocks and mempool activity, optimized for low-latency data display.
3.  (Future Work) Develops an AI model to generatively predict entire subsequent blocks in a sequence, using the collected data.

## Objectives

**Phase 1: Data Pipeline & Real-time Monitor (COMPLETE)**
-   Fetch historical and real-time full block data from Maya Protocol's Mayanode API and Tendermint RPC endpoints.
-   Implement robust Protobuf decoding for transaction data obtained from Tendermint RPC (`cosmos.tx.v1beta1.Tx` via `betterproto`).
-   Store all parsed block components (headers, transactions, messages, events, attributes) in a fully relational SQLite database, enabling efficient querying and reconstruction.
-   Develop a Flask web application (`app.py`, `templates/latest_blocks.html`) to:
    -   Display the latest 10 confirmed blocks, with details, dynamically updating.
    -   Display live mempool transaction activity, refreshing periodically.
    -   Achieve high performance for data retrieval and display, suitable for arbitrage monitoring (e.g., **5-9ms API response for single block reconstruction**).

**Phase 2: Generative Block Prediction Model (Future Work)**
-   Preprocess parsed block data from the database into a rich feature set suitable for AI model training.
-   Build an AI model (e.g., Transformer-based `GenerativeBlockModel`) that learns to predict the full feature vector of the next block in a sequence.
-   Develop a composite loss function and evaluation metrics for the block prediction model.
-   Implement real-time inference capabilities for the trained model.

## Functional Requirements

### Data Collection & Management (Phase 1 - COMPLETE)
-   **Data Sources:** Mayanode API (`https://mayanode.mayachain.info/mayachain/block`) and Tendermint RPC (`https://tendermint.mayachain.info/block`, `/unconfirmed_txs`).
-   `src/api_connections.py`: Provides functions to connect and fetch data from these sources.
-   `src/fetch_realtime_transactions.py`: Orchestrates data fetching (historical catch-up with `aiohttp`, specific range/count/target, continuous polling) and manages data insertion into the database.
-   `src/database_utils.py`: Handles SQLite database operations, including schema creation, granular data insertion for all block components (fully relational), and **highly optimized reconstruction of Mayanode API-like JSON from the database (`reconstruct_block_as_mayanode_api_json`)**.
-   `src/common_utils.py`: Provides core parsing logic for block data, including `betterproto`-based Protobuf decoding and transformation of Tendermint transactions.

### Web UI & API (`app.py`, `templates/latest_blocks.html`) (Phase 1 - COMPLETE)
-   **Main Display Page (`/latest-blocks`):**
    -   Shows the 10 most recent blocks from the database.
    -   Dynamically updates to fetch and display new blocks since the last known height.
    -   Displays live mempool transactions, refreshing at a set interval.
    -   UI is responsive and provides collapsible sections for detailed block information.
-   **API Endpoints:**
    -   `/api/latest-blocks-data`: Returns JSON for the latest 10 blocks (reconstructed).
    -   `/api/blocks-since/<int:from_height>`: Returns JSON for blocks newer than `from_height`.
    -   `/api/mempool`: Returns JSON for current mempool transactions (decoded).

### Data Preprocessing (`src/preprocess_ai_data.py`) (Phase 2 - Future Work)
-   (Details remain similar to previous version, focusing on block-level features, ID mapping, hashing, scaling, sequence generation, and artifact creation for the AI model.)

### AI Model (`src/model.py` - `GenerativeBlockModel`) (Phase 2 - Future Work)
-   (Details remain similar to previous version, outlining a Transformer-based architecture, embedding layers, input/output projections, and feature output information.)

### Model Training (`src/train_model.py`) (Phase 2 - Future Work)
-   (Details remain similar to previous version, covering data loading, model instantiation, composite loss, optimizer, and saving model weights.)

### Model Evaluation (`src/evaluate_model_generative_block.py`) (Phase 2 - Future Work)
-   (Details remain similar to previous version, focusing on per-feature metrics, overall loss, and saving results.)

## Non-Functional Requirements
-   **Data Integrity:** Accurate parsing and storage of all block fields. (VERIFIED for Phase 1)
-   **System Performance (Web App):** API endpoints for block data should respond with very low latency (target <10ms for single block reconstruction). (ACHIEVED: 5-9ms)
-   **Model Performance (AI):** (Future Work) Model should demonstrate capability in predicting features of the next block with reasonable accuracy/error rates.
-   **Reproducibility:** Data fetching, parsing (including Protobuf decoding), storage, and (future) AI processes should be reproducible.
-   **Scalability:** System designed for potential expansion to handle larger datasets.
-   **Modularity:** Code is well-separated for data handling, API, UI, and (future) AI components.

## Success Criteria

**Phase 1: Data Pipeline & Real-time Monitor (COMPLETE)**
-   `src/fetch_realtime_transactions.py` successfully fetches and stores block data into the relational database. (ACHIEVED)
-   `src/common_utils.py` and `src/database_utils.py` accurately parse, store, and reconstruct block data. (ACHIEVED)
-   Protobuf decoding for Tendermint transactions is robust and integrated. (ACHIEVED)
-   The Flask application (`app.py`, `latest_blocks.html`):
    -   Dynamically displays the 10 latest blocks from the database. (ACHIEVED)
    -   Dynamically displays live mempool activity. (ACHIEVED)
    -   API endpoints respond with high performance (e.g., **5-9ms for single block reconstruction** via `/api/latest-blocks-data` if only one new block is fetched by the underlying logic of `get_latest_blocks_with_details`). (ACHIEVED)
    -   UI is responsive and user-friendly. (ACHIEVED)

**Phase 2: Generative Block Prediction Model (Future Work)**
-   `src/preprocess_ai_data.py` successfully processes stored block data into NPZ format and generates artifacts.
-   The `GenerativeBlockModel` trains, and composite loss decreases.
-   `src/evaluate_model_generative_block.py` produces meaningful metrics.
-   (Future) Analysis of predicted block sequences yields insights.