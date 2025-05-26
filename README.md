# Maya Protocol Arbitrage Scanner

**MAJOR PROJECT PIVOT: This project is undergoing a significant change in direction.**

Previously, the project aimed to predict individual transactions using data from the Midgard API. This README section reflects that now **SUPERSEDED** approach.

**NEW DIRECTION (Phase 10 onwards): Generative Block Prediction Model using Mayanode API & Tendermint RPC**

The project has pivoted to a more fundamental and potentially powerful approach: **predicting entire blocks of the Maya Protocol blockchain.** This involves:
-   **Data Sources:**
    -   Mayanode API (`https://mayanode.mayachain.info/mayachain/block`): For fetching comprehensive confirmed block data. This endpoint often provides transactions as **pre-decoded JSON objects**.
    -   Tendermint RPC (`https://tendermint.mayachain.info`): For fetching confirmed blocks (which contain transactions as **base64 encoded Protobuf strings**) and for mempool data (also Protobuf).
-   **Prediction Target:** The AI model will be trained to predict the *entire next block* (or a sequence of blocks), including block headers, all contained transactions, and associated event data.
-   **Rationale:** Predicting full blocks offers a richer, more structured data source that encodes more information about blockchain dynamics, potentially leading to a more accurate and insightful generative model. Arbitrage opportunities and other analyses will be emergent properties of these block predictions.
-   **Status & Key Achievements:**
    -   This is a **fresh start**. Code related to Midgard data fetching and individual transaction prediction (Phases 1-5) has been overhauled or removed.
    -   API clients in `src/api_connections.py` for Mayanode and Tendermint RPC are implemented and tested.
    -   Core parsing logic in `src/common_utils.py` for Mayanode JSON block structures is in place.
    -   **Protobuf Decoding Solution for Tendermint Data:**
        -   A robust solution for decoding `cosmos.tx.v1beta1.Tx` messages (from Tendermint RPC) in Python has been successfully implemented using `betterproto`.
        -   This involves a curated set of `.proto` files (`proto/src/`) and generated Python stubs (`proto/generated/pb_stubs/`).
        -   **Crucially, both the source `.proto` files and the generated Python stubs are committed to Git** to ensure reproducibility and ease of setup for developers, avoiding the complexities of replicating the `protoc` compilation environment.
        -   For historically problematic or highly specific Mayanode types (e.g., `types.MsgObservedTxOut`), a fallback method using `protoc --decode` via `subprocess` is documented and available (`scripts/decode_local_block_for_comparison.py`).
        -   A comprehensive guide, `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md`, details both of these decoding methods.
    -   A script `scripts/compare_block_data.py` has been developed to fetch blocks from both Mayanode (JSON and reconstructed Tendermint-like) and Tendermint (Protobuf, then decoded), saving them for comparison.
    -   Database schema (`src/database_utils.py`) is designed, and data ingestion (`src/fetch_realtime_transactions.py`) is under development.

Further details of this new approach are documented in `Docs/Implementation Plan.md` and the `.cursor/scratchpad.md` files.

---
*The following sections describe the OLD (superseded) approach for historical context only.*

## Project Phases (OLD - SUPERSEDED - Details Omitted for Brevity)

## Current Status (OLD - SUPERSEDED - Details Omitted for Brevity)

## Getting Started: Step-by-Step Instructions (NEW - Phase 10 Focus)

Follow these steps to set up the project environment. The primary focus is on data fetching, parsing (including Protobuf decoding), database storage, and comparison, which are prerequisites for AI model development.

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
      This will install `betterproto[compiler]`, `protobuf==4.21.12`, `grpclib`, and other necessary packages for the current Protobuf decoding strategy.

   d. **Protobuf Handling (IMPORTANT - Pre-configured for `betterproto`):
      -   **Source `.proto` files** are located in `proto/src/`.
      -   **Pre-generated Python stubs** (using `betterproto`) are located in `proto/generated/pb_stubs/`.
      -   **Both of these directories are committed to Git.** This means you **DO NOT need to run a `protoc` compilation step yourself** to use the primary `CosmosTx` decoding functionality.
      -   The Python scripts (e.g., `src/api_connections.py`, `src/common_utils.py`) are configured to use these pre-generated stubs by dynamically adding `proto/generated/` (the parent of `pb_stubs/`) to `sys.path` at runtime. This allows imports like `from pb_stubs.cosmos.tx.v1beta1 import Tx`.
      -   **Linter Configuration (Pylance/Pyright):** To help linters resolve these dynamic imports, add `"./proto/generated"` to your `python.analysis.extraPaths` in your VS Code workspace settings (`.vscode/settings.json`). See `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md` (Troubleshooting section) for more details.
      -   For detailed information on how these were generated, or if you need to regenerate them or decode other message types, refer to `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md`.

   e. **Troubleshooting Note - Stale Code Execution / Python Caching:**
      - If you make changes to Python files (especially utility modules like `src/common_utils.py`) and these changes don't seem to take effect, you might be encountering a Python caching issue.
      - This topic is now covered in more detail in `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md` under the "General Python Caching / Stale Code Execution" troubleshooting section.

### 2. Key Scripts & Data

   -   **Data Sources:**
        -   Mayanode API: `https://mayanode.mayachain.info/mayachain/block`
        -   Tendermint RPC: `https://tendermint.mayachain.info/block`
   -   **Core Logic & Utilities:**
        -   `src/api_connections.py`: Fetches data from Mayanode and Tendermint.
        -   `src/common_utils.py`: General parsing utilities for block and transaction data, including Protobuf transformations.
        -   `src/database_utils.py`: SQLite database operations (schema, insertion, reconstruction).
        -   `src/fetch_realtime_transactions.py`: Main script for fetching and ingesting block data into the database. Supports historical catch-up (async with `aiohttp`), specific range/count/target fetching, and continuous polling. Features improved console output with `tqdm`.
        -   `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md`: Essential reading for understanding Protobuf decoding and Python environment troubleshooting.
        -   `proto/src/`: Source `.proto` files.
        -   `proto/generated/pb_stubs/`: Pre-generated Python stubs for `betterproto`.
   -   **Example & Test Scripts:**
        -   `scripts/compare_block_data.py`: Fetches a block from both Mayanode and Tendermint, decodes Tendermint transactions using `betterproto` stubs, and saves all versions (decoded Tendermint, raw Mayanode, reconstructed Mayanode) to `comparison_outputs/` for analysis.
        -   `scripts/test_mayanode_decoding.py`: Demonstrates deep decoding of `CosmosTx` messages, including nested `Any` types and address derivation, using the `betterproto` stubs.
        -   `scripts/decode_local_block_for_comparison.py`: Demonstrates the fallback `subprocess` method for decoding complex types like `MsgObservedTxOut` (uses sample data strings within the script).

### 3. Running the Comparison Script (Example Usage)

   To see the current data fetching, decoding, and comparison in action:
   ```bash
   python scripts/compare_block_data.py --block <BLOCK_HEIGHT>
   ```
   Replace `<BLOCK_HEIGHT>` with a recent block number (e.g., 11320570).
   This will:
   1.  Fetch the specified block from Tendermint RPC.
   2.  Decode its transactions using the `betterproto` stubs via `src/common_utils.py`.
   3.  Save the decoded and transformed Tendermint block to `comparison_outputs/tendermint_block_<BLOCK_HEIGHT>_decoded_transformed.json`.
   4.  Fetch the same block from the Mayanode API and save the raw response to `comparison_outputs/mayanode_api_block_<BLOCK_HEIGHT>_raw.json`.
   5.  Parse the raw Mayanode API response using `src/common_utils.py` and save it to `comparison_outputs/mayanode_api_block_<BLOCK_HEIGHT>_parsed.json`.
   6.  Print a summary comparison of key fields and transaction alignment.

   You can then inspect the JSON files in the `comparison_outputs/` directory.

### 4. Running the Data Ingestion Service

   `src/fetch_realtime_transactions.py` is used to populate the SQLite database (`mayanode_blocks.db`).

   **Example: Historical Catch-up (recommended for initial population)**
   To fetch the last 20,000 blocks and store them:
   ```bash
   python -m src.fetch_realtime_transactions --historical-catchup 20000
   ```
   This uses asynchronous fetching for speed and provides a `tqdm` progress bar.

   **Example: Continuous Polling (after historical catch-up)**
   To continuously monitor for and fetch new blocks:
   ```bash
   python -m src.fetch_realtime_transactions
   ```
   Other options like `--fetch-range`, `--fetch-count`, and `--target-height` are available. Use `python -m src.fetch_realtime_transactions --help` for details.

### 5. Next Steps in Development (Block Prediction Model & Flask App)

   With data fetching, decoding, and database ingestion established:
   -   **Flask App for CACAO Dividends (Task 10.2.E):**
       -   Implement database queries in `src/database_utils.py` to identify CACAO dividend transactions.
       -   Develop the Flask application (`app.py`) to display this information.
   -   **AI Model Development (Tasks 10.3 onwards):**
       -   Developing `src/preprocess_ai_data.py` for feature engineering from block data.
       -   Adapting/creating the generative AI model (`src/model.py`) for block prediction.
       -   Updating training (`src/train_model.py`) and evaluation scripts.

## Project Structure (Reflecting Cleaned State & New Focus)

-   `src/`:
    -   `fetch_realtime_transactions.py`: For downloading historical Mayanode blocks and future continuous database ingestion.
    -   `api_connections.py`: Functions to connect to Mayanode REST API and Tendermint RPC.
    -   `common_utils.py`: Parsing block and transaction data, including Protobuf transformations.
    -   `database_utils.py`: SQLite database operations (schema, insertion, reconstruction).
    -   `preprocess_ai_data.py`: (To be developed) Feature engineering for block prediction.
    -   `model.py`: (To be developed) Generative block prediction model.
    -   `train_model.py`: (To be adapted) Training script for the block model.
    -   `app.py`: (To be developed) Flask application for CACAO dividend viewer and future model insights.
    -   *(Other evaluation/inference scripts to be added for block model)*
-   `