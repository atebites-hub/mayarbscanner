# .cursor/scratchpad.md - Maya Protocol Arbitrage Scanner

## Background and Motivation

**MAJOR PROJECT PIVOT: Fresh Start - Generative Block Prediction Model using Mayanode API**

The project is undergoing a significant pivot. All previous work on transaction-level prediction using Midgard API data (Phases 1-5) is now **SUPERSEDED**.
The new goal is to build a **Generative Block Prediction Model**. This involves:
1.  Fetching full block data directly from the Mayanode API (`https://mayanode.mayachain.info/mayachain/doc`).
2.  The model will aim to predict the *entire next block*.
3.  Arbitrage identification will be an emergent property.

The Mayanode REST API endpoint `/mayachain/block` provides transactions as pre-decoded JSON. However, other sources like the Tendermint RPC endpoint (`/block` or `/unconfirmed_txs`) provide base64 encoded protobuf strings for transactions (typically `cosmos.tx.v1beta1.Tx`). Thus, a robust protobuf decoding capability is essential for handling these sources.

**Protobuf Decoding Journey Summary:**
Initial attempts to decode these transactions using standard Python `protobuf` library and various `protoc` versions were unsuccessful.
The breakthrough for `cosmos.tx.v1beta1.Tx` (from Tendermint RPC) came from:
1.  Using `betterproto` to generate Python stubs from a curated set of `.proto` files (Cosmos SDK, gogoproto, google protos).
2.  Installing specific versions of dependencies: `protobuf==4.21.12` (compatible with `betterproto==2.0.0b7`), `betterproto[compiler]==2.0.0b7`, and `grpclib`.
3.  Compiling protos using `python -m grpc_tools.protoc ... --python_betterproto_out=...`.
This allows native Python deserialization of `CosmosTx` messages.

For more complex, Mayanode-specific types like `types.MsgObservedTxOut` (which historically caused significant issues):
1.  Using the exact `.proto` files from a specific Mayanode source commit (`59743feb...` from mid-2023) was key.
2.  The correct outer message type was identified as `types.MsgObservedTxOut`.
3.  A manual modification to `Tx.chain` field (from `string` to `bytes`) in the source `common.proto` was necessary.
4.  `protoc --decode=types.MsgObservedTxOut ...` (e.g., v3.20.1) then successfully parsed these messages.
5.  Native Python deserialization with standard libraries still failed for these complex types.
6.  A workaround using `subprocess` to call `protoc --decode` from Python and parse its text output was implemented (`scripts/decode_local_block_for_comparison.py`) and remains a documented fallback for such specific cases.

**Current Goal for Protobuf:**
-   **Primary Solution for Tendermint Tx (`CosmosTx`):** Utilize the `betterproto`-generated stubs for native Python deserialization. This is now implemented and working.
-   **Fallback for Complex/Specific Mayanode Types (e.g., `MsgObservedTxOut`):** Rely on the documented `subprocess` method calling `protoc --decode`, as detailed in `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md`.
-   The workspace now contains the necessary `proto/src` and `proto/generated/pb_stubs` (which should be committed to Git) for the `betterproto` approach.

## Key Challenges and Analysis (NEW - For Block Prediction)

-   **Mayanode API Interaction:** Robust interaction with Mayanode API for block data. (Largely addressed)
-   **Tendermint RPC for Mempool & Blocks:** Successfully fetching unconfirmed Txs and full blocks. (Addressed)
-   **API Response Variability (`/mayachain/block`):** Understood; Mayanode REST provides decoded JSON, Tendermint RPC provides protobuf.
-   **Protobuf Deserialization (Native Python):**
    -   For `cosmos.tx.v1beta1.Tx` from Tendermint RPC: **Addressed** using `betterproto`.
    -   For complex Mayanode-specific types like `types.MsgObservedTxOut`: Addressed via documented fallback (`subprocess` with `protoc --decode`).
-   **Handling Empty Blocks:** Addressed in parsing logic.
-   **Toolchain for Protobuf:**
    -   **Primary:** `betterproto[compiler]==2.0.0b7`, `protobuf==4.21.12`, `grpclib`, and `python -m grpc_tools.protoc` for generating stubs for `CosmosTx`.
    -   **Fallback/Diagnostic:** Standalone `protoc` (e.g., v3.20.1) for `protoc --decode` with specific Mayanode protos.

## Database Design Philosophy (NEW)

-   **Core: Transaction Ledger:** The blockchain is fundamentally a transaction ledger. The database schema reflects this by centering around `blocks` and `transactions` tables, with other tables linking to them.
-   **Normalization:** The schema aims for a reasonable level of normalization (approx. 3NF) to ensure data integrity and reduce redundancy. For example, block header information is stored once per block, not repeated for every transaction within that block.
-   **Relational Integrity:** Foreign key constraints are defined to maintain relationships between tables (e.g., a transaction must belong to a valid block).
-   **Querying for Account History:** Account transaction history is derived by querying the `transaction_address_link` table for a given address, then joining with `transactions` and `blocks` to retrieve detailed information. The `addresses` table (with `first_seen`/`last_seen`) provides a summary but links are definitive.
-   **Data for AI:** While the stored schema is normalized, data will be denormalized via joins during preprocessing for AI model training to create comprehensive feature sets.
-   **Comprehensive Data Storage:** The goal is to store all relevant data from blocks. Complex nested JSON objects (like message bodies or event attributes) are stored as JSON strings (TEXT) in SQLite for now. Future migration to PostgreSQL could leverage JSONB for more advanced querying within these structures.

## High-level Task Breakdown

**Phases 1-5: Transaction-Level Prediction (Midgard API) - SUPERSEDED**
*   All tasks related to fetching data from Midgard and predicting individual transactions are now considered superseded by the new block-level prediction approach using the Mayanode API.

---

**Phase 10: Generative Block Prediction Model (Mayanode API & Tendermint RPC) - NEW FOCUS**

*   **Task 10.1: Mayanode API Research & Client Implementation (Data Fetching & Next Block Templating) (COMPLETE - VERIFIED)**
    *   Sub-tasks:
        *   Identify and test Mayanode REST API endpoints for confirmed blocks (by height, latest). (COMPLETE - `fetch_mayanode_block` using `https://mayanode.mayachain.info/mayachain/block`)
        *   Identify and test Tendermint RPC endpoints for unconfirmed transactions (mempool) and mempool statistics. (COMPLETE - `fetch_tendermint_unconfirmed_txs`, `get_tendermint_num_unconfirmed_txs` using `https://tendermint.mayachain.info`)
        *   Implement Python functions in `src/api_connections.py` to query these endpoints. (COMPLETE)
        *   Implement `construct_next_block_template()` in `src/api_connections.py`. (COMPLETE)
        *   Update `src/fetch_realtime_transactions.py` for downloading historical blocks. (COMPLETE)
        *   Update `Docs/Mayanode_API_Cheat_Sheet.md`. (COMPLETE)
    *   Success Criteria: Able to fetch and save data for specific blocks, latest block, and current mempool. `construct_next_block_template` produces a valid template.

*   **Task 10.2.A.0: Investigate and Confirm Mayanode `/block` Endpoint Response Structure (COMPLETE)**
    *   **Objective:** Determine the consistent data structure returned by `https://mayanode.mayachain.info/mayachain/block` when called from `src/api_connections.py`. Specifically, ascertain if transactions (`txs`) within confirmed blocks are provided as decoded JSON objects or as base64 encoded protobuf strings that our script receives.
    *   **Method:**
        1.  Temporarily modified `src/api_connections.py` to fetch and print parts of the raw JSON response and the processed transaction data from the `/mayachain/block` endpoint for height 11255442.
        2.  Live script output analyzed.
    *   **Findings & Conclusion:** The Mayanode endpoint `https://mayanode.mayachain.info/mayachain/block` currently returns confirmed block data where the `txs` array (at the top level of the response, alongside `id`, `header` etc.) contains **already decoded JSON objects**. Each object represents a transaction and includes `hash`, `result` (with events), and a `tx` field which contains the fully parsed transaction body and auth info (Cosmos SDK Tx structure).
    *   **Implication:** Protobuf deserialization is **NOT required** for transactions sourced from `https://mayanode.mayachain.info/mayachain/block` via our current `fetch_mayanode_block` implementation. The function correctly identifies this as a "direct block data structure" and returns it as is.
    *   **Success Criteria:** Achieved. We have a clear, verified understanding of the transaction data structure provided by this endpoint to our script.

*   **Task 10.2.A: Block & Transaction Data Parsing Logic (`common_utils.py`) (REVISED - SIMPLIFIED)**
    *   Sub-tasks:
        *   Create `src/common_utils.py`. (COMPLETE)
        *   Implement `parse_iso_datetime()` helper. (COMPLETE)
        *   Implement `parse_confirmed_block()`: (CORE LOGIC MOSTLY COMPLETE)
            *   Parse block header (from `block_json_data.get("header")`). (COMPLETE)
            *   Parse block-level events (`begin_block_events`, `end_block_events` from `block_json_data`) using `_try_decode_base64` for attribute values. (COMPLETE for Mayanode direct attribute style)
        *   **Verify `parse_transaction_data()` structure:** (VERIFICATION COMPLETE - MINOR ADJUSTMENTS/CLARIFICATIONS IF ANY)
            *   Input: Each item from the `block_json_data.get("txs", [])` list. This item is confirmed to be a dictionary (e.g., `{"hash": "...", "result": {...}, "tx": {...}}`) when data is from `/mayachain/block`.
            *   `parse_transaction_data` currently correctly extracts `tx_hash` from `tx_obj.get("hash")`.
            *   It also correctly assigns `tx_content_json` from `tx_obj.get("tx")`. **This aligns with the current API output, no major change needed here.**
            *   It correctly parses transaction-level result events from `tx_obj.get("result", {}).get("events", [])`.
            *   For auditing, `tx_obj` (the entire transaction dictionary including hash, result, and tx) could be stored if a full raw audit object per transaction is needed. `parse_transaction_data` doesn't explicitly store this, but the caller `parse_confirmed_block` originally had `transactions_raw` which held these `tx_obj` items. We can decide if a separate raw field is needed in the final *parsed transaction output* or if `tx_content_json` (which is `tx_obj.get("tx")`) is sufficient for "raw" tx body audit.
            *   **Protobuf Deserialization Sub-tasks from previous plan are CONFIRMED NOT NEEDED for `/mayachain/block` data source.**
        *   Integrate `parse_transaction_data()` into `parse_confirmed_block()` to process the list of transaction dictionaries. (COMPLETE - current logic iterates `block_json_data.get("txs", [])` and calls `parse_transaction_data` for each item.)
        *   Analyze transaction counts and content variability per block across a larger dataset (e.g., 1000 blocks) to inform strategies for handling variable lengths in model input (relevant for Task 10.3).
    *   Success Criteria: `parse_confirmed_block()` successfully transforms a raw block JSON (from `fetch_mayanode_block`) into a structured dictionary. `parse_transaction_data` correctly processes the pre-parsed transaction dictionaries. Tested with representative block data.

*   **Task 10.2.B: Database Implementation and Data Ingestion Pipeline (IN PROGRESS - CORE INSERTION LOGIC COMPLETE, PENDING QUERY FUNCTIONS & FULL TEST WITH DECODED TX DATA)**
    *   **Description:**
        *   Design a relational database schema (e.g., using SQLite initially) to store parsed confirmed block data, individual transactions, block-level events, transaction-level events, and unique addresses.
        *   Implement Python functions (e.g., in `src/database_utils.py`) for database initialization, data insertion (blocks, transactions, events, addresses), and querying.
        *   Modify `src/fetch_realtime_transactions.py` (or create a new `src/data_ingestion_service.py`) to:
            *   Continuously poll the Mayanode API for new confirmed blocks (compared to the latest in DB).
            *   Implement a configurable delay (e.g., 5-10 seconds) between polls to respect API rate limits.
            *   Fetch, parse (using 10.2.A logic), and insert new confirmed block data into the database.
            *   Ensure the system avoids redundant API calls for data already present in the local database.
        *   Develop utility functions to query the database, including constructing an address's transaction history from locally stored transactions.
        *   **Mempool Data Strategy:** For inference, live mempool data will be fetched on demand (as per `construct_next_block_template` in `api_connections.py`). Storing historical mempool snapshots in the database is a secondary consideration if a clear need for training/analysis emerges.
    *   **Success Criteria:**
        *   A SQLite database is created with a well-defined schema.
        *   The data ingestion script (`fetch_realtime_transactions.py` or equivalent) continuously and politely populates the database with new confirmed blocks.
        *   Data can be queried efficiently, including address-specific transaction histories derived from stored blocks.
        *   The system avoids redundant API calls for confirmed blocks.

*   **Task 10.2.C: Protobuf Decoding Setup & Implementation (REVISED - Strategy Defined & Implemented)**
    *   **Overall Objective:** Achieve reliable native Python deserialization for `cosmos.tx.v1beta1.Tx` from Tendermint RPC, and have a documented fallback for complex Mayanode-specific types.
    *   **Current Status & Approach:**
        *   **For `cosmos.tx.v1beta1.Tx` (Tendermint RPC):** Successfully implemented using `betterproto` generated Python stubs. The necessary `.proto` files are in `proto/src/`, generated stubs in `proto/generated/pb_stubs/`. This is used in `scripts/compare_block_data.py` and `scripts/test_mayanode_decoding.py`.
        *   **For `types.MsgObservedTxOut` (and similar complex Mayanode types):** The `subprocess` method calling `protoc --decode` (using specific Mayanode protos from commit `59743feb...` and the `Tx.chain` fix) remains the reliable fallback. This is demonstrated in `scripts/decode_local_block_for_comparison.py`.
        *   Both methods are documented in `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md`.
    *   **Sub-Tasks:**
        *   **Task 10.2.C.1: Curate `.proto` files for `betterproto` (COMPLETE)** (Gathered necessary Cosmos, gogo, google protos in `proto/src/`)
        *   **Task 10.2.C.2: Generate Python stubs using `betterproto` (COMPLETE)** (Using `python -m grpc_tools.protoc ...`)
        *   **Task 10.2.C.3: Integrate `betterproto` stubs for `CosmosTx` decoding (COMPLETE)** (Implemented in `scripts/compare_block_data.py` and `scripts/test_mayanode_decoding.py`)
        *   **Task 10.2.C.4: Document `protoc --decode` fallback for `MsgObservedTxOut` (COMPLETE)** (Covered in the new guide and `scripts/decode_local_block_for_comparison.py` preserved)
        *   **Task 10.2.C.9: Experimentation - Alternative Python Protobuf Deserialization (PARTIALLY COMPLETE - `betterproto` successful for CosmosTx, `subprocess` for MsgObservedTxOut, further native solutions for complex types not pursued for now)**
    *   **Success Criteria:** `cosmos.tx.v1beta1.Tx` messages from Tendermint RPC are reliably decoded in Python. A robust, documented method exists for handling other problematic types.

*   **Task 10.3: Preprocessing for Block Prediction (`preprocess_ai_data.py`)**
    *   **Description:** Rewrite `src/preprocess_ai_data.py` to:
        *   Load the parsed block data.
        *   Implement feature engineering based on the new block schema. This will involve creating sequences of blocks (X) where the target (Y) is the next block in the sequence.
        *   Handle ID mapping, numerical scaling, and any new feature types specific to block data (e.g., event types, consensus hashes).
        *   Generate `sequences_and_targets.npz` and `model_config.json` artifacts tailored for block prediction.
    *   **Success Criteria:** `preprocess_ai_data.py` can generate training and test data (`.npz` files) and a `model_config.json` suitable for training a block prediction model.

*   **Task 10.4: Adapt Generative Model for Block Prediction (`model.py`)**
    *   **Description:** Update the `GenerativeTransactionModel` (or create a new `GenerativeBlockModel`) in `src/model.py` to accept sequences of processed block features and output a prediction for the next block's features.
    *   This may involve changes to input embedding layers, output projection layers, and potentially the core Transformer architecture if block structure demands it (e.g., hierarchical prediction).
    *   **Success Criteria:** `model.py` defines a model architecture capable of learning from block sequences and predicting subsequent blocks.

*   **Task 10.5: Update Training Script for Block Model (`train_model.py`)**
    *   **Description:** Modify `src/train_model.py` to:
        *   Load the new block-based NPZ data and `model_config.json`.
        *   Instantiate and train the (potentially new) generative block model.
        *   Adapt the composite loss function if necessary to handle the structure of predicted blocks.
    *   **Success Criteria:** `train_model.py` can successfully train a generative block prediction model and save its weights.

*   **Task 10.6: Develop Evaluation Suite for Block Model (`evaluate_model_block.py` - New File)**
    *   **Description:** Create a new script `src/evaluate_model_block.py` to evaluate the performance of the block prediction model.
    *   Define and implement metrics suitable for block-level predictions (e.g., accuracy of header fields, F1 score for transaction types within the block, metrics for overall block structure similarity).
    *   **Success Criteria:** `evaluate_model_block.py` can load a trained block model and test data, and output meaningful evaluation metrics.

*   **Task 10.7: Develop Realtime Inference Suite for Block Prediction (`realtime_inference_block.py` - New File)**
    *   **Description:** Create a new script `src/realtime_inference_block.py` (analogous to the deleted `realtime_inference_suite.py` but for blocks).
    *   Implement functions to:
        *   Load a trained block model and artifacts.
        *   Fetch live block data to form context.
        *   Preprocess live block sequences.
        *   Predict the next block.
        *   Decode the predicted block back into a human-readable/Mayanode-like JSON format.
        *   Implement simulation and live prediction modes.
    *   **Success Criteria:** `realtime_inference_block.py` can run simulations generating sequences of blocks and attempt live prediction of the next block.

*   **Task 10.8: Iteration, Refinement, and Documentation (Phase 10)**
    *   **Description:** Ongoing improvements, bug fixing, performance optimization, and comprehensive documentation.
    *   **Sub-tasks (examples):**
        *   Code refactoring and cleanup.
        *   Performance profiling and optimization for data ingestion and querying.
        *   Add more detailed logging.
        *   Comprehensive `README.md` updates.
        *   Update `Docs/Implementation Plan.md` and `Docs/Project Requirements.md`.
        *   **Create Comprehensive Protobuf Decoding Guide (COMPLETE - `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md`)**

## Project Status Board

**Overall Status:** In Progress - Core data fetching from Mayanode and Tendermint, Tendermint transaction decoding, and initial data comparison successfully implemented. Protobuf decoding strategies (native for `CosmosTx`, fallback for complex types) established and thoroughly documented.

**Recent Accomplishments & Key Milestones:**
- Successfully fetched block data from Tendermint RPC and Mayanode API, and compared them (`scripts/compare_block_data.py`). This included saving raw Mayanode API responses for full data visibility.
- Implemented robust, native Python decoding of `cosmos.tx.v1beta1.Tx` (from Tendermint block data) using `betterproto` generated stubs. This resolved a major hurdle.
- Preserved and documented the `subprocess` method using `protoc --decode` for the historically problematic `types.MsgObservedTxOut` as a reliable fallback.
- Created the comprehensive `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md` detailing both successful Protobuf decoding methods, including setup, compilation, usage, and rationale for committing generated stubs.
- Refactored data fetching in comparison scripts to use centralized functions from `src/api_connections.py`.
- Cleaned up workspace, ensuring essential `.proto` source files (`proto/src/`) and generated `betterproto` stubs (`proto/generated/pb_stubs/`) are preserved and ready for version control.

**Current Focus:**
- Finalizing documentation updates related to Protobuf handling and recent progress.
- Preparing for the next phase of the block prediction model: data preprocessing.

**Upcoming Tasks (from Implementation Plan):**
- [ ] Task 10.3: Preprocessing for Block Prediction (`preprocess_ai_data.py`)
- [ ] Task 10.4: Adapt Generative Model for Block Prediction (`model.py`)
- [ ] Task 10.5: Update Training Script for Block Model (`train_model.py`)
- [ ] Task 10.6: Develop Evaluation Suite for Block Model (`evaluate_model_block.py`)
- [ ] Task 10.7: Develop Realtime Inference Suite for Block Prediction (`realtime_inference_block.py`)

## Executor's Feedback or Assistance Requests

-   The Protobuf decoding challenge for `CosmosTx` is now resolved using `betterproto`.
-   The comprehensive guide `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md` is created.
-   Requesting review of the new guide and other documentation updates.
-   Ready to proceed with the next tasks in the block prediction model development, starting with `Task 10.3: Preprocessing for Block Prediction`.

## Lessons

-   **Protobuf in Python (Mayanode/Cosmos):**
    -   Standard `google-protobuf` Python library can struggle with gogoproto extensions commonly used in Cosmos SDK / Mayanode, leading to empty messages even with correct `.proto` files and `protoc` compilation.
    -   `betterproto` (specifically `betterproto[compiler]==2.0.0b7` with `protobuf==4.21.12` and `grpclib`) provides a viable path for native Python deserialization of standard Cosmos messages like `cosmos.tx.v1beta1.Tx` when stubs are generated using `python -m grpc_tools.protoc`.
    -   For highly specific/complex Mayanode types (e.g., `types.MsgObservedTxOut`), `protoc --decode` (e.g., v3.20.1) with specific `.proto` files (and sometimes manual fixes like `string` to `bytes` for certain fields) invoked via `subprocess` is a robust fallback if native libraries fail.
    -   **Commit Generated Stubs:** Due to the specific environment (protoc version, library versions, plugin versions, source proto nuances) required for successful Protobuf stub generation with `betterproto` for these types of projects, **committing the generated Python stubs (`proto/generated/pb_stubs/` in our case) to version control is highly recommended.** This ensures reproducibility and simplifies setup for others or for oneself on new environments.
    -   A curated set of source `.proto` files (`proto/src/`) including necessary `gogoproto`, `cosmos_proto`, `google/api`, `google/protobuf`, and relevant Cosmos SDK/Mayanode versions is crucial.
-   **API Endpoints:**
    -   The Mayanode API (`https://mayanode.mayachain.info/mayachain/block`) provides transactions pre-decoded as JSON, simplifying parsing for confirmed blocks from this source.
    -   Tendermint RPC endpoints (`/block`) provide transactions as base64-encoded protobuf strings (`cosmos.tx.v1beta1.Tx`), necessitating the `betterproto` decoding solution.
-   **Iterative Debugging:** Deep-diving into byte-level data, trying different `protoc` versions, and comparing `protoc --decode_raw` vs `protoc --decode` vs Python library outputs was essential to pinpointing protobuf issues.
-   **Dependency Management:** Pinning versions (e.g., `protobuf==4.21.12`, `betterproto==2.0.0b7`) is important once a working combination is found.

**Date:** <YYYY-MM-DD>

**Feedback/Request:**

*   **Protobuf Compilation Success, Decoding Failure (Task 10.2.C.4 & 10.2.C.5):** We successfully compiled a comprehensive set of protos: `protoc` v3.20.0, Python `protobuf` v3.20.0, Cosmos SDK v0.45.9 for `proto_defs/cosmos/`, Mayanode commit `59743feb...` for `proto_defs/mayachain/`, and compatible versions for `ibc`, `tendermint`, `gogoproto`, `ics23`, `cosmos_proto`, `google/api`.
*   However, Python `CosmosTx().FromString()` still fails to populate fields.
*   **Crucially, `protoc --decode=cosmos.tx.v1beta1.Tx proto_defs/cosmos/tx/v1beta1/tx.proto < /tmp/tx.bin` also fails.** This points to a schema definition issue, not just a Python library problem. The Mayanode-specific versions of core Cosmos protos (likely in their `proto/gaia/` from the commit we have) are probably necessary and different from the standard SDK versions.
*   **Assistance Needed:** The next attempt should involve using Mayanode's `proto/gaia/` protos as the base for `proto_defs/cosmos/` and then very carefully resolving any ensuing dependency/compilation issues.

## Files & Scripts

*   `src/api_connections.py`: Fetches data from Mayanode and Tendermint RPC.
*   `src/common_utils.py`: Parses block and transaction data.
*   `src/database_utils.py`: Database interactions (SQLite).
*   `src/fetch_realtime_transactions.py`: Main script for data ingestion.
*   `src/preprocess_ai_data.py`: Prepares data for AI model.
*   `src/model.py`: Generative AI model definition.
*   `src/train_model.py`: Script to train the AI model.
*   `src/evaluate_model_block.py`: (To be created) Evaluates block prediction model.
*   `src/realtime_inference_block.py`: (To be created) Real-time block prediction.
*   `scripts/generate_protos.sh`: Compiles `.proto` files to Python stubs.
*   `proto_defs/`: Contains all `.proto` definition files.
*   `bin/protoc_compiler/`: Contains the `protoc` binary and its includes.
*   `requirements.txt`: Python package dependencies.
*   `decode_local_block_for_comparison.py`: Script for testing protobuf decoding.

---
(Older content below, largely superseded or for historical context on proto issues)
---