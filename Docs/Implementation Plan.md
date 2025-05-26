# Implementation Plan

**MAJOR PROJECT PIVOT NOTIFICATION:**
This Implementation Plan is being substantially revised. The project has pivoted from predicting individual transactions using the Midgard API to **predicting full blocks using the direct Mayanode API (`https://mayanode.mayachain.info/mayachain/doc`) and Tendermint RPC (`https://tendermint.mayachain.info`)**. This is a fresh start, leveraging more comprehensive and reliable data for a potentially more powerful generative model.

**Phases 1 through 5 (Transaction Prediction - Midgard API) are now considered SUPERSEDED.** Their details are retained below for historical context but are no longer the active plan.

---

## Phase 1-5: Transaction Prediction (SUPERSEDED - Details Omitted for Brevity)

---

## Phase 10: Generative Block Prediction Model (Mayanode API & Tendermint RPC) - NEW FOCUS

**Overarching Goal:** Develop an AI model capable of predicting sequences of blocks from the Maya Protocol blockchain, using data fetched directly from the Mayanode API (for confirmed blocks, often pre-decoded) and Tendermint RPC (for confirmed blocks with Protobuf transactions and for mempool data).

**Developer Setup Note: Protobuf Handling (REVISED & STABILIZED)**

Successfully handling Protobuf messages, particularly `cosmos.tx.v1beta1.Tx` from Tendermint RPC and potentially other Mayanode-specific types, is crucial.

**Current Approach & Rationale:**
-   **Primary Method for `cosmos.tx.v1beta1.Tx` (Tendermint RPC):** Native Python deserialization using `betterproto`.
    -   A curated set of source `.proto` files is maintained in `proto/src/` (includes Cosmos SDK, gogoproto, google protos).
    -   Python stubs are generated into `proto/generated/pb_stubs/` using `python -m grpc_tools.protoc ... --python_betterproto_out=...` (see `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md` for the exact command).
    -   **Crucially, both `proto/src/` and `proto/generated/pb_stubs/` ARE COMMITTED TO GIT.** This ensures that developers can use the pre-generated, working stubs without needing to replicate the `protoc` compilation environment, which proved to be highly sensitive to versions of `protoc`, `betterproto`, and dependent libraries.
    -   Required Python packages: `protobuf==4.21.12`, `betterproto[compiler]==2.0.0b7`, `grpclib`.
    -   **Python Import Strategy:** To avoid conflicts (like the `types` module collision), Python scripts (e.g., `src/api_connections.py`, `src/common_utils.py`) should add the *parent directory* `proto/generated/` to `sys.path` and then import using the full path from `pb_stubs` (e.g., `from pb_stubs.cosmos.tx.v1beta1 import Tx as CosmosTx`). See `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md` for a detailed explanation under Troubleshooting.
-   **Fallback Method for Complex/Specific Mayanode Types (e.g., `types.MsgObservedTxOut`):** Using `protoc --decode` via `subprocess`.
    -   This method requires specific Mayanode `.proto` files (e.g., from commit `59743feb...`) and sometimes manual edits to these protos (like the `Tx.chain` `string` to `bytes` fix).
    -   A compatible `protoc` binary (e.g., v3.20.1) is needed.
    -   This approach is documented in `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md` and demonstrated in `scripts/decode_local_block_for_comparison.py`.

This dual approach provides robust decoding for common cases and a reliable fallback for more challenging ones, with an emphasis on reproducibility by committing generated stubs.

**Key Technical Shifts:**
-   **Data Sources:** Mayanode API (`https://mayanode.mayachain.info/mayachain/block` often provides pre-decoded JSON) and Tendermint RPC (`https://tendermint.mayachain.info/block` provides Protobuf encoded transactions).
-   **Prediction Target:** Entire blocks (including headers, transactions, events) instead of individual transactions.
-   **File Impact:** `fetch_realtime_transactions.py` (updated for block fetching), `api_connections.py`, `common_utils.py`, `preprocess_ai_data.py`, `model.py`, `train_model.py` will require major rewrites or new implementations. `realtime_inference_suite.py` was deleted and will be replaced by a new block-focused version.

**Task Breakdown & Status:**

*   **Task 10.1: Mayanode API & Tendermint RPC Client Implementation (Data Fetching)**
    *   **Description:** Deep dive into Mayanode API & Tendermint RPC for block data. Implement robust functions in `src/api_connections.py` to fetch blocks by height, latest block, and historical ranges. Rewrite `src/fetch_realtime_transactions.py` to use these functions to download and store block data.
    *   **Endpoints:** Mayanode `/mayachain/block`, Tendermint `/block`, `/unconfirmed_txs`, `/num_unconfirmed_txs`.
    *   **Success Criteria:** `api_connections.py` has reliable block and mempool fetching. `src/fetch_realtime_transactions.py` can create a dataset of historical blocks.
    *   **Status: COMPLETE & VERIFIED** (Historical catch-up, specific range/count/target fetching, and continuous polling modes implemented in `fetch_realtime_transactions.py` with improved console output using `tqdm` and `aiohttp` for async historical fetching).

*   **Task 10.2.A.0: Investigate and Confirm Mayanode `/block` Endpoint Response Structure**
    *   **Description:** Determine the structure of data from `https://mayanode.mayachain.info/mayachain/block`.
    *   **Outcome:** Mayanode endpoint `/mayachain/block` returns transactions as already decoded JSON objects. Protobuf deserialization is NOT required for transactions sourced from this specific endpoint.
    *   **Status: COMPLETE**

*   **Task 10.2.A: Block Data Parsing and Schema Definition (`common_utils.py`)**
    *   **Description:** Based on API responses, develop parsing logic in `src/common_utils.py`. For Tendermint RPC data (base64 protobuf transactions), integrate the output of Protobuf decoding (Task 10.2.C). Includes robust `signer` derivation, `camel_to_snake` conversion, and transformation of decoded Tendermint Tx to Mayanode-like JSON.
    *   **Status: COMPLETE & VERIFIED** (Verbose debug logs removed, functions refined).

*   **Task 10.2.B: Database Implementation and Data Ingestion Pipeline (`database_utils.py`, `fetch_realtime_transactions.py`)**
    *   **Description:** Design a fully relational schema (SQLite), implement DB utilities in `database_utils.py` (including table creation, data insertion, and Mayanode JSON reconstruction). Modify `fetch_realtime_transactions.py` for continuous polling and data ingestion, using various fetching strategies (historical, range, count, target height, continuous polling) and improved console output.
    *   **Status: COMPLETE & VERIFIED** (Core relational schema, insertion logic, and reconstruction function are implemented and tested. `fetch_realtime_transactions.py` uses these utilities for robust data ingestion with improved CLI feedback).

*   **Task 10.2.C: Protobuf Decoding Setup & Implementation**
    *   **Description:** Establish and document reliable Protobuf decoding for `cosmos.tx.v1beta1.Tx` (from Tendermint) and a fallback for other complex types.
    *   **Sub-tasks & Status:**
        *   Curate `.proto` files for `betterproto` (`proto/src/`): **COMPLETE**
        *   Generate Python stubs using `betterproto` (`proto/generated/pb_stubs/`): **COMPLETE**
        *   Integrate `betterproto` stubs for `CosmosTx` decoding in scripts (e.g., `compare_block_data.py`, `test_mayanode_decoding.py`): **COMPLETE**
        *   Document `protoc --decode` fallback for `MsgObservedTxOut`: **COMPLETE** (in new guide)
        *   Create comprehensive `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md`: **COMPLETE**
    *   **Success Criteria:** `cosmos.tx.v1beta1.Tx` messages from Tendermint RPC are reliably decoded in Python. A robust, documented method exists for handling other problematic types. The entire setup is reproducible.
    *   **Overall Status for 10.2.C: COMPLETE**

*   **Task 10.2.D: Data Comparison Script (`scripts/compare_block_data.py`)**
    *   **Description:** Create a script to fetch a block from Tendermint RPC and Mayanode API, decode Tendermint transactions, and save both blocks to JSON files for comparison. Also save raw Mayanode API response.
    *   **Success Criteria:** Script successfully fetches, decodes, and saves data. Output files allow for manual comparison and verification.
    *   **Status: COMPLETE**

*   **Task 10.2.E: Flask App for Maya (CACAO) Dividend Viewer (New - replaces old 10.2.D)**
    *   **Description:** Develop a Flask application to display Maya (CACAO) dividend payouts identified from Mayanode blocks. This involves fetching a significant number of blocks, storing them in the database, and then querying/displaying the dividend information.
    *   **Sub-tasks & Status:**
        *   **10.2.E.1: Research Maya (CACAO) Dividend Identification:** **COMPLETE**
        *   **10.2.E.2: Enhance Data Ingestion for Bulk Block Fetching (as part of Task 10.2.B):** **COMPLETE** (`fetch_realtime_transactions.py` supports bulk historical catch-up).
        *   **10.2.E.3: Implement Database Queries for Dividends (`database_utils.py`):** **PENDING**
        *   **10.2.E.4: Develop Flask Application Structure & Logic (`app.py`):** **PENDING**
        *   **10.2.E.5: Test End-to-End Pipeline & Data Accuracy:** **PENDING**
    *   **Status: IN PROGRESS**

*   **Task 10.3: Preprocessing for Block Prediction (`preprocess_ai_data.py`)**
    *   **Description:** Rewrite `src/preprocess_ai_data.py` to load parsed block data, implement feature engineering for block sequences, and generate training artifacts.
    *   **Status: PENDING**

*   **Task 10.4: Adapt Generative Model for Block Prediction (`model.py`)**
    *   **Description:** Update `GenerativeTransactionModel` or create `GenerativeBlockModel` in `src/model.py` for block sequence prediction.
    *   **Status: PENDING**

*   **Task 10.5: Update Training Script for Block Model (`train_model.py`)**
    *   **Description:** Modify `src/train_model.py` to use new block-based data and model.
    *   **Status: PENDING**

*   **Task 10.6: Develop Evaluation Suite for Block Model (`evaluate_model_block.py` - New File)**
    *   **Description:** Create script to evaluate block prediction model performance.
    *   **Status: PENDING**

*   **Task 10.7: Develop Realtime Inference Suite for Block Prediction (`realtime_inference_block.py` - New File)**
    *   **Description:** Create script for live block prediction and simulation.
    *   **Status: PENDING**

*   **Task 10.8: Iteration, Refinement, and Documentation (Phase 10)**
    *   **Description:** Ongoing improvements, bug fixing, performance optimization, and comprehensive documentation.
    *   **Sub-tasks (examples):**
        *   Code refactoring and cleanup.
        *   Performance profiling and optimization.
        *   Add detailed logging.
        *   Comprehensive `README.md` updates.
        *   Maintain `Docs/Implementation Plan.md` and `Docs/Project Requirements.md`.
        *   Maintain and update `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md` as needed.
    *   **Status: ONGOING**

## Risks and Mitigations
- **Risk:** `edit_file` tool instability. (Mitigation: Simpler edits, careful verification).
- **Risk:** Model performance with real-world block data complexity. (Mitigation: Iterative refinement in Task 10.8).
- **Risk:** Model size and inference speed. (Mitigation: Future optimization).
- **Risk:** New or variant Protobuf messages requiring decoding adjustments. (Mitigation: Established `betterproto` workflow and `protoc --decode` fallback are adaptable. The guide provides a strong foundation).

## Next Steps
- **Current Focus:** Begin development of **Task 10.2.E.3: Implement Database Queries for Dividends** as part of the Flask application task.
- Proceed with remaining sub-tasks for the Flask App (10.2.E.4, 10.2.E.5).
- Then move to **Task 10.3: Preprocessing for Block Prediction**.