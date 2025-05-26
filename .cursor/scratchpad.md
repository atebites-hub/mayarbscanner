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

**Resolution of `signer` Field Issue & Caching Mystery (Key Learning):**
A persistent issue where the `signer` field in decoded Tendermint transactions (specifically in `body.messages[0].signer`) was not being correctly populated by deriving it from `authInfo.signerInfos` has been **RESOLVED**.
The root causes were twofold:
1.  **Test Data:** The primary test transaction string being used likely resulted in an `AuthInfo` object that was empty or `None` *after* the initial `CosmosTx().parse()` step. When `betterproto`'s `to_dict(include_default_values=False)` was called on such an empty `AuthInfo` object, it would correctly yield an empty dictionary, leading to no `signerInfos` being available for the derivation logic.
2.  **Python Caching/Environment:** A significant challenge was a suspected Python caching or environment issue that prevented updates to `src/common_utils.py` (especially new `print` statements) from being reflected when `scripts/compare_block_data.py` was run. This was resolved by:
    *   Clearing `__pycache__` directories and `.pyc` files.
    *   Temporarily using a faulty import in `compare_block_data.py` to confirm it *was* reading the latest `common_utils.py` (it crashed as expected).
    *   Finally, isolating the test by creating a new script (`temp_test_common_utils.py`) and running it in a fresh terminal, which consistently showed the latest code changes.
Once a new test transaction string (known to contain `signer_infos`) was used with the fresh environment, the existing `signer` derivation logic in `src/common_utils.py` was confirmed to be working correctly.

**Current Goal for Protobuf:**
-   **Primary Solution for Tendermint Tx (`CosmosTx`):** Utilize the `betterproto`-generated stubs for native Python deserialization. This is now implemented and **VALIDATED, including correct `signer` field population.**
-   **Fallback for Complex/Specific Mayanode Types (e.g., `MsgObservedTxOut`):** Rely on the documented `subprocess` method calling `protoc --decode`, as detailed in `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md`.
-   The workspace now contains the necessary `proto/src` and `proto/generated/pb_stubs` (which should be committed to Git) for the `betterproto` approach.

## Key Challenges and Analysis (NEW - For Block Prediction)

-   **Mayanode API Interaction:** Robust interaction with Mayanode API for block data. (Largely addressed)
-   **Tendermint RPC for Mempool & Blocks:** Successfully fetching unconfirmed Txs and full blocks. (Addressed)
-   **API Response Variability (`/mayachain/block`):** Understood; Mayanode REST provides decoded JSON, Tendermint RPC provides protobuf.
-   **Protobuf Deserialization (Native Python):**
    -   For `cosmos.tx.v1beta1.Tx` from Tendermint RPC: **VALIDATED** using `betterproto`, including robust handling of `auth_info` and derivation of `signer` fields.
    -   For complex Mayanode-specific types like `types.MsgObservedTxOut`: Addressed via documented fallback (`subprocess` with `protoc --decode`).
-   **Handling Empty Blocks:** Addressed in parsing logic.
-   **Toolchain for Protobuf:**
    -   **Primary:** `betterproto[compiler]==2.0.0b7`, `protobuf==4.21.12`, `grpclib`, and `python -m grpc_tools.protoc` for generating stubs for `CosmosTx`.
    -   **Fallback/Diagnostic:** Standalone `protoc` (e.g., v3.20.1) for `protoc --decode` with specific Mayanode protos.
-   **Python Environment/Caching:** Previously a challenge, now understood and mitigated through direct testing and cache clearing when necessary.
-   **Complex JSON Reconstruction:** Designing and implementing the logic to accurately reconstruct the nested JSON structure of a Mayanode API block from purely relational database tables will be a complex task, requiring careful mapping of all fields and potentially impacting query performance for this specific function.

## Database Design Philosophy (NEW)

-   **Core: Transaction Ledger:** The blockchain is fundamentally a transaction ledger. The database schema reflects this by centering around `blocks` and `transactions` tables, with other tables linking to them.
-   **Normalization & Relational Purity:** The schema aims for a **fully relational design** to maximize data integrity, query flexibility, and adherence to database best practices. All discrete data points from blocks, transactions, messages, events, and their attributes will be stored in dedicated tables and columns. **No raw JSON blobs for transactions or their sub-components will be stored directly in the database.**
-   **Relational Integrity:** Foreign key constraints are defined to maintain relationships between tables (e.g., a transaction must belong to a valid block, an event attribute must belong to a valid event).
-   **JSON Reconstruction on Demand:** Complex JSON structures, such as the original Mayanode API block response, will be **reconstructed by querying the relational tables and assembling the data in Python**. A key utility (`reconstruct_block_as_mayanode_api_json`) will be responsible for this.
-   **Querying for Account History:** Account transaction history is derived by querying the `transaction_address_link` table for a given address, then joining with `transactions` and `blocks` to retrieve detailed information. The `addresses` table (with `first_seen`/`last_seen`) provides a summary but links are definitive.
-   **Data for AI:** While the stored schema is normalized, data will be denormalized via joins during preprocessing for AI model training to create comprehensive feature sets. The relational source allows flexible feature engineering.
-   **Comprehensive Data Storage:** The goal is to store all relevant data from blocks. Complex nested JSON objects (like message bodies or event attributes) are stored as JSON strings (TEXT) in SQLite for now. Future migration to PostgreSQL could leverage JSONB for more advanced querying within these structures. **REVISED: All data points will be stored in discrete fields. JSON strings will not be used for structured data within the database; reconstruction of original JSON is done on demand.**

## High-level Task Breakdown

**Phases 1-5: Transaction-Level Prediction (Midgard API) - SUPERSEDED**
*   All tasks related to fetching data from Midgard and predicting individual transactions are now considered superseded by the new block-level prediction approach using the Mayanode API.

---

**Phase 0: Project Setup & Prerequisites**
*   **Task 0.1: Setup/Verify Python Virtual Environment and Install Dependencies (COMPLETE)**
    *   **Description:** Ensure a Python virtual environment (e.g., `venv`) is active in the project root. Install or update all dependencies listed in `requirements.txt`.
    *   **Success Criteria:** `pip install -r requirements.txt` completes successfully within an activated virtual environment. Key scripts (e.g., `src/api_connections.py`) can import their required libraries without error.

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

*   **Task 10.2.A: Block & Transaction Data Parsing Logic (`common_utils.py`) & Comparison (`compare_block_data.py`) (COMPLETE & VERIFIED)**
    *   Sub-tasks:
        *   Enhance `scripts/compare_block_data.py`:
            *   Make `BLOCK_HEIGHT` a command-line argument. (COMPLETE)
            *   Implement a dedicated transaction-by-transaction comparison function. (COMPLETE)
            *   Implement a full parsed block structure comparison function. (COMPLETE)
        *   Run `scripts/compare_block_data.py` for a test block (e.g., 11320570) and analyze differences. (COMPLETE)
            *   **Finding 1 (Transaction Content):** The `compare_transaction_lists` function shows a **MATCH** for all 22 transactions in block 11320570 after Tendermint transactions are decoded and transformed. This validates the core Protobuf decoding and transformation logic in `src/common_utils.py`.
            *   **Finding 2 (Parsed Fee Attribute):** The `compare_full_parsed_blocks` function initially showed a type mismatch for `transactions[X].result_events_parsed[0].attributes.fee` (string from Mayanode API vs. None from Tendermint RPC).
                *   Investigation revealed that the Tendermint RPC `/block_results` for this block provides `value: null` for the `fee` attribute in the raw `tx` event.
                *   The `AuthInfo.fee.amount` list in the decoded Protobuf transaction was also empty for these transactions.
                *   Therefore, `src/common_utils.py` correctly parses the Tendermint event fee as `None` and, finding no fee in `AuthInfo` to populate it from, leaves it as `None`.
                *   This mismatch is a genuine data presentation difference between the Mayanode API (which provides a fee string) and the combination of Tendermint RPC + AuthInfo (which don't for this block/tx type). The parsing accurately reflects this. A comment was added to `common_utils.py` to document this.
            *   **Finding 3 (Other Parsed Block Diffs):** Other differences in `compare_full_parsed_blocks` (e.g., `end_block_events_raw`, `transactions_raw_base64`, `transactions_source`, raw event structures within `transactions[X].result_events_raw`) are understood to be artifacts of how the two different data sources present their raw data or how `parse_confirmed_block` stores source-specific raw information. The *parsed transactions* themselves align.
    *   Success Criteria: `scripts/compare_block_data.py` runs successfully, producing detailed comparison reports. The core transaction decoding and transformation logic is validated. Differences in full block structures are understood and documented, distinguishing between parsing issues and genuine data source variations. (ACHIEVED)

*   **Task 10.2.B: Database Implementation and Data Ingestion Pipeline (Relational Refactor)**
    *   **Description:**
        *   Design a **fully relational** database schema (e.g., using SQLite initially) to store every component of parsed confirmed block data. This includes block headers, individual transactions (core details like hash, index, success status), transaction messages (with type, index within transaction, and links to message-specific detail tables), message-specific attributes (e.g., sender, recipient, amounts, memo for `MsgSend`; details for `MsgSwap`, `MsgDeposit`, `MsgOutboundTx`, etc., likely in separate tables per message type or a structured EAV-like model if message types are too diverse), transaction-level events (type, index, attributes), event attributes (key, value, index, potentially nested if necessary and manageable), block-level events (begin/end block, type, attributes), unique addresses involved in transactions/events, and any other discrete data points. **No raw JSON blobs for transactions, messages, or events will be stored.**
        *   Implement Python functions in `src/database_utils.py` for:
            *   Database initialization (`create_tables`) with the fully relational schema.
            *   Granular data insertion functions for each distinct entity (e.g., `insert_block_header`, `insert_transaction_core`, `insert_transaction_message_base`, `insert_msg_send_details`, `insert_event`, `insert_event_attribute`, `insert_address_link`, etc.). These will be orchestrated by a main `insert_block_and_components` function.
            *   Comprehensive query functions to retrieve data for analysis and reconstruction.
            *   A key function: `reconstruct_block_as_mayanode_api_json(conn, block_height)` that queries all relevant tables and meticulously reassembles the complete block JSON structure to match the Mayanode API's `/mayachain/block` output as closely as feasible.
        *   Modify `src/fetch_realtime_transactions.py` (or create a new `src/data_ingestion_service.py`) to:
            *   Continuously poll the Mayanode API for new confirmed blocks (compared to the latest in DB).
            *   Implement a configurable delay (e.g., 5-10 seconds) between polls to respect API rate limits.
            *   Fetch, parse (using 10.2.A logic), and **insert new confirmed block data by decomposing it into the new relational database schema using the updated `database_utils.py` functions.**
            *   Ensure the system avoids redundant API calls for data already present in the local database.
        *   Develop utility functions to query the database, including constructing an address's transaction history from locally stored transactions.
        *   **Mempool Data Strategy:** For inference, live mempool data will be fetched on demand (as per `construct_next_block_template` in `api_connections.py`). Storing historical mempool snapshots in the database is a secondary consideration if a clear need for training/analysis emerges.
    *   **Success Criteria:**
        *   A SQLite database is created with a **fully relational schema capable of storing all discrete data points from a Mayanode block.**
        *   The data ingestion script (`fetch_realtime_transactions.py` or equivalent) continuously and politely populates the database with new confirmed blocks, correctly decomposing them into the relational schema.
        *   The `reconstruct_block_as_mayanode_api_json` function in `database_utils.py` can successfully query the database for a given block height and produce a JSON output that is a structural and informational **carbon copy** (as much as feasible) of the Mayanode API's response for that block.
        *   Data can be queried efficiently for various analytical purposes.
        *   The system avoids redundant API calls for confirmed blocks.

*   **Task 10.2.C: Protobuf Decoding Setup & Implementation (COMPLETE & VALIDATED)**
    *   **Overall Objective:** Achieve reliable native Python deserialization for `cosmos.tx.v1beta1.Tx` from Tendermint RPC, ensuring the transformed output matches the Mayanode API transaction structure. Have a documented fallback for complex Mayanode-specific types.
    *   **Current Status & Approach:**
        *   **For `cosmos.tx.v1beta1.Tx` (Tendermint RPC):** Successfully implemented and **VALIDATED** using `betterproto` generated Python stubs. The `decode_cosmos_tx_string_to_dict` and `transform_decoded_tm_tx_to_mayanode_format` functions in `src/common_utils.py` now produce output that matches the Mayanode API's JSON structure for transactions (tested with all 22 transactions in Mayanode block 11320570). This includes correct `signer` field population and alignment of other fields. The necessary `.proto` files are in `proto/src/`, generated stubs in `proto/generated/pb_stubs/`.
        *   **For `types.MsgObservedTxOut` (and similar complex Mayanode types):** The `subprocess` method calling `protoc --decode` (using specific Mayanode protos from commit `59743feb...` and the `Tx.chain` fix) remains the reliable fallback. This is demonstrated in `scripts/decode_local_block_for_comparison.py`.
        *   Both methods are documented in `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md`.
    *   **Sub-Tasks:**
        *   **Task 10.2.C.1: Curate `.proto` files for `betterproto` (COMPLETE)**
        *   **Task 10.2.C.2: Generate Python stubs using `betterproto` (COMPLETE)**
        *   **Task 10.2.C.3: Integrate `betterproto` stubs for `CosmosTx` decoding (COMPLETE & VALIDATED)**
        *   **Task 10.2.C.4: Document `protoc --decode` fallback for `MsgObservedTxOut` (COMPLETE)**
        *   **Task 10.2.C.9: Experimentation - Alternative Python Protobuf Deserialization (COMPLETE - `betterproto` validated for CosmosTx, `subprocess` validated for MsgObservedTxOut)**
    *   **Success Criteria:** `cosmos.tx.v1beta1.Tx` messages from Tendermint RPC are reliably decoded and transformed in Python to match the Mayanode API's transaction JSON structure, including correct population of the `signer` field and other structural alignments. A robust, documented method exists for handling other problematic types.

*   **Task 10.2.D: Flask App for Maya (CACAO) Dividend Viewer (NEW)**
    *   **Description:** Develop a Flask application to display Maya (CACAO) dividend payouts identified from Mayanode blocks. This involves fetching a significant number of blocks (e.g., 10,000), storing them in the database, and then querying/displaying the dividend information. The app should allow users to select a wallet address and view its received dividends, including the block height, date, and time of issuance, and amount. This task serves as an end-to-end test for the data pipeline (fetch -> parse -> store -> query -> display).
    *   **Sub-tasks:**
        *   **10.2.D.1: Research Maya (CACAO) Dividend Identification (COMPLETE)**
            *   **Objective:** Define how to identify CACAO "dividend" (reward/emission) payouts to Node Operators, Liquidity Providers, and potentially $MAYA token holders from parsed Mayanode block data.
            *   **Research Findings (Summary & Finalized Identification Logic):**
                *   CACAO rewards/dividends (for LPs & Node Operators) are primarily distributed via mechanisms reflected in `end_block_events`.
                *   **Key Identifier:** `transfer` events within `end_block_events` originating from a known Mayanode protocol module address (e.g., the Reserve or a dedicated rewards module).
                    *   **Example Sender (Reserve/Rewards Module):** `maya1dheycdevq39qlkxs2a6wuuzyn4aqxhve4hc8sm` (identified from block 11317539 analysis). This address receives CACAO from fee collection (e.g., from a fee collector module like `maya1g98cy3n9mmjrpn0sxmn63lztelera37n8yyjwl`) and then distributes it.
                    *   **Recipient:** The `recipient` attribute of such a `transfer` event is the address receiving the CACAO dividend.
                    *   **Amount:** The `amount` attribute of the `transfer` event (e.g., `"126498452cacao"`) indicates the raw CACAO amount (typically 8 decimal places).
                *   **Corroborating Event:** The `rewards` event, also in `end_block_events`, contains a `bond_reward` attribute (e.g., `"126498452"`) which shows the CACAO amount allocated for bonders in that block. This often matches one of the `transfer` amounts from the Reserve to an individual address.
                *   **Fee Collection Path:** Fees (e.g., from swaps) are collected (seen as `pool_deduct` in a `fee` event), then often transferred from a fee collector module to the Reserve/Rewards module, then distributed.
                *   **CACAO Denomination in Events:** `XXXXcacao` (e.g., `12345cacao`).
                *   **$MAYA Token Holder Dividends:** This is a separate mechanism. Identification would require finding the specific CACAO distribution address for $MAYA token revenue share and observing its `transfer` events. *This is out of scope for the immediate CACAO dividend identification for LPs/Nodes but noted for future potential.* 
            *   **Identification Strategy for Parsed Block Data (to be implemented in DB queries):
                1.  Iterate through `end_block_events` of each parsed block.
                2.  Look for `event` objects with `type: "transfer"`.
                3.  For each such `transfer` event, check its `attributes`:
                    *   An attribute with `key: "sender"` whose `value` matches a known Mayanode Reserve/Rewards module address (e.g., `"maya1dheycdevq39qlkxs2a6wuuzyn4aqxhve4hc8sm"`). This list of sender addresses might need to be configurable or expanded if more are discovered.
                    *   An attribute with `key: "amount"` whose `value` ends with `"cacao"` (e.g., `"12345cacao"`). Extract the numeric part as the raw amount.
                    *   An attribute with `key: "recipient"` whose `value` is the address that received the CACAO dividend.
                4.  Store these identified dividend transfers (recipient, amount, block height, block time) in the database.**
            *   Determine how "Maya dividends" (paid in CACAO) are represented in Mayanode block transaction data (e.g., specific message types, event types, event attributes like `rewards` or `transfer` from known protocol addresses, or specific module interactions). (COMPLETE - As above)
            *   Document the exact criteria for identifying a CACAO dividend payout, the amount, and the recipient address from the parsed block and transaction data. (COMPLETE - As above)
            *   Success Criteria: Clear, documented logic for identifying CACAO dividend transactions, their amounts, and recipients from parsed block data. (ACHIEVED)
        *   **10.2.D.2: Enhance Data Ingestion for Bulk Block Fetching (COMPLETE)**
            *   Modify/ensure `src/fetch_realtime_transactions.py` (or the primary data ingestion script) can efficiently fetch and store a large number of historical blocks (e.g., target 10,000, or up to a specified block height) if they are not already in the database.
            *   Implemented `--fetch-range START:END`, `--fetch-count N`, and `--target-height H` command-line arguments in `src/fetch_realtime_transactions.py` to enable specific fetching tasks. These modes cause the script to exit after completion, making it suitable for bulk fetching.
            *   Implement or verify robust error handling and progress indication for this bulk fetching process. (Existing script has good error handling and progress prints per block).
            *   Success Criteria: The local SQLite database is populated with data from at least 10,000 Mayanode blocks (or a comparable large dataset). (Script functionality achieved, actual population pending execution by user for Flask app test).
        *   **10.2.D.3: Implement Database Queries for Dividends:**
            *   Add functions to `src/database_utils.py` to:
                *   Query and list unique wallet addresses that have received CACAO dividends based on the criteria from 10.2.D.1.
                *   Query all CACAO dividend transactions for a specific wallet address, retrieving details like block height, block timestamp, and dividend amount.
            *   Ensure these queries are reasonably optimized for performance against the potentially large dataset.
            *   Success Criteria: Database utility functions can efficiently retrieve the list of dividend-receiving addresses and detailed dividend information for a given address.
        *   **10.2.D.4: Develop Flask Application Structure & Logic:**
            *   Set up a basic Flask application (`app.py` or similar in the root or a new `flask_app/` directory).
            *   Implement Flask routes and view functions:
                *   A main page (`/`) to list all unique wallet addresses that have received dividends.
                *   A detail page (e.g., `/address/<wallet_address>`) for a selected wallet address, displaying a table of its dividends (block height, date/time, CACAO amount).
            *   Create simple HTML templates (e.g., using Jinja2) for rendering these views.
            *   Integrate the Flask app with `src/database_utils.py` to fetch the necessary data.
            *   Success Criteria: A functional Flask web application with distinct pages for listing addresses and viewing detailed dividends for a selected address. Data is dynamically fetched from the database.
        *   **10.2.D.5: Test End-to-End Pipeline & Data Accuracy:**
            *   Perform a full run: Fetch 10,000 blocks, let them be parsed and inserted into the database.
            *   Use the Flask application to browse the dividend data.
            *   If possible, manually verify a few dividend transactions by cross-referencing with a block explorer or known dividend events to confirm the accuracy of the identified data and amounts.
            *   Success Criteria: The Flask application correctly displays CACAO dividend information sourced from the locally populated database of 10,000 Mayanode blocks, validating the integrity of the data pipeline (fetch, parse, store, query, display).
    *   **Success Criteria (Overall for 10.2.D):** A running Flask application correctly displays CACAO dividend information for various addresses, sourced from the locally populated database of Mayanode blocks. The process validates the data fetching, parsing, storage, and retrieval mechanisms.

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

**Overall Status:** In Progress - Core data fetching, Protobuf decoding, relational database schema, and initial data ingestion pipeline (`src/fetch_realtime_transactions.py` with historical catch-up and polling) are implemented and functioning. Console output for historical catch-up has been significantly improved.

**Recent Accomplishments & Key Milestones:**
- Successfully fetched block data from Tendermint RPC and Mayanode API, and compared them (`scripts/compare_block_data.py`).
- Implemented robust, native Python decoding of `cosmos.tx.v1beta1.Tx` using `betterproto`.
- Documented Protobuf decoding methods in `Docs/Python_Protobuf_Decoding_Guide_Mayanode.md`.
- **RESOLVED: The persistent issue with `signer` field population in `src/common_utils.py` and associated Python environment/caching mystery.**
- **Database Refactor:** Successfully redesigned `src/database_utils.py` with a fully relational schema. Implemented and tested `insert_block_and_components` and `reconstruct_block_as_mayanode_api_json`.
- **Data Ingestion Enhancements (`src/fetch_realtime_transactions.py`):**
    - Added various fetching modes: `--historical-catchup`, `--fetch-range`, `--fetch-count`, `--target-height`.
    - Implemented asynchronous fetching for `--historical-catchup` using `aiohttp` and `asyncio`.
    - **Improved Console Output:** Refined logging in historical catch-up to use `tqdm` effectively, providing a clean, single-line animated progress bar with status updates and ETA, and removed verbose debug prints from `common_utils.py`.
- **`common_utils.py` Refinements:** Removed unused `_parse_event_attributes`, improved `parse_iso_datetime` and `transform_decoded_tm_tx_to_mayanode_format`, enhanced `camel_to_snake`, and added more tests.

**Current Tasks & Focus:**
-   Task 10.2.D (Flask App for CACAO Dividends): This is the next major development task.
    -   10.2.D.3: Implement Database Queries for Dividends (PENDING)
    -   10.2.D.4: Develop Flask Application Structure & Logic (PENDING)
    -   10.2.D.5: Test End-to-End Pipeline & Data Accuracy (PENDING)

**Upcoming Tasks:**
-   Task 10.3: Preprocessing for Block Prediction (`preprocess_ai_data.py`)
-   Task 10.4: Adapt Generative Model for Block Prediction (`model.py`)

## Executor's Feedback or Assistance Requests
-   The console output for `src/fetch_realtime_transactions.py` in historical catch-up mode should now be significantly cleaner. Requesting user to test this.
-   Ready to proceed with Task 10.2.D.3 (Database Queries for CACAO Dividends) once user confirms satisfaction with current state or provides further feedback.

## Lessons
-   **Manual DB Deletion:** When schema changes are significant, especially removing columns or tables that SQLite might not handle well with `ALTER TABLE`, it's often cleaner to delete the `.db` file manually and let the `create_tables` function rebuild it from scratch.
-   **Protobuf Availability:** The `PROTOBUF_AVAILABLE` (now `PROTO_TYPES_AVAILABLE`) flag is crucial for scripts that depend on protobuf compiled types. Ensure it's correctly checked and that scripts handle its state gracefully.
-   **Python Module Execution:** Scripts intended to be part of a package (e.g., using relative imports like `from . import common_utils`) should be run as modules using `python -m src.script_name` from the project root if they are in a subdirectory like `src/`.
-   **Linter Error Iteration:** If linter errors persist after 2-3 attempts on the same block of code, especially with `cursor.execute` calls, it's better to pause, simplify the problematic SQL or Python logic, or ask the user for a fresh perspective rather than repeatedly trying minor variations. The issue might be more fundamental than a simple syntax error.
-   **Focus on One Task:** When in Executor mode, complete one task from the "Project Status Board" at a time. Inform the user upon completion, detailing the milestone achieved based on success criteria and test results, and await manual testing/verification before proceeding.
-   **Scratchpad for Resilience:** The IDE (Cursor) can be unstable. Regularly update the scratchpad, especially with the current task context, to ensure work can be resumed efficiently after crashes or context loss.
-   **`tqdm` for Clean CLI Progress:** For long-running batch processes like historical data fetching, `tqdm` provides excellent progress bars. Careful management of its `postfix` and `description` attributes, along with conditional printing in helper functions, can lead to a clean, single-line updating display, improving user experience. Avoid printing unrelated messages from within loops that `tqdm` is managing, as this breaks the animation.
-   **Verbose Logging Control:** Debug print statements are useful during development but should be conditional or easily removable for cleaner operational output.

## Current Context / Last Task Worked On

**Overall Goal:** Build a generative block prediction model using Mayanode API and Tendermint RPC data. Focus is on robust data acquisition, parsing, storage, and retrieval.

**Current Phase:** Data Layer Implementation (Task 10.2), specifically refinements to data ingestion and preparing for the Flask app (Task 10.2.D).

**Last Major Task Group Completed (Refinements to `common_utils.py` and `fetch_realtime_transactions.py`):**
-   **`common_utils.py`:**
    -   Removed unused `_parse_event_attributes`.
    -   Improved `parse_iso_datetime` for robustness.
    -   Refined `camel_to_snake` for better edge case handling.
    -   Enhanced `transform_decoded_tm_tx_to_mayanode_format` for more accurate Mayanode API structure replication (removal of default/empty fields, handling of internally added keys).
    -   Added more comprehensive tests for `camel_to_snake` and `transform_decoded_tm_tx_to_mayanode_format`.
    -   Commented out verbose debug print statements (`[DEBUG parse_confirmed_block]`, `[TM_RPC_DEBUG]`).
-   **`fetch_realtime_transactions.py` (for improved historical catch-up console output):**
    -   Modified `fetch_and_store_block` to only print the initial "Processing block..." message if `tqdm_bar` is *not* active.
    -   Enhanced the `run_historical_catchup` function:
        *   When a block is skipped because it already exists, `tqdm` postfix is updated (e.g., "Block 12345 exists. Skipped.").
        *   After fetching and attempting to store a block, `tqdm` postfix is updated with the status (e.g., "Processing: 12346 OK", "Processing: 12347 FAILED (fetch)", or "Processing: 12348 FAILED (store)").
        *   The `tqdm` progress bar initialization now uses `bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'` for a standard, informative layout.
        *   The ETA calculation is updated more frequently within the loop and updates the `tqdm` bar's description (e.g., "Historical Catch-up (ETA: 0:00:XX)").
        *   The `processed_count_in_catchup` (used for ETA) now more accurately reflects blocks that were actually fetched/processed, not just skipped, for better ETA calculation.
        *   Ensured `overall_progress_bar.set_description_str` is called to reset or set the description before setting the postfix, which helps keep the `tqdm` line cleaner.

**Reasoning:** The primary goal was to address the user's feedback regarding verbose debug logs and to improve the console user experience during historical data fetching. Removing debug prints declutters the output. The `tqdm` enhancements aim to provide a clean, single-line animated progress bar that gives clear status updates for each block and a more accurate ETA, rather than multiple scrolling lines.

**Awaiting User Input:** Confirmation of improved console output for historical catch-up in `fetch_realtime_transactions.py`. Then, ready to proceed with Task 10.2.D.3 (Database Queries for CACAO Dividends).

## Last Task Worked On Context

**File:** `src/common_utils.py` and `src/fetch_realtime_transactions.py`
**Summary of Changes:**

*   `src/common_utils.py`:
    *   Commented out all `print(f"[DEBUG parse_confirmed_block] ...")` and `print(f"  [TM_RPC_DEBUG] ...")` statements to reduce console verbosity during block parsing.

*   `fetch_realtime_transactions.py` (for improved historical catch-up console output):
    -   Modified `fetch_and_store_block` to only print the initial "Processing block..." message if `tqdm_bar` is *not* active.
    -   Enhanced the `run_historical_catchup` function:
        *   When a block is skipped because it already exists, `tqdm` postfix is updated (e.g., "Block 12345 exists. Skipped.").
        *   After fetching and attempting to store a block, `tqdm` postfix is updated with the status (e.g., "Processing: 12346 OK", "Processing: 12347 FAILED (fetch)", or "Processing: 12348 FAILED (store)").
        *   The `tqdm` progress bar initialization now uses `bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'` for a standard, informative layout.
        *   The ETA calculation is updated more frequently within the loop and updates the `tqdm` bar's description (e.g., "Historical Catch-up (ETA: 0:00:XX)").
        *   The `processed_count_in_catchup` (used for ETA) now more accurately reflects blocks that were actually fetched/processed, not just skipped, for better ETA calculation.
        *   Ensured `overall_progress_bar.set_description_str` is called to reset or set the description before setting the postfix, which helps keep the `tqdm` line cleaner.

**Reasoning:** The primary goal was to address the user's feedback regarding verbose debug logs and to improve the console user experience during historical data fetching. Removing debug prints declutters the output. The `tqdm` enhancements aim to provide a clean, single-line animated progress bar that gives clear status updates for each block and a more accurate ETA, rather than multiple scrolling lines.