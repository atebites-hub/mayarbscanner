# Mayanode & Tendermint API Cheat Sheet (Block & Mempool Data)

This document provides a quick reference for Mayanode REST API and Tendermint RPC endpoints. This is crucial for **Phase 10: Generative Block Prediction Model**.

**Confirmed block data will be fetched using the Mayanode REST API. Unconfirmed transaction (mempool) data will be fetched using the Tendermint RPC.**

## Mayanode REST API (for Confirmed Blocks)

**Base URL (Mainnet):** `https://mayanode.mayachain.info/mayachain`

**Official API Documentation (as referenced by user):** `https://mayanode.mayachain.info/mayachain/doc`

### 1. Get Block by Height

Provides detailed information about a block at a specific height. The response is expected to match the user-provided `BlockResponse` schema, which includes structured transaction data.

*   **Endpoint:** `/block`
*   **Method:** `GET`
*   **Query Parameters:**
    *   `height` (integer): The block height to fetch.
*   **Example:** `https://mayanode.mayachain.info/mayachain/block?height=1000`
*   **Expected Response Structure (User-Provided `BlockResponse`):**
    ```json
    {
      "id": {
        "hash": "...",
        "parts": { ... }
      },
      "header": {
        "version": { ... },
        "chain_id": "...",
        "height": "...", // string integer
        "time": "...",
        // ... other header fields
        "proposer_address": "..."
      },
      "begin_block_events": [ { ... } ],
      "end_block_events": [ { ... } ],
      "txs": [ 
        // Array of BlockTx objects with structured transaction details
        {
          "hash": "...",
          "tx": { /* Can be complex, potentially base64 or further structured */ }, 
          "result": { /* Transaction execution result */ }
        }
      ]
    }
    ```
    *Note: The exact structure of `txs[].tx` needs to be confirmed upon successful fetching. It might still require some decoding/parsing if it's not fully expanded JSON.*

### 2. Get Latest Block

Provides detailed information about the most recent block.

*   **Endpoint:** `/block` (without the `height` parameter)
*   **Method:** `GET`
*   **Query Parameters:** None
*   **Example:** `https://mayanode.mayachain.info/mayachain/block`
*   **Note:** Verified to work as expected.

## Tendermint RPC (for Unconfirmed/Mempool Transactions)

**Base URL (Mainnet):** `https://tendermint.mayachain.info` (Confirmed working)

**Official Tendermint RPC Documentation:** `https://docs.tendermint.com/v0.34/rpc/` (or the version relevant to Mayanode)

### 1. Get Unconfirmed Transactions

Returns a list of transactions waiting in the mempool.

*   **Endpoint:** `/unconfirmed_txs`
*   **Method:** `GET` (Tendermint RPC is JSON-RPC over HTTP)
*   **Query Parameters:**
    *   `limit` (integer, optional, e.g., `30`): Maximum number of transactions to return.
*   **Example:** `https://tendermint.mayachain.info/unconfirmed_txs?limit=30`
*   **Expected Response Structure (subset for `result.txs`):
    ```json
    {
      "jsonrpc": "2.0",
      "id": -1,
      "result": {
        "n_txs": "...",       // String count of txs in response
        "total": "...",       // String count of total txs in mempool
        "total_bytes": "...", // String total bytes of txs in mempool
        "txs": [
          "BASE64_ENCODED_TRANSACTION_STRING_1",
          "BASE64_ENCODED_TRANSACTION_STRING_2",
          // ...
        ]
      }
    }
    ```
    *Note: The `txs` are typically base64 encoded raw transaction bytes. These will need to be decoded and further parsed if their internal structure is required (though for just knowing *what* is pending, the opaque strings might be enough initially for the model if it learns to treat them as unique identifiers/features).* 

### 2. Get Number of Unconfirmed Transactions

Returns the count of transactions waiting in the mempool.

*   **Endpoint:** `/num_unconfirmed_txs`
*   **Method:** `GET`
*   **Query Parameters:** None
*   **Example:** `https://tendermint.mayachain.info/num_unconfirmed_txs`
*   **Expected Response Structure:
    ```json
    {
      "jsonrpc": "2.0",
      "id": -1,
      "result": {
        "n_txs": "...",       // String count of txs in response
        "total": "...",       // String count of total txs in mempool
        "total_bytes": "..." // String total bytes of txs in mempool
      }
    }
    ```

## Important Considerations

*   **Rate Limiting:** Be mindful of potential API rate limits when fetching a large number of historical blocks.
*   **Error Handling:** Implement robust error handling for network issues, API errors (e.g., block not found), and unexpected response formats.
*   **Transaction Detail (`txs[].tx`):** The exact format of the `tx` field within the `txs` array in the `BlockResponse` needs careful inspection once data is fetched. It could be fully structured JSON, or it might contain base64 encoded data that requires further decoding, similar to raw Tendermint transactions but hopefully structured as per `BlockTx`.

## Next Steps for Task 10.1 (Post-Documentation Update)

1.  **Rebuild `src/api_connections.py`:** Implement functions to call these Mayanode API endpoints.
    *   Add functions for Tendermint RPC `/unconfirmed_txs` and `/num_unconfirmed_txs`.
2.  **Test `src/api_connections.py`:** 
    *   Verify fetching specific blocks (e.g., height 1000) and the latest block from Mayanode API.
    *   Verify fetching unconfirmed transactions and their count from Tendermint RPC.
3.  **Update `src/fetch_realtime_transactions.py`:** Adapt the script to use the new API functions for confirmed blocks (already done).
4.  **Test `src/fetch_realtime_transactions.py`:** Confirm historical block downloading (already done). 