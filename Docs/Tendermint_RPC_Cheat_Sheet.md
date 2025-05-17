# Tendermint RPC Cheat Sheet (MayaChain)

This document provides a quick reference for commonly used Tendermint RPC endpoints relevant to accessing mempool data for MayaChain.

**Base URL (Mainnet Example):** `https://tendermint.mayachain.info` (Port: 27147 if connecting directly to a node)
**General API Documentation:** `https://docs.tendermint.com/v0.34/rpc/` (or `/master/` for latest)

## Key Endpoints for Mempool Data

### 1. Unconfirmed Transactions

Retrieves a list of unconfirmed (pending) transactions in the mempool.

*   **Endpoint:** `/unconfirmed_txs`
*   **Method:** `GET`
*   **Key Parameters:**
    *   `limit`: Maximum number of transactions to return (e.g., `30`, default is `30`, max is `100`).
*   **Example (using a public node):** `https://tendermint.mayachain.info/unconfirmed_txs?limit=50`
*   **Example (direct to node):** `http://<node_ip>:27147/unconfirmed_txs?limit=50`
*   **Response:** Contains a list of transactions, each typically base64 encoded. These will need to be decoded and further parsed (often as protobuf messages specific to MayaChain/Cosmos SDK transaction types).

### 2. Number of Unconfirmed Transactions

Retrieves the count of unconfirmed (pending) transactions in the mempool.

*   **Endpoint:** `/num_unconfirmed_txs`
*   **Method:** `GET`
*   **Example (using a public node):** `https://tendermint.mayachain.info/num_unconfirmed_txs`
*   **Example (direct to node):** `http://<node_ip>:27147/num_unconfirmed_txs`
*   **Response:** Provides the total number of transactions in the mempool, the number of transactions, and the total bytes.

### 3. Broadcast Transaction (Async, Sync, Commit)

Used to submit new transactions to the network. While not for *reading* mempool data, it's how transactions get *into* the mempool.

*   **Endpoints:**
    *   `/broadcast_tx_async`: Returns immediately.
    *   `/broadcast_tx_sync`: Waits for `CheckTx`.
    *   `/broadcast_tx_commit`: Waits for the transaction to be committed in a block.
*   **Method:** `GET` (can also be `POST`)
*   **Key Parameters:**
    *   `tx`: The transaction payload, usually as a hex-encoded string prefixed with `0x`, or a base64 encoded string for POST requests.
*   **Example (async):** `https://tendermint.mayachain.info/broadcast_tx_async?tx=0x...`
*   **Note:** The transaction format is specific to MayaChain (Cosmos SDK based) and needs to be properly constructed and serialized.

## General Notes

*   The actual transaction data within the mempool responses will be MayaChain-specific (likely Cosmos SDK `StdTx` or similar, protobuf encoded and then often base64 encoded in the JSON RPC response).
*   You will need appropriate libraries to decode and interpret these transactions (e.g., corresponding to MayaChain's protobuf definitions).
*   Public RPC endpoints might have rate limits or restrictions.
*   For reliable and high-throughput access, running a local MayaChain node and querying its Tendermint RPC is recommended.
*   The default port for Tendermint RPC on MayaChain mainnet nodes is `27147`.
*   Refer to the general Tendermint RPC documentation and MayaChain-specific resources for details on transaction structures and encoding. 