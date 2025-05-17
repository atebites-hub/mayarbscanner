# Midgard API Cheat Sheet (MayaChain)

This document provides a quick reference for commonly used Midgard API endpoints relevant to collecting historical data for MayaChain.

**Base URL (Mainnet):** `https://midgard.mayachain.info/v2`
**API Documentation:** `https://midgard.mayachain.info/v2/doc`

## Key Endpoints for Historical Data

### 1. Actions

Provides a list of actions (swaps, add/remove liquidity, etc.). Useful for a detailed transaction history.

*   **Endpoint:** `/actions`
*   **Method:** `GET`
*   **Key Parameters:**
    *   `limit`: Number of actions to return (e.g., `50`).
    *   `offset`: Number of actions to skip (for pagination).
    *   `asset`: Comma-separated list of assets to filter by (e.g., `BTC.BTC,ETH.ETH`).
    *   `type`: Comma-separated list of action types (e.g., `swap,addLiquidity`).
    *   `txid`: Filter by a specific transaction ID.
    *   `address`: Filter by a specific address involved in the action.
*   **Example:** `https://midgard.mayachain.info/v2/actions?limit=10&offset=0&asset=MAYA.CACAO`

### 2. Transactions

Provides details for a specific transaction by its hash.

*   **Endpoint:** `/tx/{txid}`
*   **Method:** `GET`
*   **Key Parameters:**
    *   `txid`: The transaction hash (required path parameter).
*   **Example:** `https://midgard.mayachain.info/v2/tx/YOUR_TRANSACTION_ID`

### 3. Pool History

Provides historical data for a specific liquidity pool (depth, volume, price, etc.).

*   **Endpoint:** `/history/pools` (general trends) or `/history/depths/{pool}` (specific pool depth over time)
*   **Method:** `GET`
*   **Key Parameters for `/history/depths/{pool}`:**
    *   `pool`: The pool identifier (e.g., `BTC.BTC`).
    *   `interval`: Time interval for data points (e.g., `day`, `hour`).
    *   `from`: Start timestamp (Unix timestamp).
    *   `to`: End timestamp (Unix timestamp).
    *   `count`: Number of data points to return.
*   **Example:** `https://midgard.mayachain.info/v2/history/depths/BTC.BTC?interval=day&count=30`

### 4. Network Data

Provides general network information.

*   **Endpoint:** `/network`
*   **Method:** `GET`
*   **Use:** Can be used to get current block height, active nodes, etc.

### 5. Pools

Provides a list of all available liquidity pools and their current status.

*   **Endpoint:** `/pools`
*   **Method:** `GET`
*   **Key Parameters:**
    *   `status`: Filter by pool status (e.g., `available`).
*   **Example:** `https://midgard.mayachain.info/v2/pools?status=available`

## General Notes

*   Refer to the official API documentation for the most up-to-date details, full parameter lists, and response structures.
*   Pagination is typically handled with `limit` and `offset` parameters.
*   Timestamps are generally in Unix timestamp format (seconds). 