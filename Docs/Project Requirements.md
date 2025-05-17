# Product Requirements Document (PRD)

## Title
Transformer Model for Arbitrage Prediction between Maya Protocol and Uniswap

## Problem Statement
Arbitrageurs (arbs) exploit price differences between Maya Protocol and Uniswap to profit from slip fees. To maximize profit, arbs need to predict competitor trades in the next block and set their trade sizes (ΔX) to outbid competitors, securing priority in transaction processing. A transformer model can analyze historical and real-time data to predict these trades, enabling a profitable arbitrage strategy.

## Objectives
- Build a transformer model to predict:
  - `p_j`: Probability of trading for each active arb in the next block.
  - `μ_j`: Expected trade size for each active arb.
- Incorporate real-time mempool data to refine predictions.
- Enable efficient real-time inference on a MacBook M3 Max with 48GB VRAM.
- Ensure adaptability to evolving arb behavior through periodic retraining.

## Functional Requirements

### Data Collection
- Fetch historical transaction data from the Maya blockchain:
  - Fields: `arb_ID`, `ΔX` (trade size), `block_ID`, `timestamp`, `asset`, `memo`.
- Collect external market data:
  - Uniswap pool price (`P_u`), Bitcoin price (e.g., via CoinGecko).
- Access real-time mempool data for pending transactions targeting the Maya pool.

### Data Preprocessing
- Encode each transaction into a vector with:
  - `arb_ID` embedding, normalized `ΔX`, normalized `block_ID`, `timestamp`, global features (`P_m`, `P_u`).
- Construct input sequences of `M` trades (e.g., last 1000 trades or last 10 blocks).

### Model Architecture
- Use a transformer encoder:
  - 4-8 layers, 8 attention heads, hidden dimension of 1024.
- **Input:** Sequence of `M` transaction vectors.
- **Output:** For each of `K` active arbs, predict `p_j` (probability) and `μ_j` (trade size).

### Training
- Prepare a dataset of historical sequences with corresponding targets (actual trades).
- Use a combined loss function:
  - Binary Cross-Entropy (BCE) for `p_j`.
  - Mean Squared Error (MSE) for `μ_j` (when `p_j=1`).
- Train on a GPU or cloud instance, optimizing for MacBook deployment.

### Real-Time Inference
- Process the last `M` trades to predict `p_j` and `μ_j` for active arbs.
- Adjust predictions using mempool data (e.g., override if a trade is pending).
- Set `ΔX` to outbid the largest predicted competitor trade (e.g., `ΔX = max(μ_j) + ε`).

## Non-Functional Requirements
- **Performance:** Complete inference within seconds for real-time use.
- **Scalability:** Support up to 100 active arbs.
- **Adaptability:** Retrain weekly with new data.
- **Efficiency:** Optimize for the MacBook M3 Max (48GB VRAM), using techniques like float16 precision.

## Success Criteria
- Model predicts arb trades with ≥80% precision.
- Arbitrage strategy yields positive profit in simulations.
- System runs within hardware memory and performance constraints.