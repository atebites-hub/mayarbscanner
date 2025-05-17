import pandas as pd
import os
import time
from api_connections import fetch_recent_maya_actions

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "realtime_maya_transactions.csv")
TRANSACTION_LIMIT = 50 # Fetch 50 most recent actions (within API limits)

def parse_action(action):
    """Parses a single action from Midgard API into a flat dictionary."""
    parsed = {
        "date": action.get("date"),
        "height": action.get("height"),
        "status": action.get("status"),
        "type": action.get("type"),
        "in_tx_count": len(action.get("in", [])),
        "out_tx_count": len(action.get("out", [])),
        "pools": ",".join(action.get("pools", [])),
        "first_in_tx_id": action.get("in", [{}])[0].get("txID", "") if action.get("in") else "",
        "first_out_tx_id": action.get("out", [{}])[0].get("txID", "") if action.get("out") else "",
    }

    # Extract coins from the first IN transaction
    if action.get("in"):
        first_in_tx = action["in"][0]
        if first_in_tx.get("coins"):
            first_in_coin = first_in_tx["coins"][0]
            parsed["in_asset"] = first_in_coin.get("asset")
            parsed["in_amount"] = first_in_coin.get("amount")
        parsed["in_address"] = first_in_tx.get("address")

    # Extract coins from the first OUT transaction
    if action.get("out"):
        first_out_tx = action["out"][0]
        if first_out_tx.get("coins"):
            first_out_coin = first_out_tx["coins"][0]
            parsed["out_asset"] = first_out_coin.get("asset")
            parsed["out_amount"] = first_out_coin.get("amount")
        parsed["out_address"] = first_out_tx.get("address")

    # Extract metadata for swaps
    if action.get("type") == "swap" and action.get("metadata", {}).get("swap"):
        swap_meta = action["metadata"]["swap"]
        parsed["swap_liquidity_fee"] = swap_meta.get("liquidityFee")
        parsed["swap_slip_bps"] = swap_meta.get("swapSlip") # This is swapSlip in bps (basis points)
        parsed["swap_target_asset"] = swap_meta.get("swapTarget") # This might be useful (target asset for the swap)
        network_fees = swap_meta.get("networkFees", [])
        if network_fees:
            parsed["swap_network_fee_asset"] = network_fees[0].get("asset")
            parsed["swap_network_fee_amount"] = network_fees[0].get("amount")
        
        # arb_ID and ΔX are specific to arbitrage identification.
        # For now, we don't have direct arb_ID. ΔX would be calculated from in_amount vs out_amount considering prices.
        # We can use in_tx_id or a combination of in/out tx_ids as a proxy for a unique transaction identifier for now.
        # Actual ΔX calculation would require price data at the time of the swap.
        # We will use the first_in_tx_id as a general transaction_id for now.
        parsed["transaction_id"] = parsed["first_in_tx_id"]


    # Placeholder for arb_ID and ΔX (which needs more complex calculation)
    # For ΔX (change in X), if it's a single asset swap, this would be related to in_amount vs out_amount
    # For arb_ID, this would be a unique identifier for an arbitrage event, which we are not yet identifying.
    # We can use the transaction ID as a general identifier.

    return parsed

def main():
    print(f"Fetching the latest {TRANSACTION_LIMIT} Maya Protocol actions...")
    
    actions_data = fetch_recent_maya_actions(limit=TRANSACTION_LIMIT)

    if not actions_data or "actions" not in actions_data:
        print("Failed to fetch actions or actions data is not in the expected format.")
        return

    actions_list = actions_data["actions"]
    print(f"Successfully fetched {len(actions_list)} actions.")

    if not actions_list:
        print("No actions found.")
        return

    parsed_actions = [parse_action(action) for action in actions_list]
    
    df = pd.DataFrame(parsed_actions)

    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved {len(df)} realtime transactions to {OUTPUT_FILE}")
    
    # Add a small delay to allow file system operations to complete if needed by subsequent steps
    time.sleep(1)


if __name__ == "__main__":
    main() 