import pandas as pd
import os
import time
from api_connections import fetch_recent_maya_actions # Removed get_maya_last_aggregated_block_height as it's not directly used for 24hr fetch
from common_utils import parse_action, DF_COLUMNS # Import from common_utils

OUTPUT_DIR = "data"
# Update output file name
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "historical_24hr_maya_transactions.csv")
FETCH_ACTION_BATCH_SIZE = 50
# MAX_PAGINATION_PAGES_TO_SCAN = 5 # Not strictly needed for 24hr scan, we go by time
# MAX_ACTIONS_TO_SAVE = 100    # Not needed, save all from 24hr window
MAX_PAGES_FOR_24_HOUR_SCAN = 200 # Safety break for pagination, 200 pages * 50 actions/page = 10,000 actions

# def parse_action(action): # Removed: Will use from common_utils
#     """Parses a single action from Midgard API into a flat dictionary."""
#     parsed = {
#         "date": action.get("date"), # This is a nanosecond timestamp string
#         "height": action.get("height"),
#         "status": action.get("status"),
#         "type": action.get("type"),
#         "in_tx_count": len(action.get("in", [])),
#         "out_tx_count": len(action.get("out", [])),
#         "pools": ",".join(action.get("pools", [])),
#         "first_in_tx_id": action.get("in", [{}])[0].get("txID", "") if action.get("in") else "",
#         "first_out_tx_id": action.get("out", [{}])[0].get("txID", "") if action.get("out") else "",
#     }
#
#     # Extract coins from the first IN transaction
#     if action.get("in"):
#         first_in_tx = action["in"][0]
#         if first_in_tx.get("coins"):
#             first_in_coin = first_in_tx["coins"][0]
#             parsed["in_asset"] = first_in_coin.get("asset")
#             parsed["in_amount"] = first_in_coin.get("amount")
#         parsed["in_address"] = first_in_tx.get("address")
#
#     # Extract coins from the first OUT transaction
#     if action.get("out"):
#         first_out_tx = action["out"][0]
#         if first_out_tx.get("coins"):
#             first_out_coin = first_out_tx["coins"][0]
#             parsed["out_asset"] = first_out_coin.get("asset")
#             parsed["out_amount"] = first_out_coin.get("amount")
#         parsed["out_address"] = first_out_tx.get("address")
#
#     # Extract metadata for swaps
#     if action.get("type") == "swap" and action.get("metadata", {}).get("swap"):
#         swap_meta = action["metadata"]["swap"]
#         parsed["swap_liquidity_fee"] = swap_meta.get("liquidityFee")
#         parsed["swap_slip_bps"] = swap_meta.get("swapSlip") 
#         parsed["swap_target_asset"] = swap_meta.get("swapTarget") 
#         network_fees = swap_meta.get("networkFees", [])
#         if network_fees:
#             parsed["swap_network_fee_asset"] = network_fees[0].get("asset")
#             parsed["swap_network_fee_amount"] = network_fees[0].get("amount")
#         
#         parsed["transaction_id"] = parsed["first_in_tx_id"]
#
#     return parsed

# Define DF_COLUMNS globally # Removed: Will use from common_utils
# DF_COLUMNS = [
#     "date", "height", "status", "type", "in_tx_count", "out_tx_count", "pools", 
#     "first_in_tx_id", "first_out_tx_id", "in_asset", "in_amount", "in_address",
#     "out_asset", "out_amount", "out_address", "swap_liquidity_fee", 
#     "swap_slip_bps", "swap_target_asset", "swap_network_fee_asset", 
#     "swap_network_fee_amount", "transaction_id"
# ]

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}", flush=True)
    elif os.path.exists(OUTPUT_FILE):
        try:
            os.remove(OUTPUT_FILE)
            print(f"Removed existing output file: {OUTPUT_FILE}", flush=True)
        except OSError as e:
            print(f"Error removing existing file {OUTPUT_FILE}: {e}.", flush=True)

    # Calculate timestamp for 24 hours ago (in nanoseconds)
    # Midgard 'date' is a Unix timestamp in nanoseconds as a string.
    now_ns = time.time_ns()
    twenty_four_hours_ago_ns = now_ns - (24 * 60 * 60 * 1_000_000_000)

    print(f"Fetching all actions from Maya Protocol from the last 24 hours (since timestamp: {twenty_four_hours_ago_ns}).", flush=True)
    
    all_actions_in_window = []
    current_offset = 0
    pages_fetched = 0
    stop_fetching = False

    while pages_fetched < MAX_PAGES_FOR_24_HOUR_SCAN and not stop_fetching:
        print(f"Fetching actions page {pages_fetched + 1} (offset: {current_offset}, limit: {FETCH_ACTION_BATCH_SIZE})...", flush=True)
        # We don't filter by height here, we filter by timestamp client-side
        actions_data = fetch_recent_maya_actions(limit=FETCH_ACTION_BATCH_SIZE, offset=current_offset)
        pages_fetched += 1

        if not actions_data or "actions" not in actions_data or not actions_data["actions"]:
            print("No more actions returned from API. Stopping scan.", flush=True)
            stop_fetching = True
            break
        
        current_batch_actions = actions_data["actions"]
        if not current_batch_actions: # Should be caught by the above, but defensive
            print("API returned an empty 'actions' list. Assuming end of feed.", flush=True)
            stop_fetching = True
            break
        
        print(f"Fetched {len(current_batch_actions)} actions on page {pages_fetched}.", flush=True)
        
        oldest_action_in_batch_date_ns = 0
        for action in current_batch_actions:
            action_date_str = action.get("date")
            if action_date_str:
                try:
                    action_date_ns = int(action_date_str)
                    if action_date_ns >= twenty_four_hours_ago_ns:
                        all_actions_in_window.append(action)
                    # Update the oldest timestamp seen in this batch to check for pagination cutoff
                    if oldest_action_in_batch_date_ns == 0 or action_date_ns < oldest_action_in_batch_date_ns:
                         oldest_action_in_batch_date_ns = action_date_ns
                except ValueError:
                    print(f"Warning: Could not parse date string '{action_date_str}' to int for action.", flush=True)
                    continue # Skip this action if date is unparseable
        
        if oldest_action_in_batch_date_ns == 0 and current_batch_actions: # All actions in batch had bad dates
             print("Warning: All actions in the current batch had unparseable dates. Cannot determine if we should continue. Stopping to be safe.", flush=True)
             stop_fetching = True # Safety break if we can't determine age
             break

        if oldest_action_in_batch_date_ns < twenty_four_hours_ago_ns:
            print(f"Oldest action in batch (date: {oldest_action_in_batch_date_ns}) is older than 24-hour window. Stopping pagination.", flush=True)
            stop_fetching = True
        
        if len(current_batch_actions) < FETCH_ACTION_BATCH_SIZE:
            print(f"Fetched fewer actions ({len(current_batch_actions)}) than limit ({FETCH_ACTION_BATCH_SIZE}). Assuming end of API data.", flush=True)
            stop_fetching = True # No more data
        
        if not stop_fetching:
            current_offset += FETCH_ACTION_BATCH_SIZE
            time.sleep(0.5) # Be respectful to the API

    print(f"Collected a total of {len(all_actions_in_window)} actions from the last 24 hours after scanning {pages_fetched} pages.", flush=True)
    
    if not all_actions_in_window:
        print(f"No actions found within the last 24 hours.", flush=True)
        df = pd.DataFrame(columns=DF_COLUMNS)
    else:
        print(f"Parsing {len(all_actions_in_window)} collected actions...", flush=True)
        parsed_actions = [parse_action(action) for action in all_actions_in_window]
        df = pd.DataFrame(parsed_actions, columns=DF_COLUMNS)
        # Convert date column from string nanoseconds to more readable format or keep as is for precision
        # For now, keeping as nanosecond string as per original parsing.
        # If conversion to datetime is needed later:
        # df['date'] = pd.to_datetime(df['date'].astype(float), unit='ns')

    print(f"DEBUG: DataFrame created with {len(df)} rows. About to save to CSV: {OUTPUT_FILE}", flush=True)
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"DEBUG: df.to_csv call completed for {OUTPUT_FILE}.", flush=True)
        if df.empty:
            print(f"Successfully saved an empty CSV (no relevant actions found in the last 24 hours) to {OUTPUT_FILE}", flush=True)
        else:
            # For nanosecond strings, min/max might not be directly comparable without conversion
            # For simplicity, we'll just report count.
            print(f"Successfully saved {len(df)} actions from the last 24 hours to {OUTPUT_FILE}", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to save DataFrame to CSV {OUTPUT_FILE}. Error: {e}", flush=True)
    
    time.sleep(1) # Small delay before script exits

if __name__ == "__main__":
    main() 