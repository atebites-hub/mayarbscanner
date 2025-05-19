import pandas as pd
import os
import time
import argparse # Added
from api_connections import fetch_recent_maya_actions # Removed get_maya_last_aggregated_block_height as it's not directly used for 24hr fetch
from common_utils import parse_action, DF_COLUMNS # Import from common_utils

OUTPUT_DIR = "data"
# OUTPUT_FILE = os.path.join(OUTPUT_DIR, "historical_24hr_maya_transactions.csv") # Will be set by args
FETCH_ACTION_BATCH_SIZE = 50
# MAX_PAGINATION_PAGES_TO_SCAN = 5 # Not strictly needed for 24hr scan, we go by time
# MAX_ACTIONS_TO_SAVE = 100    # Not needed, save all from 24hr window
MAX_PAGES_FOR_TIME_WINDOW_SCAN = 400 # Safety break: 400 pages * 50 = 20,000 actions

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

def main(args): # Added args parameter
    output_file = args.output_file # Use arg
    hours_ago_start = args.hours_ago_start # Use arg
    duration_hours = args.duration_hours # Use arg

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}", flush=True)
    
    output_file_path = os.path.join(OUTPUT_DIR, output_file) # Construct full path

    if os.path.exists(output_file_path):
        try:
            os.remove(output_file_path)
            print(f"Removed existing output file: {output_file_path}", flush=True)
        except OSError as e:
            print(f"Error removing existing file {output_file_path}: {e}.", flush=True)

    now_ns = time.time_ns()
    # Calculate the end of the window (e.g., 24 hours ago from now)
    window_end_ns = now_ns - (hours_ago_start * 60 * 60 * 1_000_000_000)
    # Calculate the start of the window (e.g., 24 hours prior to window_end_ns)
    window_start_ns = window_end_ns - (duration_hours * 60 * 60 * 1_000_000_000)

    print(f"Fetching actions from Maya Protocol for a {duration_hours}-hour window starting {hours_ago_start + duration_hours} hours ago and ending {hours_ago_start} hours ago.", flush=True)
    print(f"Time window: {window_start_ns} ns to {window_end_ns} ns.", flush=True)
    
    all_actions_in_window = []
    current_offset = 0
    pages_fetched = 0
    stop_fetching = False
    # Heuristic: Start scanning from a point slightly before our window_end_ns might suggest.
    # Midgard API gives latest first. We need to go back in time.
    # The `offset` parameter is key. We keep fetching pages until the oldest action in a batch
    # is older than our `window_start_ns`.

    # The logic needs to ensure we capture actions *within* the [window_start_ns, window_end_ns] range.
    # We paginate backwards in time.
    # We stop fetching a page if ALL transactions in that page are older than window_start_ns.
    # We also stop if the NEWEST transaction in a page is older than window_start_ns (implies we passed the window).
    # More robust: keep fetching as long as some part of the batch could be in the window.
    # Stop when oldest_action_in_batch_date_ns < window_start_ns.

    while pages_fetched < MAX_PAGES_FOR_TIME_WINDOW_SCAN and not stop_fetching:
        print(f"Fetching actions page {pages_fetched + 1} (offset: {current_offset}, limit: {FETCH_ACTION_BATCH_SIZE})...", flush=True)
        actions_data = fetch_recent_maya_actions(limit=FETCH_ACTION_BATCH_SIZE, offset=current_offset)
        pages_fetched += 1

        if not actions_data or "actions" not in actions_data or not actions_data["actions"]:
            print("No more actions returned from API. Stopping scan.", flush=True)
            stop_fetching = True
            break
        
        current_batch_actions = actions_data["actions"]
        if not current_batch_actions:
            print("API returned an empty 'actions' list. Assuming end of feed.", flush=True)
            stop_fetching = True
            break
        
        print(f"Fetched {len(current_batch_actions)} actions on page {pages_fetched}.", flush=True)
        
        # Track the timestamps of actions in the current batch
        newest_action_in_batch_date_ns = 0
        oldest_action_in_batch_date_ns = float('inf') # Initialize with infinity
        actions_to_add_from_this_batch = []

        for action in current_batch_actions:
            action_date_str = action.get("date")
            if action_date_str:
                try:
                    action_date_ns = int(action_date_str)
                    
                    # Update batch boundary timestamps
                    if action_date_ns > newest_action_in_batch_date_ns:
                        newest_action_in_batch_date_ns = action_date_ns
                    if action_date_ns < oldest_action_in_batch_date_ns:
                         oldest_action_in_batch_date_ns = action_date_ns

                    # Check if the action falls within our desired time window
                    if window_start_ns <= action_date_ns < window_end_ns: # Use < for window_end_ns to make it exclusive end
                        actions_to_add_from_this_batch.append(action)
                except ValueError:
                    print(f"Warning: Could not parse date string '{action_date_str}' to int for action.", flush=True)
                    continue
        
        all_actions_in_window.extend(actions_to_add_from_this_batch)

        if oldest_action_in_batch_date_ns == float('inf') and current_batch_actions: 
             print("Warning: All actions in the current batch had unparseable dates. Cannot determine if we should continue. Stopping to be safe.", flush=True)
             stop_fetching = True
             break
        
        # Stop condition: if the oldest action in the current batch is already older than our window's start time,
        # then subsequent pages (even older actions) will definitely be outside our window.
        if oldest_action_in_batch_date_ns < window_start_ns:
            print(f"Oldest action in current batch (timestamp: {oldest_action_in_batch_date_ns}) is older than the target window start ({window_start_ns}). Stopping pagination.", flush=True)
            stop_fetching = True
        
        if len(current_batch_actions) < FETCH_ACTION_BATCH_SIZE:
            print(f"Fetched fewer actions ({len(current_batch_actions)}) than limit ({FETCH_ACTION_BATCH_SIZE}). Assuming end of API data for this period.", flush=True)
            stop_fetching = True
        
        if not stop_fetching:
            current_offset += FETCH_ACTION_BATCH_SIZE
            time.sleep(0.5)

    print(f"Collected a total of {len(all_actions_in_window)} actions within the specified time window after scanning {pages_fetched} pages.", flush=True)
    
    if not all_actions_in_window:
        print(f"No actions found within the specified time window.", flush=True)
        df = pd.DataFrame(columns=DF_COLUMNS)
    else:
        print(f"Parsing {len(all_actions_in_window)} collected actions...", flush=True)
        # Sort actions by date (ascending) before parsing, good practice though parse_action is per action
        all_actions_in_window.sort(key=lambda x: int(x.get("date", "0")))
        parsed_actions = [parse_action(action) for action in all_actions_in_window]
        df = pd.DataFrame(parsed_actions, columns=DF_COLUMNS)

    print(f"DEBUG: DataFrame created with {len(df)} rows. About to save to CSV: {output_file_path}", flush=True)
    try:
        df.to_csv(output_file_path, index=False)
        print(f"DEBUG: df.to_csv call completed for {output_file_path}.", flush=True)
        if df.empty:
            print(f"Successfully saved an empty CSV (no relevant actions found in the window) to {output_file_path}", flush=True)
        else:
            print(f"Successfully saved {len(df)} actions from the specified time window to {output_file_path}", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to save DataFrame to CSV {output_file_path}. Error: {e}", flush=True)
    
    time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Maya Protocol actions for a specified time window.")
    parser.add_argument("--output-file", type=str, default="historical_transactions.csv", 
                        help="Name of the output CSV file to be saved in the 'data/' directory.")
    parser.add_argument("--hours-ago-start", type=int, default=24, 
                        help="Defines the end of the fetching window, in hours ago from now. E.g., 24 means the window ends 24 hours ago.")
    parser.add_argument("--duration-hours", type=int, default=24, 
                        help="Duration of the fetching window in hours, extending backwards from 'hours-ago-start'. E.g., if hours-ago-start is 24 and duration is 24, it fetches data from 48 hours ago to 24 hours ago.")
    
    cli_args = parser.parse_args()
    main(cli_args) 