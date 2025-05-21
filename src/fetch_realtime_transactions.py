import os
import time
import json
import argparse
from api_connections import fetch_recent_maya_actions

OUTPUT_DIR = "data"
ACTIONS_PER_PAGE = 50  # Midgard API limit per call for /actions
API_DELAY_SECONDS = 0.2 # Small delay between API calls

def fetch_actions_in_batches(total_actions_to_fetch, start_offset):
    """
    Fetches a specified total number of actions from the Midgard API,
    starting from a given offset, in batches of ACTIONS_PER_PAGE.

    Args:
        total_actions_to_fetch (int): The total number of actions to retrieve.
        start_offset (int): The starting offset for fetching actions.

    Returns:
        list: A list of action dictionaries, or None if an error occurs.
    """
    all_fetched_actions = []
    current_offset = start_offset
    pages_to_fetch = (total_actions_to_fetch + ACTIONS_PER_PAGE - 1) // ACTIONS_PER_PAGE # Ceiling division

    print(f"Starting fetch: {total_actions_to_fetch} actions, starting offset {start_offset}, in {pages_to_fetch} pages.")

    for page_num in range(pages_to_fetch):
        print(f"  Fetching page {page_num + 1}/{pages_to_fetch} (offset: {current_offset}, limit: {ACTIONS_PER_PAGE})...", flush=True)
        try:
            actions_data = fetch_recent_maya_actions(limit=ACTIONS_PER_PAGE, offset=current_offset)
            
            if not actions_data or "actions" not in actions_data:
                print(f"Warning: No 'actions' key in response or empty response on page {page_num + 1}. Offset: {current_offset}", flush=True)
                # Decide if we should stop or continue trying next pages
                if page_num < pages_to_fetch -1 : # If not the last expected page
                    print("  Attempting to continue to next page.", flush=True)
                else: # If this was the last expected page, likely just end of data
                     print("  Reached end of expected pages.", flush=True)
                current_batch_actions = [] # Ensure it's an empty list
            else:
                current_batch_actions = actions_data["actions"]

            if not current_batch_actions and page_num < pages_to_fetch -1 :
                print(f"Warning: API returned an empty 'actions' list on page {page_num + 1} (Offset: {current_offset}) before all expected actions were fetched. Continuing...", flush=True)
                # This might happen if we request an offset beyond available data.
            
            all_fetched_actions.extend(current_batch_actions)
            print(f"  Fetched {len(current_batch_actions)} actions. Total collected so far: {len(all_fetched_actions)} actions.", flush=True)

            if len(current_batch_actions) < ACTIONS_PER_PAGE and page_num < pages_to_fetch - 1:
                print(f"  Warning: Fetched fewer actions ({len(current_batch_actions)}) than limit ({ACTIONS_PER_PAGE}) on page {page_num + 1}. Reached end of available data early.", flush=True)
                # break # Stop if we get less than a full page, means no more data past this.

            current_offset += ACTIONS_PER_PAGE
            time.sleep(API_DELAY_SECONDS)

        except Exception as e:
            print(f"ERROR: An exception occurred during API call for offset {current_offset}: {e}", flush=True)
            # Optionally, decide if we should retry or just return what we have
            return None # Indicate failure

    print(f"Finished fetching. Total actions collected: {len(all_fetched_actions)} (requested: {total_actions_to_fetch}).")
    return all_fetched_actions

def save_actions_to_json(actions_list, filename):
    """
    Saves a list of action dictionaries to a JSON file in the OUTPUT_DIR.
    The JSON structure will be {"actions": [...]}.
    Replaces the file if it exists.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}", flush=True)
    
    output_file_path = os.path.join(OUTPUT_DIR, filename)

    print(f"Saving {len(actions_list)} actions to {output_file_path}...", flush=True)
    try:
        with open(output_file_path, 'w') as f:
            json.dump({"actions": actions_list}, f, indent=4) # Add indent for readability
        print(f"Successfully saved actions to {output_file_path}", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to save actions to {output_file_path}. Error: {e}", flush=True)

def main(args):
    # --- 1. Fetch and Save Test Data ---
    print("\n--- Fetching Test Data (data/test_data.json) ---")
    test_actions_to_fetch = 2500
    test_start_offset = 0
    test_actions = fetch_actions_in_batches(test_actions_to_fetch, test_start_offset)
    
    if test_actions is not None:
        save_actions_to_json(test_actions, "test_data.json")
    else:
        print("Failed to fetch test data. Skipping save.")

    # --- 2. Fetch and Save Training Data ---
    print("\n--- Fetching Training Data (data/transactions_data.json) ---")
    train_actions_to_fetch = 25000
    # Training data starts after test data, so offset is the number of test actions fetched.
    # If test_actions was None or incomplete, this might need adjustment or a fixed offset.
    # For simplicity, assuming test_actions_to_fetch is the correct offset.
    train_start_offset = test_actions_to_fetch 
    train_actions = fetch_actions_in_batches(train_actions_to_fetch, train_start_offset)
    
    if train_actions is not None:
        save_actions_to_json(train_actions, "transactions_data.json")
    else:
        print("Failed to fetch training data. Skipping save.")
        
    print("\n--- Data Fetching Process Complete ---")

if __name__ == "__main__":
    # No arguments needed for this specific script anymore,
    # as the behavior is fixed (fetch for test_data.json and transactions_data.json)
    # However, keeping argparse structure in case we want to add options later.
    parser = argparse.ArgumentParser(description="Fetch Maya Protocol actions and save as raw JSON for training and test datasets.")
    # Example: parser.add_argument("--api-key", type=str, help="Optional API key if Midgard requires it in the future.")
    cli_args = parser.parse_args()
    main(cli_args) 