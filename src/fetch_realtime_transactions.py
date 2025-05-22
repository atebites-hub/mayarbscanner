#!/usr/bin/env python3

import argparse
import json
import os
import time
from datetime import datetime
import sys
import sqlite3

# Project-specific imports
from src.api_connections import (
    get_mayanode_latest_block_height, 
    fetch_mayanode_block,
    construct_next_block_template # Added for initial mempool view
)
from src.common_utils import parse_confirmed_block
from src.database_utils import (
    get_db_connection,
    create_tables,
    get_latest_block_height_from_db,
    get_all_block_heights_from_db, # Added for gap detection
    insert_block,
    check_if_block_exists
)

DEFAULT_POLL_INTERVAL_SECONDS = 10
API_REQUEST_DELAY_SECONDS = 0.5 # Delay between individual block fetches

def fetch_and_store_block(conn, height_to_fetch):
    """Fetches, parses, and stores a single block. Returns True on success, False on failure."""
    print(f"Processing block {height_to_fetch}...")
    
    # Optional: Double check if block exists before fetch+parse+insert attempt.
    if check_if_block_exists(conn, height_to_fetch):
        print(f"Block {height_to_fetch} already exists in DB (checked). Skipping redundant processing.")
        return True # Count as success for moving to next block in a sequence

    raw_block_data = fetch_mayanode_block(height=height_to_fetch)
    if raw_block_data:
        parsed_data = parse_confirmed_block(raw_block_data)
        if parsed_data:
            if insert_block(conn, parsed_data):
                print(f"Successfully fetched, parsed, and inserted block {height_to_fetch}.")
                return True
            else:
                print(f"Failed to insert block {height_to_fetch} into DB.")
                return False
        else:
            print(f"Failed to parse block {height_to_fetch}. Skipping.")
            return False # Indicate failure to process this block, allows retry or specific handling
    else:
        print(f"Failed to fetch block {height_to_fetch} from API.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Continuously fetch new Mayanode blocks, filling gaps, and store them in a database.")
    parser.add_argument(
        "--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS,
        help=f"Polling interval in seconds to check for new blocks (default: {DEFAULT_POLL_INTERVAL_SECONDS})."
    )
    parser.add_argument(
        "--start-at-latest", action='store_true',
        help="If set, on first run, will only fetch from the current latest API block onwards, skipping historical backfill."
    )
    parser.add_argument(
        "--max-blocks-per-cycle", type=int, default=100,
        help="Maximum number of blocks to attempt to fetch in a single (gap-filling or polling) cycle (default: 100)."
    )
    parser.add_argument(
        "--disable-initial-mempool", action='store_true',
        help="If set, disables fetching and printing the initial mempool-based next block template."
    )

    args = parser.parse_args()

    print(f"--- Mayanode Continuous Block Ingestion Service --- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Polling interval: {args.poll_interval} seconds")
    print(f"Max blocks per cycle: {args.max_blocks_per_cycle}")
    if args.start_at_latest: print("Mode: Start at latest API block, skipping historical backfill on first run.")
    if args.disable_initial_mempool: print("Initial mempool block construction: DISABLED")

    conn = None
    try:
        conn = get_db_connection()
        create_tables(conn)
        print("Database connection established and tables ensured.")

        # 1. Initial Mempool Block (if not disabled)
        if not args.disable_initial_mempool:
            print("\n--- Constructing Initial Next Block Template (from Mempool) ---")
            next_block_template = construct_next_block_template()
            if next_block_template:
                print("Successfully constructed next block template.")
                # Optionally print parts of it or the whole thing
                print(json.dumps(next_block_template, indent=2))
                print("--- End of Initial Next Block Template ---")
            else:
                print("Could not construct initial next block template.")

        # 2. Startup Synchronization: Fill Gaps
        print("\n--- Startup Synchronization: Checking for missing historical blocks ---")
        latest_api_height = get_mayanode_latest_block_height()
        if latest_api_height is None:
            print("Critical: Could not determine latest API block height. Cannot perform sync. Exiting.")
            return

        print(f"Current latest API block height: {latest_api_height}")
        
        stored_heights = get_all_block_heights_from_db(conn)
        print(f"Found {len(stored_heights)} blocks in the local database.")

        first_block_to_consider_for_gaps = 1
        if args.start_at_latest and not stored_heights: # Only applies if DB is empty
            print(f"--start-at-latest is set and DB is empty. Sync will begin from {latest_api_height}.")
            first_block_to_consider_for_gaps = latest_api_height
        
        missing_heights = []
        for h in range(first_block_to_consider_for_gaps, latest_api_height + 1):
            if h not in stored_heights:
                missing_heights.append(h)
        
        if missing_heights:
            print(f"Found {len(missing_heights)} missing historical blocks (from {missing_heights[0]} to {missing_heights[-1]} in segments). Attempting to fetch...")
            blocks_synced_total = 0
            for i in range(0, len(missing_heights), args.max_blocks_per_cycle):
                batch_to_fetch = missing_heights[i:i + args.max_blocks_per_cycle]
                print(f"Syncing batch of {len(batch_to_fetch)} missing blocks starting from {batch_to_fetch[0]}...")
                batch_success = True
                for height in batch_to_fetch:
                    if not fetch_and_store_block(conn, height):
                        print(f"Failed to sync block {height}. Stopping current batch sync.")
                        batch_success = False
                        break 
                    blocks_synced_total +=1
                    time.sleep(API_REQUEST_DELAY_SECONDS)
                if not batch_success:
                    print("Error during batch sync. Will retry on next full polling cycle if issues persist.")
                    break # Stop further batch processing in this startup sync
            print(f"Startup synchronization completed. Synced {blocks_synced_total} missing blocks.")
        else:
            print("No missing historical blocks found up to current API height. Database is synchronized.")

        # 3. Continuous Polling Loop
        print("\n--- Starting Continuous Polling for New Blocks ---")
        while True:
            current_db_latest = get_latest_block_height_from_db(conn)
            next_block_to_fetch = (current_db_latest + 1) if current_db_latest else 1
            # If --start-at-latest was used and DB was empty, ensure we don't re-fetch by starting from the known latest
            if args.start_at_latest and not current_db_latest : # and implies DB was empty at start
                 api_latest_for_poll_start = get_mayanode_latest_block_height()
                 if api_latest_for_poll_start:
                    next_block_to_fetch = api_latest_for_poll_start +1 
            
            print(f"\n--- Polling Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} --- Next block to fetch: {next_block_to_fetch}")
            latest_api_height_poll = get_mayanode_latest_block_height()

            if latest_api_height_poll is None:
                print("Could not fetch latest block height from API during poll. Retrying after interval.")
                time.sleep(args.poll_interval)
                continue
            
            print(f"Latest API height: {latest_api_height_poll}. DB latest: {current_db_latest if current_db_latest else 'N/A'}")

            blocks_fetched_this_cycle = 0
            if latest_api_height_poll >= next_block_to_fetch:
                print(f"New blocks potentially available. Checking from {next_block_to_fetch} up to {latest_api_height_poll}.")
                
                target_end_height_for_cycle = min(latest_api_height_poll, next_block_to_fetch + args.max_blocks_per_cycle - 1)
                
                for height in range(next_block_to_fetch, target_end_height_for_cycle + 1):
                    if not fetch_and_store_block(conn, height):
                        print(f"Failed to process block {height} during polling. Will retry in next cycle.")
                        break # Stop this cycle's fetching on failure
                    blocks_fetched_this_cycle += 1
                    time.sleep(API_REQUEST_DELAY_SECONDS)
                
                if blocks_fetched_this_cycle > 0:
                    print(f"Fetched {blocks_fetched_this_cycle} block(s) in this polling cycle.")
                elif latest_api_height_poll >= next_block_to_fetch:
                    print("No blocks fetched this cycle, but still catching up or API hasn't advanced.")
                else:
                    print("Database appears up to date with API latest.")
            else:
                print("Database is up to date with the latest API height.")

            print(f"Waiting for {args.poll_interval} seconds before next poll...")
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nShutdown signal received. Exiting gracefully...")
    except sqlite3.Error as db_err:
        print(f"Database error occurred: {db_err}. Shutting down.")    
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Shutting down.")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")
        print(f"--- Mayanode Continuous Block Ingestion Service SHUT DOWN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) 
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    main()
