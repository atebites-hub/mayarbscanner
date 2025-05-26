#!/usr/bin/env python3

import argparse
import json
import os
import time
from datetime import datetime, timedelta
import sys
import sqlite3
import asyncio
import aiohttp
from typing import Optional, Deque
from collections import deque
from tqdm import tqdm

# --- BEGIN relocated sys.path modification ---
# Ensure the project root is in sys.path for absolute imports from 'src'.
# This needs to be done before any 'from src...' imports.
# Using script-specific variable names to avoid potential global conflicts.
_FRT_SCRIPT_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRT_SCRIPT_PROJECT_ROOT = os.path.dirname(_FRT_SCRIPT_CURRENT_DIR)
if _FRT_SCRIPT_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _FRT_SCRIPT_PROJECT_ROOT)
# Clean up temporary variables from the global namespace after they've been used.
del _FRT_SCRIPT_CURRENT_DIR
del _FRT_SCRIPT_PROJECT_ROOT
# --- END relocated sys.path modification ---

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
    insert_block_and_components, # Key change: use this for comprehensive block insertion
    check_if_block_exists
)

DEFAULT_POLL_INTERVAL_SECONDS = 5.5
API_REQUEST_DELAY_SECONDS = 0.5 # Delay between individual block fetches in sequential modes
CONCURRENT_FETCH_REQUESTS = 10  # Number of concurrent requests for historical catch-up
FETCH_BATCH_SIZE = 50 # Number of blocks to fetch in one async batch before sequential processing

async def fetch_block_data_async(session: aiohttp.ClientSession, height: int) -> tuple[int, Optional[dict]]:
    """Asynchronously fetches raw block data for a given height from Mayanode API.
    Returns a tuple (height, data_dict_or_None).
    """
    base_url = "https://mayanode.mayachain.info/mayachain/block"
    params = {"height": height}
    try:
        async with session.get(base_url, params=params, timeout=10) as response:
            if response.status == 200:
                return height, await response.json()
            else:
                print(f"Error: Mayanode API async request for block {height} failed. Status: {response.status}, Reason: {response.reason}", flush=True)
                return height, None
    except asyncio.TimeoutError:
        print(f"Error: Mayanode API async request for block {height} timed out.", flush=True)
        return height, None
    except aiohttp.ClientError as e:
        print(f"Error: Mayanode API async request for block {height} client error: {e}", flush=True)
        return height, None
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred during async fetch for block {height}: {e}", flush=True)
        return height, None

def fetch_and_store_block(conn, height_to_fetch: int, raw_block_data_override: Optional[dict] = None, tqdm_bar: Optional[tqdm] = None) -> bool:
    """Fetches (or uses provided raw data), parses, and stores a single block.
    
    Args:
        conn (sqlite3.Connection): Active database connection.
        height_to_fetch (int): The block height to process.
        raw_block_data_override (Optional[dict]): Optional raw block data to use instead of fetching from API.
        tqdm_bar (Optional[tqdm]): If provided, status messages will be written via tqdm_bar.write() or set as postfix.

    Returns:
        bool: True on success or if block already exists, False on failure.
    """
    # Use tqdm_bar.write if available for detailed messages, or print for non-tqdm mode.
    # For tqdm mode, simple status updates can be done via postfix.
    write_func = tqdm_bar.write if tqdm_bar else lambda msg: print(msg, flush=True)
    # For simple status updates to tqdm postfix, we'll call tqdm_bar.set_postfix_str directly.

    # No initial "Processing block..." print here if using tqdm, as tqdm will handle the current item display.
    if not tqdm_bar:
        print(f"Processing block {height_to_fetch}...", flush=True)
    
    if check_if_block_exists(conn, height_to_fetch):
        if tqdm_bar:
            # Postfix update is handled by the calling loop (run_historical_catchup)
            pass # No specific message here if tqdm is managing it.
        else:
            write_func(f"Block {height_to_fetch} already exists in DB. Skipping redundant processing.")
        return True

    raw_block_data = raw_block_data_override
    if raw_block_data is None:
        # write_func(f"Fetching block {height_to_fetch} via synchronous call...") # Can be verbose
        raw_block_data = fetch_mayanode_block(height=height_to_fetch) # Synchronous call if no override
    
    if raw_block_data:
        parsed_data = parse_confirmed_block(raw_block_data, source_type="mayanode_api")
        
        if parsed_data:
            if insert_block_and_components(conn, parsed_data):
                # write_func(f"Successfully fetched, parsed, and inserted block {height_to_fetch}.") # Keep this for non-tqdm, tqdm will update postfix
                if not tqdm_bar: # Only print success if not using tqdm (tqdm shows status in postfix)
                    write_func(f"Successfully fetched, parsed, and inserted block {height_to_fetch}.")
                return True
            else:
                write_func("Failed to insert block {height_to_fetch} and its components into DB.")
                return False
        else:
            write_func("Failed to parse block {height_to_fetch} from API data. Skipping.")
            return False
    else:
        write_func("Failed to fetch block {height_to_fetch} from API.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Continuously fetch new Mayanode blocks, filling gaps, and store them in a database. Can also perform specific fetch tasks.")
    parser.add_argument(
        "--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS,
        help=f"Polling interval in seconds to check for new blocks (default: {DEFAULT_POLL_INTERVAL_SECONDS}). Used in continuous mode."
    )
    parser.add_argument(
        "--start-at-latest", action='store_true',
        help="If set, on first run in continuous mode, will only fetch from the current latest API block onwards, skipping historical backfill."
    )
    parser.add_argument(
        "--max-blocks-per-cycle", type=int, default=100,
        help="Maximum number of blocks to attempt to fetch in a single (gap-filling or polling) cycle (default: 100)."
    )
    parser.add_argument(
        "--disable-initial-mempool", action='store_true',
        help="If set, disables fetching and printing the initial mempool-based next block template."
    )
    parser.add_argument(
        "--historical-catchup", type=int, metavar="NUM_BLOCKS", default=None,
        help="Fetch the last NUM_BLOCKS from the current API height using concurrent requests. Script will exit after completion."
    )

    args = parser.parse_args()

    print(f"--- Mayanode Block Ingestion Service --- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---", flush=True)
    print(f"Polling interval: {args.poll_interval} seconds", flush=True)
    print(f"Max blocks per cycle: {args.max_blocks_per_cycle}", flush=True)
    if args.start_at_latest: print("Mode: Start at latest API block (continuous mode startup).", flush=True)
    if args.disable_initial_mempool: print("Initial mempool block construction: DISABLED", flush=True)
    if args.historical_catchup: print(f"Mode: Historical catch-up for last {args.historical_catchup} blocks.", flush=True)

    conn = None
    try:
        conn = get_db_connection()
        create_tables(conn)
        print("Database connection established and tables ensured.", flush=True)

        # --- New Historical Catch-up Mode ---
        if args.historical_catchup:
            num_blocks_to_catchup = args.historical_catchup
            if num_blocks_to_catchup <= 0:
                print("Error: --historical-catchup NUM_BLOCKS must be a positive integer.", flush=True)
                return

            print(f"\n--- Executing Historical Catch-up for last {num_blocks_to_catchup} blocks ---", flush=True)
            latest_api_height = get_mayanode_latest_block_height()
            if latest_api_height is None:
                print("Critical: Could not determine latest API block height. Cannot perform historical catch-up. Exiting.", flush=True)
                return
            
            start_height_catchup = max(1, latest_api_height - num_blocks_to_catchup + 1)
            end_height_catchup = latest_api_height
            total_blocks_to_process_in_catchup = end_height_catchup - start_height_catchup + 1

            print(f"Targeting blocks from {start_height_catchup} to {end_height_catchup} (inclusive) for historical catch-up: {total_blocks_to_process_in_catchup} blocks total.", flush=True)
            
            overall_progress_bar = tqdm(total=total_blocks_to_process_in_catchup, unit="block", desc="Historical Catch-up", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]', dynamic_ncols=True, leave=False)
            start_time_catchup = time.time()
            blocks_fetched_historical = 0
            blocks_failed_historical = 0

            async def run_historical_catchup():
                nonlocal blocks_fetched_historical, blocks_failed_historical, overall_progress_bar # Allow modification of outer scope vars
                
                processed_count_in_catchup = 0

                timeout = aiohttp.ClientTimeout(total=60) # 60 seconds total timeout for all ops in a session request
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    all_heights_to_process = list(range(start_height_catchup, end_height_catchup + 1))
                    
                    # Removed the tqdm wrapper from the batch loop, will update the single overall_progress_bar
                    for i in range(0, len(all_heights_to_process), FETCH_BATCH_SIZE):
                        batch_heights = all_heights_to_process[i:i + FETCH_BATCH_SIZE]
                        if not batch_heights: continue

                        # Screen clearing and status update can be done here before each batch
                        # For simplicity now, tqdm handles its own line. More complex UI would clear and redraw.
                        # print(f"Fetching batch of {len(batch_heights)} blocks: {batch_heights[0]} to {batch_heights[-1]}...")
                        
                        tasks = []
                        # heights_for_this_batch_tasks = [] # Keep track of heights actually added to tasks
                        for height in batch_heights:
                            if not check_if_block_exists(conn, height): # Pre-check to avoid unnecessary async calls
                                tasks.append(fetch_block_data_async(session, height))
                                # heights_for_this_batch_tasks.append(height) # Not strictly needed if results_with_heights gives height
                            else:
                                # Block already exists, update progress bar and set postfix
                                overall_progress_bar.set_postfix_str(f"Exists: H{height}", refresh=False) # No need to refresh too often for this
                                overall_progress_bar.update(1)
                                # No need to increment processed_count_in_catchup here or blocks_fetched/failed

                        if not tasks:
                            # All blocks in this batch might already exist.
                            continue
                        
                        results_with_heights = await asyncio.gather(*tasks)
                        
                        # print(f"Processing {len(results_with_heights)} fetched results for batch...")
                        for fetched_height, block_data_or_none in results_with_heights:
                            processed_count_in_catchup += 1
                            current_block_display = f"H{fetched_height}" # Shortened for postfix
                            overall_progress_bar.set_description_str(f"Historical Catch-up") # Reset description before postfix
                            if block_data_or_none is None:
                                blocks_failed_historical += 1
                                overall_progress_bar.set_postfix_str(f"{current_block_display} FAIL(fetch)", refresh=True)
                            else:
                                # Pass the overall_progress_bar to fetch_and_store_block for postfix updates
                                # fetch_and_store_block will use tqdm_bar.write for its own errors if any, or be silent on success
                                if fetch_and_store_block(conn, fetched_height, raw_block_data_override=block_data_or_none, tqdm_bar=overall_progress_bar):
                                    blocks_fetched_historical += 1
                                    overall_progress_bar.set_postfix_str(f"{current_block_display} OK", refresh=True)
                                else:
                                    blocks_failed_historical += 1
                                    overall_progress_bar.set_postfix_str(f"{current_block_display} FAIL(store)", refresh=True)
                            overall_progress_bar.update(1)
                            
                            # ETA calculation - Moved inside the loop to update more frequently
                            elapsed_time = time.time() - start_time_catchup
                            if processed_count_in_catchup > 0 and overall_progress_bar.total is not None and overall_progress_bar.total > 0:
                                # Calculate rate based on blocks actually processed (fetched + stored/failed)
                                # not just loop iterations that might include skips.
                                current_rate = processed_count_in_catchup / elapsed_time if elapsed_time > 0 else 0
                                remaining_blocks = overall_progress_bar.total - overall_progress_bar.n
                                if current_rate > 0 and remaining_blocks > 0:
                                    eta_seconds = remaining_blocks / current_rate
                                    eta_formatted = str(timedelta(seconds=int(eta_seconds)))
                                    # Update description with ETA
                                    overall_progress_bar.set_description_str(f"Historical Catch-up (ETA: {eta_formatted})") 
                                else:
                                    overall_progress_bar.set_description_str(f"Historical Catch-up") # Default if ETA can't be calc

            asyncio.run(run_historical_catchup())
            
            overall_progress_bar.close() # Close the progress bar once catch-up is fully done.
            elapsed_total_catchup = time.time() - start_time_catchup
            print(f"\n--- Historical Catch-up Task Completed in {elapsed_total_catchup:.2f} seconds ---", flush=True)
            print(f"Successfully fetched/stored during catch-up: {blocks_fetched_historical} blocks.", flush=True)
            print(f"Failed to fetch/store during catch-up: {blocks_failed_historical} blocks.", flush=True)
            print("\n--- Transitioning to Live Block Monitoring ---", flush=True) # Added message

        # --- Default Continuous Operation (Gap Filling & Polling) ---
        # This part executes if no specific mode like --historical-catchup (or the removed ones) was specified.
        # Or, if --historical-catchup was specified, it would have returned already.
        # Thus, if we reach here, it means no exit condition was met (like --historical-catchup completion or errors in other modes).

        # We need to ensure that if --historical-catchup was *not* specified, the script proceeds to continuous mode.
        # If args.historical_catchup was specified and completed, the script will have already exited.
        # So, if we are here, it means no exit condition was met (like --historical-catchup completion or errors in other modes).

        # Initial Mempool Block (if not disabled)
        if not args.disable_initial_mempool:
            print("\n--- Constructing Initial Next Block Template (from Mempool) ---", flush=True)
            next_block_template = construct_next_block_template()
            if next_block_template:
                print("Successfully constructed next block template.", flush=True)
                # Optionally print parts of it or the whole thing
                # print(json.dumps(next_block_template, indent=2)) # Can be very verbose
                print("--- End of Initial Next Block Template ---", flush=True)
            else:
                print("Could not construct initial next block template.", flush=True)

        # 2. Startup Synchronization: Fill Gaps
        print("\n--- Startup Synchronization: Checking for missing historical blocks ---", flush=True)
        latest_api_height_sync = get_mayanode_latest_block_height()
        if latest_api_height_sync is None:
            print("Critical: Could not determine latest API block height. Cannot perform sync. Exiting.", flush=True)
            return

        print(f"Current latest API block height: {latest_api_height_sync}", flush=True)
        
        stored_heights = get_all_block_heights_from_db(conn)
        print(f"Found {len(stored_heights)} blocks in the local database.", flush=True)

        first_block_to_consider_for_gaps_sync = 1
        # If DB is empty and start_at_latest is true, begin sync from current API latest.
        if args.start_at_latest and not stored_heights: 
            print(f"--start-at-latest is set and DB is empty. Sync will begin from {latest_api_height_sync}.", flush=True)
            first_block_to_consider_for_gaps_sync = latest_api_height_sync 
        
        missing_heights_sync = []
        # Determine the upper bound for gap checking: either latest_api_height_sync or current DB max if higher (unlikely for gaps)
        upper_bound_for_gaps = latest_api_height_sync
        if stored_heights:
            upper_bound_for_gaps = max(latest_api_height_sync, max(stored_heights) if stored_heights else 0)

        for h_sync in range(first_block_to_consider_for_gaps_sync, upper_bound_for_gaps + 1):
            if h_sync not in stored_heights:
                missing_heights_sync.append(h_sync)
        
        if missing_heights_sync:
            print(f"Found {len(missing_heights_sync)} missing historical block segments. Attempting to fetch...", flush=True)
            blocks_synced_total_mode = 0
            for i_sync in range(0, len(missing_heights_sync), args.max_blocks_per_cycle):
                batch_to_fetch_sync = missing_heights_sync[i_sync:i_sync + args.max_blocks_per_cycle]
                print(f"Syncing batch of {len(batch_to_fetch_sync)} missing blocks starting from {batch_to_fetch_sync[0]}...", flush=True)
                batch_success_sync = True
                for height_sync_item in batch_to_fetch_sync:
                    # Ensure we don't try to fetch beyond the current API height in this gap fill
                    if height_sync_item > get_mayanode_latest_block_height(): # Re-check API height
                        print(f"Block {height_sync_item} is beyond current API latest. Stopping batch for now.", flush=True)
                        batch_success_sync = False
                        break
                    # In sync mode, we don't pass tqdm_bar, so it uses regular print
                    if not fetch_and_store_block(conn, height_sync_item):
                        print(f"Failed to sync block {height_sync_item}. Stopping current batch sync.", flush=True)
                        batch_success_sync = False
                        break 
                    blocks_synced_total_mode +=1
                if not batch_success_sync:
                    print("Error during batch sync. Will retry on next full polling cycle if issues persist.", flush=True)
                    break # Stop further batch processing in this startup sync
            print(f"Startup synchronization completed. Attempted to sync {blocks_synced_total_mode} missing blocks.", flush=True)
        else:
            print("No missing historical blocks found up to current API height. Database is synchronized.", flush=True)

        # 3. Continuous Polling Loop
        print("\n--- Starting Continuous Polling for New Blocks ---", flush=True)
        recently_fetched_blocks_polling: Deque[int] = deque(maxlen=10) # Store last 10 fetched block heights

        while True:
            os.system('cls' if os.name == 'nt' else 'clear') # Clear screen
            print(f"--- Mayanode Block Polling --- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---", flush=True)
            
            current_db_latest_poll = get_latest_block_height_from_db(conn)
            next_block_to_fetch_poll = (current_db_latest_poll + 1) if current_db_latest_poll else 1
            
            if args.start_at_latest and not current_db_latest_poll : 
                 api_latest_for_poll_start_cont = get_mayanode_latest_block_height()
                 if api_latest_for_poll_start_cont:
                    next_block_to_fetch_poll = api_latest_for_poll_start_cont
            
            print(f"Status: Monitoring for new blocks...", flush=True)
            print(f"Database Latest: {current_db_latest_poll if current_db_latest_poll else 'N/A'}", flush=True)
            print(f"Next block to check: {next_block_to_fetch_poll}", flush=True)
            
            latest_api_height_main_poll = get_mayanode_latest_block_height()

            if latest_api_height_main_poll is None:
                print("\nCould not fetch latest block height from API. Retrying...", flush=True)
                # No screen clear here, just wait and loop
            else:
                print(f"Current API Latest: {latest_api_height_main_poll}", flush=True)
                if recently_fetched_blocks_polling:
                    print(f"Recently Fetched: {list(recently_fetched_blocks_polling)}", flush=True)
                else:
                    print("Recently Fetched: None yet in this session.", flush=True)

                blocks_fetched_this_cycle_mode = 0
                if latest_api_height_main_poll >= next_block_to_fetch_poll:
                    print(f"\nNew blocks potentially available. Checking from {next_block_to_fetch_poll} up to {latest_api_height_main_poll}.", flush=True)
                    
                    num_blocks_to_target = min(args.max_blocks_per_cycle, latest_api_height_main_poll - next_block_to_fetch_poll + 1)
                    target_end_height_for_cycle_poll = next_block_to_fetch_poll + num_blocks_to_target - 1
                    
                    print(f"Attempting to fetch up to {num_blocks_to_target} block(s) in this cycle (up to height {target_end_height_for_cycle_poll})...", flush=True)
                    for height_poll_item in range(next_block_to_fetch_poll, target_end_height_for_cycle_poll + 1):
                        # Temporary print before actual fetch for this block
                        print(f"  Fetching block {height_poll_item}...", flush=True) 
                        # In polling mode, we don't pass tqdm_bar, so it uses regular print
                        if fetch_and_store_block(conn, height_poll_item):
                            blocks_fetched_this_cycle_mode += 1
                            recently_fetched_blocks_polling.append(height_poll_item)
                            print(f"  Successfully processed block {height_poll_item}.", flush=True)
                        else:
                            print(f"  Failed to process block {height_poll_item}. Will retry in next cycle.", flush=True)
                            break 
                        time.sleep(API_REQUEST_DELAY_SECONDS) # Small delay between sequential fetches in polling
                    
                    if blocks_fetched_this_cycle_mode > 0:
                        print(f"\nFetched {blocks_fetched_this_cycle_mode} block(s) in this polling cycle.", flush=True)
                    # No explicit 'else' needed, status will be updated on next screen clear
                else:
                    print("\nDatabase is up to date with the latest API height.", flush=True)

            print(f"\nWaiting for {args.poll_interval} seconds before next poll...", flush=True)
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nShutdown signal received. Exiting gracefully...", flush=True)
    except sqlite3.Error as db_err:
        print(f"Database error occurred: {db_err}. Shutting down.", flush=True)    
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Shutting down.", flush=True)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.", flush=True)
        print(f"--- Mayanode Continuous Block Ingestion Service SHUT DOWN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---", flush=True)

if __name__ == "__main__":
    # The sys.path modification logic previously here has been moved to the 
    # top of the script to ensure 'src.' imports work correctly when the 
    # script is run directly.
    main()
