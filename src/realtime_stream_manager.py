import time
import collections
import threading # Import threading module
from datetime import datetime, timezone
import logging
import os # For file operations
import pandas as pd # For DataFrame and CSV operations

from api_connections import fetch_recent_maya_actions # For fetching actions
from common_utils import parse_action, DF_COLUMNS # For parsing and structure

CONFIRMED_ACTIONS_MAXLEN = 100    # Store last 100 confirmed actions
PENDING_ACTION_EXPIRY_SECONDS = 300 # Consider a pending action stale after 5 minutes
POLLING_INTERVAL_CONFIRMED = 10   # Seconds to wait before polling for new confirmed actions
POLLING_INTERVAL_PENDING = 5      # Seconds to wait before polling for pending actions
FETCH_ACTION_BATCH_SIZE_STREAM = 10 # Smaller batch size for frequent polling

# Define path for historical data CSV, consistent with app.py
# Assuming this script is in src/, and data/ is a sibling to src/
DATA_DIR_RSM = os.path.join(os.path.dirname(__file__), '..', 'data')
HISTORICAL_DATA_CSV_RSM = os.path.join(DATA_DIR_RSM, 'historical_24hr_maya_transactions.csv')

logger = logging.getLogger(__name__)

class RealtimeStreamManager:
    def __init__(self):
        self.live_confirmed_actions = collections.deque(maxlen=CONFIRMED_ACTIONS_MAXLEN)
        self.pending_actions_pool = {} # tx_id -> {"action_data": parsed_action, "seen_at_ns": ns_timestamp}
        self.last_known_confirmed_action_date_ns = self._get_initial_timestamp_ns()
        
        # Threading control
        self._stop_event = threading.Event() # Use a single event for all polling loops
        self._confirmed_thread = None
        self._pending_thread = None

    def _get_initial_timestamp_ns(self):
        """Gets a starting timestamp (e.g., now or from a config/state file)."""
        # For simplicity, start by looking for actions from the last few minutes
        # In a production system, this might be persisted or set to current time on first run.
        # return int((datetime.now(timezone.utc).timestamp() - (5 * 60)) * 1_000_000_000) # 5 minutes ago in ns
        return 0 # Start with 0 to fetch all recent actions on first poll

    def get_live_confirmed_actions(self):
        """Returns a list of the most recent confirmed actions."""
        return list(self.live_confirmed_actions)

    def get_current_pending_block(self):
        """Returns a list of current pending actions."""
        # Filter out stale pending actions before returning if needed, or do it in the polling loop
        return [item['action_data'] for item in self.pending_actions_pool.values()]

    # Task 1.3.2: Confirmed Actions Stream
    def poll_confirmed_actions(self):
        """Polls Midgard for newly confirmed actions and updates the local store."""
        logger.info(f"Polling for confirmed actions newer than {self.last_known_confirmed_action_date_ns}")
        try:
            # Fetch a small batch of recent actions
            response_data = fetch_recent_maya_actions(limit=FETCH_ACTION_BATCH_SIZE_STREAM, offset=0)
            actions_from_api = response_data.get("actions", [])
            logger.debug(f"Confirmed poll: Fetched {len(actions_from_api)} actions raw from API.")

            if not actions_from_api:
                logger.debug("Confirmed poll: No actions returned from API.")
                return

            newly_confirmed_this_poll = []
            max_date_this_poll = self.last_known_confirmed_action_date_ns # Initialize with current last known

            for action in actions_from_api: # API returns newest first
                action_date_str = action.get("date")
                if not action_date_str:
                    logger.warning(f"Confirmed poll: Skipping action with missing date: {action.get('in', [{}])[0].get('txID', 'Unknown TXID')}")
                    continue
                try:
                    action_date_ns = int(action_date_str)
                except ValueError:
                    logger.warning(f"Confirmed poll: Skipping action with unparseable date '{action_date_str}': {action.get('in', [{}])[0].get('txID', 'Unknown TXID')}")
                    continue

                if action_date_ns <= self.last_known_confirmed_action_date_ns:
                    # All subsequent actions will be older or same, stop processing this batch
                    logger.debug(f"Confirmed poll: Action date {action_date_ns} is not newer than last known {self.last_known_confirmed_action_date_ns}. Stopping processing for this batch.")
                    break

                if action.get("status") in ["success", "refund"]:
                    parsed_action = parse_action(action)
                    if parsed_action and parsed_action.get("transaction_id"):
                        logger.info(f"Confirmed poll: New confirmed action: {parsed_action['transaction_id']} ({parsed_action['type']}) at {action_date_ns}")
                        newly_confirmed_this_poll.append(parsed_action)
                        max_date_this_poll = max(max_date_this_poll, action_date_ns) # Update max_date seen in this poll batch
                        
                        # Remove from pending pool if it exists there
                        tx_id_to_remove = parsed_action.get("transaction_id") or parsed_action.get("first_in_tx_id")
                        if tx_id_to_remove and tx_id_to_remove in self.pending_actions_pool:
                            del self.pending_actions_pool[tx_id_to_remove]
                            logger.info(f"Confirmed poll: Removed {tx_id_to_remove} from pending pool.")
                    else:
                        logger.warning(f"Confirmed poll: Failed to parse a confirmed action or no TXID: {action}")
                # else: # Action is not 'success' or 'refund', or is older
                    # logger.debug(f"Confirmed poll: Skipping action status {action.get('status')} or older date for tx {action.get('in', [{}])[0].get('txID', 'Unknown TXID')}")

            # Prepend newly confirmed actions to our deque and append to CSV
            if newly_confirmed_this_poll:
                # Update in-memory deque (for live view)
                for pa in reversed(newly_confirmed_this_poll):
                    self.live_confirmed_actions.appendleft(pa)
                
                # Append to historical CSV
                try:
                    df_new_confirmed = pd.DataFrame(newly_confirmed_this_poll)
                    # Ensure columns match DF_COLUMNS, handling missing ones with None
                    for col in DF_COLUMNS:
                        if col not in df_new_confirmed.columns:
                            df_new_confirmed[col] = None
                    df_new_confirmed = df_new_confirmed[DF_COLUMNS] # Reorder to match standard

                    # Check if file exists and is not empty to decide on writing header
                    file_exists_and_has_content = os.path.exists(HISTORICAL_DATA_CSV_RSM) and os.path.getsize(HISTORICAL_DATA_CSV_RSM) > 0
                    
                    df_new_confirmed.to_csv(
                        HISTORICAL_DATA_CSV_RSM, 
                        mode='a', 
                        header=not file_exists_and_has_content, # Write header only if file is new/empty
                        index=False
                    )
                    logger.info(f"Appended {len(df_new_confirmed)} new confirmed actions to {HISTORICAL_DATA_CSV_RSM}")
                except Exception as e:
                    logger.error(f"Error appending to historical CSV {HISTORICAL_DATA_CSV_RSM}: {e}", exc_info=True)

                # Update last known date for polling logic
                if max_date_this_poll > self.last_known_confirmed_action_date_ns:
                    self.last_known_confirmed_action_date_ns = max_date_this_poll
                    logger.info(f"Confirmed poll: Added {len(newly_confirmed_this_poll)} new actions. Last known date updated to {self.last_known_confirmed_action_date_ns}.")
                else:
                    logger.info(f"Confirmed poll: Added {len(newly_confirmed_this_poll)} new actions, but their max date ({max_date_this_poll}) was not newer than current last known ({self.last_known_confirmed_action_date_ns}). Last known date not changed.")
            else:
                logger.debug("Confirmed poll: No new confirmed actions to add in this cycle.")

        except requests.exceptions.RequestException as e:
            logger.error(f"ERROR in confirmed polling loop (RequestException): {e}")
        except Exception as e:
            logger.error(f"ERROR in confirmed polling loop (General Exception): {e}", exc_info=True)

    # Task 1.3.3: Pending Actions Stream
    def poll_pending_actions(self):
        """Polls Midgard for pending actions (specifically swaps) and updates the local pool."""
        logger.info("Polling for pending swap actions...")
        try:
            # Fetch a small batch of recent swap actions
            response_data = fetch_recent_maya_actions(limit=FETCH_ACTION_BATCH_SIZE_STREAM, offset=0, action_type="swap")
            actions_from_api = response_data.get("actions", [])
            logger.debug(f"Pending poll: Fetched {len(actions_from_api)} actions raw from API (type=swap).")

            if not actions_from_api:
                logger.debug("Pending poll: No swap actions returned from API.")
                self._prune_stale_pending_actions() # Still prune even if no new ones
                return

            current_time_ns = time.time_ns()
            added_or_updated_count = 0

            for action in actions_from_api:
                if action.get("status") == "pending":
                    parsed_action = parse_action(action)
                    if not parsed_action or not parsed_action.get("transaction_id"):
                        logger.warning(f"Pending poll: Skipping pending action due to parsing failure or no TXID: {action}")
                        continue
                    
                    tx_id = parsed_action["transaction_id"]

                    # Check if this transaction is already confirmed; if so, skip and ensure it's not in pending
                    if any(confirmed_tx["transaction_id"] == tx_id for confirmed_tx in self.live_confirmed_actions):
                        logger.debug(f"Pending poll: Action {tx_id} is already confirmed, ensuring it's removed from pending if present.")
                        if tx_id in self.pending_actions_pool:
                            del self.pending_actions_pool[tx_id]
                        continue

                    if tx_id not in self.pending_actions_pool:
                        logger.info(f"Pending poll: New pending action: {tx_id} ({parsed_action.get('type')})")
                        added_or_updated_count += 1
                    else:
                        logger.debug(f"Pending poll: Refreshing seen_at for existing pending action: {tx_id}")
                        added_or_updated_count +=1 # Count as update
                    
                    self.pending_actions_pool[tx_id] = {
                        "action_data": parsed_action,
                        "seen_at_ns": current_time_ns
                    }
                # else:
                    # logger.debug(f"Pending poll: Skipping action with status {action.get('status')} for tx {action.get('in', [{}])[0].get('txID', 'Unknown TXID')}")

            if added_or_updated_count > 0:
                logger.info(f"Pending poll: Added or updated {added_or_updated_count} pending actions.")
            else:
                logger.debug("Pending poll: No new pending actions to add or update in this cycle.")

            self._prune_stale_pending_actions()

        except requests.exceptions.RequestException as e:
            logger.error(f"ERROR in pending polling loop (RequestException): {e}")
        except Exception as e:
            logger.error(f"ERROR in pending polling loop (General Exception): {e}", exc_info=True)

    def _prune_stale_pending_actions(self):
        """Removes pending actions older than PENDING_ACTION_EXPIRY_SECONDS."""
        current_time_ns = time.time_ns()
        expiry_threshold_ns = PENDING_ACTION_EXPIRY_SECONDS * 1_000_000_000
        stale_keys = []
        for tx_id, item_data in self.pending_actions_pool.items():
            if current_time_ns - item_data["seen_at_ns"] > expiry_threshold_ns:
                stale_keys.append(tx_id)
        
        if stale_keys:
            for tx_id in stale_keys:
                del self.pending_actions_pool[tx_id]
                logger.info(f"Pruned stale pending action: TXID {tx_id}")
            logger.info(f"Pruned {len(stale_keys)} stale pending actions.")

    # Task 1.3.4: Pending Block Management (partially covered in poll methods and pruning)
    # No separate method needed for now as logic is integrated.

    def _confirmed_polling_loop(self):
        """Target function for the confirmed actions polling thread."""
        logger.info("Confirmed actions polling thread started.")
        while not self._stop_event.is_set():
            try:
                self.poll_confirmed_actions()
            except Exception as e:
                logger.error(f"ERROR in confirmed polling loop: {e}")
            # Wait for the polling interval, but check stop_event frequently
            self._stop_event.wait(POLLING_INTERVAL_CONFIRMED) 
        logger.info("Confirmed actions polling thread stopped.")

    def _pending_polling_loop(self):
        """Target function for the pending actions polling thread."""
        logger.info("Pending actions polling thread started.")
        while not self._stop_event.is_set():
            try:
                self.poll_pending_actions()
            except Exception as e:
                logger.error(f"ERROR in pending polling loop: {e}")
            # Wait for the polling interval, but check stop_event frequently
            self._stop_event.wait(POLLING_INTERVAL_PENDING)
        logger.info("Pending actions polling thread stopped.")

    def start_streaming(self):
        if self._confirmed_thread is not None and self._confirmed_thread.is_alive():
            logger.info("Confirmed streaming thread already running.")
            return
        if self._pending_thread is not None and self._pending_thread.is_alive():
            logger.info("Pending streaming thread already running.")
            return

        self._stop_event.clear() # Clear event in case of restart

        self._confirmed_thread = threading.Thread(target=self._confirmed_polling_loop, daemon=True)
        self._pending_thread = threading.Thread(target=self._pending_polling_loop, daemon=True)

        logger.info("Starting stream manager polling threads...")
        self._confirmed_thread.start()
        self._pending_thread.start()

    def stop_streaming(self):
        logger.info("Stream manager stopping polling threads...")
        self._stop_event.set() # Signal threads to stop

        if self._confirmed_thread is not None and self._confirmed_thread.is_alive():
            self._confirmed_thread.join(timeout=POLLING_INTERVAL_CONFIRMED + 2) # Wait for thread to finish
            if self._confirmed_thread.is_alive():
                logger.warning("Confirmed polling thread did not stop in time.")
        
        if self._pending_thread is not None and self._pending_thread.is_alive():
            self._pending_thread.join(timeout=POLLING_INTERVAL_PENDING + 2) # Wait for thread to finish
            if self._pending_thread.is_alive():
                logger.warning("Pending polling thread did not stop in time.")
        
        logger.info("Stream manager polling threads shut down.")

if __name__ == '__main__':
    manager = RealtimeStreamManager()
    logger.info(f"Initial last_known_confirmed_action_date_ns: {manager.last_known_confirmed_action_date_ns}")
    logger.info(f"Live confirmed actions (initial): {manager.get_live_confirmed_actions()}")
    logger.info(f"Pending block (initial): {manager.get_current_pending_block()}")
    
    # Example of how polling might be manually invoked for a single cycle (not threaded yet)
    # print("\n--- Simulating one poll cycle for confirmed --- ")
    # manager.poll_confirmed_actions() # This will be implemented next
    # print(f"Live confirmed after poll: {manager.get_live_confirmed_actions()}")
    # print(f"Last known confirmed date after poll: {manager.last_known_confirmed_action_date_ns}")

    # print("\n--- Simulating one poll cycle for pending --- ")
    # manager.poll_pending_actions() # This will be implemented next
    # print(f"Pending block after poll: {manager.get_current_pending_block()}")

    manager.start_streaming()
    try:
        logger.info("\nStream manager started. Running for 30 seconds for demonstration...")
        time.sleep(30) # Example: run for a bit
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    finally:
        manager.stop_streaming()
        logger.info("Main program finished.") 