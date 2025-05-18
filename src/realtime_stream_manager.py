import time
import collections
import threading # Import threading module
from datetime import datetime, timezone

from api_connections import fetch_recent_maya_actions # For fetching actions
from common_utils import parse_action, DF_COLUMNS # For parsing and structure

CONFIRMED_ACTIONS_MAXLEN = 100    # Store last 100 confirmed actions
PENDING_ACTION_EXPIRY_SECONDS = 300 # Consider a pending action stale after 5 minutes
POLLING_INTERVAL_CONFIRMED = 10   # Seconds to wait before polling for new confirmed actions
POLLING_INTERVAL_PENDING = 5      # Seconds to wait before polling for pending actions
FETCH_ACTION_BATCH_SIZE_STREAM = 10 # Smaller batch size for frequent polling

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
        return int((datetime.now(timezone.utc).timestamp() - (5 * 60)) * 1_000_000_000) # 5 minutes ago in ns

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
        print(f"[{datetime.now(timezone.utc)}] Polling for confirmed actions newer than {self.last_known_confirmed_action_date_ns}", flush=True)
        actions_data = fetch_recent_maya_actions(limit=FETCH_ACTION_BATCH_SIZE_STREAM, offset=0) # Always fetch latest

        if not actions_data or "actions" not in actions_data or not actions_data["actions"]:
            print(f"[{datetime.now(timezone.utc)}] No actions returned from Midgard for confirmed poll.", flush=True)
            return

        newly_confirmed_count = 0
        processed_actions_this_poll = []

        # Actions from API are newest first. Iterate and stop if we see an action older than or same as last known.
        for action in actions_data["actions"]:
            action_date_str = action.get("date")
            action_status = action.get("status")

            if not action_date_str:
                print(f"[{datetime.now(timezone.utc)}] Skipping action due to missing date: {action.get('type')}", flush=True)
                continue
            
            try:
                action_date_ns = int(action_date_str)
            except ValueError:
                print(f"[{datetime.now(timezone.utc)}] Could not parse date string '{action_date_str}' for action.", flush=True)
                continue

            # If this action is older than or same as the newest one we've already processed, 
            # and since the API returns newest first, we can stop checking this batch.
            if action_date_ns <= self.last_known_confirmed_action_date_ns:
                # print(f"[{datetime.now(timezone.utc)}] Action (date: {action_date_ns}) is not newer than last known ({self.last_known_confirmed_action_date_ns}). Stopping processing for this poll.", flush=True)
                break # Stop processing this batch
            
            processed_actions_this_poll.append(action_date_ns) # Keep track of what we've looked at in this poll

            if action_status in ["success", "refund"]:
                parsed = parse_action(action)
                # Ensure it has a transaction_id for removal from pending pool
                tx_id = parsed.get("transaction_id") or parsed.get("first_in_tx_id")
                
                # Add to live_confirmed_actions deque (it handles sorting by insertion order - newest)
                # To ensure deque has newest at the left (index 0 upon conversion to list):
                self.live_confirmed_actions.appendleft(parsed) 
                newly_confirmed_count += 1
                
                print(f"[{datetime.now(timezone.utc)}] Added new confirmed action: Type={parsed.get('type')}, Date={action_date_ns}, TXID={tx_id}", flush=True)

                # If this action was in the pending pool, remove it
                if tx_id and tx_id in self.pending_actions_pool:
                    del self.pending_actions_pool[tx_id]
                    print(f"[{datetime.now(timezone.utc)}] Removed action TXID {tx_id} from pending pool as it is now confirmed.", flush=True)
        
        # Update last_known_confirmed_action_date_ns with the newest action processed in this poll
        if processed_actions_this_poll:
            self.last_known_confirmed_action_date_ns = max(processed_actions_this_poll)
            # print(f"[{datetime.now(timezone.utc)}] Updated last_known_confirmed_action_date_ns to {self.last_known_confirmed_action_date_ns}", flush=True)

        if newly_confirmed_count > 0:
            print(f"[{datetime.now(timezone.utc)}] Finished polling confirmed. Added {newly_confirmed_count} new confirmed actions.", flush=True)
        # else:
            # print(f"[{datetime.now(timezone.utc)}] Finished polling confirmed. No new confirmed actions found newer than {self.last_known_confirmed_action_date_ns}.")

    # Task 1.3.3: Pending Actions Stream
    def poll_pending_actions(self):
        """Polls Midgard for pending actions (specifically swaps) and updates the local pool."""
        print(f"[{datetime.now(timezone.utc)}] Polling for pending swap actions...", flush=True)
        # Fetch actions of type 'swap'. We will filter for 'pending' status client-side.
        actions_data = fetch_recent_maya_actions(limit=FETCH_ACTION_BATCH_SIZE_STREAM, offset=0, action_type="swap")

        if not actions_data or "actions" not in actions_data or not actions_data["actions"]:
            print(f"[{datetime.now(timezone.utc)}] No swap actions returned from Midgard for pending poll.", flush=True)
            self._prune_stale_pending_actions() # Still prune even if fetch fails
            return

        added_or_updated_count = 0
        current_time_ns = time.time_ns()

        for action in actions_data["actions"]:
            action_status = action.get("status")

            if action_status == 'pending':
                parsed = parse_action(action)
                # Use first_in_tx_id as the key, should be present for swaps.
                tx_id = parsed.get("transaction_id") or parsed.get("first_in_tx_id") 

                if not tx_id:
                    print(f"[{datetime.now(timezone.utc)}] Skipping pending action due to missing tx_id: {parsed.get('type')}, Date: {parsed.get('date')}", flush=True)
                    continue
                
                # If action already confirmed and in our live deque, don't add to pending.
                # This is a safeguard, poll_confirmed_actions should remove it if it confirms later.
                if any(confirmed_action.get("transaction_id") == tx_id or confirmed_action.get("first_in_tx_id") == tx_id for confirmed_action in self.live_confirmed_actions):
                    # print(f"[{datetime.now(timezone.utc)}] Pending action TXID {tx_id} is already in confirmed list. Skipping.", flush=True)
                    if tx_id in self.pending_actions_pool: # Clean up if it somehow lingered
                        del self.pending_actions_pool[tx_id]
                    continue

                if tx_id not in self.pending_actions_pool:
                    self.pending_actions_pool[tx_id] = {
                        "action_data": parsed,
                        "seen_at_ns": current_time_ns
                    }
                    added_or_updated_count += 1
                    print(f"[{datetime.now(timezone.utc)}] Added new pending action to pool: TXID {tx_id}, Type={parsed.get('type')}, Date={parsed.get('date')}", flush=True)
                else:
                    # Action already in pool. We could update its `seen_at_ns` or re-parse if data might change.
                    # For now, just update `seen_at_ns` to keep it fresh.
                    self.pending_actions_pool[tx_id]["seen_at_ns"] = current_time_ns
                    # print(f"[{datetime.now(timezone.utc)}] Refreshed pending action in pool: TXID {tx_id}", flush=True)
        
        if added_or_updated_count > 0:
            print(f"[{datetime.now(timezone.utc)}] Finished polling pending. Added/updated {added_or_updated_count} pending actions.", flush=True)
        # else:
            # print(f"[{datetime.now(timezone.utc)}] Finished polling pending. No new pending swap actions found or all were already tracked.")

        self._prune_stale_pending_actions()

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
                print(f"[{datetime.now(timezone.utc)}] Pruned stale pending action: TXID {tx_id}", flush=True)
            print(f"[{datetime.now(timezone.utc)}] Pruned {len(stale_keys)} stale pending actions.", flush=True)

    # Task 1.3.4: Pending Block Management (partially covered in poll methods and pruning)
    # No separate method needed for now as logic is integrated.

    def _confirmed_polling_loop(self):
        """Target function for the confirmed actions polling thread."""
        print(f"[{datetime.now(timezone.utc)}] Confirmed actions polling thread started.", flush=True)
        while not self._stop_event.is_set():
            try:
                self.poll_confirmed_actions()
            except Exception as e:
                print(f"[{datetime.now(timezone.utc)}] ERROR in confirmed polling loop: {e}", flush=True)
            # Wait for the polling interval, but check stop_event frequently
            self._stop_event.wait(POLLING_INTERVAL_CONFIRMED) 
        print(f"[{datetime.now(timezone.utc)}] Confirmed actions polling thread stopped.", flush=True)

    def _pending_polling_loop(self):
        """Target function for the pending actions polling thread."""
        print(f"[{datetime.now(timezone.utc)}] Pending actions polling thread started.", flush=True)
        while not self._stop_event.is_set():
            try:
                self.poll_pending_actions()
            except Exception as e:
                print(f"[{datetime.now(timezone.utc)}] ERROR in pending polling loop: {e}", flush=True)
            # Wait for the polling interval, but check stop_event frequently
            self._stop_event.wait(POLLING_INTERVAL_PENDING)
        print(f"[{datetime.now(timezone.utc)}] Pending actions polling thread stopped.", flush=True)

    def start_streaming(self):
        if self._confirmed_thread is not None and self._confirmed_thread.is_alive():
            print("[{datetime.now(timezone.utc)}] Confirmed streaming thread already running.", flush=True)
            return
        if self._pending_thread is not None and self._pending_thread.is_alive():
            print("[{datetime.now(timezone.utc)}] Pending streaming thread already running.", flush=True)
            return

        self._stop_event.clear() # Clear event in case of restart

        self._confirmed_thread = threading.Thread(target=self._confirmed_polling_loop, daemon=True)
        self._pending_thread = threading.Thread(target=self._pending_polling_loop, daemon=True)

        print(f"[{datetime.now(timezone.utc)}] Starting stream manager polling threads...", flush=True)
        self._confirmed_thread.start()
        self._pending_thread.start()

    def stop_streaming(self):
        print(f"[{datetime.now(timezone.utc)}] Stream manager stopping polling threads...", flush=True)
        self._stop_event.set() # Signal threads to stop

        if self._confirmed_thread is not None and self._confirmed_thread.is_alive():
            self._confirmed_thread.join(timeout=POLLING_INTERVAL_CONFIRMED + 2) # Wait for thread to finish
            if self._confirmed_thread.is_alive():
                print(f"[{datetime.now(timezone.utc)}] Confirmed polling thread did not stop in time.", flush=True)
        
        if self._pending_thread is not None and self._pending_thread.is_alive():
            self._pending_thread.join(timeout=POLLING_INTERVAL_PENDING + 2) # Wait for thread to finish
            if self._pending_thread.is_alive():
                print(f"[{datetime.now(timezone.utc)}] Pending polling thread did not stop in time.", flush=True)
        
        print(f"[{datetime.now(timezone.utc)}] Stream manager polling threads shut down.", flush=True)

if __name__ == '__main__':
    manager = RealtimeStreamManager()
    print(f"Initial last_known_confirmed_action_date_ns: {manager.last_known_confirmed_action_date_ns}")
    print(f"Live confirmed actions (initial): {manager.get_live_confirmed_actions()}")
    print(f"Pending block (initial): {manager.get_current_pending_block()}")
    
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
        print("\nStream manager started. Running for 30 seconds for demonstration...")
        time.sleep(30) # Example: run for a bit
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        manager.stop_streaming()
        print("Main program finished.") 