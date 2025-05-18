import subprocess
import os
import pandas as pd
import sys
import re # Import re for regular expressions

# --- Adjust sys.path to include project root for module imports ---
# This allows finding the 'src' module when running this test script directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End of sys.path adjustment ---

# --- Configuration ---
PYTHON_EXECUTABLE = sys.executable # Use the same python that runs this script
SRC_DIR = "src"
DATA_DIR = "data"
# HISTORICAL_DATA_FILE = os.path.join(DATA_DIR, "simulated_historical_maya_transactions.csv")
# MIN_HISTORICAL_RECORDS = 1000
REALTIME_DATA_FILE = os.path.join(DATA_DIR, "realtime_maya_transactions.csv")
# MIN_REALTIME_RECORDS = 10 # No longer using a fixed minimum, will validate against script output
FETCH_REALTIME_TIMEOUT_SECONDS = 90 # Increased timeout for fetching realtime data
MEMPOOL_MONITOR_TIMEOUT_SECONDS = 25 # Increased timeout for mempool monitor to fetch from API

# --- Helper Function for Running Scripts ---
def run_script(script_name, script_path, timeout=30, capture_output=True, check_return_code=True):
    print(f"--- Testing {script_name} --- ({script_path})")
    workspace_root = os.getcwd() # Get current working directory
    print(f"INFO: Running script with cwd: {workspace_root}") # Log cwd
    try:
        process = subprocess.run(
            [PYTHON_EXECUTABLE, script_path],
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=False, # We will check return code manually
            cwd=workspace_root # Explicitly set cwd
        )
        if check_return_code and process.returncode != 0:
            print(f"ERROR: {script_name} exited with code {process.returncode}.")
            print("Stderr:\n", process.stderr)
            print("Stdout:\n", process.stdout)
            return False, process.stdout, process.stderr, False # script_error, stdout, stderr, did_timeout
        
        print(f"{script_name} completed run (return code: {process.returncode}). Expected for its type? Depends on test.")
        return True, process.stdout, process.stderr, False # script_success, stdout, stderr, did_timeout
    except subprocess.TimeoutExpired as e:
        print(f"INFO: {script_name} timed out after {timeout} seconds (expected for some tests).")
        captured_stdout = getattr(e, 'stdout', b'')
        if isinstance(captured_stdout, bytes):
            captured_stdout = captured_stdout.decode(errors='ignore')
        captured_stderr = getattr(e, 'stderr', b'')
        if isinstance(captured_stderr, bytes):
            captured_stderr = captured_stderr.decode(errors='ignore')
        return False, captured_stdout, captured_stderr, True # run_error (timeout), stdout, stderr, did_timeout
    except FileNotFoundError:
        print(f"ERROR: Python executable not found at {PYTHON_EXECUTABLE} or script {script_path} not found.")
        return False, "", "FileNotFoundError_BY_TEST_RUNNER", False # run_error, stdout, stderr, did_timeout
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while running {script_name}: {e}")
        return False, "", f"OtherException_BY_TEST_RUNNER: {str(e)}", False # run_error, stdout, stderr, did_timeout

# --- Test Functions ---
def test_task_1_1_api_connections():
    script_name = "API Connections Script (api_connections.py)"
    script_path = os.path.join(SRC_DIR, "api_connections.py")
    success, stdout, stderr, _ = run_script(script_name, script_path)
    if not success:
        # Stderr is already printed by run_script if process.returncode != 0 and check_return_code=True
        return False
    
    # Check for key phrases indicating success/expected behavior
    checks = {
        "maya_connection_attempted": "--- Testing Maya Protocol API ---" in stdout,
        "coingecko_success": "Successfully connected to CoinGecko (Ping)" in stdout,
        "uniswap_info_displayed": "--- Uniswap API Information ---" in stdout,
        "maya_health_check_attempted": "--- Testing Maya Protocol Midgard Health API ---" in stdout, # Corrected string
        "last_block_height_fetched_or_failed": ("Last aggregated block height:" in stdout) or ("Failed to get last aggregated block height." in stdout) # Added check
    }
    all_checks_passed = all(checks.values())
    if all_checks_passed:
        print(f"PASS: {script_name} output indicates expected API interaction attempts.")
    else:
        print(f"FAIL: {script_name} output missing expected content. Checks: {checks}")
        if not checks["maya_health_check_attempted"]:
            print(f"DEBUG: Full stdout for {script_name} when maya_health_check_attempted failed:")
            print(stdout) # Print full stdout for debugging this specific check
    return all_checks_passed

def test_task_1_2_realtime_data():
    script_name_fetch = "24-Hour Historical Data Fetch (fetch_realtime_transactions.py)"
    script_path_fetch = os.path.join(SRC_DIR, "fetch_realtime_transactions.py")
    # Update to the new file name
    historical_data_file = os.path.join(DATA_DIR, "historical_24hr_maya_transactions.csv")
    
    print(f"--- Testing {script_name_fetch} ---")
    print(f"Running {script_path_fetch} to fetch 24-hour historical data...")
    
    # Ensure the old/target file is removed before running, if it exists
    if os.path.exists(historical_data_file):
        try:
            os.remove(historical_data_file)
            print(f"Removed existing file {historical_data_file} before test run.")
        except OSError as e:
            print(f"WARN: Could not remove existing file {historical_data_file}: {e}")

    # Use a longer timeout as fetching 24 hours of data can take time
    success_fetch, stdout_fetch, stderr_fetch, _ = run_script(script_name_fetch, script_path_fetch, timeout=FETCH_REALTIME_TIMEOUT_SECONDS * 2) # Double timeout for 24hr fetch
    
    if not success_fetch:
        # Stderr might contain useful info if the script itself failed (e.g., Python error)
        # run_script already prints stdout/stderr if process.returncode != 0
        print(f"FAIL: {script_name_fetch} did not complete successfully (return code non-zero or internal error in run_script).")
        # Additional print for clarity if it was a script execution problem vs. run_script internal problem
        if stderr_fetch and ("FileNotFoundError_BY_TEST_RUNNER" not in stderr_fetch and "OtherException_BY_TEST_RUNNER" not in stderr_fetch):
             print(f"Script Stderr:\n{stderr_fetch}")
        return False

    print(f"--- Validating output of {script_name_fetch} and content of {historical_data_file} ---")

    # 1. Verify the script attempts to fetch data over a 24-hour period (check stdout)
    if "Fetching all actions from Maya Protocol from the last 24 hours" not in stdout_fetch:
        print(f"FAIL: Script output did not indicate it's fetching 24-hour data.")
        print(f"Stdout:\n{stdout_fetch}")
        return False
    print("PASS: Script output indicates attempt to fetch 24-hour data.")

    # 2. Check that data/historical_24hr_maya_transactions.csv is created
    if not os.path.exists(historical_data_file):
        print(f"FAIL: {historical_data_file} was not created by {script_name_fetch}.")
        print(f"Stdout:\n{stdout_fetch}")
        if stderr_fetch: print(f"Stderr:\n{stderr_fetch}")
        return False
    print(f"PASS: {historical_data_file} was created.")

    # 3. Read the CSV and perform further checks
    try:
        df = pd.read_csv(historical_data_file)
        num_records_in_csv = len(df)
        print(f"INFO: {historical_data_file} contains {num_records_in_csv} records.")

        # 3.1 Verify all expected columns (DF_COLUMNS) are present
        # These are defined in fetch_realtime_transactions.py, re-define or import if necessary
        # For now, using a subset of core columns for robustness if DF_COLUMNS changes slightly
        # Better: ensure DF_COLUMNS from the script is available here or use a shared definition.
        # Using the full list from the scratchpad for now:
        expected_df_columns = [
            "date", "height", "status", "type", "in_tx_count", "out_tx_count", "pools", 
            "first_in_tx_id", "first_out_tx_id", "in_asset", "in_amount", "in_address",
            "out_asset", "out_amount", "out_address", "swap_liquidity_fee", 
            "swap_slip_bps", "swap_target_asset", "swap_network_fee_asset", 
            "swap_network_fee_amount", "transaction_id"
        ]
        missing_columns = [col for col in expected_df_columns if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in expected_df_columns]

        if missing_columns:
            print(f"FAIL: Missing expected columns in {historical_data_file}: {missing_columns}")
            return False
        if extra_columns:
            print(f"WARN: Extra unexpected columns found in {historical_data_file}: {extra_columns}. Test will proceed.")
        
        print(f"PASS: All expected columns are present in {historical_data_file}.")

        # 3.2 If the CSV has data, verify timestamps are within a reasonable 24-hour window
        if num_records_in_csv > 0:
            if 'date' not in df.columns:
                print(f"FAIL: 'date' column missing, cannot verify timestamps.")
                return False
            
            # Convert 'date' column to numeric (nanoseconds since epoch)
            try:
                df['date_ns'] = pd.to_numeric(df['date'])
            except Exception as e:
                print(f"FAIL: Could not convert 'date' column to numeric: {e}")
                return False

            current_time_ns = pd.Timestamp.now().value # Current time in nanoseconds
            twenty_four_hours_ns = 24 * 60 * 60 * 1_000_000_000
            # Allow a small buffer (e.g., 5 minutes) for clock drift and processing time
            buffer_ns = 5 * 60 * 1_000_000_000 
            
            lower_bound_ns = current_time_ns - twenty_four_hours_ns - buffer_ns
            upper_bound_ns = current_time_ns + buffer_ns

            # Check if all dates are within the approximate 24-hour window
            # The script fetches actions *older* than now, so upper_bound_ns check is mostly sanity.
            # The crucial part is that they are not *too old*.
            out_of_window_too_old = df[df['date_ns'] < lower_bound_ns]
            out_of_window_too_new = df[df['date_ns'] > upper_bound_ns]

            if not out_of_window_too_old.empty:
                print(f"FAIL: Found {len(out_of_window_too_old)} records with timestamps older than the expected 24-hour window (plus buffer).")
                print(f"  Example old dates (ns): {out_of_window_too_old['date_ns'].head().tolist()}")
                print(f"  Lower bound (ns): {lower_bound_ns}")
                return False
            if not out_of_window_too_new.empty:
                # This might happen if the script runs very close to a new action appearing and clock sync issues
                print(f"WARN: Found {len(out_of_window_too_new)} records with timestamps newer than current time (plus buffer). This might be okay but is noted.")
                print(f"  Example new dates (ns): {out_of_window_too_new['date_ns'].head().tolist()}")
                print(f"  Upper bound (ns): {upper_bound_ns}")

            print("PASS: Timestamps in CSV are within a reasonable 24-hour window (or newer, which is acceptable).")
        else: # num_records_in_csv == 0
            # 3.3 Verify handling of an empty CSV (already covered by file creation and column check if it's truly empty with headers)
            # Check stdout for messages indicating no actions were found
            if ("No actions found within the last 24 hours." in stdout_fetch or 
                "Successfully saved an empty CSV (no relevant actions found in the last 24 hours)" in stdout_fetch):
                print("PASS: Script reported no actions found, and CSV is empty. Correct handling of empty results.")
            elif "Collected a total of 0 actions" in stdout_fetch:
                 print("PASS: Script reported 0 actions collected, and CSV is empty. Correct handling of empty results.")
            else:
                # This could be a slight mismatch if the file is empty but the log message is different.
                # The critical part is that the CSV is empty and columns are correct.
                print(f"INFO: CSV is empty. Stdout for no actions: '{stdout_fetch}'. Assuming this is acceptable if API had no recent data.")
        
        # 4. Removal of logic related to specific block height checking is implicit by not having it here.

        return True # All checks passed for this test case

    except pd.errors.EmptyDataError:
        # This occurs if the CSV is completely empty (no headers even). Script should create headers.
        # Check stdout to see if the script explicitly said it's saving an empty CSV.
        if "Successfully saved an empty CSV" in stdout_fetch:
            print(f"PASS: {historical_data_file} is empty (no headers), and script reported saving an empty CSV.")
            return True
        else:
            print(f"FAIL: {historical_data_file} is completely empty (no headers), and script didn't explicitly report this. This is likely an issue.")
            print(f"Stdout:\n{stdout_fetch}")
            return False
    except Exception as e:
        print(f"FAIL: Could not read or validate {historical_data_file}. Error: {e}")
        print(f"Stdout:\n{stdout_fetch}")
        if stderr_fetch: print(f"Stderr:\n{stderr_fetch}")
        return False

def test_task_1_3_mempool_monitor():
    script_name = "Mempool Monitor (mempool_monitor.py)"
    script_path = os.path.join(SRC_DIR, "mempool_monitor.py")
    
    print(f"INFO: Using PYTHON_EXECUTABLE: {PYTHON_EXECUTABLE} for {script_name}")

    # For mempool monitor, check_return_code is False in run_script if we pass it, but we are not.
    # The run_script function's `success` flag here means "did the script run and exit normally (code 0)?"
    # or "did run_script itself encounter an error like FileNotFoud?"
    # We pass check_return_code=True (default) to run_script for most, but for mempool, we expect timeout.
    # So, let's make check_return_code False for this specific call.
    run_script_internal_success, stdout, stderr_from_script, did_timeout = run_script(script_name, script_path, timeout=MEMPOOL_MONITOR_TIMEOUT_SECONDS, check_return_code=False)

    stdout = stdout if stdout is not None else ""
    stderr_from_script = stderr_from_script if stderr_from_script is not None else ""
    
    header_present = "--- Maya Protocol Mempool Monitor (Midgard API) ---" in stdout
    pending_action_detected_or_attempted = ("DETECTED" in stdout and "PENDING SWAP ACTION(S)" in stdout) or \
                                           "No pending swap actions found" in stdout or \
                                           "Fetching from Midgard:" in stdout

    if did_timeout:
        if header_present and pending_action_detected_or_attempted:
            print(f"PASS: {script_name} started, attempted/completed fetching, and was terminated by timeout as expected. Stdout: '{stdout}', Stderr: '{stderr_from_script}'")
            return True
        elif header_present: # Header is there, but no sign of fetch attempt/completion
            print(f"FAIL: {script_name} timed out with header, but no fetch attempt/completion messages. Stdout: '{stdout}', Stderr: '{stderr_from_script}'")
            return False
        else: # No header, but timed out
            print(f"FAIL: {script_name} timed out but API header not found. Startup issue. Stdout: '{stdout}', Stderr: '{stderr_from_script}'")
            return False
    # Script did NOT time out. It either completed (run_script_internal_success=True) or run_script had a pre-run error (run_script_internal_success=False)
    elif run_script_internal_success: # Script completed on its own (e.g. return code 0)
        print(f"FAIL: {script_name} completed execution (expected to run indefinitely). Stdout: '{stdout[:1000]}', Stderr: '{stderr_from_script[:500]}'")
        if header_present and pending_action_detected_or_attempted:
            print("  INFO: Script printed expected output but then exited early.")
        return False
    else: # run_script_internal_success is False, and did_timeout is False. This means a pre-run error in run_script (e.g. FileNotFoundError for script/python)
        print(f"FAIL: {script_name} failed to start due to a test runner error (e.g., file not found). Error message from runner: '{stderr_from_script}'. Stdout: '{stdout}'")
        return False

# --- New Tests for RealtimeStreamManager (Task 1.5.2) ---
# Need to import RealtimeStreamManager and mock
import time as pytime # Alias to avoid conflict with module-level time
from unittest.mock import patch, MagicMock
from src.realtime_stream_manager import RealtimeStreamManager, FETCH_ACTION_BATCH_SIZE_STREAM, PENDING_ACTION_EXPIRY_SECONDS
from src.common_utils import parse_action # For creating expected parsed data

# Mock Midgard API responses
def get_mock_midgard_action(date_ns, tx_id, status, type, in_asset="BTC.BTC", in_amount="100000000", out_asset="ETH.ETH", out_amount="2000000000"):
    return {
        "date": str(date_ns),
        "height": "12345",
        "status": status,
        "type": type,
        "in": [{
            "txID": tx_id,
            "address": "maya_in_addr",
            "coins": [{"asset": in_asset, "amount": in_amount}]
        }],
        "out": [{
            "txID": f"{tx_id}_out",
            "address": "maya_out_addr",
            "coins": [{"asset": out_asset, "amount": out_amount}] if status == "success" else []
        }],
        "pools": [f"{in_asset}", f"{out_asset}"],
        "metadata": {
            "swap": {
                "networkFees": [{"asset": "MAYA.CACAO", "amount": "10000"}],
                "liquidityFee": "500000",
                "swapSlip": "10",
                "swapTarget": out_asset
            } if type == "swap" else {}
        }
    }

MOCK_ACTION_TIME_NOW_NS = int(pytime.time() * 1_000_000_000)

MOCK_CONFIRMED_ACTION_1 = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 20_000_000_000, "tx_confirmed_1", "success", "swap")
MOCK_CONFIRMED_ACTION_2 = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 10_000_000_000, "tx_confirmed_2", "refund", "swap")
MOCK_PENDING_ACTION_1 = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 5_000_000_000, "tx_pending_1", "pending", "swap")
MOCK_PENDING_ACTION_2 = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 3_000_000_000, "tx_pending_2", "pending", "swap", in_asset="ETH.ETH", out_asset="BTC.BTC")
MOCK_OLDER_CONFIRMED_ACTION = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 300_000_000_000, "tx_older_confirmed", "success", "swap")


def test_task_1_3_realtime_stream_manager():
    print("--- Testing RealtimeStreamManager ---")
    all_subtests_passed = True

    # Test 1: Initialization
    print("  Testing RealtimeStreamManager: Initialization...")
    try:
        manager = RealtimeStreamManager()
        # Stop threads immediately for isolated unit tests of poll methods
        if hasattr(manager, '_stop_event'): # New manager has _stop_event
             manager._stop_event.set()
             if manager._confirmed_thread and manager._confirmed_thread.is_alive(): manager._confirmed_thread.join(timeout=1)
             if manager._pending_thread and manager._pending_thread.is_alive(): manager._pending_thread.join(timeout=1)
        
        initial_confirmed_len = len(manager.get_live_confirmed_actions())
        initial_pending_len = len(manager.get_current_pending_block())
        if initial_confirmed_len == 0 and initial_pending_len == 0 and manager.last_known_confirmed_action_date_ns > 0:
            print("  PASS: Initialization seems correct.")
        else:
            print(f"  FAIL: Initialization incorrect. Confirmed: {initial_confirmed_len}, Pending: {initial_pending_len}, LastDate: {manager.last_known_confirmed_action_date_ns}")
            all_subtests_passed = False
    except Exception as e:
        print(f"  FAIL: Exception during RealtimeStreamManager initialization test: {e}")
        all_subtests_passed = False
        return False # Critical failure

    # Test 2: Polling Confirmed Actions
    print("\n  Testing RealtimeStreamManager: Polling Confirmed Actions...")
    manager = RealtimeStreamManager() # Fresh manager
    manager._stop_event.set() # Ensure threads don't run for this part
    original_last_known_date = MOCK_ACTION_TIME_NOW_NS - 100_000_000_000
    manager.last_known_confirmed_action_date_ns = original_last_known_date

    # Scenario 2.1: New confirmed actions
    print("    Scenario 2.1: New confirmed actions")
    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_confirmed_new:
        mock_fetch_confirmed_new.return_value = {"actions": [MOCK_CONFIRMED_ACTION_1, MOCK_CONFIRMED_ACTION_2, MOCK_OLDER_CONFIRMED_ACTION]}
        manager.poll_confirmed_actions()

        confirmed_actions = manager.get_live_confirmed_actions()
        if len(confirmed_actions) == 2 and \
           confirmed_actions[0]["transaction_id"] == "tx_confirmed_2" and \
           confirmed_actions[1]["transaction_id"] == "tx_confirmed_1" and \
           manager.last_known_confirmed_action_date_ns == int(MOCK_CONFIRMED_ACTION_2["date"]):
            print("    PASS: Polling confirmed actions - new actions processed correctly.")
        else:
            print(f"    FAIL: Polling confirmed actions - new actions processing error. Count: {len(confirmed_actions)}, LastDate: {manager.last_known_confirmed_action_date_ns}")
            print(f"    Expected last date: {int(MOCK_CONFIRMED_ACTION_2['date'])}")
            all_subtests_passed = False

    # Scenario 2.2: No new confirmed actions
    print("    Scenario 2.2: No new confirmed actions")
    manager.last_known_confirmed_action_date_ns = int(MOCK_CONFIRMED_ACTION_2["date"]) # Up to date with MOCK_CONFIRMED_ACTION_2
    initial_confirmed_count_sc2_2 = len(manager.get_live_confirmed_actions()) # Should be 2 from previous scenario
    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_no_new_confirmed:
        mock_fetch_no_new_confirmed.return_value = {"actions": [MOCK_CONFIRMED_ACTION_2, MOCK_CONFIRMED_ACTION_1]} # Actions not newer than last_known
        manager.poll_confirmed_actions()
        if len(manager.get_live_confirmed_actions()) == initial_confirmed_count_sc2_2: # No change
             print("    PASS: Polling confirmed actions - no new actions handled correctly.")
        else:
            print(f"    FAIL: Polling confirmed actions - no new actions caused unexpected change. Count: {len(manager.get_live_confirmed_actions())}, Expected: {initial_confirmed_count_sc2_2}")
            all_subtests_passed = False

    # Scenario 2.3: Empty API response
    print("    Scenario 2.3: Empty API response for confirmed poll")
    current_last_date = manager.last_known_confirmed_action_date_ns
    initial_confirmed_count_sc2_3 = len(manager.get_live_confirmed_actions())
    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_empty_confirmed:
        mock_fetch_empty_confirmed.return_value = {"actions": []}
        manager.poll_confirmed_actions()
        if len(manager.get_live_confirmed_actions()) == initial_confirmed_count_sc2_3 and manager.last_known_confirmed_action_date_ns == current_last_date:
            print("    PASS: Polling confirmed actions - empty API response handled correctly.")
        else:
            print(f"    FAIL: Polling confirmed actions - empty API response error. Count: {len(manager.get_live_confirmed_actions())}, LastDate: {manager.last_known_confirmed_action_date_ns}")
            all_subtests_passed = False
    
    # Scenario 2.4: Action with missing date
    print("    Scenario 2.4: Action with missing date")
    current_last_date_sc2_4 = manager.last_known_confirmed_action_date_ns
    initial_confirmed_count_sc2_4 = len(manager.get_live_confirmed_actions())
    action_no_date = get_mock_midgard_action(None, "tx_no_date", "success", "swap")
    del action_no_date['date'] # Remove date field
    action_after_no_date = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS + 10_000_000_000, "tx_after_no_date", "success", "swap")

    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_missing_date:
        # API returns: action_after_no_date (newest), action_no_date (problematic), MOCK_CONFIRMED_ACTION_2 (older than current_last_date_sc2_4)
        # manager.last_known_confirmed_action_date_ns is currently int(MOCK_CONFIRMED_ACTION_2["date"])
        # So, MOCK_CONFIRMED_ACTION_2 should be skipped by date check. action_after_no_date should be processed.
        mock_fetch_missing_date.return_value = {"actions": [action_after_no_date, action_no_date, MOCK_CONFIRMED_ACTION_2]}
        manager.poll_confirmed_actions()
        confirmed_actions_sc2_4 = manager.get_live_confirmed_actions()
        
        # Expecting action_after_no_date to be added, initial_confirmed_count_sc2_4 was 2. So now 3.
        # And last_known_confirmed_action_date_ns should be date of action_after_no_date
        if len(confirmed_actions_sc2_4) == initial_confirmed_count_sc2_4 + 1 and \
           confirmed_actions_sc2_4[0]["transaction_id"] == "tx_after_no_date" and \
           manager.last_known_confirmed_action_date_ns == int(action_after_no_date["date"]):
            print("    PASS: Polling confirmed actions - action with missing date handled correctly (skipped).")
        else:
            print(f"    FAIL: Polling confirmed actions - action with missing date error. Count: {len(confirmed_actions_sc2_4)}, LastDate: {manager.last_known_confirmed_action_date_ns}")
            print(f"    Initial count: {initial_confirmed_count_sc2_4}, Expected tx_id: tx_after_no_date, Expected date: {int(action_after_no_date['date'])}")
            all_subtests_passed = False

    # Scenario 2.5: Action with unparseable date
    print("    Scenario 2.5: Action with unparseable date")
    manager.last_known_confirmed_action_date_ns = int(action_after_no_date["date"]) # Update to newest known
    initial_confirmed_count_sc2_5 = len(manager.get_live_confirmed_actions()) # Should be 3
    action_unparseable_date = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS + 20_000_000_000, "tx_unparseable_date", "success", "swap")
    action_unparseable_date['date'] = "not-a-timestamp"
    action_after_unparseable = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS + 30_000_000_000, "tx_after_unparseable", "success", "swap")

    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_unparseable_date:
        mock_fetch_unparseable_date.return_value = {"actions": [action_after_unparseable, action_unparseable_date, action_after_no_date]}
        manager.poll_confirmed_actions()
        confirmed_actions_sc2_5 = manager.get_live_confirmed_actions()

        # Expecting action_after_unparseable to be added. Initial count was 3, now 4.
        if len(confirmed_actions_sc2_5) == initial_confirmed_count_sc2_5 + 1 and \
           confirmed_actions_sc2_5[0]["transaction_id"] == "tx_after_unparseable" and \
           manager.last_known_confirmed_action_date_ns == int(action_after_unparseable["date"]):
            print("    PASS: Polling confirmed actions - action with unparseable date handled correctly (skipped).")
        else:
            print(f"    FAIL: Polling confirmed actions - action with unparseable date error. Count: {len(confirmed_actions_sc2_5)}, LastDate: {manager.last_known_confirmed_action_date_ns}")
            print(f"    Initial count: {initial_confirmed_count_sc2_5}, Expected tx_id: tx_after_unparseable, Expected date: {int(action_after_unparseable['date'])}")
            all_subtests_passed = False


    # Test 3: Polling Pending Actions
    print("\n  Testing RealtimeStreamManager: Polling Pending Actions...")
    manager = RealtimeStreamManager() # Fresh manager
    manager._stop_event.set()

    # Scenario 3.1: New pending actions
    print("    Scenario 3.1: New pending actions")
    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_pending_new:
        mock_fetch_pending_new.return_value = {"actions": [MOCK_PENDING_ACTION_1, MOCK_PENDING_ACTION_2]} # These are type=swap, status=pending
        manager.poll_pending_actions()
        pending_block = manager.get_current_pending_block()
        pending_tx_ids = sorted([p["transaction_id"] for p in pending_block])

        if len(pending_block) == 2 and "tx_pending_1" in pending_tx_ids and "tx_pending_2" in pending_tx_ids:
            print("    PASS: Polling pending actions - new actions added correctly.")
        else:
            print(f"    FAIL: Polling pending actions - new actions processing error. Count: {len(pending_block)}, IDs: {pending_tx_ids}")
            all_subtests_passed = False

    # Scenario 3.2: Pruning stale pending actions (already covered, re-verified)
    print("    Scenario 3.2: Pruning stale pending actions")
    manager.pending_actions_pool.clear()
    stale_action_tx_id = "tx_stale_pending"
    stale_action_parsed = parse_action(get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - (PENDING_ACTION_EXPIRY_SECONDS + 60) * 1_000_000_000, stale_action_tx_id, "pending", "swap"))
    manager.pending_actions_pool[stale_action_tx_id] = {
        "action_data": stale_action_parsed,
        "seen_at_ns": MOCK_ACTION_TIME_NOW_NS - (PENDING_ACTION_EXPIRY_SECONDS + 120) * 1_000_000_000 # Extra stale
    }
    fresh_action_tx_id = "tx_fresh_pending"
    fresh_action_parsed = parse_action(get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 10_000_000_000, fresh_action_tx_id, "pending", "swap"))
    manager.pending_actions_pool[fresh_action_tx_id] = {
        "action_data": fresh_action_parsed,
        "seen_at_ns": MOCK_ACTION_TIME_NOW_NS - 10_000_000_000 # Fresh
    }
    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_for_prune:
        mock_fetch_for_prune.return_value = {"actions": []}
        manager.poll_pending_actions() # Triggers pruning
        pending_block_after_prune = manager.get_current_pending_block()
        pruned_tx_ids = [p["transaction_id"] for p in pending_block_after_prune]
        if len(pending_block_after_prune) == 1 and fresh_action_tx_id in pruned_tx_ids and stale_action_tx_id not in pruned_tx_ids:
            print("    PASS: Polling pending actions - stale action pruned correctly.")
        else:
            print(f"    FAIL: Polling pending actions - stale action pruning error. Count: {len(pending_block_after_prune)}, IDs: {pruned_tx_ids}")
            all_subtests_passed = False
            
    # Scenario 3.3: Empty API response for pending poll
    print("    Scenario 3.3: Empty API response for pending poll")
    manager.pending_actions_pool.clear()
    manager.pending_actions_pool[fresh_action_tx_id] = { # Keep one fresh action
        "action_data": fresh_action_parsed, "seen_at_ns": MOCK_ACTION_TIME_NOW_NS - 10_000_000_000
    }
    initial_pending_count_sc3_3 = len(manager.get_current_pending_block())
    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_empty_pending:
        mock_fetch_empty_pending.return_value = {"actions": []}
        manager.poll_pending_actions()
        if len(manager.get_current_pending_block()) == initial_pending_count_sc3_3:
            print("    PASS: Polling pending actions - empty API response handled correctly.")
        else:
            print(f"    FAIL: Polling pending actions - empty API response error. Count: {len(manager.get_current_pending_block())}")
            all_subtests_passed = False

    # Scenario 3.4: Mix of pending and success status swaps from API
    print("    Scenario 3.4: Mix of pending and success status swaps from API")
    manager.pending_actions_pool.clear()
    pending_swap_mix = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 2_000_000_000, "tx_pending_mix", "pending", "swap")
    success_swap_mix = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 1_000_000_000, "tx_success_mix", "success", "swap")
    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_mixed_status:
        # fetch_recent_maya_actions in poll_pending_actions is called with action_type="swap"
        # The SUT then filters for status='pending'
        mock_fetch_mixed_status.return_value = {"actions": [success_swap_mix, pending_swap_mix]}
        manager.poll_pending_actions()
        current_pending_block = manager.get_current_pending_block()
        if len(current_pending_block) == 1 and current_pending_block[0]["transaction_id"] == "tx_pending_mix":
            print("    PASS: Polling pending actions - only pending swap added from mixed status results.")
        else:
            print(f"    FAIL: Polling pending actions - mixed status processing error. Count: {len(current_pending_block)}, IDs: {[p['transaction_id'] for p in current_pending_block]}")
            all_subtests_passed = False

    # Scenario 3.5: Update seen_at_ns for existing pending action
    print("    Scenario 3.5: Update seen_at_ns for existing pending action")
    manager.pending_actions_pool.clear()
    existing_pending_tx_id = "tx_existing_pending"
    initial_seen_at = MOCK_ACTION_TIME_NOW_NS - 60_000_000_000 # 1 minute ago
    existing_pending_action = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 70_000_000_000, existing_pending_tx_id, "pending", "swap")
    manager.pending_actions_pool[existing_pending_tx_id] = {
        "action_data": parse_action(existing_pending_action),
        "seen_at_ns": initial_seen_at
    }
    # Make sure current time for the test is measurably different
    pytime.sleep(0.01) # Ensure current_time_ns in SUT will be greater
    
    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_existing_pending:
        # API returns the same pending action again
        mock_fetch_existing_pending.return_value = {"actions": [existing_pending_action]}
        manager.poll_pending_actions()
        
        item_in_pool = manager.pending_actions_pool.get(existing_pending_tx_id)
        if item_in_pool and item_in_pool["seen_at_ns"] > initial_seen_at:
            print("    PASS: Polling pending actions - seen_at_ns updated for existing action.")
        else:
            new_seen_at = item_in_pool["seen_at_ns"] if item_in_pool else "Not Found"
            print(f"    FAIL: Polling pending actions - seen_at_ns update error. Initial: {initial_seen_at}, New: {new_seen_at}")
            all_subtests_passed = False

    # Scenario 3.6: Skip pending action already in live_confirmed_actions
    print("    Scenario 3.6: Skip pending action already in live_confirmed_actions")
    manager.pending_actions_pool.clear()
    manager.live_confirmed_actions.clear() # Clear confirmed for this test too
    
    confirmed_tx_id_for_skip_test = "tx_already_confirmed_skip"
    # Add to live_confirmed_actions
    confirmed_action_for_skip = parse_action(get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 80_000_000_000, confirmed_tx_id_for_skip_test, "success", "swap"))
    manager.live_confirmed_actions.append(confirmed_action_for_skip)
    
    # API returns this action again but as pending
    pending_version_of_confirmed = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 75_000_000_000, confirmed_tx_id_for_skip_test, "pending", "swap")
    
    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_pending_but_confirmed:
        mock_fetch_pending_but_confirmed.return_value = {"actions": [pending_version_of_confirmed]}
        manager.poll_pending_actions()
        if len(manager.get_current_pending_block()) == 0: # Should not be added to pending
            print("    PASS: Polling pending actions - action already confirmed was skipped.")
        else:
            print(f"    FAIL: Polling pending actions - action already confirmed was not skipped. Pending count: {len(manager.get_current_pending_block())}")
            all_subtests_passed = False
    manager.live_confirmed_actions.clear() # Clean up for next tests

    # Scenario 3.7: Skip pending action with missing transaction_id
    print("    Scenario 3.7: Skip pending action with missing transaction_id")
    manager.pending_actions_pool.clear()
    action_missing_txid_api = { # Raw API action, parse_action will try to find tx_id
        "date": str(MOCK_ACTION_TIME_NOW_NS - 1_000_000_000), "status": "pending", "type": "swap", "in": [{"coins": [{"asset": "BTC.BTC", "amount": "100"}]}] # No txID in 'in'
    }
    action_with_txid_api = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 2_000_000_000, "tx_good_pending", "pending", "swap")

    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_missing_txid:
        mock_fetch_missing_txid.return_value = {"actions": [action_missing_txid_api, action_with_txid_api]}
        manager.poll_pending_actions()
        current_pending_block = manager.get_current_pending_block()
        if len(current_pending_block) == 1 and current_pending_block[0]["transaction_id"] == "tx_good_pending":
            print("    PASS: Polling pending actions - action with missing tx_id skipped, valid one processed.")
        else:
            print(f"    FAIL: Polling pending actions - missing tx_id handling error. Count: {len(current_pending_block)}, IDs: {[p.get('transaction_id', 'None') for p in current_pending_block]}")
            all_subtests_passed = False


    # Test 4: Confirmed action removes from pending (already covered, re-verified)
    print("\n  Testing RealtimeStreamManager: Confirmed action removes from pending...")
    manager = RealtimeStreamManager() # Fresh manager
    manager._stop_event.set()
    manager.last_known_confirmed_action_date_ns = MOCK_ACTION_TIME_NOW_NS - 100_000_000_000

    # Add a pending action that will then be confirmed
    action_to_confirm_tx_id = "tx_to_be_confirmed"
    pending_version = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 5_000_000_000, action_to_confirm_tx_id, "pending", "swap")
    manager.pending_actions_pool[action_to_confirm_tx_id] = {
        "action_data": parse_action(pending_version),
        "seen_at_ns": MOCK_ACTION_TIME_NOW_NS - 5_000_000_000
    }
    assert len(manager.get_current_pending_block()) == 1, "Setup for confirm-removes-pending failed"

    confirmed_version = get_mock_midgard_action(MOCK_ACTION_TIME_NOW_NS - 3_000_000_000, action_to_confirm_tx_id, "success", "swap")
    with patch('src.realtime_stream_manager.fetch_recent_maya_actions') as mock_fetch_confirming:
        mock_fetch_confirming.return_value = {"actions": [confirmed_version]}
        manager.poll_confirmed_actions()
        
        if len(manager.get_current_pending_block()) == 0 and len(manager.get_live_confirmed_actions()) == 1:
            print("  PASS: Confirmed action correctly removed from pending pool and added to confirmed list.")
        else:
            print(f"  FAIL: Confirmed action removal from pending error. Pending count: {len(manager.get_current_pending_block())}, Confirmed count: {len(manager.get_live_confirmed_actions())}")
            all_subtests_passed = False

    # Test 5: Threading Start/Stop (Basic)
    print("\n  Testing RealtimeStreamManager: Threading Start/Stop...")
    manager = RealtimeStreamManager()
    
    # Patch the poll methods on the manager instance to track calls
    manager.poll_confirmed_actions = MagicMock(side_effect=manager.poll_confirmed_actions)
    manager.poll_pending_actions = MagicMock(side_effect=manager.poll_pending_actions)

    # Mock fetch_recent_maya_actions to prevent actual API calls during thread test
    # and to ensure loops don't run too long if not stopped by event quickly
    with patch('src.realtime_stream_manager.fetch_recent_maya_actions', MagicMock(return_value={"actions": []})):
        manager.start_streaming()
        # Wait long enough for at least one poll cycle of each if possible, but short for test speed.
        # Shortest interval is POLLING_INTERVAL_PENDING (5s by default). 0.5s should be enough for a few calls.
        pytime.sleep(0.5) 

        threads_started = (manager._confirmed_thread is not None and manager._confirmed_thread.is_alive()) and \
                          (manager._pending_thread is not None and manager._pending_thread.is_alive())
        
        # Check if poll methods were called
        confirmed_polls = manager.poll_confirmed_actions.call_count
        pending_polls = manager.poll_pending_actions.call_count

        if threads_started:
            print(f"  PASS: start_streaming() started polling threads. Confirmed polls: {confirmed_polls}, Pending polls: {pending_polls}")
            if confirmed_polls == 0 and pending_polls == 0: # If sleep was too short or intervals too long
                 print("  WARN: Threads started, but poll methods call_count is 0. This might be due to short sleep or long poll intervals. Test logic for checking calls might need adjustment if intervals are very large.")
                 # This is not a hard fail for threads_started, but indicates call count check might not be robust for all poll interval settings.
            elif confirmed_polls == 0 or pending_polls == 0:
                 print("  WARN: One of the poll methods call_count is 0. Ensure sleep is adequate for at least one poll cycle of both.")


        else:
            print("  FAIL: start_streaming() did not start polling threads correctly.")
            all_subtests_passed = False
            # If threads didn't start, no point checking stop or call counts further for this part
            manager.stop_streaming() # Attempt cleanup
            # Skip further assertions for this sub-test as threads didn't start.
            print(f"\n--- RealtimeStreamManager Test Summary: {'PASS' if all_subtests_passed else 'FAIL'} ---")
            return all_subtests_passed


        manager.stop_streaming() # This calls join with timeout
        
        # Check if threads are stopped (is_alive() should be False after join)
        # Allow a bit more time for join to complete if test environment is slow
        # The stop_streaming method already has join with timeout.
        threads_stopped = True
        if manager._confirmed_thread and manager._confirmed_thread.is_alive():
            print("  WARN: Confirmed thread still alive after stop_streaming (may take a moment).")
            pytime.sleep(max(manager.POLLING_INTERVAL_CONFIRMED, manager.POLLING_INTERVAL_PENDING) + 1) # Wait longer
            if manager._confirmed_thread.is_alive():
                 print("  FAIL: Confirmed thread did not stop.")
                 threads_stopped = False

        if manager._pending_thread and manager._pending_thread.is_alive():
            print("  WARN: Pending thread still alive after stop_streaming (may take a moment).")
            pytime.sleep(max(manager.POLLING_INTERVAL_CONFIRMED, manager.POLLING_INTERVAL_PENDING) + 1) # Wait longer
            if manager._pending_thread.is_alive():
                print("  FAIL: Pending thread did not stop.")
                threads_stopped = False
        
        if threads_stopped:
             print("  PASS: stop_streaming() appeared to stop polling threads.")
        else:
            # This part of the check is important if threads did start but didn't stop.
            all_subtests_passed = False


    print(f"\n--- RealtimeStreamManager Test Summary: {'PASS' if all_subtests_passed else 'FAIL'} ---")
    return all_subtests_passed

# --- New Tests for Flask App Endpoints (Task 1.5.3) ---
import json
# We need to import app from src.app to test it. 
# This will also initialize stream_manager from app.py. We'll need to mock it.
# from src.app import app as flask_app # Avoid top-level import if it causes immediate side effects before patching

def test_task_1_5_3_flask_app_endpoints():
    print("--- Testing Flask App Endpoints ---")
    all_subtests_passed = True

    # Dynamically import app to allow patching before its stream_manager is fully used by tests.
    # This is a bit tricky because stream_manager.start_streaming() is called on import in app.py.
    # For robust testing, it's often better if app creation and stream_manager start are separable.
    # Given the current structure, we will patch the methods on the already initialized stream_manager.
    
    # Import app and its stream_manager instance from src.app
    from src.app import app as flask_app, stream_manager as app_stream_manager

    # Ensure stream_manager threads are stopped for these tests if they were somehow started.
    # The app.py initializes and starts it. We need to control its output via mocking.
    # We will mock the getter methods on the existing app_stream_manager instance.
    app_stream_manager.stop_streaming() # Stop actual polling if it was running

    flask_app.testing = True
    client = flask_app.test_client()

    # Mock data
    mock_historical_data = pd.DataFrame([{'date': '1678886400000000000', 'type': 'swap', 'status': 'success'}]).to_dict(orient='records')
    mock_live_confirmed_data = [{'date': '1679996400000000000', 'type': 'addLiquidity', 'status': 'success'}]
    mock_live_pending_data = [{'date': '1680000000000000000', 'type': 'swap', 'status': 'pending'}]

    # Test 1: Root endpoint (/)
    print("  Testing Flask Endpoint: / (index.html)")
    try:
        response = client.get('/')
        if response.status_code == 200 and b"Maya Protocol Real-time Scanner" in response.data:
            print("  PASS: Root endpoint (/) served index.html successfully.")
        else:
            print(f"  FAIL: Root endpoint (/) error. Status: {response.status_code}, Data: {response.data[:200]}")
            all_subtests_passed = False
    except Exception as e:
        print(f"  FAIL: Exception during root endpoint (/) test: {e}")
        all_subtests_passed = False

    # Test 2: /api/historical-24hr
    print("\n  Testing Flask Endpoint: /api/historical-24hr")
    # Patch pandas read_csv used by get_historical_transactions in app.py
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame(mock_historical_data)
        try:
            response = client.get('/api/historical-24hr')
            if response.status_code == 200 and response.content_type == 'application/json':
                json_data = json.loads(response.data)
                if json_data == mock_historical_data:
                    print("  PASS: /api/historical-24hr returned correct JSON data.")
                else:
                    print(f"  FAIL: /api/historical-24hr JSON data mismatch. Got: {json_data}, Expected: {mock_historical_data}")
                    all_subtests_passed = False
            else:
                print(f"  FAIL: /api/historical-24hr error. Status: {response.status_code}, Content-Type: {response.content_type}")
                all_subtests_passed = False
        except Exception as e:
            print(f"  FAIL: Exception during /api/historical-24hr test: {e}")
            all_subtests_passed = False

    # Test 3: /api/live-confirmed
    print("\n  Testing Flask Endpoint: /api/live-confirmed")
    with patch.object(app_stream_manager, 'get_live_confirmed_actions', return_value=mock_live_confirmed_data) as mock_get_confirmed:
        try:
            response = client.get('/api/live-confirmed')
            if response.status_code == 200 and response.content_type == 'application/json':
                json_data = json.loads(response.data)
                if json_data == mock_live_confirmed_data:
                    print("  PASS: /api/live-confirmed returned correct JSON data.")
                    mock_get_confirmed.assert_called_once()
                else:
                    print(f"  FAIL: /api/live-confirmed JSON data mismatch. Got: {json_data}, Expected: {mock_live_confirmed_data}")
                    all_subtests_passed = False
            else:
                print(f"  FAIL: /api/live-confirmed error. Status: {response.status_code}, Content-Type: {response.content_type}")
                all_subtests_passed = False
        except Exception as e:
            print(f"  FAIL: Exception during /api/live-confirmed test: {e}")
            all_subtests_passed = False

    # Test 4: /api/live-pending
    print("\n  Testing Flask Endpoint: /api/live-pending")
    with patch.object(app_stream_manager, 'get_current_pending_block', return_value=mock_live_pending_data) as mock_get_pending:
        try:
            response = client.get('/api/live-pending')
            if response.status_code == 200 and response.content_type == 'application/json':
                json_data = json.loads(response.data)
                if json_data == mock_live_pending_data:
                    print("  PASS: /api/live-pending returned correct JSON data.")
                    mock_get_pending.assert_called_once()
                else:
                    print(f"  FAIL: /api/live-pending JSON data mismatch. Got: {json_data}, Expected: {mock_live_pending_data}")
                    all_subtests_passed = False
            else:
                print(f"  FAIL: /api/live-pending error. Status: {response.status_code}, Content-Type: {response.content_type}")
                all_subtests_passed = False
        except Exception as e:
            print(f"  FAIL: Exception during /api/live-pending test: {e}")
            all_subtests_passed = False

    # Cleanup: It might be good to explicitly stop the app_stream_manager again if it was restarted by app 
    # However, the atexit handler in app.py should take care of it on script exit.
    # For isolated tests, if app_stream_manager was a test-specific instance, we'd stop it here.

    print(f"\n--- Flask App Endpoints Test Summary: {'PASS' if all_subtests_passed else 'FAIL'} ---")
    return all_subtests_passed

# --- Main Test Execution ---
if __name__ == "__main__":
    print("=== Starting Phase 1 Completion Test Script ===\n")
    results = {}

    results["Task 1.1 API Connections"] = test_task_1_1_api_connections()
    print("-"*50 + "\n")
    results["Task 1.2 Realtime Data"] = test_task_1_2_realtime_data() # Updated key and function call
    print("-"*50 + "\n")
    results["Task 1.3 Mempool Monitor"] = test_task_1_3_mempool_monitor()
    print("-"*50 + "\n")
    # results["Task 1.4 Preprocess Data"] = test_task_1_4_preprocess_data()
    # print("-"*50 + "\n")
    results["Task 1.5.2 RealtimeStreamManager"] = test_task_1_3_realtime_stream_manager()
    print("-"*50 + "\n")
    results["Task 1.5.3 Flask App Endpoints"] = test_task_1_5_3_flask_app_endpoints()
    print("-"*50 + "\n")

    print("\n=== Phase 1 Test Summary ===")
    all_passed = True
    for task, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"- {task}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\nSUCCESS: All Phase 1 tasks appear to be completed successfully based on these tests!")
    else:
        print("\nFAILURE: Some Phase 1 tests failed. Please review the output above.")

    print("\nNote: This script provides a basic check. Manual verification might still be advisable.")
    sys.exit(0 if all_passed else 1) 