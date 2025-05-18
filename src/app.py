from flask import Flask, jsonify, render_template
import pandas as pd
import os
import atexit
import time # For nanosecond timestamps if needed directly in app
import logging # Import logging module
import subprocess # For running fetch_realtime_transactions.py
import sys # To get Python executable path

# --- Configure basic logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ---

# Import the RealtimeStreamManager
from realtime_stream_manager import RealtimeStreamManager

app = Flask(__name__)

# --- Data File Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
HISTORICAL_DATA_CSV = os.path.join(DATA_DIR, 'historical_24hr_maya_transactions.csv')
FETCH_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'fetch_realtime_transactions.py')

# --- Initialize Realtime Stream Manager (but don't start it yet) ---
stream_manager = RealtimeStreamManager()

# --- Function to ensure historical data exists ---
def ensure_historical_data_exists():
    """Checks if historical data CSV exists, if not, runs the fetch script."""
    logger.info(f"Checking for historical data file: {HISTORICAL_DATA_CSV}")
    if not os.path.exists(HISTORICAL_DATA_CSV):
        logger.warning(f"Historical data file not found. Attempting to generate it by running {FETCH_SCRIPT_PATH}...")
        try:
            # Ensure data directory exists
            if not os.path.exists(DATA_DIR):
                logger.info(f"Data directory {DATA_DIR} not found, creating it.")
                os.makedirs(DATA_DIR)
            
            # Get the path to the current Python interpreter that's running app.py
            python_executable = sys.executable
            
            # It's good practice to use the absolute path to the script if sys.executable is a full path
            # or if the script_path is not guaranteed to be found via relative paths by the subprocess's CWD.
            # Assuming FETCH_SCRIPT_PATH is correctly relative to this app.py's location, or an absolute path.
            
            logger.info(f"Running script: {python_executable} {FETCH_SCRIPT_PATH}")
            process = subprocess.run(
                [python_executable, FETCH_SCRIPT_PATH],
                capture_output=True,
                text=True,
                timeout=180,  # Increased timeout, fetching can take a while
                check=False, # We will check return code manually
                cwd=os.path.dirname(__file__) # Run fetch script from its own directory or project root if appropriate
            )
            if process.returncode == 0:
                logger.info(f"Successfully ran {FETCH_SCRIPT_PATH} to generate historical data.")
                logger.debug(f"Fetch script stdout:\n{process.stdout}")
            else:
                logger.error(f"Failed to run {FETCH_SCRIPT_PATH}. Return code: {process.returncode}")
                logger.error(f"Fetch script stderr:\n{process.stderr}")
                logger.error(f"Fetch script stdout:\n{process.stdout}")
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout expired while running {FETCH_SCRIPT_PATH}.")
        except Exception as e:
            logger.error(f"An error occurred while trying to run {FETCH_SCRIPT_PATH}: {e}", exc_info=True)
    else:
        logger.info("Historical data file already exists.")

# --- Register shutdown hook for the stream manager --- 
@atexit.register
def shutdown_app_services():
    logger.info("Flask app shutting down...")
    # Stop Stream Manager
    if stream_manager:
        logger.info("Stopping RealtimeStreamManager...")
        stream_manager.stop_streaming()
        logger.info("RealtimeStreamManager stopped.")
    
    # Clear historical data if it exists
    # As per user request: "it should be deleted / cleared after the server closes"
    if os.path.exists(HISTORICAL_DATA_CSV):
        try:
            logger.info(f"Attempting to delete historical data file: {HISTORICAL_DATA_CSV}")
            os.remove(HISTORICAL_DATA_CSV)
            logger.info(f"Successfully deleted {HISTORICAL_DATA_CSV}.")
        except Exception as e:
            logger.error(f"Error deleting {HISTORICAL_DATA_CSV} on shutdown: {e}", exc_info=True)
    else:
        logger.info(f"Historical data file {HISTORICAL_DATA_CSV} not found at shutdown, nothing to delete.")

# --- Data Access Functions ---
def get_historical_transactions():
    """Loads the 24-hour historical transaction data from the CSV file."""
    logger.debug(f"HISTORICAL_DATA_CSV value: {HISTORICAL_DATA_CSV}")
    try:
        logger.info(f"Attempting to load historical data from: {HISTORICAL_DATA_CSV}")
        if os.path.exists(HISTORICAL_DATA_CSV):
            df = pd.read_csv(HISTORICAL_DATA_CSV)
            logger.info(f"CSV loaded. DataFrame shape: {df.shape}")
            if df.empty:
                logger.info("DataFrame is empty after loading CSV.")
                return []
            
            # Handle potential NaN values by converting them to None, which is JSON null
            # This is important because pandas.NA, np.nan, etc., are not directly JSON serializable by default in all contexts.
            # Flask's jsonify usually handles np.nan by converting to null, but being explicit can help.
            df = df.astype(object) # Convert all columns to object type to better handle mixed types and NaNs
            df_filled = df.where(pd.notnull(df), None)
            
            data = df_filled.to_dict(orient='records')
            # print(f"Data prepared for JSON (first 1 record if exists): {str(data[:1])[:500]}...", flush=True) # Potentially very long
            logger.info(f"Successfully prepared {len(data)} records from historical CSV for JSON.")
            return data
        else:
            logger.warning(f"Historical data CSV not found: {HISTORICAL_DATA_CSV}")
            return [] # Return empty list if file doesn't exist
    except pd.errors.EmptyDataError:
        logger.warning(f"Historical data CSV is completely empty (no headers): {HISTORICAL_DATA_CSV}")
        return [] # Return empty list if file is empty and causes pandas error
    except Exception as e:
        logger.error(f"Error reading or processing historical data CSV: {e}", exc_info=True)
        # import traceback
        # traceback.print_exc()
        return [] # Return empty list on error

# --- Flask API Endpoints ---
@app.route('/')
def serve_index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/ping')
def ping():
    """A simple ping endpoint to check if Flask is responding."""
    return "pong", 200

@app.route('/api/historical-24hr')
def api_historical_transactions():
    """API endpoint to get 24-hour historical transaction data."""
    data = get_historical_transactions()
    return jsonify(data)

@app.route('/api/live-confirmed')
def api_live_confirmed_transactions():
    """API endpoint to get live confirmed transaction data."""
    data = stream_manager.get_live_confirmed_actions()
    return jsonify(data)

@app.route('/api/live-pending')
def api_live_pending_transactions():
    """API endpoint to get live pending transaction data."""
    data = stream_manager.get_current_pending_block()
    return jsonify(data)

if __name__ == '__main__':
    logger.info("Initializing Flask application...")
    
    # Step 1: Ensure historical data is present (generate if needed)
    ensure_historical_data_exists()
    
    # Step 2: Start the RealtimeStreamManager polling threads
    logger.info("Starting RealtimeStreamManager polling threads...")
    stream_manager.start_streaming()
    logger.info("RealtimeStreamManager streaming started.")
    
    # Step 3: Run the Flask app
    logger.info("Starting Flask app server (debug=True, use_reloader=False)...")
    app.run(debug=True, use_reloader=False) 