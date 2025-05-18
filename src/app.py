from flask import Flask, jsonify, render_template
import pandas as pd
import os
import atexit

# Import the RealtimeStreamManager
from realtime_stream_manager import RealtimeStreamManager

app = Flask(__name__)

# --- Data File Configuration ---
HISTORICAL_DATA_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_24hr_maya_transactions.csv')

# --- Initialize and Start Realtime Stream Manager ---
stream_manager = RealtimeStreamManager()
stream_manager.start_streaming() # Start polling threads

# --- Register shutdown hook for the stream manager ---
@atexit.register
def shutdown_stream_manager():
    print("Flask app shutting down. Stopping stream manager...")
    stream_manager.stop_streaming()
    print("Stream manager stopped.")

# --- Data Access Functions ---
def get_historical_transactions():
    """Loads the 24-hour historical transaction data from the CSV file."""
    try:
        print(f"Attempting to load historical data from: {HISTORICAL_DATA_CSV}", flush=True)
        if os.path.exists(HISTORICAL_DATA_CSV):
            df = pd.read_csv(HISTORICAL_DATA_CSV)
            print(f"CSV loaded. DataFrame shape: {df.shape}", flush=True)
            if df.empty:
                print("DataFrame is empty after loading CSV.", flush=True)
                return []
            
            # Handle potential NaN values by converting them to None, which is JSON null
            # This is important because pandas.NA, np.nan, etc., are not directly JSON serializable by default in all contexts.
            # Flask's jsonify usually handles np.nan by converting to null, but being explicit can help.
            df = df.astype(object) # Convert all columns to object type to better handle mixed types and NaNs
            df_filled = df.where(pd.notnull(df), None)
            
            data = df_filled.to_dict(orient='records')
            # print(f"Data prepared for JSON (first 1 record if exists): {str(data[:1])[:500]}...", flush=True) # Potentially very long
            print(f"Successfully prepared {len(data)} records from historical CSV for JSON.", flush=True)
            return data
        else:
            print(f"Historical data CSV not found: {HISTORICAL_DATA_CSV}", flush=True)
            return [] # Return empty list if file doesn't exist
    except Exception as e:
        print(f"Error reading or processing historical data CSV: {e}", flush=True)
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
    # Note: app.run(debug=True) can cause the script to run twice in debug mode,
    # which might initialize the stream_manager twice if not careful.
    # For production, use a proper WSGI server like gunicorn or waitress.
    # For development, if issues with double init, set use_reloader=False.
    app.run(debug=True, use_reloader=False) # Added use_reloader=False to prevent double init of stream_manager 