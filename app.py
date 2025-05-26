import os
import sys
from flask import Flask, render_template, jsonify, request
import sqlite3
import time # Import time module

# Ensure src is in path to import database_utils
_APP_PY_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming app.py is in the project root, and database_utils is in mayarbscanner/src/
_PROJECT_ROOT_FOR_APP = _APP_PY_CURRENT_DIR 
# Construct path to src directory correctly
_SRC_DIR_PATH = os.path.join(_PROJECT_ROOT_FOR_APP, 'src')

if _SRC_DIR_PATH not in sys.path:
    sys.path.insert(0, _SRC_DIR_PATH)

try:
    from database_utils import (
        get_db_connection, 
        get_latest_blocks_with_details,
        get_blocks_since_height
    )
    from api_connections import fetch_decoded_tendermint_mempool_txs # For Phase 2
    # Check if protobufs are available for mempool decoding (used by fetch_decoded_tendermint_mempool_txs)
    from common_utils import PROTO_TYPES_AVAILABLE as COMMON_UTILS_PROTOBUF_AVAILABLE
except ImportError as e:
    print(f"Error: Could not import necessary modules: {e}")
    print(f"PROJECT_ROOT_FOR_APP: {_PROJECT_ROOT_FOR_APP}")
    print(f"Attempted SRC_DIR_PATH: {_SRC_DIR_PATH}")
    print(f"Current sys.path: {sys.path}")
    # Fallback if running from within src directory (e.g., during some tests, though not typical for app.py)
    if os.path.basename(_APP_PY_CURRENT_DIR) == "src":
        _PROJECT_ROOT_FOR_APP_FALLBACK = os.path.dirname(_APP_PY_CURRENT_DIR)
        if _PROJECT_ROOT_FOR_APP_FALLBACK not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT_FOR_APP_FALLBACK)
            print(f"Fallback: Added {_PROJECT_ROOT_FOR_APP_FALLBACK} to sys.path")
            try:
                from database_utils import (
                    get_db_connection, 
                    get_latest_blocks_with_details,
                    get_blocks_since_height
                )
                from api_connections import fetch_decoded_tendermint_mempool_txs
                from common_utils import PROTO_TYPES_AVAILABLE as COMMON_UTILS_PROTOBUF_AVAILABLE
            except ImportError as e2:
                print(f"Fallback import also failed: {e2}")
                sys.exit(1) # Exit if critical imports fail
    else:
        sys.exit(1) # Exit if critical imports fail

app = Flask(__name__)
DATABASE_FILE = 'mayanode_blocks.db' # Make sure this matches your DB file name

@app.route('/')
@app.route('/latest-blocks')
def latest_blocks_page():
    """Serves the main page which will dynamically load block data."""
    return render_template('latest_blocks.html')

@app.route('/api/latest-blocks-data')
def api_latest_blocks_data():
    """API endpoint to get the latest 10 blocks as JSON."""
    route_start_time = time.time()
    # print(f"[API /api/latest-blocks-data] Request received at {route_start_time}") # Optional: keep for basic request logging
    conn = None
    try:
        conn_start_time = time.time()
        conn = get_db_connection(DATABASE_FILE)
        conn_end_time = time.time()
        # print(f"[API /api/latest-blocks-data] DB connection established in {conn_end_time - conn_start_time:.4f} seconds.") # Removed

        fetch_details_start_time = time.time()
        latest_blocks = get_latest_blocks_with_details(conn, limit=10)
        fetch_details_end_time = time.time()
        # print(f"[API /api/latest-blocks-data] get_latest_blocks_with_details took {fetch_details_end_time - fetch_details_start_time:.4f} seconds.") # Removed

        jsonify_start_time = time.time()
        response = jsonify(latest_blocks)
        jsonify_end_time = time.time()
        # print(f"[API /api/latest-blocks-data] jsonify took {jsonify_end_time - jsonify_start_time:.4f} seconds.") # Removed
        
        route_end_time = time.time()
        # print(f"[API /api/latest-blocks-data] Total processing time for request: {route_end_time - route_start_time:.4f} seconds.") # Optional: keep or remove
        return response
    except sqlite3.Error as e:
        print(f"Database error in /api/latest-blocks-data: {e}")
        return jsonify({"error": "Database error", "details": str(e)}), 500
    except Exception as e:
        print(f"Unexpected error in /api/latest-blocks-data: {e}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/blocks-since/<int:from_height>')
def api_blocks_since(from_height):
    """API endpoint to get all blocks with height greater than from_height."""
    # print(f"[API /api/blocks-since/{from_height}] Request received")
    conn = None
    try:
        conn = get_db_connection(DATABASE_FILE)
        # The get_blocks_since_height function returns blocks sorted ASC by height
        new_blocks = get_blocks_since_height(conn, from_height)
        # print(f"[API /api/blocks-since/{from_height}] Found {len(new_blocks)} new blocks.")
        return jsonify(new_blocks)
    except sqlite3.Error as e:
        print(f"Database error in /api/blocks-since/{from_height}: {e}")
        return jsonify({"error": "Database error", "details": str(e)}), 500
    except Exception as e:
        print(f"Unexpected error in /api/blocks-since/{from_height}: {e}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/mempool')
def api_mempool_data():
    """API endpoint to get current mempool data as JSON."""
    route_start_time = time.time()
    # print(f"[API /api/mempool] Request received at {route_start_time}") # Can be verbose

    if not COMMON_UTILS_PROTOBUF_AVAILABLE:
        print("Warning: /api/mempool called, but protobufs for decoding are not available via common_utils.")
        return jsonify({"error": "Mempool decoding not available due to missing protobufs components."}), 503 # Service Unavailable

    try:
        fetch_start_time = time.time()
        # fetch_decoded_tendermint_mempool_txs typically returns a list of dicts (decoded txs)
        # or a list of strings (base64 if internal decoding failed or was skipped).
        # It might return None on total fetch failure from Tendermint RPC.
        mempool_txs = fetch_decoded_tendermint_mempool_txs(limit=50) # Fetch a reasonable number for display
        fetch_end_time = time.time()
        # print(f"[API /api/mempool] fetch_decoded_tendermint_mempool_txs took {fetch_end_time - fetch_start_time:.4f} seconds.") # Removed

        if mempool_txs is None:
            # This indicates an issue fetching from the Tendermint RPC endpoint itself.
            print(f"[API /api/mempool] Error: fetch_decoded_tendermint_mempool_txs returned None. Cannot connect or fetch from RPC.")
            return jsonify({"error": "Failed to fetch mempool data from Mayanode (Tendermint RPC)."}), 502 # Bad Gateway

        # The function `fetch_decoded_tendermint_mempool_txs` already attempts decoding.
        # The result `mempool_txs` will be a list of dictionaries if successful, 
        # or a list of base64 strings if protobuf decoding failed internally for some items.
        # If `COMMON_UTILS_PROTOBUF_AVAILABLE` was false at the time `api_connections` was imported, 
        # then `fetch_decoded_tendermint_mempool_txs` would likely return raw base64 strings directly.

        jsonify_start_time = time.time()
        response = jsonify(mempool_txs)
        jsonify_end_time = time.time()
        # print(f"[API /api/mempool] jsonify took {jsonify_end_time - jsonify_start_time:.4f} seconds.") # Removed

        route_end_time = time.time()
        # print(f"[API /api/mempool] Total processing time for request: {route_end_time - route_start_time:.4f} seconds.") # Removed
        return response
        
    except Exception as e:
        print(f"[API /api/mempool] Unexpected error: {e}")
        # Log the full traceback for server-side debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An unexpected error occurred while fetching mempool data", "details": str(e)}), 500

if __name__ == '__main__':
    print("Checking if protobufs are available for mempool decoding before starting Flask app...")
    if not COMMON_UTILS_PROTOBUF_AVAILABLE:
        print("WARNING: CosmosTx protobuf object NOT available via common_utils.")
        print("The /api/mempool endpoint will likely fail or return undecoded data.")
        print("Ensure protobufs are compiled and 'common_utils.py' can import 'CosmosTx'.")
    else:
        print("Protobufs for mempool decoding appear to be available.")
    
    app.run(debug=True, host='0.0.0.0', port=5001) 