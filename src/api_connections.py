import requests
import json
import time
from datetime import datetime, timezone # Added timezone

import sys
import os

# --- Adjust sys.path for protobuf imports ---
# This ensures that the generated protobuf stubs can be found.
# The stubs are expected to be in 'proto/generated/pb_stubs/' relative to the project root.

# Get the directory of the current script (e.g., mayarbscanner/src/api_connections.py)
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (e.g., 'mayarbscanner/') by going up one level from 'src/'
_project_root = os.path.dirname(_current_script_dir)
# Path to the parent of pb_stubs, i.e., 'proto/generated/'
_proto_generated_path = os.path.join(_project_root, "proto", "generated")

# Prepend the stubs path to sys.path if it's not already there
if _proto_generated_path not in sys.path:
    sys.path.insert(0, _proto_generated_path)
    # print(f"[api_connections.py] Prepended to sys.path for proto stubs (via proto/generated/): {_proto_generated_path}")

# Try to import CosmosTx after adjusting path
try:
    # Import Tx from the __init__.py of pb_stubs.cosmos.tx.v1beta1
    from pb_stubs.cosmos.tx.v1beta1 import Tx as CosmosTx
    PROTOBUF_COSMOS_TX_AVAILABLE = True
    # print("[api_connections.py] Successfully imported CosmosTx (from pb_stubs.cosmos.tx.v1beta1 after sys.path mod to proto/generated/).")
except ImportError as e:
    print(f"[api_connections.py] ImportError during protobuf setup (from pb_stubs.cosmos.tx.v1beta1): {e}") # Keep this error print
    print(f"[api_connections.py] Warning: CosmosTx not available. Ensure stubs exist and check package structure in '{os.path.join(_proto_generated_path, 'pb_stubs', 'cosmos', 'tx', 'v1beta1')}'.") # Keep
    PROTOBUF_COSMOS_TX_AVAILABLE = False

# Import the centralized decoder and its availability flag from common_utils
# This will be used by functions in this module that need to decode.
if _project_root not in sys.path: # Ensure project root is in path for src import
    sys.path.insert(0, _project_root)
try:
    from src.common_utils import decode_cosmos_tx_string_to_dict, PROTO_TYPES_AVAILABLE as COMMON_UTILS_PROTOBUF_AVAILABLE
    # print("[api_connections.py] Successfully imported decode_cosmos_tx_string_to_dict and PROTO_TYPES_AVAILABLE from src.common_utils.")
except ImportError as e_cu:
    print(f"[api_connections.py] Failed to import from src.common_utils: {e_cu}. Decoding will likely fail.") # Keep this error print
    # Define fallbacks if common_utils import fails, so script doesn't crash on undefined names
    def decode_cosmos_tx_string_to_dict(tx_b64_string, default_hrp="maya"):
        print("Critical Error: common_utils.decode_cosmos_tx_string_to_dict is not available!") # Keep
        return tx_b64_string # Return original string
    COMMON_UTILS_PROTOBUF_AVAILABLE = False

# Configuration
MAYANODE_BASE_URL = "https://mayanode.mayachain.info/mayachain"
# TENDERMINT_RPC_BASE_URL = "https://rpc.mayachain.info"       # For Tendermint RPC (mempool, etc.) - This failed NameResolution
TENDERMINT_RPC_BASE_URL = "https://tendermint.mayachain.info" # Developer provided, trying HTTPS first
# --- Placeholder for Tendermint RPC Base URL until a working one is identified ---
# TENDERMINT_RPC_BASE_URL = "" # Set to a valid URL when known

# --- Helper Function (similar to previous, can be adapted if needed) ---
def fetch_api_data(url, api_name, params=None):
    """Fetches data from a given API URL and prints status."""
    # print(f"--- Querying {api_name} API ---")
    # print(f"URL: {url}")
    # if params:
    #     print(f"Params: {params}")
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        # print(f"Successfully connected and got response from {api_name}.")

        try:
            data = response.json()
            # For block data, the response can be very large.
            # We will avoid printing samples here to keep the output clean.
            # Callers can decide how to handle/log the data.
            return data
        except requests.exceptions.JSONDecodeError:
            print("Error: Response was not in JSON format, or content is empty.") # Keep for errors
            print("Response text (first 500 chars):", response.text[:500]) # Keep for errors
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to {api_name} ({url}): {e}") # Keep for errors
        return None
    # finally: # Removing the separator print as well
        # print("-" * 40 + "\n")

# --- Mayanode API Functions ---

def fetch_mayanode_block(height=None):
    """Fetches a specific block by height from Mayanode, or the latest block if height is None."""
    endpoint = "/block"
    params = {}
    api_description = "Mayanode API for "

    if height is not None:
        try:
            params["height"] = int(height)
            api_description += f"Block {height}"
        except ValueError:
            print(f"Error: Invalid height '{height}'. Must be an integer.") # Keep
            return None
    else:
        api_description += "Latest Block"
        # No specific parameter for latest, the API should return latest if height is omitted.
    
    url = f"{MAYANODE_BASE_URL}{endpoint}"
    
    response_data = fetch_api_data(url, api_description, params=params)
    
    # Check for the JSON-RPC wrapper structure first
    if response_data and isinstance(response_data, dict) and "result" in response_data and \
       isinstance(response_data["result"], dict) and "block" in response_data["result"] and \
       isinstance(response_data["result"]["block"], dict) and "header" in response_data["result"]["block"] and \
       "data" in response_data["result"]["block"]:
        
        block_content = response_data["result"]["block"]
        block_id_info = response_data["result"].get("block_id", {}) # Get block_id separately
        
        reconstructed_for_parser = {
            "id": block_id_info,
            "header": block_content.get("header"),
            "data": block_content.get("data"),
            "evidence": block_content.get("evidence"),
            "last_commit": block_content.get("last_commit"),
            "begin_block_events": response_data["result"].get("begin_block_events"), 
            "end_block_events": response_data["result"].get("end_block_events"),     
        }

        fetched_height_val = reconstructed_for_parser.get("header", {}).get("height")
        # print(f"Successfully fetched and processed RPC block data. Reported height: {fetched_height_val}")
        if height is not None and str(height) != str(fetched_height_val): # Compare as string as API returns string
            print(f"Warning: Requested height {height} but block data reports height {fetched_height_val}.") # Keep
            pass
        
        return reconstructed_for_parser

    elif response_data and isinstance(response_data, dict) and "header" in response_data and "id" in response_data: 
        fetched_height_val = response_data.get("header", {}).get("height") 
        # print(f"Successfully fetched direct block data. Reported height: {fetched_height_val}")
        if height is not None and str(height) != str(fetched_height_val): # Compare as string
            print(f"Warning: Requested height {height} but block data reports height {fetched_height_val}.") # Keep
            pass
        
        return response_data
    elif response_data:
        print("Error: Fetched data, but it does not match expected BlockResponse structure (missing key fields or not matching RPC wrapper).") # Keep
        print("Sample of unexpected data (first 500 chars):", str(response_data)[:500]) # Keep
        return None
    else:
        # print(f"Failed to fetch block data for {'height ' + str(height) if height else 'latest'}.") # Commented out
        return None

def get_mayanode_latest_block_height():
    """Fetches the latest block from Mayanode and returns its height."""
    # print("Attempting to determine latest block height from Mayanode...")
    latest_block = fetch_mayanode_block() 
    if latest_block and isinstance(latest_block, dict):
        header = latest_block.get("header")
        if header and "height" in header:
            try:
                height_str = header["height"]
                latest_height = int(height_str)
                # print(f"Successfully determined latest block height: {latest_height}")
                return latest_height
            except ValueError:
                print(f"Error: 'header.height' ('{height_str}') is not a valid integer.") # Keep
                return None
        else:
            print("Error: 'header' or 'header.height' not found in latest block response.") # Keep
            return None
    else:
        # print("Failed to fetch latest block to determine height.") # Commented out
        return None

def fetch_mayanode_unconfirmed_txs(limit: int = 30):
    """Fetches unconfirmed transactions from the Mayanode mempool.
    The \`limit\` parameter behavior is based on standard Tendermint RPC; actual behavior may vary.
    """
    endpoint = "/unconfirmed_txs"
    params = {"limit": str(limit)} 
    api_description = f"Mayanode API for Unconfirmed Txs (limit {limit})"
    url = f"{MAYANODE_BASE_URL}{endpoint}"

    unconfirmed_txs_data = fetch_api_data(url, api_description, params=params)

    if unconfirmed_txs_data and isinstance(unconfirmed_txs_data, dict):
        if "result" in unconfirmed_txs_data and "txs" in unconfirmed_txs_data["result"]:
            txs_list = unconfirmed_txs_data["result"]["txs"]
            total_txs = unconfirmed_txs_data["result"].get("total", "N/A")
            # print(f"Successfully fetched unconfirmed transactions. Count in response: {len(txs_list if txs_list else [])}, Total in mempool: {total_txs}")
            if txs_list is None: 
                 print("Mempool is likely empty or 'txs' is null.") # Keep
                 return [] 
            return txs_list 
        else:
            print("Error: Fetched data for unconfirmed_txs, but it does not match expected structure (missing 'result.txs').") # Keep
            print("Sample of unexpected data (first 500 chars):", str(unconfirmed_txs_data)[:500]) # Keep
            return None
    elif unconfirmed_txs_data: 
        print("Error: Fetched data for unconfirmed_txs, but it is not a dictionary or is in an unexpected format.") # Keep
        print("Sample of unexpected data (first 500 chars):", str(unconfirmed_txs_data)[:500]) # Keep
        return None
    else:
        # print("Failed to fetch unconfirmed transactions.") # Commented out
        return None

# --- Tendermint RPC Functions ---

def fetch_tendermint_rpc_block_raw(height: int):
    """Fetches a raw block by height directly from the Tendermint RPC endpoint.
    This should return block data with transactions as base64 encoded strings.
    """
    if not TENDERMINT_RPC_BASE_URL:
        print("ERROR: fetch_tendermint_rpc_block_raw - TENDERMINT_RPC_BASE_URL is not set.") # Keep
        return None

    endpoint = "/block"
    params = {"height": str(height)}
    api_description = f"Tendermint RPC for Raw Block {height}"
    url = f"{TENDERMINT_RPC_BASE_URL}{endpoint}"

    response_data = fetch_api_data(url, api_description, params=params)

    if response_data and isinstance(response_data, dict) and "result" in response_data and \
       isinstance(response_data["result"], dict) and "block" in response_data["result"]:
        # print(f"Successfully fetched raw block data from Tendermint RPC for height {height}.")
        return response_data["result"] 
    elif response_data:
        print(f"Error: Fetched data from Tendermint RPC for block {height}, but it does not match expected structure.") # Keep
        print(f"Sample of unexpected data (first 500 chars): {str(response_data)[:500]}") # Keep
        return None
    else:
        # print(f"Failed to fetch raw block data from Tendermint RPC for height {height}.") # Commented out
        return None

def fetch_tendermint_rpc_block_results(height: int) -> dict | None:
    """Fetches block results by height directly from the Tendermint RPC endpoint.
    This includes begin_block_events, end_block_events, and tx_results.
    """
    if not TENDERMINT_RPC_BASE_URL:
        print("ERROR: fetch_tendermint_rpc_block_results - TENDERMINT_RPC_BASE_URL is not set.") # Keep
        return None

    endpoint = "/block_results"
    params = {"height": str(height)}
    api_description = f"Tendermint RPC for Block Results {height}"
    url = f"{TENDERMINT_RPC_BASE_URL}{endpoint}"

    response_data = fetch_api_data(url, api_description, params=params)

    if response_data and isinstance(response_data, dict) and "result" in response_data:
        if isinstance(response_data["result"], dict) and "height" in response_data["result"]:
            if response_data["result"]["height"] == str(height):
                # print(f"Successfully fetched Tendermint block results for height {height}.")
                return response_data["result"] 
            else:
                print(f"Warning: Requested block results for height {height}, but response is for height {response_data['result']['height']}.") # Keep
                return response_data["result"]
        else:
            print(f"Error: Tendermint block results for height {height} does not have a valid 'result' dictionary with 'height'.") # Keep
            print("Sample of unexpected result (first 500 chars):", str(response_data.get("result"))[:500]) # Keep
            return None
    elif response_data: 
        print(f"Error: Unexpected structure for Tendermint block results for height {height}.") # Keep
        print("Sample of unexpected data (first 500 chars):", str(response_data)[:500]) # Keep
        return None
    else:
        # print(f"Failed to fetch Tendermint block results for height {height}.") # Commented out
        return None

def fetch_decoded_tendermint_mempool_txs(limit: int = 100): 
    """Fetches unconfirmed transactions from the Tendermint RPC /unconfirmed_txs endpoint
    and decodes them into dictionaries.
    Fetches up to the specified limit (max typically 100 for public Tendermint RPCs).
    """
    if not TENDERMINT_RPC_BASE_URL:
        print("WARNING: fetch_decoded_tendermint_mempool_txs - TENDERMINT_RPC_BASE_URL is not set.") # Keep
        print("This function will return an empty list. Update when a valid endpoint is known.") # Keep
        return [] 

    endpoint = "/unconfirmed_txs"
    params = {"limit": str(limit)} if limit is not None else {}
    api_description = f"Tendermint RPC for Unconfirmed Txs (limit {limit if limit is not None else 'default'})"
    url = f"{TENDERMINT_RPC_BASE_URL}{endpoint}"

    response_data = fetch_api_data(url, api_description, params=params)
    decoded_tx_list = []

    if response_data and isinstance(response_data, dict):
        if "result" in response_data and isinstance(response_data["result"], dict) and "txs" in response_data["result"]:
            txs_base64_list = response_data["result"]["txs"]
            n_txs = response_data["result"].get("n_txs", "N/A") 
            total_txs = response_data["result"].get("total", "N/A") 
            # print(f"Successfully fetched unconfirmed transaction strings. Count in response: {n_txs}, Total in mempool: {total_txs}")
            if n_txs != total_txs:
                print(f"WARNING: Fetched {n_txs} transaction strings, but mempool reports {total_txs} total. Endpoint may have a limit.") # Keep for now
                pass
            
            if txs_base64_list is None: 
                 print("Mempool is likely empty or 'txs' is null in response.") # Keep
                 return [] 

            if COMMON_UTILS_PROTOBUF_AVAILABLE: 
                # print(f"Decoding {len(txs_base64_list)} transaction strings using common_utils.decode_cosmos_tx_string_to_dict...")
                for i, tx_b64_string in enumerate(txs_base64_list):
                    decoded_tx = decode_cosmos_tx_string_to_dict(tx_b64_string)
                    if isinstance(decoded_tx, dict):
                        decoded_tx_list.append(decoded_tx)
                    else:
                        print(f"Warning: Tx index {i} failed to decode into a dict. Original b64 (preview): {tx_b64_string[:50]}...") # Keep
                        pass
                # print(f"Successfully decoded {len(decoded_tx_list)} transactions out of {len(txs_base64_list)} strings.")
                return decoded_tx_list
            else:
                print("Warning: Protobuf decoding not available via common_utils. Returning raw base64 strings instead.") # Keep
                return txs_base64_list 
        else:
            print("Error: Fetched data for unconfirmed_txs, but it does not match expected Tendermint structure (missing 'result.txs').") # Keep
            print("Full response (first 500 chars):", str(response_data)[:500]) # Keep
            return None 
    elif response_data: 
        print("Error: Fetched data for unconfirmed_txs, but it is not a dictionary or is in an unexpected format.") # Keep
        print("Sample of unexpected data (first 500 chars):", str(response_data)[:500]) # Keep
        return None
    else:
        # print("Failed to fetch unconfirmed transactions (no data returned from fetch_api_data).") # Commented out
        return None

def get_tendermint_num_unconfirmed_txs():
    """Fetches the number of unconfirmed transactions from Tendermint RPC /num_unconfirmed_txs endpoint."""
    if not TENDERMINT_RPC_BASE_URL:
        print("WARNING: get_tendermint_num_unconfirmed_txs - TENDERMINT_RPC_BASE_URL is not set.") # Keep
        print("This function will return 0. Update when a valid endpoint is known.") # Keep
        return 0

    endpoint = "/num_unconfirmed_txs"
    api_description = "Tendermint RPC for Num Unconfirmed Txs"
    url = f"{TENDERMINT_RPC_BASE_URL}{endpoint}"

    response_data = fetch_api_data(url, api_description)

    if response_data and isinstance(response_data, dict):
        if "result" in response_data and isinstance(response_data["result"], dict):
            result_data = response_data["result"]
            n_txs = result_data.get("n_txs", "N/A")
            total_txs = result_data.get("total", "N/A")
            total_bytes = result_data.get("total_bytes", "N/A")
            # print(f"Successfully fetched num_unconfirmed_txs. In response: {n_txs}, Total in mempool: {total_txs}, Total bytes: {total_bytes}")
            try:
                return int(total_txs) if total_txs != "N/A" else (int(n_txs) if n_txs != "N/A" else 0)
            except ValueError:
                print(f"Error: Could not parse count from num_unconfirmed_txs response: n_txs='{n_txs}', total='{total_txs}'") # Keep
                return None
        else:
            print("Error: Fetched data for num_unconfirmed_txs, but it does not match expected Tendermint structure (missing 'result').") # Keep
            print("Full response (first 500 chars):", str(response_data)[:500]) # Keep
            return None
    elif response_data:
        print("Error: Fetched data for num_unconfirmed_txs, but it is not a dictionary or is in an unexpected format.") # Keep
        print("Sample of unexpected data (first 500 chars):", str(response_data)[:500]) # Keep
        return None
    else:
        # print("Failed to fetch num_unconfirmed_txs (no data returned from fetch_api_data).") # Commented out
        return None

def construct_next_block_template():
    """Constructs a template for the next block using the latest confirmed block info
    and unconfirmed transactions from the mempool.

    Note: Transactions in the template are raw base64 strings and are NOT yet ordered by slip/fee.
    """
    # print("\n--- Constructing Next Block Template ---")

    latest_confirmed_block = fetch_mayanode_block() 
    if not latest_confirmed_block:
        print("Error: Could not fetch latest confirmed block to build template. Aborting.") # Keep
        return None

    chain_id = latest_confirmed_block.get("header", {}).get("chain_id", "mayachain-mainnet-v1") 
    last_block_id_hash = latest_confirmed_block.get("id", {}).get("hash", "UNKNOWN_LAST_BLOCK_HASH")
    last_block_id_parts_hash = latest_confirmed_block.get("id", {}).get("parts", {}).get("hash", "UNKNOWN_LAST_BLOCK_PARTS_HASH")
    last_block_id_parts_total = latest_confirmed_block.get("id", {}).get("parts", {}).get("total", 0)
    last_confirmed_height_str = latest_confirmed_block.get("header", {}).get("height", "0")
    try:
        next_block_height_val = int(last_confirmed_height_str) + 1
    except ValueError:
        print(f"Warning: Could not parse last confirmed height '{last_confirmed_height_str}'. Using placeholder for next height.") # Keep
        next_block_height_val = "PENDING_CALCULATION"

    unconfirmed_txs = fetch_decoded_tendermint_mempool_txs(limit=100) 
    if unconfirmed_txs is None: 
        print("Warning: Could not fetch and decode unconfirmed transactions. Proceeding with empty tx list for template.") # Keep
        unconfirmed_tx_list_for_template = []
    elif all(isinstance(tx, dict) for tx in unconfirmed_txs): 
        # print(f"Retrieved {len(unconfirmed_txs)} DECODED unconfirmed transactions for the template.")
        unconfirmed_tx_list_for_template = unconfirmed_txs 
    else: 
        print(f"Warning: Retrieved {len(unconfirmed_txs)} items from mempool, but not all are decoded dicts. Using as is for template.") # Keep
        unconfirmed_tx_list_for_template = unconfirmed_txs 
    
    next_block_template = {
        "id": {
            "hash": "PENDING_CONSTRUCTION", 
            "parts": {"total": 1, "hash": "PENDING_CONSTRUCTION"} 
        },
        "header": {
            "version": {"block": latest_confirmed_block.get("header",{}).get("version",{}).get("block","UNKNOWN"), "app": latest_confirmed_block.get("header",{}).get("version",{}).get("app","UNKNOWN")},
            "chain_id": chain_id,
            "height": str(next_block_height_val),
            "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "last_block_id": {
                "hash": last_block_id_hash,
                "parts": {
                    "total": last_block_id_parts_total,
                    "hash": last_block_id_parts_hash
                }
            },
            "last_commit_hash": "PENDING_CONSTRUCTION",
            "data_hash": "PENDING_CONSTRUCTION", 
            "validators_hash": "PENDING_CONSTRUCTION",
            "next_validators_hash": "PENDING_CONSTRUCTION",
            "consensus_hash": "PENDING_CONSTRUCTION",
            "app_hash": "PENDING_CONSTRUCTION", 
            "last_results_hash": "PENDING_CONSTRUCTION",
            "evidence_hash": "PENDING_CONSTRUCTION",
            "proposer_address": "UNKNOWN_UNTIL_PROPOSED"
        },
        "begin_block_events": [], 
        "end_block_events": [],   
        "txs": unconfirmed_tx_list_for_template 
    }
    
    # print("Next block template constructed.")
    return next_block_template

# --- Main Execution (for testing) ---
if __name__ == "__main__":
    # print("--- Testing Mayanode API Connections (src/api_connections.py) ---")

    # print("\n--- Preliminary Step: Getting latest block height for reference ---")
    latest_ref_height = get_mayanode_latest_block_height()

    if latest_ref_height:
        # print(f"Latest reference height determined: {latest_ref_height}")
        recent_historical_height = latest_ref_height - 10
        if recent_historical_height < 1:
            # print("Cannot test recent historical block, latest height is too low.")
            pass
        else:
            # print(f"\n--- Test 1: Fetching specific recent historical block (height {recent_historical_height}) ---")
            block_historical = fetch_mayanode_block(height=recent_historical_height)
            if block_historical:
                # print(f"Successfully fetched block {recent_historical_height}. Keys:", list(block_historical.keys()))
                if str(block_historical.get("header", {}).get("height")) == str(recent_historical_height): # Compare as string
                    # print(f"Block {recent_historical_height}: Header height matches requested height.")
                    pass
                else:
                    # print(f"Block {recent_historical_height}: MISMATCH - Fetched block header height {block_historical.get('header', {}).get('height')} vs requested {recent_historical_height}")
                    pass
            else:
               # print(f"Failed to fetch block {recent_historical_height}.")
               pass
    else:
        # print("Could not determine latest reference height. Skipping Test 1 for specific historical block.")
        pass
    
    time.sleep(1) 

    # print("\n--- Test 2: Fetching the latest block ---")
    latest_block_data = fetch_mayanode_block()
    if latest_block_data:
        latest_height_from_data_val = latest_block_data.get("header", {}).get("height") 
        # print(f"Successfully fetched latest block. Reported height: {latest_height_from_data_val}. Keys: {list(latest_block_data.keys())}")
    else:
        # print("Failed to fetch the latest block.")
        pass

    time.sleep(1) 

    # print("\n--- Test 3: Getting the latest block height ---")
    latest_retrieved_height_val = get_mayanode_latest_block_height() 
    if latest_retrieved_height_val:
        # print(f"Latest block height successfully retrieved: {latest_retrieved_height_val}")
        if latest_block_data and latest_height_from_data_val is not None: 
             if str(latest_retrieved_height_val) == str(latest_height_from_data_val): # Compare as string
                 # print("Latest height from get_mayanode_latest_block_height() matches height from fetched latest block.")
                 pass
             elif abs(latest_retrieved_height_val - int(latest_height_from_data_val)) <= 5: # Compare as int for diff
                 # print(f"Note: Latest height from get_mayanode_latest_block_height() ({latest_retrieved_height_val}) is close to latest block's height ({latest_height_from_data_val}). This is acceptable.")
                 pass
             else:
                 # print(f"Warning: Latest height from get_mayanode_latest_block_height() ({latest_retrieved_height_val}) differs significantly from latest block's height ({latest_height_from_data_val}).")
                 pass
    else:
        # print("Failed to retrieve the latest block height.")
        pass

    time.sleep(1)
    # print("\n--- Test 4: Fetching DECODED unconfirmed transactions (Tendermint RPC) ---")
    decoded_mempool_txs = fetch_decoded_tendermint_mempool_txs(limit=5)
    if decoded_mempool_txs is not None: 
        # print(f"Fetched {len(decoded_mempool_txs)} decoded unconfirmed transaction(s) via Tendermint RPC.")
        if len(decoded_mempool_txs) > 0 and isinstance(decoded_mempool_txs[0], dict):
            # print("Sample of first decoded unconfirmed tx (first 200 chars of JSON):")
            # print(json.dumps(decoded_mempool_txs[0], indent=2)[:200] + "...")
            pass
        elif len(decoded_mempool_txs) > 0: 
            # print("Sample of first item (not a dict, likely raw base64 string due to decoding issue):")
            # print(str(decoded_mempool_txs[0])[:100] + "...")
            pass
    else:
        # print("Call to fetch_decoded_tendermint_mempool_txs failed or returned None (API error).")
        pass

    time.sleep(1)
    # print("\n--- Test 5: Fetching number of unconfirmed transactions (Tendermint RPC) ---\")
    num_txs = get_tendermint_num_unconfirmed_txs()
    if num_txs is not None:
        # print(f"Number of unconfirmed transactions in mempool (Tendermint RPC): {num_txs}")
        pass
    else:
        # print("Failed to fetch or parse number of unconfirmed transactions from Tendermint RPC.")
        pass

    time.sleep(1)
    # print("\n--- Test 6: Constructing next block template ---")
    block_template = construct_next_block_template()
    if block_template:
        # print(f"Successfully constructed block template for tentative height: {block_template.get('header',{}).get('height')}")
        # print(f"Template includes {len(block_template.get('txs',[]))} transactions.")
        if block_template.get('txs'):
            # print("First transaction string (first 50 chars):", str(block_template['txs'][0])[:50] + "...") # Ensure it's a string for slicing
            pass
    else:
        # print("Failed to construct block template.")
        pass

    # print("\n--- Mayanode API Connection tests finished. ---")
