import requests
import json
import time
from datetime import datetime, timezone # Added timezone

# Configuration
MAYANODE_BASE_URL = "https://mayanode.mayachain.info/mayachain"
# TENDERMINT_RPC_BASE_URL = "https://rpc.mayachain.info"       # For Tendermint RPC (mempool, etc.) - This failed NameResolution
TENDERMINT_RPC_BASE_URL = "https://tendermint.mayachain.info" # Developer provided, trying HTTPS first
# --- Placeholder for Tendermint RPC Base URL until a working one is identified ---
# TENDERMINT_RPC_BASE_URL = "" # Set to a valid URL when known

# --- Helper Function (similar to previous, can be adapted if needed) ---
def fetch_api_data(url, api_name, params=None):
    """Fetches data from a given API URL and prints status."""
    print(f"--- Querying {api_name} API ---")
    print(f"URL: {url}")
    if params:
        print(f"Params: {params}")
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        print(f"Successfully connected and got response from {api_name}.")

        try:
            data = response.json()
            # For block data, the response can be very large.
            # We will avoid printing samples here to keep the output clean.
            # Callers can decide how to handle/log the data.
            return data
        except requests.exceptions.JSONDecodeError:
            print("Error: Response was not in JSON format, or content is empty.")
            print("Response text (first 500 chars):", response.text[:500])
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to {api_name} ({url}): {e}")
        return None
    finally:
        print("-" * 40 + "\n")

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
            print(f"Error: Invalid height '{height}'. Must be an integer.")
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
       "data" in response_data["result"]["block"]: # "id" is not in result.block, it's result.block_id
        
        block_content = response_data["result"]["block"]
        block_id_info = response_data["result"].get("block_id", {}) # Get block_id separately
        
        # For compatibility with how parse_confirmed_block might expect 'id' at top level
        # we can add it to the block_content.
        # parse_confirmed_block uses block_json_data.get("id", {}) for block_hash
        # and header.get("last_block_id", {}).get("hash") for last_block_hash
        # The new structure is response_data.result.block_id.hash and response_data.result.block.header.last_block_id.hash
        # So, we need to make sure parse_confirmed_block can find these.
        # Let's pass block_content and block_id_info to a revised parsing logic later,
        # or adjust parse_confirmed_block.
        # For now, let's just return the core 'block' content and ensure 'id' (block_id) is handled by the parser.
        # The parser expects a single dict. We should combine them or update parser.
        # The current `parse_confirmed_block` expects block_json_data.get("id", {}) for the block hash.
        # And block_json_data.get("header",{}) for most other things.
        # The structure from the API is:
        # response_data.result.block_id (for the block's own hash)
        # response_data.result.block.header (for height, time, last_block_id etc.)
        # response_data.result.block.data (for txs)
        # response_data.result.block.evidence
        # response_data.result.block.last_commit
        
        # We need to give parse_confirmed_block something that has "id" and "header" at the top.
        # Let's reconstruct a dictionary that parse_confirmed_block can work with.
        
        reconstructed_for_parser = {
            "id": block_id_info,  # This contains the block's hash
            "header": block_content.get("header"),
            "data": block_content.get("data"),
            "evidence": block_content.get("evidence"),
            "last_commit": block_content.get("last_commit"),
            "begin_block_events": response_data["result"].get("begin_block_events"), # Tendermint v0.34+
            "end_block_events": response_data["result"].get("end_block_events"),     # Tendermint v0.34+
            # Note: some older tendermint versions might have events inside block_content.results
            # The sample block 11255511.json had begin/end_block_events at the same level as 'block' and 'block_id'
            # which means directly under 'result'.
        }

        fetched_height = reconstructed_for_parser.get("header", {}).get("height")
        print(f"Successfully fetched and processed RPC block data. Reported height: {fetched_height}")
        if height is not None and str(height) != fetched_height:
            print(f"Warning: Requested height {height} but block data reports height {fetched_height}.")
        
        return reconstructed_for_parser

    # Fallback for direct block data structure (if not wrapped in JSON-RPC)
    # This was the original assumption.
    elif response_data and isinstance(response_data, dict) and "header" in response_data and "id" in response_data: # "id" was used for block_id
        fetched_height = response_data.get("header", {}).get("height")
        print(f"Successfully fetched direct block data. Reported height: {fetched_height}")
        if height is not None and str(height) != fetched_height:
            print(f"Warning: Requested height {height} but block data reports height {fetched_height}.")
        
        return response_data
    elif response_data:
        print("Error: Fetched data, but it does not match expected BlockResponse structure (missing key fields or not matching RPC wrapper).")
        print("Sample of unexpected data (first 500 chars):", str(response_data)[:500])
        return None
    else:
        print(f"Failed to fetch block data for {'height ' + str(height) if height else 'latest'}.")
        return None

def get_mayanode_latest_block_height():
    """Fetches the latest block from Mayanode and returns its height."""
    print("Attempting to determine latest block height from Mayanode...")
    latest_block = fetch_mayanode_block() # No height specified, should fetch latest
    if latest_block and isinstance(latest_block, dict):
        header = latest_block.get("header")
        if header and "height" in header:
            try:
                height_str = header["height"]
                latest_height = int(height_str)
                print(f"Successfully determined latest block height: {latest_height}")
                return latest_height
            except ValueError:
                print(f"Error: 'header.height' ('{height_str}') is not a valid integer.")
                return None
        else:
            print("Error: 'header' or 'header.height' not found in latest block response.")
            return None
    else:
        print("Failed to fetch latest block to determine height.")
        return None

def fetch_mayanode_unconfirmed_txs(limit: int = 30):
    """Fetches unconfirmed transactions from the Mayanode mempool.
    The `limit` parameter behavior is based on standard Tendermint RPC; actual behavior may vary.
    """
    endpoint = "/unconfirmed_txs"
    params = {"limit": str(limit)} # Tendermint RPC limit is often a string
    api_description = f"Mayanode API for Unconfirmed Txs (limit {limit})"
    url = f"{MAYANODE_BASE_URL}{endpoint}"

    unconfirmed_txs_data = fetch_api_data(url, api_description, params=params)

    if unconfirmed_txs_data and isinstance(unconfirmed_txs_data, dict):
        # Standard Tendermint structure: result.n_txs, result.total, result.total_bytes, result.txs
        if "result" in unconfirmed_txs_data and "txs" in unconfirmed_txs_data["result"]:
            txs_list = unconfirmed_txs_data["result"]["txs"]
            total_txs = unconfirmed_txs_data["result"].get("total", "N/A")
            print(f"Successfully fetched unconfirmed transactions. Count in response: {len(txs_list if txs_list else [])}, Total in mempool: {total_txs}")
            if txs_list is None: # Handle case where 'txs' key exists but is null
                 print("Mempool is likely empty or 'txs' is null.")
                 return [] # Return empty list for consistency
            return txs_list # This will be a list of base64 encoded tx strings
        else:
            print("Error: Fetched data for unconfirmed_txs, but it does not match expected structure (missing 'result.txs').")
            print("Sample of unexpected data (first 500 chars):", str(unconfirmed_txs_data)[:500])
            return None
    elif unconfirmed_txs_data: # e.g. if it's a list directly, or unexpected format
        print("Error: Fetched data for unconfirmed_txs, but it is not a dictionary or is in an unexpected format.")
        print("Sample of unexpected data (first 500 chars):", str(unconfirmed_txs_data)[:500])
        return None
    else:
        print("Failed to fetch unconfirmed transactions.")
        return None

# --- Tendermint RPC Functions ---
def fetch_tendermint_unconfirmed_txs(limit: int = 100): # Default to 100 (common max)
    """Fetches unconfirmed transactions from the Tendermint RPC /unconfirmed_txs endpoint.
    Fetches up to the specified limit (max typically 100 for public Tendermint RPCs).
    """
    if not TENDERMINT_RPC_BASE_URL:
        print("WARNING: fetch_tendermint_unconfirmed_txs - TENDERMINT_RPC_BASE_URL is not set.")
        print("This function will return an empty list. Update when a valid endpoint is known.")
        return [] 

    endpoint = "/unconfirmed_txs"
    # Tendermint RPC often expects numbers as strings in query params
    params = {"limit": str(limit)} if limit is not None else {}
    api_description = f"Tendermint RPC for Unconfirmed Txs (limit {limit if limit is not None else 'default'})"
    url = f"{TENDERMINT_RPC_BASE_URL}{endpoint}"

    response_data = fetch_api_data(url, api_description, params=params)

    if response_data and isinstance(response_data, dict):
        if "result" in response_data and isinstance(response_data["result"], dict) and "txs" in response_data["result"]:
            txs_list = response_data["result"]["txs"]
            n_txs = response_data["result"].get("n_txs", "N/A") # Count in this response
            total_txs = response_data["result"].get("total", "N/A") # Total in mempool
            print(f"Successfully fetched unconfirmed transactions. Count in response: {n_txs}, Total in mempool: {total_txs}")
            if n_txs != total_txs:
                print(f"WARNING: Fetched {n_txs} transactions, but mempool reports {total_txs} total. Endpoint may have a limit.")
            if txs_list is None: # Handle case where 'txs' key exists but is null (empty mempool)
                 print("Mempool is likely empty or 'txs' is null in response.")
                 return [] # Return empty list for consistency
            return txs_list # This will be a list of base64 encoded tx strings
        else:
            print("Error: Fetched data for unconfirmed_txs, but it does not match expected Tendermint structure (missing 'result.txs').")
            print("Full response (first 500 chars):", str(response_data)[:500])
            return None
    elif response_data: # e.g. if it's not a dictionary or unexpected format
        print("Error: Fetched data for unconfirmed_txs, but it is not a dictionary or is in an unexpected format.")
        print("Sample of unexpected data (first 500 chars):", str(response_data)[:500])
        return None
    else:
        print("Failed to fetch unconfirmed transactions (no data returned from fetch_api_data).")
        return None

def get_tendermint_num_unconfirmed_txs():
    """Fetches the number of unconfirmed transactions from Tendermint RPC /num_unconfirmed_txs endpoint."""
    if not TENDERMINT_RPC_BASE_URL:
        print("WARNING: get_tendermint_num_unconfirmed_txs - TENDERMINT_RPC_BASE_URL is not set.")
        print("This function will return 0. Update when a valid endpoint is known.")
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
            print(f"Successfully fetched num_unconfirmed_txs. In response: {n_txs}, Total in mempool: {total_txs}, Total bytes: {total_bytes}")
            try:
                # Return the 'total' count, as it represents the whole mempool size
                return int(total_txs) if total_txs != "N/A" else (int(n_txs) if n_txs != "N/A" else 0)
            except ValueError:
                print(f"Error: Could not parse count from num_unconfirmed_txs response: n_txs='{n_txs}', total='{total_txs}'")
                return None
        else:
            print("Error: Fetched data for num_unconfirmed_txs, but it does not match expected Tendermint structure (missing 'result').")
            print("Full response (first 500 chars):", str(response_data)[:500])
            return None
    elif response_data:
        print("Error: Fetched data for num_unconfirmed_txs, but it is not a dictionary or is in an unexpected format.")
        print("Sample of unexpected data (first 500 chars):", str(response_data)[:500])
        return None
    else:
        print("Failed to fetch num_unconfirmed_txs (no data returned from fetch_api_data).")
        return None

def construct_next_block_template():
    """Constructs a template for the next block using the latest confirmed block info
    and unconfirmed transactions from the mempool.

    Note: Transactions in the template are raw base64 strings and are NOT yet ordered by slip/fee.
    """
    print("\n--- Constructing Next Block Template ---")

    # 1. Fetch latest confirmed block for context
    latest_confirmed_block = fetch_mayanode_block() # Fetches the latest
    if not latest_confirmed_block:
        print("Error: Could not fetch latest confirmed block to build template. Aborting.")
        return None

    # Extract necessary info from latest confirmed block
    chain_id = latest_confirmed_block.get("header", {}).get("chain_id", "mayachain-mainnet-v1") # Default if not found
    last_block_id_hash = latest_confirmed_block.get("id", {}).get("hash", "UNKNOWN_LAST_BLOCK_HASH")
    last_block_id_parts_hash = latest_confirmed_block.get("id", {}).get("parts", {}).get("hash", "UNKNOWN_LAST_BLOCK_PARTS_HASH")
    last_block_id_parts_total = latest_confirmed_block.get("id", {}).get("parts", {}).get("total", 0)
    last_confirmed_height_str = latest_confirmed_block.get("header", {}).get("height", "0")
    try:
        next_block_height_val = int(last_confirmed_height_str) + 1
    except ValueError:
        print(f"Warning: Could not parse last confirmed height '{last_confirmed_height_str}'. Using placeholder for next height.")
        next_block_height_val = "PENDING_CALCULATION"

    # 2. Fetch unconfirmed transactions
    # Fetches up to 100 by default, which is a common max for /unconfirmed_txs.
    # If more sophisticated fetching of *all* txs is needed (e.g. if mempool > 100 and endpoint limits),
    # fetch_tendermint_unconfirmed_txs would need to handle pagination if the RPC supports it (not standard for this call).
    unconfirmed_tx_strings = fetch_tendermint_unconfirmed_txs(limit=100) 
    if unconfirmed_tx_strings is None:
        print("Warning: Could not fetch unconfirmed transactions. Proceeding with empty tx list for template.")
        unconfirmed_tx_strings = []
    
    print(f"Retrieved {len(unconfirmed_tx_strings)} unconfirmed transaction strings for the template.")

    # 3. Construct the template
    next_block_template = {
        "id": {
            "hash": "PENDING_CONSTRUCTION", 
            "parts": {"total": 1, "hash": "PENDING_CONSTRUCTION"} # Simplified parts
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
            "data_hash": "PENDING_CONSTRUCTION", # Will depend on the actual transactions included and their order
            "validators_hash": "PENDING_CONSTRUCTION",
            "next_validators_hash": "PENDING_CONSTRUCTION",
            "consensus_hash": "PENDING_CONSTRUCTION",
            "app_hash": "PENDING_CONSTRUCTION", 
            "last_results_hash": "PENDING_CONSTRUCTION",
            "evidence_hash": "PENDING_CONSTRUCTION",
            "proposer_address": "UNKNOWN_UNTIL_PROPOSED"
        },
        "begin_block_events": [], # Placeholder, typically generated during block processing
        "end_block_events": [],   # Placeholder, typically generated during block processing
        "txs": unconfirmed_tx_strings # Raw base64 encoded strings
        # The actual "BlockTx" structure with hash, tx, result will be formed when block is finalized.
        # For a template, just having the raw txs is the main goal from mempool.
    }
    
    print("Next block template constructed.")
    # print(json.dumps(next_block_template, indent=2)) # For debugging, can be very verbose
    return next_block_template

# --- Main Execution (for testing) ---
if __name__ == "__main__":
    print("--- Testing Mayanode API Connections (src/api_connections.py) ---")

    # First, try to get the latest block height to use as a reference for historical fetch
    print("\n--- Preliminary Step: Getting latest block height for reference ---")
    latest_ref_height = get_mayanode_latest_block_height()

    if latest_ref_height:
        print(f"Latest reference height determined: {latest_ref_height}")
        # 1. Test fetching a specific RECENT historical block
        recent_historical_height = latest_ref_height - 10
        if recent_historical_height < 1:
            print("Cannot test recent historical block, latest height is too low.")
        else:
            print(f"\n--- Test 1: Fetching specific recent historical block (height {recent_historical_height}) ---")
            block_historical = fetch_mayanode_block(height=recent_historical_height)
            if block_historical:
                print(f"Successfully fetched block {recent_historical_height}. Keys:", list(block_historical.keys()))
                if block_historical.get("header", {}).get("height") == str(recent_historical_height):
                    print(f"Block {recent_historical_height}: Header height matches requested height.")
                else:
                    print(f"Block {recent_historical_height}: MISMATCH - Fetched block header height {block_historical.get('header', {}).get('height')} vs requested {recent_historical_height}")
            else:
               print(f"Failed to fetch block {recent_historical_height}.")
    else:
        print("Could not determine latest reference height. Skipping Test 1 for specific historical block.")
    
    time.sleep(1) # Pause between API calls

    # 2. Test fetching the latest block
    print("\n--- Test 2: Fetching the latest block ---")
    latest_block_data = fetch_mayanode_block()
    if latest_block_data:
        latest_height_from_data = latest_block_data.get("header", {}).get("height")
        print(f"Successfully fetched latest block. Reported height: {latest_height_from_data}. Keys: {list(latest_block_data.keys())}")
        # print(json.dumps(latest_block_data, indent=2)) # Potentially very large output
    else:
        print("Failed to fetch the latest block.")

    time.sleep(1) # Pause between API calls

    # 3. Test getting the latest block height directly
    print("\n--- Test 3: Getting the latest block height ---")
    latest_retrieved_height = get_mayanode_latest_block_height()
    if latest_retrieved_height:
        print(f"Latest block height successfully retrieved: {latest_retrieved_height}")
        if latest_block_data and latest_height_from_data:
             if str(latest_retrieved_height) == latest_height_from_data:
                 print("Latest height from get_mayanode_latest_block_height() matches height from fetched latest block.")
             elif abs(latest_retrieved_height - int(latest_height_from_data)) <= 5: # Allow for new blocks in between calls
                 print(f"Note: Latest height from get_mayanode_latest_block_height() ({latest_retrieved_height}) is close to latest block's height ({latest_height_from_data}). This is acceptable.")
             else:
                 print(f"Warning: Latest height from get_mayanode_latest_block_height() ({latest_retrieved_height}) differs significantly from latest block's height ({latest_height_from_data}).")
    else:
        print("Failed to retrieve the latest block height.")

    # print("\n--- Specific Test for Block 11255442 ---") # Keep this commented out or remove
    # block_11255442 = fetch_mayanode_block(height=11255442)
    # if block_11255442:
    #     print("Successfully fetched block 11255442 in main test.")
    # else:
    #     print("Failed to fetch block 11255442 in main test.")

    time.sleep(1)
    # 4. Test fetching unconfirmed transactions from Tendermint RPC
    print("\n--- Test 4: Fetching unconfirmed transactions (Tendermint RPC) ---")
    unconfirmed_txs = fetch_tendermint_unconfirmed_txs(limit=5)
    if unconfirmed_txs is not None: # Check for None explicitly, as empty list is a valid success (empty mempool)
        print(f"Fetched {len(unconfirmed_txs)} unconfirmed transaction(s) via Tendermint RPC.")
        if len(unconfirmed_txs) > 0:
            print("Sample of first unconfirmed tx (base64 encoded string) from Tendermint RPC:")
            print(unconfirmed_txs[0][:100] + "...") # Print first 100 chars of the first tx
        else:
            print("Failed to fetch or parse unconfirmed transactions response from Tendermint RPC.")

    time.sleep(1)
    # 5. Test fetching number of unconfirmed transactions from Tendermint RPC
    print("\n--- Test 5: Fetching number of unconfirmed transactions (Tendermint RPC) ---")
    num_txs = get_tendermint_num_unconfirmed_txs()
    if num_txs is not None:
        print(f"Number of unconfirmed transactions in mempool (Tendermint RPC): {num_txs}")
    else:
        print("Failed to fetch or parse number of unconfirmed transactions from Tendermint RPC.")

    time.sleep(1)
    # 6. Test constructing next block template
    print("\n--- Test 6: Constructing next block template ---")
    block_template = construct_next_block_template()
    if block_template:
        print(f"Successfully constructed block template for tentative height: {block_template.get('header',{}).get('height')}")
        print(f"Template includes {len(block_template.get('txs',[]))} transactions.")
        if block_template.get('txs'):
            print("First transaction string (first 50 chars):", block_template['txs'][0][:50] + "...")
    else:
        print("Failed to construct block template.")

    print("\n--- Mayanode API Connection tests finished. ---")
