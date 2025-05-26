"""
Script to fetch a block from Tendermint RPC and a Mayanode API,
decode the transactions in the Tendermint block, and then save both
block structures for comparison.
"""
import sys
import os
import requests
import base64
import json
from pathlib import Path
import argparse # Add argparse
# import inspect # Removed for now
# Removed hashlib and bech32 as they are now in common_utils

# --- Adjust sys.path for local imports ---
# Get the absolute path of the script's directory
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = _SCRIPT_DIR.parent # Project root is one level up from 'scripts'
PROTO_GEN_PARENT_DIR = PROJECT_ROOT_DIR / "proto" / "generated"
OUTPUT_DIR = PROJECT_ROOT_DIR / "comparison_outputs"

# Add project root to sys.path to allow importing from src
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# Add the parent directory of our 'pb_stubs' package to sys.path
if str(PROTO_GEN_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_GEN_PARENT_DIR))

# Attempt to import a non-existent function to test if the latest common_utils is read
# from src.common_utils import this_function_should_not_exist_12345 # <<< REMOVED AFTER TESTING

# Configuration
DEFAULT_BLOCK_HEIGHT = 11320570  # Using block found by find_test_tx_with_authinfo.py
# BLOCK_HEIGHT will be set by argparse
TENDERMINT_RPC_URL = "https://tendermint.mayachain.info"
# MAYANODE_API_URL_FORMAT is now handled by the imported function from api_connections

# Define the parent directory of the generated protobuf package
PROTO_GEN_PARENT_DIR = str(Path(__file__).parent.parent / "proto" / "generated")
PROJECT_ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT_DIR / "comparison_outputs"

# Add project root to sys.path to allow importing from src
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# Add the parent directory of our 'pb_stubs' package to sys.path
if PROTO_GEN_PARENT_DIR not in sys.path:
    sys.path.insert(0, PROTO_GEN_PARENT_DIR)

# Import after sys.path modification
try:
    from pb_stubs.cosmos.tx.v1beta1 import Tx as CosmosTx
    from pb_stubs.types import MsgNetworkFee
    from pb_stubs.cosmos.crypto.secp256k1 import PubKey as Secp256k1PubKey
    from src.api_connections import (
        fetch_mayanode_block,
        fetch_tendermint_rpc_block_raw,
        fetch_tendermint_rpc_block_results,
    )
    from src.common_utils import (
        decode_cosmos_tx_string_to_dict,
        transform_decoded_tm_tx_to_mayanode_format,
        parse_confirmed_block,
        PROTO_TYPES_AVAILABLE as common_utils_proto_types_available,
        TYPE_URL_TO_MAYANODE_TYPE
    )
    
    # +++ Debugging common_utils loading +++
    # print("--- Debugging src.common_utils loading ---")
    # ... inspect logic removed ...
    # print("--- End Debugging src.common_utils loading ---\n")

    PROTOBUF_AVAILABLE_MAIN = True # Assume available if imports succeed
    print("Successfully imported generated protobuf modules and api_connections.")

except ImportError as e:
    print(f"ImportError in compare_block_data.py: {e}")
    print("Please ensure all dependencies are installed and protobufs are compiled.")
    PROTOBUF_AVAILABLE_MAIN = False
    # Define dummy functions if imports fail, so the script might partially run or exit gracefully
    def fetch_mayanode_block(height,_): return None
    def transform_decoded_tm_tx_to_mayanode_format(data): return data
    def decode_cosmos_tx_string_to_dict(s): return s
    CosmosTx = None
    MsgNetworkFee = None
    Secp256k1PubKey = None

# Map chain identifiers to their Human-Readable Part (HRP)
CHAIN_TO_HRP = {
    "MAYA": "maya", "THOR": "thor", "COSMOS": "cosmos", "GAIA": "cosmos",
    "BTC": "bc", "ETH": None, "TERRA": "terra", "XRD": None,
}

# --- Block Fetching and Processing Functions ---

def fetch_block_from_tendermint_rpc_full(height: int) -> dict | None:
    endpoint = f"{TENDERMINT_RPC_URL}/block?height={height}"
    print(f"Fetching Tendermint block from: {endpoint}")
    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        # The Tendermint RPC /block endpoint nests the block data under "result.block"
        return response.json().get("result", {}).get("block", {})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Tendermint block: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Tendermint RPC: {e}")
    return None

# fetch_block_from_mayanode_api is now imported from src.api_connections

def main(block_height_to_process: int):
    if not fetch_mayanode_block: # Check if essential import loaded
        print("Critical definition (fetch_mayanode_block) not loaded. Exiting.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # This check is now redundant as api_connections.py handles its own check and common_utils also.
    # if not api_proto_types_available or not common_utils_proto_types_available:
    #     print("PROTOBUF_AVAILABLE is false in api_connections or common_utils. Exiting.")
    #     return
    
    print(f"--- Processing block {block_height_to_process} ---")
    
    # Fetch block from Tendermint RPC (includes base64 encoded txs)
    tm_block_rpc_result = fetch_tendermint_rpc_block_raw(height=block_height_to_process)
    if not tm_block_rpc_result:
        print(f"Failed to fetch block {block_height_to_process} from Tendermint RPC. Exiting.")
        return
    
    # Fetch block from Mayanode API (already decoded JSON txs)
    mayanode_block_json = fetch_mayanode_block(height=block_height_to_process)
    if not mayanode_block_json:
        print(f"Failed to fetch block {block_height_to_process} from Mayanode API. Exiting.")
        return
    
    # Extract transactions from Tendermint block
    # The fetch_tendermint_rpc_block_raw function returns the content of response_data["result"]
    # So, tm_block_rpc_result directly contains "block" and "block_id"
    tendermint_txs_b64 = tm_block_rpc_result.get("block", {}).get("data", {}).get("txs", [])
    if not tendermint_txs_b64:
        # It's possible for blocks to have no transactions.
        print(f"No transactions found in Tendermint block {block_height_to_process} data.txs list.")
        # Continue processing as an empty block is valid.
    else:
        print(f"Found {len(tendermint_txs_b64)} transactions in Tendermint block {block_height_to_process}.")

    # Save raw Tendermint transactions (base64 strings)
    tm_output_path = OUTPUT_DIR / f"tendermint_block_{block_height_to_process}_rpc_txs.json"
    with open(tm_output_path, 'w') as f:
        json.dump({"block_height": block_height_to_process, "tendermint_txs_b64": tendermint_txs_b64}, f, indent=2)
    print(f"Saved Tendermint RPC block transactions to {tm_output_path}")

    # 2. Fetch Mayanode block
    # Save Mayanode block data (full response)
    mn_output_path = OUTPUT_DIR / f"mayanode_block_{block_height_to_process}_api_full.json"
    with open(mn_output_path, 'w') as f:
        json.dump(mayanode_block_json, f, indent=2)
    print(f"Saved Mayanode API block data to {mn_output_path}")

    # Parse the full Mayanode block using common_utils.parse_confirmed_block
    parsed_mayanode_block = parse_confirmed_block(mayanode_block_json)
    if parsed_mayanode_block:
        mn_parsed_full_output_path = OUTPUT_DIR / f"mayanode_block_{block_height_to_process}_PARSED_full.json"
        with open(mn_parsed_full_output_path, 'w') as f:
            json.dump(parsed_mayanode_block, f, indent=2, default=str) # Use default=str for datetime objects
        print(f"Saved fully parsed Mayanode block to {mn_parsed_full_output_path}")
    else:
        print(f"Failed to parse Mayanode block {block_height_to_process} using parse_confirmed_block.")

    # Extract transactions from Mayanode data
    # Based on prior analysis, Mayanode /mayachain/block returns already decoded JSON in 'txs' field at top level
    mayanode_txs_list = mayanode_block_json.get("txs", []) 
    if not isinstance(mayanode_txs_list, list):
        print(f"Warning: Mayanode txs field is not a list as expected. Got: {type(mayanode_txs_list)}")
        # Attempt to check common nested paths if the direct 'txs' is not a list or not found
        if isinstance(mayanode_block_json.get("result", {}).get("block", {}).get("data", {}), dict):
            mayanode_txs_list = mayanode_block_json.get("result", {}).get("block", {}).get("data", {}).get("txs", [])
        elif isinstance(mayanode_block_json.get("block", {}).get("data", {}), dict):
            mayanode_txs_list = mayanode_block_json.get("block", {}).get("data", {}).get("txs", [])
        else:
            mayanode_txs_list = [] # Default to empty if still not found or not a list
        
        if not isinstance(mayanode_txs_list, list): # Final check
            print(f"Error: Could not extract a list of transactions from Mayanode block data. Found: {type(mayanode_txs_list)}")
            mayanode_txs_list = []

    # Save Mayanode transactions list specifically
    mn_txs_only_output_path = OUTPUT_DIR / f"mayanode_block_{block_height_to_process}_api_txs_only.json"
    with open(mn_txs_only_output_path, 'w') as f:
        json.dump(mayanode_txs_list, f, indent=2) # This saves the list of items which might have {hash, result, tx}
    print(f"Saved Mayanode API transaction list to {mn_txs_only_output_path}")

    # Prepare the list of actual Mayanode transaction content for comparison
    # Each item in mayanode_txs_list is like {"hash": "...", "result": {...}, "tx": {...}}
    # We need a list of the "tx" parts for direct comparison with transformed Tendermint txs
    actual_mayanode_tx_content_list = []
    if isinstance(mayanode_txs_list, list):
        for tx_item in mayanode_txs_list:
            if isinstance(tx_item, dict) and "tx" in tx_item:
                actual_mayanode_tx_content_list.append(tx_item["tx"])
            else:
                # If an item is not a dict or doesn't have a "tx" key, add a placeholder or error
                actual_mayanode_tx_content_list.append({"error": "missing_tx_field_in_mayanode_tx_item", "original_item": tx_item})
    
    # Fetch Tendermint block results for events
    tm_block_results = fetch_tendermint_rpc_block_results(height=block_height_to_process)
    if not tm_block_results:
        print(f"Failed to fetch Tendermint block results for height {block_height_to_process}. Event data will be missing for Tendermint parsed block.")
        # Assign empty structures for events if fetch fails, so parsing can proceed
        tm_begin_block_events = []
        tm_end_block_events = []
        # Save empty results for audit if fetch failed
        tm_block_results_raw_path = OUTPUT_DIR / f"tendermint_block_{block_height_to_process}_RAW_block_results.json"
        with open(tm_block_results_raw_path, 'w') as f:
            json.dump({"error": "fetch_failed", "data": None}, f, indent=2)
        print(f"Saved empty/error Tendermint block results to {tm_block_results_raw_path}")
    else:
        tm_begin_block_events = tm_block_results.get("begin_block_events", [])
        tm_end_block_events = tm_block_results.get("end_block_events", [])
        
        # Save the raw tm_block_results to a file for inspection
        tm_block_results_raw_path = OUTPUT_DIR / f"tendermint_block_{block_height_to_process}_RAW_block_results.json"
        with open(tm_block_results_raw_path, 'w') as f:
            json.dump(tm_block_results, f, indent=2)
        print(f"Saved raw Tendermint block results to {tm_block_results_raw_path}")

        # DEBUG: Print info about txs_results before passing to parse_confirmed_block
        txs_results_from_fetch = tm_block_results.get("txs_results")
        if txs_results_from_fetch is None:
            print(f"[DEBUG compare_block_data.py] tm_block_results.txs_results is None.")
        elif isinstance(txs_results_from_fetch, list):
            print(f"[DEBUG compare_block_data.py] len(tm_block_results.txs_results): {len(txs_results_from_fetch)}")
            if len(txs_results_from_fetch) > 0 and isinstance(txs_results_from_fetch[0], dict):
                print(f"[DEBUG compare_block_data.py] First tx_result keys: {list(txs_results_from_fetch[0].keys())}")
        else:
            print(f"[DEBUG compare_block_data.py] tm_block_results.txs_results is not a list. Type: {type(txs_results_from_fetch)}")

    # Prepare Tendermint block data for parsing by common_utils.parse_confirmed_block
    # For Tendermint RPC, we pass the raw /block response and the /block_results response separately.
    parsed_tendermint_block = parse_confirmed_block(
        block_json_data=None, # Not used directly when source_type is tendermint_rpc and tm_block_raw_data is given
        source_type="tendermint_rpc",
        tm_block_raw_data=tm_block_rpc_result, # This is the /block response
        tm_block_results_data=tm_block_results # This is the /block_results response
    )

    if parsed_tendermint_block:
        tm_parsed_full_output_path = OUTPUT_DIR / f"tendermint_block_{block_height_to_process}_PARSED_full_from_tm.json"
        with open(tm_parsed_full_output_path, 'w') as f:
            json.dump(parsed_tendermint_block, f, indent=2, default=str) # Use default=str for datetime objects
        print(f"Saved fully parsed Tendermint-sourced block to {tm_parsed_full_output_path}")
    else:
        print(f"Failed to parse Tendermint-sourced block {block_height_to_process} using parse_confirmed_block.")

    # 3. Decode Tendermint transactions and transform them
    decoded_and_transformed_tm_txs = []
    
    print(f"\n--- Decoding and Transforming {len(tendermint_txs_b64)} Tendermint Transactions ---")
    for i, tm_tx_b64 in enumerate(tendermint_txs_b64):
        print(f"Processing Tendermint Tx {i+1}/{len(tendermint_txs_b64)}...")
        decoded_tm_tx = decode_cosmos_tx_string_to_dict(tm_tx_b64) # Using direct import
        if isinstance(decoded_tm_tx, str): # Decoding failed, returned original string
            print(f"  Failed to decode Tendermint Tx {i+1}. Original b64: {decoded_tm_tx[:60]}...")
            # Store as an error object or skip
            decoded_and_transformed_tm_txs.append({"error": "failed_to_decode_tendermint_tx", "original_b64": decoded_tm_tx})
            continue
        
        transformed_tx = transform_decoded_tm_tx_to_mayanode_format(decoded_tm_tx)
        decoded_and_transformed_tm_txs.append(transformed_tx)

    # Save decoded and transformed Tendermint transactions
    transformed_tm_output_path = OUTPUT_DIR / f"tendermint_block_{block_height_to_process}_decoded_transformed.json"
    with open(transformed_tm_output_path, 'w') as f:
        json.dump(decoded_and_transformed_tm_txs, f, indent=2)
    print(f"Saved decoded and transformed Tendermint transactions to {transformed_tm_output_path}")

    # 4. Compare (Placeholder for now - focus is on getting decoding right)
    print("\n--- Comparison (Placeholder) ---")
    print(f"Number of Tendermint transactions (decoded & transformed): {len(decoded_and_transformed_tm_txs)}")
    print(f"Number of Mayanode transactions (from API): {len(mayanode_txs_list)}")

    # Basic check: Compare signer of first message if available
    if decoded_and_transformed_tm_txs and isinstance(decoded_and_transformed_tm_txs[0], dict) and \
       mayanode_txs_list and isinstance(mayanode_txs_list[0], dict):
        
        tm_first_tx = decoded_and_transformed_tm_txs[0]
        mn_first_tx = mayanode_txs_list[0]

        # Path to signer in transformed Tendermint tx: body.messages[0].signer
        tm_signer = None
        if tm_first_tx.get('body') and isinstance(tm_first_tx['body'].get('messages'), list) and tm_first_tx['body']['messages']:
            tm_signer = tm_first_tx['body']['messages'][0].get('signer')

        # Path to signer in Mayanode tx (often pre-decoded): tx.body.messages[0].signer
        # Or directly messages[0].signer if 'tx' is the top-level list item and contains messages itself
        mn_signer = None
        # Check if mn_first_tx is the tx content itself or if it's nested under a 'tx' key
        tx_content_for_mn = mn_first_tx.get('tx') if isinstance(mn_first_tx.get('tx'), dict) else mn_first_tx
        
        if tx_content_for_mn.get('body') and isinstance(tx_content_for_mn['body'].get('messages'), list) and tx_content_for_mn['body']['messages']:
            mn_signer = tx_content_for_mn['body']['messages'][0].get('signer')
        
        print(f"Signer of first message (Tendermint transformed): {tm_signer}")
        print(f"Signer of first message (Mayanode API): {mn_signer}")

        if tm_signer and mn_signer and tm_signer == mn_signer:
            print("Signers match for the first transaction! :-) ")
        else:
            print("Signers DO NOT match for the first transaction, or one/both are missing. :-(")
    else:
        print("Could not compare signers of first transaction (lists empty or first item not a dict).")

    print(f"\n--- Block {block_height_to_process} processing complete. Outputs are in {OUTPUT_DIR} ---")

    # Call detailed_comparison with the processed data
    # detailed_comparison(decoded_and_transformed_tm_txs, mayanode_txs_list, BLOCK_HEIGHT) # Old call, to be replaced

    # New step: Compare the transaction lists directly
    compare_transaction_lists(decoded_and_transformed_tm_txs, actual_mayanode_tx_content_list, block_height_to_process)

    # New step: Compare the fully parsed block structures
    compare_full_parsed_blocks(block_height_to_process)

def compare_transaction_lists(tm_tx_list, mn_tx_list, block_height):
    print(f"\n--- Comparing Transaction Lists for Block {block_height} ---")
    comparison_output_path = OUTPUT_DIR / f"block_{block_height}_transaction_comparison_report.txt"
    report_lines = []

    report_lines.append(f"Comparing {len(tm_tx_list)} transformed Tendermint transactions with {len(mn_tx_list)} Mayanode API transactions.")

    if len(tm_tx_list) != len(mn_tx_list):
        report_lines.append(f"  MISMATCH: Number of transactions differs: Tendermint ({len(tm_tx_list)}) vs Mayanode ({len(mn_tx_list)})")
        # Further comparison might not be meaningful if counts differ, but we could try partial if needed.
    else:
        report_lines.append("  Number of transactions matches. Comparing element-wise...")
        mismatched_tx_count = 0
        for i in range(len(tm_tx_list)):
            report_lines.append(f"\n  Comparing Transaction {i+1}/{len(tm_tx_list)}:")
            tm_tx = tm_tx_list[i]
            mn_tx = mn_tx_list[i]

            # Check if either transaction is an error placeholder before comparing
            if isinstance(tm_tx, dict) and tm_tx.get("error"):
                report_lines.append(f"    Tendermint Tx {i+1} is an error object: {tm_tx.get('error')}")
                mismatched_tx_count +=1
                continue
            if isinstance(mn_tx, dict) and mn_tx.get("error"):
                report_lines.append(f"    Mayanode Tx {i+1} is an error object: {mn_tx.get('error')}")
                mismatched_tx_count +=1
                continue
            
            # Ensure both are dicts before calling compare_dicts_recursive
            if not isinstance(tm_tx, dict) or not isinstance(mn_tx, dict):
                report_lines.append(f"    MISMATCH: Transaction {i+1} types are not both dictionaries for comparison. TM: {type(tm_tx)}, MN: {type(mn_tx)}")
                mismatched_tx_count +=1
                continue

            differences = compare_dicts_recursive(tm_tx, mn_tx, path=f"tx[{i}].")
            if not differences:
                report_lines.append(f"    Transaction {i+1}: MATCHES.")
            else:
                report_lines.append(f"    Transaction {i+1}: MISMATCHES FOUND:")
                mismatched_tx_count += 1
                for diff in differences:
                    report_lines.append(f"      - {diff}")
        
        if mismatched_tx_count == 0:
            report_lines.append("\n  OVERALL TRANSACTION LISTS: All transactions MATCH.")
        else:
            report_lines.append(f"\n  OVERALL TRANSACTION LISTS: {mismatched_tx_count}/{len(tm_tx_list)} transactions had mismatches.")

    with open(comparison_output_path, 'w') as f:
        for line in report_lines:
            f.write(line + "\n")
            print(line) # Also print to console
    print(f"Transaction comparison report saved to {comparison_output_path}")
    print("--- End of Transaction List Comparison ---")

def compare_dicts_recursive(d1, d2, path=""):
    """Recursively compares two dictionaries and returns a list of differences."""
    differences = []

    # Keys unique to d1
    for k in d1.keys():
        if k not in d2:
            differences.append(f"Key '{path}{k}' found in first dict but not in second.")
    # Keys unique to d2
    for k in d2.keys():
        if k not in d1:
            differences.append(f"Key '{path}{k}' found in second dict but not in first.")

    # Compare common keys
    for k in d1.keys():
        if k in d2:
            new_path = f"{path}{k}."
            if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                differences.extend(compare_dicts_recursive(d1[k], d2[k], new_path))
            elif isinstance(d1[k], list) and isinstance(d2[k], list):
                # Basic list comparison: length and then element-wise if simple
                if len(d1[k]) != len(d2[k]):
                    differences.append(f"List length mismatch for key '{new_path[:-1]}': {len(d1[k])} vs {len(d2[k])}")
                else:
                    # For simplicity, only do deep compare if lists contain dicts, otherwise just value check
                    for i in range(len(d1[k])):
                        if isinstance(d1[k][i], dict) and isinstance(d2[k][i], dict):
                            differences.extend(compare_dicts_recursive(d1[k][i], d2[k][i], f"{new_path[:-1]}[{i}]."))
                        elif d1[k][i] != d2[k][i]:
                            differences.append(f"Value mismatch for list item '{new_path[:-1]}[{i}]': '{d1[k][i]}' vs '{d2[k][i]}'")
            elif type(d1[k]) != type(d2[k]):
                differences.append(f"Type mismatch for key '{new_path[:-1]}': {type(d1[k])} vs {type(d2[k])}")
            elif d1[k] != d2[k]:
                val1_str = str(d1[k])[:100] + ('...' if len(str(d1[k])) > 100 else '')
                val2_str = str(d2[k])[:100] + ('...' if len(str(d2[k])) > 100 else '')
                differences.append(f"Value mismatch for key '{new_path[:-1]}': '{val1_str}' vs '{val2_str}'")
    return differences

# This function is being replaced by compare_full_parsed_blocks
# def detailed_comparison(tm_txs_transformed_list, mn_txs_api_list, block_height):
#     print(f"\n--- Detailed Comparison for Block {block_height} ---")
# ... (rest of old detailed_comparison function removed for brevity)

def compare_full_parsed_blocks(block_height):
    print(f"\n--- Comparing Full Parsed Block Structures for Block {block_height} ---")

    mn_parsed_path = OUTPUT_DIR / f"mayanode_block_{block_height}_PARSED_full.json"
    tm_parsed_path = OUTPUT_DIR / f"tendermint_block_{block_height}_PARSED_full_from_tm.json"

    if not mn_parsed_path.exists():
        print(f"Error: Mayanode parsed block file not found: {mn_parsed_path}")
        return
    if not tm_parsed_path.exists():
        print(f"Error: Tendermint parsed block file not found: {tm_parsed_path}")
        return

    with open(mn_parsed_path, 'r') as f_mn, open(tm_parsed_path, 'r') as f_tm:
        try:
            mayanode_parsed_block = json.load(f_mn)
            tendermint_parsed_block = json.load(f_tm)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from parsed block files: {e}")
            return

    print(f"Comparing Mayanode parsed block ({mn_parsed_path.name}) with Tendermint parsed block ({tm_parsed_path.name})...")
    
    # Before recursive comparison, handle potential datetime string vs. object differences if not already str
    # For this comparison, we assume parse_confirmed_block consistently outputs datetime objects as strings due to default=str in json.dump
    # If not, pre-processing would be needed here to normalize them before comparison.

    differences = compare_dicts_recursive(mayanode_parsed_block, tendermint_parsed_block)

    if not differences:
        print("  SUCCESS: The fully parsed block structures from Mayanode and Tendermint sources MATCH.")
    else:
        print("  MISMATCH: The fully parsed block structures differ. Differences found:")
        for diff in differences:
            print(f"    - {diff}")
    print("--- End of Full Parsed Block Comparison ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Mayanode and Tendermint block data.")
    parser.add_argument(
        "height",
        type=int,
        nargs='?', # Makes the argument optional
        default=DEFAULT_BLOCK_HEIGHT,
        help=f"The block height to process (default: {DEFAULT_BLOCK_HEIGHT})"
    )
    args = parser.parse_args()

    BLOCK_HEIGHT_TO_PROCESS = args.height

    if fetch_mayanode_block:
        # Ensure data is fetched and saved first
        main(BLOCK_HEIGHT_TO_PROCESS) # main() will now call detailed_comparison internally
        # # Check if files exist before trying to compare. If not, run main() to generate them.
        # tm_decoded_path = OUTPUT_DIR / f"tendermint_block_{BLOCK_HEIGHT}_decoded_transformed.json"
        # mn_raw_path = OUTPUT_DIR / f"mayanode_block_{BLOCK_HEIGHT}_api_full.json"
        # if not tm_decoded_path.exists() or not mn_raw_path.exists():
        #     print(f"JSON files for block {BLOCK_HEIGHT} not found. Running main() to generate them first...")
        #     main() # Run main processing if files are missing
        #     print("--- Main processing finished. Proceeding to detailed comparison. ---")
        
        # # Now perform the detailed comparison using the (potentially just created) files
        # detailed_comparison(BLOCK_HEIGHT) # Old call style
    else:
        print("Exiting due to missing critical api_connections import.") 