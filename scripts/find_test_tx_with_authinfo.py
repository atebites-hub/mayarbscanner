import sys
import os
import base64
import json
from pathlib import Path
import time

# --- Adjust sys.path for local imports ---
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = _SCRIPT_DIR.parent
PROTO_GEN_PARENT_DIR = PROJECT_ROOT_DIR / "proto" / "generated"

# Add project root to sys.path to allow importing from src
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# Add the parent directory of our 'pb_stubs' package to sys.path
if str(PROTO_GEN_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_GEN_PARENT_DIR))

try:
    from src.api_connections import fetch_tendermint_rpc_block_raw
    from pb_stubs.cosmos.tx.v1beta1 import Tx as CosmosTx
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Ensure that proto stubs have been generated and sys.path is correct.")
    sys.exit(1)

# Configuration
# Using a block height known to have transactions. 
# You might need to adjust this if it has no txs with signer_infos.
# Or implement logic to fetch latest block and iterate backwards.
DEFAULT_BLOCK_HEIGHT = 11255442 
# Let's try a more recent block that is more likely to have various tx types
RECENT_BLOCK_HEIGHT = 11320570 # Updated to a more recent block based on api_connections test
MAX_BLOCKS_TO_SCAN = 50 # Define how many blocks to scan backwards

def find_transaction_with_signer_info(block_height: int):
    """
    Fetches a block and searches for a transaction with populated signer_infos.
    Prints the base64 encoded transaction string if found.
    """
    print(f"Attempting to fetch raw block from Tendermint RPC: {block_height}")
    # MODIFIED: Use new function for raw Tendermint data
    block_result_json = fetch_tendermint_rpc_block_raw(height=block_height) 

    if not block_result_json:
        print(f"Failed to fetch raw block data from Tendermint RPC for height {block_height}.")
        return None

    # Access transactions from the standard Tendermint RPC path: result.block.data.txs
    # block_result_json is already the "result" part of the response.
    txs_b64_list = block_result_json.get("block", {}).get("data", {}).get("txs")
    
    if txs_b64_list is None:
        print(f"Could not locate transactions list (result.block.data.txs) in Tendermint RPC response for block {block_height}.")
        print(f"Block result sample (first 500 chars): {str(block_result_json)[:500]}")
        return None

    if not txs_b64_list: # Empty list means no transactions
        print(f"No transactions found in block {block_height} (txs list was empty).")
        return None

    print(f"Found {len(txs_b64_list)} transactions in block {block_height}. Analyzing...")

    for i, tx_b64_string_from_rpc in enumerate(txs_b64_list): # Renamed tx_item to tx_b64_string_from_rpc
        print(f"  Analyzing tx {i+1}/{len(txs_b64_list)}... Item type: {type(tx_b64_string_from_rpc)}")
        
        if not isinstance(tx_b64_string_from_rpc, str):
            print(f"    Tx item from RPC is not a string as expected (type: {type(tx_b64_string_from_rpc)}). Skipping.")
            print(f"    Item sample: {str(tx_b64_string_from_rpc)[:200]}...")
            continue

        # actual_tx_b64_string is now tx_b64_string_from_rpc
        try:
            tx_bytes = base64.b64decode(tx_b64_string_from_rpc)
            decoded_tx_proto = CosmosTx()
            decoded_tx_proto.parse(tx_bytes)

            auth_info_obj = decoded_tx_proto.auth_info
            
            # Check if auth_info itself is present and not None
            if auth_info_obj is not None:
                # print(f"    Tx {i+1}: auth_info object is present.") # Debug
                # Check if signer_infos list is present and not empty
                if auth_info_obj.signer_infos and len(auth_info_obj.signer_infos) > 0:
                    print(f"\nSUCCESS: Found transaction in block {block_height} (index {i}) with populated auth_info.signer_infos!")
                    print(f"  Number of signer_infos: {len(auth_info_obj.signer_infos)}")
                    # Optionally print details of the first signer_info
                    # print(f"  First signer_info: {auth_info_obj.signer_infos[0].to_dict()}")
                    print(f"\nBase64 Encoded Transaction String:\n{tx_b64_string_from_rpc}\n")
                    return tx_b64_string_from_rpc
                # else:
                    # print(f"    Tx {i+1}: auth_info.signer_infos is None or empty.") # Debug
            # else:
                # print(f"    Tx {i+1}: auth_info object is None.") # Debug

        except Exception as e:
            print(f"Error decoding or parsing transaction {i} in block {block_height}: {e}")
            continue
    
    print(f"No transaction with populated signer_infos found in block {block_height}.")
    return None

if __name__ == "__main__":
    found_tx_string = None
    print(f"Starting scan for transaction with signer_infos, beginning from block {RECENT_BLOCK_HEIGHT} and going back up to {MAX_BLOCKS_TO_SCAN} blocks.")

    for i in range(MAX_BLOCKS_TO_SCAN):
        current_block_to_scan = RECENT_BLOCK_HEIGHT - i
        if current_block_to_scan <= 0:
            print("Reached block height 0 or less, stopping scan.")
            break
        
        print(f"\n--- Scanning Block: {current_block_to_scan} (Attempt {i+1}/{MAX_BLOCKS_TO_SCAN}) ---")
        found_tx_string = find_transaction_with_signer_info(current_block_to_scan)
        if found_tx_string:
            print(f"\nSUCCESS: Found a suitable transaction in block {current_block_to_scan}.")
            break
        else:
            # Small delay to be polite to the API if scanning many blocks quickly
            time.sleep(0.2)

    if not found_tx_string:
        print(f"\nScan complete. Could not find a suitable transaction after checking {MAX_BLOCKS_TO_SCAN} blocks (from {RECENT_BLOCK_HEIGHT} down to {RECENT_BLOCK_HEIGHT - MAX_BLOCKS_TO_SCAN + 1}).")
        print(f"Consider increasing MAX_BLOCKS_TO_SCAN or trying a different range if Mayanode activity is low or transactions are sparse.")
        # As a last resort, try the original DEFAULT_BLOCK_HEIGHT known to have txs, though maybe not signer_infos
        print(f"\n--- Last Resort: Trying original fallback block: {DEFAULT_BLOCK_HEIGHT} ---")
        found_tx_string = find_transaction_with_signer_info(DEFAULT_BLOCK_HEIGHT)
        if not found_tx_string:
            print(f"\nNo suitable transaction found in the fallback block {DEFAULT_BLOCK_HEIGHT} either.") 