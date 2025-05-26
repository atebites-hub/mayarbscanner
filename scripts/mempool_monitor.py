#!/usr/bin/env python3

import os
import sys
import time
import base64
import json # For pretty printing dicts if needed

# --- Adjust sys.path to import from src --- 
_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_SCRIPT_DIR) # This should be 'mayarbscanner/'
_SRC_PATH = os.path.join(_PROJECT_ROOT, "src")

if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# Now we can import from src
from api_connections import fetch_decoded_tendermint_mempool_txs
# Import the new centralized decoder and PROTO_TYPES_AVAILABLE check from common_utils
from common_utils import decode_cosmos_tx_string_to_dict, PROTO_TYPES_AVAILABLE as COMMON_UTILS_PROTOBUF_AVAILABLE

if not COMMON_UTILS_PROTOBUF_AVAILABLE: # Check availability from common_utils
    print("CRITICAL: CosmosTx protobuf object not available via common_utils. Mempool monitor cannot decode transactions.")
    print("Please ensure protobufs are compiled and \'common_utils.py\' can import \'CosmosTx\'.")
    sys.exit(1)

# Configuration
REFRESH_INTERVAL_SECONDS = 5.5
TRANSACTION_DISPLAY_LIMIT = 10 # Max number of transactions to display details for

def clear_console():
    """Clears the console screen."""
    # ANSI escape code for clearing screen and moving cursor to home
    # \033[H moves cursor to top-left (home)
    # \033[J clears from cursor to end of screen
    print("\033[H\033[J", end="")

def get_message_types_from_dict(decoded_tx_dict: dict):
    """Extracts message type URLs from a decoded transaction dictionary."""
    if not decoded_tx_dict or not isinstance(decoded_tx_dict, dict):
        return ["N/A"]
    
    body = decoded_tx_dict.get("body", {})
    messages = body.get("messages", [])
    
    if not messages:
        return ["NoMessages"]
    
    msg_types = []
    for msg in messages:
        if isinstance(msg, dict) and "typeUrl" in msg: # Ensure msg is a dict and has typeUrl
            msg_types.append(msg["typeUrl"])
        elif isinstance(msg, dict) and "@type" in msg: # Legacy or alternative field name
             msg_types.append(msg["@type"])
        else:
            msg_types.append("UnknownMsgFormat")
            
    return msg_types if msg_types else ["NoMessages"]

def main():
    """Main loop to fetch and display mempool data."""
    print("Starting Mempool Monitor...")
    # TENDERMINT_RPC_BASE_URL is used internally by fetch_decoded_tendermint_mempool_txs
    print(f"Fetching from Tendermint RPC (via api_connections.py)") 
    print(f"Refresh interval: {REFRESH_INTERVAL_SECONDS}s")
    time.sleep(2) # Initial pause

    try:
        while True:
            clear_console()
            print(f"--- Mayanode Mempool Monitor (via Tendermint RPC) --- {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            unconfirmed_transactions = fetch_decoded_tendermint_mempool_txs(limit=100) 

            if unconfirmed_transactions is None:
                print("Error: Could not fetch unconfirmed transactions. Check API connection and tendermint_rpc_base_url.")
                print(f"Retrying in {REFRESH_INTERVAL_SECONDS} seconds...")
            elif not unconfirmed_transactions:
                print("Mempool is empty.")
            else:
                print(f"Total unconfirmed transactions/items in response: {len(unconfirmed_transactions)} (displaying up to {TRANSACTION_DISPLAY_LIMIT})\n")
                
                displayed_count = 0
                for i, tx_item in enumerate(unconfirmed_transactions):
                    if i >= TRANSACTION_DISPLAY_LIMIT:
                        print(f"\n... and {len(unconfirmed_transactions) - TRANSACTION_DISPLAY_LIMIT} more transactions not displayed.")
                        break
                    
                    if isinstance(tx_item, dict): # Successfully decoded dictionary
                        # Create a pseudo hash from one of its fields if possible, or a generic one
                        # For example, using part of the body memo or first message type as a stand-in
                        body_memo = tx_item.get("body", {}).get("memo", "")
                        first_msg_type = get_message_types_from_dict(tx_item)[0] if get_message_types_from_dict(tx_item) else "UnknownType"
                        
                        pseudo_identifier = f"Memo: {body_memo[:20]}... / Type: {first_msg_type}" \
                            if body_memo else f"Type: {first_msg_type}"
                        
                        msg_types = get_message_types_from_dict(tx_item)
                        msg_types_str = ", ".join(m.split('.')[-1].split('/')[-1] for m in msg_types)

                        print(f"  Tx {i+1:02d}: Identifier: {pseudo_identifier}")
                        print(f"       Msg Types : {msg_types_str}")
                        
                        # DEBUG: Print full JSON if type is 'UnknownMsgFormat'
                        if msg_types and msg_types[0] == "UnknownMsgFormat":
                            print(f"         [DEBUG UnknownMsgFormat] Full Decoded TX JSON for item {i+1}:")
                            try:
                                print(json.dumps(tx_item, indent=2))
                            except TypeError:
                                print("         [DEBUG UnknownMsgFormat] Could not serialize tx_item to JSON.")
                        
                        # To see full decoded tx always (uncomment):
                        # print(f"       Decoded Tx: {json.dumps(tx_item, indent=2)}")
                        displayed_count += 1
                    elif isinstance(tx_item, str): # Original base64 string (decoding failed at API level or was skipped)
                        pseudo_hash = tx_item[:10] + "..." + tx_item[-10:]
                        print(f"  Tx {i+1:02d}: Failed to decode (received raw string). Base64 (preview): {pseudo_hash}")
                        # The decode_cosmos_tx_string_to_dict is now called *inside* fetch_decoded_tendermint_mempool_txs.
                        # If we get a string here, it means the API call itself might have returned raw strings because
                        # COMMON_UTILS_PROTOBUF_AVAILABLE was false *within api_connections.py's scope* for that function.
                        # Or, decode_cosmos_tx_string_to_dict itself returned the string due to an internal error.
                    else: # Unexpected item type
                        print(f"  Tx {i+1:02d}: Unexpected item type in list: {type(tx_item)}. Content: {str(tx_item)[:50]}...")

                if displayed_count == 0 and unconfirmed_transactions:
                    print("\nNo transactions could be displayed as fully decoded dictionaries.")
                elif displayed_count > 0:
                    print(f"\nDisplayed details for {displayed_count} decoded transactions.")

            print(f"\nNext refresh in {REFRESH_INTERVAL_SECONDS} seconds. Press Ctrl+C to exit.")
            time.sleep(REFRESH_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nExiting Mempool Monitor.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if not COMMON_UTILS_PROTOBUF_AVAILABLE: # Check flag from common_utils
        print("Mempool monitor cannot run because CosmosTx protobufs are not available via common_utils.")
        print("Please check the import errors in 'common_utils.py' or re-run protobuf generation.")
    else:
        main() 