#!/usr/bin/env python3

import json
from datetime import datetime
import base64 # Added for event attribute decoding

# Attempt to import generated protobuf classes
# These will be generated into src/ by scripts/generate_protos.sh
try:
    from cosmos.tx.v1beta1.tx_pb2 import Tx as CosmosTx
    # Use the official MessageToDict and MessageToJson from google.protobuf
    from google.protobuf.json_format import MessageToDict, MessageToJson
    # from google.protobuf.any_pb2 import Any as ProtoAny # Not directly used for basic Tx parsing
    # from mayachain.v1.x.mayachain.types.msg_network_fee_pb2 import MsgNetworkFee # Not used for basic Tx parsing
    PROTOBUF_AVAILABLE = True
    # MSG_NETWORK_FEE_AVAILABLE = True
except ImportError as e:
    print(f"ImportError during protobuf setup: {e}")
    CosmosTx = None
    MessageToDict = None
    MessageToJson = None
    # ProtoAny = None
    # MsgNetworkFee = None 
    PROTOBUF_AVAILABLE = False
    # MSG_NETWORK_FEE_AVAILABLE = False
    print("Warning: Protobuf libraries (CosmosTx or google.protobuf.json_format) not available. Base64 transaction decoding will be skipped.")

def parse_iso_datetime(datetime_str: str):
    """Safely parses an ISO 8601 datetime string, handling potential missing milliseconds or Z."""
    if not datetime_str:
        return None
    try:
        # Handle cases with or without 'Z' and varying fractional seconds
        if datetime_str.endswith('Z'):
            datetime_str = datetime_str[:-1] + '+00:00'
        
        # Common formats: with 6, 3, or 0 fractional seconds
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z", 
            "%Y-%m-%dT%H:%M:%S%z" # No fractional seconds
        ):
            try:
                # Python's %f can handle 1 to 6 digits for microseconds.
                # If more, they need to be truncated.
                if '.' in datetime_str:
                    parts = datetime_str.split('.')
                    if len(parts) > 1:
                        micros_part = parts[1].split('+')[0].split('-')[0] # get only microsecond digits
                        if len(micros_part) > 6:
                            datetime_str = parts[0] + '.' + micros_part[:6] + parts[1][len(micros_part):]
                
                return datetime.fromisoformat(datetime_str)
            except ValueError:
                continue
        print(f"Warning: Could not parse datetime string: {datetime_str} with common ISO formats.")
        return None # Or raise an error, or return the original string
    except Exception as e:
        print(f"Error parsing datetime string '{datetime_str}': {e}")
        return None

def _parse_event_attributes(attributes_list: list) -> dict:
    """
    Parses a list of event attributes, decoding base64 keys and values.
    Each attribute in the list is expected to be a dict with 'key' and 'value'.
    """
    parsed_attrs = {}
    if not isinstance(attributes_list, list):
        return parsed_attrs
        
    for attr in attributes_list:
        if not isinstance(attr, dict) or 'key' not in attr or 'value' not in attr:
            # print(f"Warning: Skipping invalid attribute format: {attr}")
            continue
        try:
            key = base64.b64decode(attr['key']).decode('utf-8')
            # Value can sometimes be null or not base64
            value_raw = attr['value']
            if value_raw:
                try:
                    value = base64.b64decode(value_raw).decode('utf-8')
                except (TypeError, base64.binascii.Error): # If not valid base64, use as is
                    # print(f"Warning: Attribute value for key '{key}' not valid base64, using raw: {value_raw}")
                    value = value_raw 
            else: # Value is None or empty string
                value = value_raw 
            parsed_attrs[key] = value
        except (TypeError, base64.binascii.Error, UnicodeDecodeError) as e:
            # Fallback for key decoding error or if value is not actually b64
            # print(f"Warning: Error decoding attribute key/value: {attr}. Error: {e}")
            # Try to use raw key if possible, or skip
            raw_key = attr.get('key')
            raw_value = attr.get('value')
            if isinstance(raw_key, str):
                 parsed_attrs[raw_key] = raw_value # Store raw if key is at least a string
            # else skip
    return parsed_attrs

def _try_decode_base64(value_raw):
    """Tries to decode a value from base64; returns original if not valid base64 or if None."""
    if value_raw is None: # Explicitly check for None
        return None
    if not isinstance(value_raw, str): # Ensure it's a string before trying to decode
        return value_raw
    try:
        # Check if it might be base64. A simple heuristic: length and char set.
        # This is not foolproof but can prevent errors on clearly non-base64 strings.
        # A more robust check might involve trying to decode and catching specific errors.
        if len(value_raw) % 4 == 0 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in value_raw):
            return base64.b64decode(value_raw).decode('utf-8')
        return value_raw # Return raw if it doesn't look like base64
    except (TypeError, base64.binascii.Error, UnicodeDecodeError):
        return value_raw # Return raw if decoding fails

def parse_confirmed_block(block_json_data: dict) -> dict:
    """
    Parses a raw JSON object representing a confirmed block from Mayanode
    into a structured dictionary.

    Args:
        block_json_data: A dictionary loaded from the block's JSON file.

    Returns:
        A dictionary containing key information extracted from the block,
        or None if essential data is missing.
    """
    if not block_json_data or not isinstance(block_json_data, dict):
        print("Error: Invalid or empty block_json_data provided.")
        return None

    header = block_json_data.get("header", {})
    block_id_info = block_json_data.get("id", {})

    parsed_block = {
        "block_height": header.get("height"),
        "block_hash": block_id_info.get("hash"),
        "block_time_str": header.get("time"), # Store original string
        "block_time_dt": parse_iso_datetime(header.get("time")), # Store datetime object
        "chain_id": header.get("chain_id"),
        "proposer_address": header.get("proposer_address"),
        "last_block_hash": header.get("last_block_id", {}).get("hash"),
        
        # Core Hashes
        "data_hash": header.get("data_hash"),
        "validators_hash": header.get("validators_hash"),
        "next_validators_hash": header.get("next_validators_hash"),
        "consensus_hash": header.get("consensus_hash"),
        "app_hash": header.get("app_hash"),
        "last_results_hash": header.get("last_results_hash"),
        "evidence_hash": header.get("evidence_hash"),

        "begin_block_events_raw": block_json_data.get("begin_block_events", []),
        "end_block_events_raw": block_json_data.get("end_block_events", []),
        "begin_block_events_parsed": [], # Added for parsed events
        "end_block_events_parsed": [],   # Added for parsed events
        
        "transactions_raw": [], # Will hold raw transaction objects from the block
        "is_template": False
    }

    # Process transactions
    final_tx_list_for_parsing = [] # This will hold DICTs for parse_transaction_data

    # Case 1: Transactions are already decoded JSON objects at the top level
    # This is what fetch_mayanode_block provides if it gets data from Mayanode REST /mayachain/block
    top_level_txs = block_json_data.get("txs")
    if isinstance(top_level_txs, list) and top_level_txs and all(isinstance(tx, dict) for tx in top_level_txs):
        # print(f"Info: Using top-level 'txs' list with {len(top_level_txs)} dictionary items.")
        final_tx_list_for_parsing = top_level_txs
        parsed_block["transactions_source"] = "top_level_json_objects"
    
    # Case 2: Transactions are in data.txs 
    # This is what fetch_mayanode_block's reconstruction provides if original source was Tendermint RPC /block
    else: # Only check data.txs if top_level_txs wasn't suitable
        data_txs = block_json_data.get("data", {}).get("txs")
        if isinstance(data_txs, list) and data_txs: # Ensure list is not empty before checking first item
            if isinstance(data_txs[0], str): # Heuristic: if first is string, assume all are base64
                parsed_block["transactions_raw_base64"] = data_txs # Keep the raw ones for audit if needed
                parsed_block["transactions_source"] = "data_txs_base64_strings_decoded" # New source type
                
                if PROTOBUF_AVAILABLE and CosmosTx and MessageToDict:
                    print(f"[DEBUG common_utils] Attempting to decode {len(data_txs)} base64 transactions using Protobuf.")
                    decoded_tx_dictionaries = []
                    for i, b64_tx_string in enumerate(data_txs):
                        try:
                            tx_bytes = base64.b64decode(b64_tx_string)
                            if i == 0: # DEBUG for the first transaction
                                print(f"\n--- DEBUG TX INDEX {i} ---")
                                print(f"Raw b64 string (first 50 chars): {b64_tx_string[:50]}")
                                print(f"tx_bytes length: {len(tx_bytes)}")
                                print(f"tx_bytes (first 20 bytes, hex): {tx_bytes[:20].hex()}")

                            cosmos_tx_message = CosmosTx()
                            cosmos_tx_message.FromString(tx_bytes)
                            
                            if i == 0: # DEBUG for the first transaction
                                print(f"cosmos_tx_message IsInitialized(): {cosmos_tx_message.IsInitialized()}")
                                populated_fields = []
                                try:
                                    populated_fields = cosmos_tx_message.ListFields()
                                    print(f"cosmos_tx_message.ListFields(): {populated_fields}")
                                    if not populated_fields:
                                        print("WARNING: ListFields() is empty after FromString!")
                                except Exception as e_list_fields:
                                    print(f"Error calling ListFields(): {e_list_fields}")
                                # Simplified debug below
                                print(f"Attempting MessageToJson for tx index {i}")

                            # Convert the protobuf message to a dictionary using json_format.MessageToJson
                            tx_dict_from_proto = {} # Default to empty
                            try:
                                # First try with including_default_value_fields=True
                                tx_json_string = MessageToJson(
                                    cosmos_tx_message,
                                    preserving_proto_field_name=True,
                                    including_default_value_fields=True
                                )
                                tx_dict_from_proto = json.loads(tx_json_string)
                            except TypeError: # If including_default_value_fields is not supported (should not happen with 4.25.3)
                                print("Warning: including_default_value_fields=True not supported by MessageToJson. Trying without it.")
                                try:
                                    tx_json_string = MessageToJson(
                                        cosmos_tx_message,
                                        preserving_proto_field_name=True
                                    )
                                    tx_dict_from_proto = json.loads(tx_json_string)
                                except Exception as e_to_json_inner:
                                    print(f"Error using MessageToJson (fallback) or json.loads: {e_to_json_inner}")
                                    tx_dict_from_proto = {} # Fallback to empty dict
                            except Exception as e_to_json_outer:
                                print(f"Error using MessageToJson or json.loads: {e_to_json_outer}")
                                tx_dict_from_proto = {} # Fallback to empty dict

                            # if i == 0: # DEBUG for the first transaction
                            #     print(f"tx_dict_from_proto (first tx) content written to decoded_tx_sample.json")
                            #     with open("decoded_tx_sample.json", "w") as f_out:
                            #         json.dump(tx_dict_from_proto, f_out, indent=2)
                            #     print("--- END DEBUG ---\n")
                            
                            # Construct the object for parse_transaction_data
                            # The actual tx_hash is usually calculated over the tx_bytes, not part of the Tx proto itself.
                            # For Tendermint RPC, the hash is available alongside the tx string in the original block data.
                            # However, here we only have the string.
                            # We will need to enhance `fetch_mayanode_block` or where these base64 strings originate
                            # to also pass along their corresponding hashes if we want to store them.
                            # For now, using a placeholder.
                            # The 'result' field (events, gas used) also comes separately in RPC responses.
                            # We will have to see how to merge this if we go this route.
                            # For now, making it an empty dict to satisfy parse_transaction_data.
                            reconstructed_tx_obj = {
                                "hash": f"PROTO_DECODED_NO_HASH_INDEX_{i}", # Placeholder
                                "tx": tx_dict_from_proto, # This is the main content
                                "result": {} # Placeholder for events, gas, etc.
                            }
                            decoded_tx_dictionaries.append(reconstructed_tx_obj)
                        except Exception as e:
                            print(f"Error decoding/parsing protobuf for tx index {i}: {e}")
                            # Optionally, add a placeholder or skip
                    # End of the loop for decoding individual transactions
                    
                    # After decoding all transactions, save the list of *just the tx content dicts*
                    # This is to match the structure of mayanode_tx_sample.json which is a list of tx contents.
                    just_the_tx_dicts = [item["tx"] for item in decoded_tx_dictionaries if isinstance(item, dict) and "tx" in item]
                    print(f"Full list of {len(just_the_tx_dicts)} decoded transaction *contents* written to decoded_tx_sample.json")
                    with open("decoded_tx_sample.json", "w") as f_out:
                        json.dump(just_the_tx_dicts, f_out, indent=2)
                    
                    final_tx_list_for_parsing = decoded_tx_dictionaries # This list still contains the full reconstructed_tx_obj for further parsing stages
                else:
                    print("[DEBUG common_utils] Protobuf libraries not available. Skipping decoding of base64 transactions.")
                    # final_tx_list_for_parsing remains empty, behavior as before.
            elif isinstance(data_txs[0], dict): # Heuristic: if first is dict, assume all are
                print(f"[DEBUG common_utils] Found 'data.txs' list with {len(data_txs)} dictionary items. Processing these.") # DEBUG
                final_tx_list_for_parsing = data_txs
                parsed_block["transactions_source"] = "data_txs_json_objects"
            else: # Non-empty list, but not all strings or all dicts based on first element
                print(f"[DEBUG common_utils] 'data.txs' is a list but its first element is neither str nor dict: {type(data_txs[0])}. Not processing txs from data.txs.") # DEBUG
        elif isinstance(data_txs, list) and not data_txs: # Empty list
             print("[DEBUG common_utils] 'data.txs' is an empty list.") # DEBUG
        # else: block_json_data.data.txs is not a list or doesn't exist.
    
    print(f"[DEBUG common_utils] final_tx_list_for_parsing length: {len(final_tx_list_for_parsing)}") # DEBUG
    parsed_transactions_for_db = []
    if final_tx_list_for_parsing: # This list contains DICTs
        for tx_obj_dict in final_tx_list_for_parsing:
            parsed_transactions_for_db.append(parse_transaction_data(tx_obj_dict))
    
    parsed_block["transactions"] = parsed_transactions_for_db
    print(f"[DEBUG common_utils] parsed_block['transactions'] length: {len(parsed_block.get('transactions', []))}, source: {parsed_block.get('transactions_source')}") # DEBUG
    
    # We can decide later if we want to keep transactions_raw or remove it 
    # For now, let's remove it from the top-level to avoid confusion if transactions key exists
    # and if we didn't populate transactions_raw_base64
    if "transactions" in parsed_block and not parsed_block.get("transactions_raw_base64"):
        parsed_block.pop("transactions_raw", None) # transactions_raw was the old key for raw tx objects

    # Parse Begin Block Events
    if parsed_block["begin_block_events_raw"]:
        for raw_event in parsed_block["begin_block_events_raw"]:
            if isinstance(raw_event, dict) and 'type' in raw_event:
                event_type = raw_event["type"]
                attributes = {}
                for key, value in raw_event.items():
                    if key != 'type':
                        attributes[key] = _try_decode_base64(value)
                
                parsed_block["begin_block_events_parsed"].append({
                    "type": event_type,
                    "attributes": attributes
                })
            else:
                # print(f"Warning: Skipping invalid begin_block_event format: {raw_event}")
                pass # Silently skip malformed events for now

    # Parse End Block Events
    if parsed_block["end_block_events_raw"]:
        for raw_event in parsed_block["end_block_events_raw"]:
            if isinstance(raw_event, dict) and 'type' in raw_event:
                event_type = raw_event["type"]
                attributes = {}
                for key, value in raw_event.items():
                    if key != 'type':
                        attributes[key] = _try_decode_base64(value)
                        
                parsed_block["end_block_events_parsed"].append({
                    "type": event_type,
                    "attributes": attributes
                })
            else:
                # print(f"Warning: Skipping invalid end_block_event format: {raw_event}")
                pass # Silently skip malformed events for now

    return parsed_block

def parse_transaction_data(tx_obj: dict) -> dict:
    """
    Parses a single raw transaction object from a block.
    - Assumes tx_obj["tx"] is already a JSON-like dict for confirmed blocks.
    - Parses the transaction result events.
    Args:
        tx_obj: A dictionary representing a single transaction from block_json_data["txs"]
    Returns:
        A dictionary containing the parsed transaction data.
    """
    parsed_tx = {
        "hash": tx_obj.get("hash"),
        "tx_content_json": tx_obj.get("tx"), # Assumed to be dict for confirmed blocks
        "result_log": tx_obj.get("result", {}).get("log"),
        "result_gas_wanted": tx_obj.get("result", {}).get("gas_wanted"),
        "result_gas_used": tx_obj.get("result", {}).get("gas_used"),
        "result_events_raw": tx_obj.get("result", {}).get("events", []), # Keep raw for now
        "result_events_parsed": []
    }

    # Parse Transaction Result Events
    # These seem to follow the same structure as block-level events in Mayanode API
    raw_result_events = parsed_tx["result_events_raw"]
    if raw_result_events:
        for raw_event in raw_result_events:
            if isinstance(raw_event, dict) and 'type' in raw_event:
                event_type = raw_event["type"]
                attributes = {}
                for key, value in raw_event.items():
                    if key != 'type':
                        attributes[key] = _try_decode_base64(value)
                parsed_tx["result_events_parsed"].append({
                    "type": event_type,
                    "attributes": attributes
                })
            # else: print(f"Skipping malformed tx result event: {raw_event}")

    return parsed_tx

# --- Address Extraction Helpers ---

def is_mayanode_address(s: str) -> bool:
    """Basic check if a string looks like a Mayanode address."""
    if not isinstance(s, str):
        return False
    # Typical Mayanode addresses start with 'maya1' and are around 43-44 characters long
    # (prefix + 38 data chars for bech32)
    # This is a basic heuristic, not a full bech32 validation.
    return s.startswith("maya1") and 40 < len(s) < 50


def extract_addresses_from_parsed_tx(parsed_tx_data: dict) -> set[str]:
    """Extracts potential Mayanode addresses from a single parsed transaction."""
    addresses = set()

    if not parsed_tx_data or not isinstance(parsed_tx_data, dict):
        return addresses

    # 1. Check common fields in messages within tx_content_json
    tx_content = parsed_tx_data.get("tx_content_json", {})
    if isinstance(tx_content, dict):
        # Safely access body and messages
        body = tx_content.get("body", {})
        messages = body.get("messages", []) if isinstance(body, dict) else []
        
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    # Check common address-holding keys in messages
                    for key, value in msg.items():
                        if isinstance(value, str) and is_mayanode_address(value):
                            addresses.add(value)
                        # Look for 'signer' or 'from_address', 'to_address', 'delegator_address', etc.
                        # This can be expanded with more known keys
                        if key in ["signer", "from_address", "to_address", "delegator_address", "validator_address", "granter", "grantee"] and isinstance(value, str) and is_mayanode_address(value):
                            addresses.add(value)
                        # Sometimes addresses are nested, e.g. in a list of signers or participants
                        if isinstance(value, list):
                            for item_in_list in value:
                                if isinstance(item_in_list, str) and is_mayanode_address(item_in_list):
                                    addresses.add(item_in_list)
    
    # 2. Check event attributes
    # Events are already parsed into 'type' and 'attributes' (dict)
    result_events = parsed_tx_data.get("result_events_parsed", [])
    if isinstance(result_events, list):
        for event in result_events:
            if isinstance(event, dict):
                attributes = event.get("attributes", {})
                if isinstance(attributes, dict):
                    for attr_key, attr_value in attributes.items():
                        if isinstance(attr_value, str) and is_mayanode_address(attr_value):
                            addresses.add(attr_value)
                        # Sometimes an attribute key might indicate an address list
                        # e.g. key "participants" and value is a comma-separated string of addresses
                        if isinstance(attr_value, str) and ',' in attr_value:
                            possible_addrs = attr_value.split(',')
                            for pa in possible_addrs:
                                if is_mayanode_address(pa.strip()):
                                    addresses.add(pa.strip())

    # 3. Check Signer Information (if available and directly as address)
    # The current parsed_tx_data.tx_content_json.auth_info.signer_infos contains pubkeys,
    # not directly addresses. Conversion is complex.
    # However, sometimes fee_payer or other fields in auth_info might be an address.
    if isinstance(tx_content, dict):
        # Safely access auth_info and fee
        auth_info = tx_content.get("auth_info", {})
        fee_info = auth_info.get("fee", {}) if isinstance(auth_info, dict) else {}

        if isinstance(fee_info, dict):
            payer = fee_info.get("payer")
            granter = fee_info.get("granter")
            if isinstance(payer, str) and is_mayanode_address(payer):
                addresses.add(payer)
            if isinstance(granter, str) and is_mayanode_address(granter):
                addresses.add(granter)

    return addresses


def extract_addresses_from_parsed_block(parsed_block_data: dict) -> set[str]:
    """Extracts all unique potential Mayanode addresses from a parsed block."""
    all_addresses = set()
    if not parsed_block_data or not isinstance(parsed_block_data, dict):
        return all_addresses

    # 1. Proposer address from block header
    proposer = parsed_block_data.get("proposer_address")
    if is_mayanode_address(proposer):
        all_addresses.add(proposer)

    # 2. Addresses from transactions
    transactions = parsed_block_data.get("transactions", [])
    if isinstance(transactions, list):
        for tx_data in transactions:
            tx_addresses = extract_addresses_from_parsed_tx(tx_data)
            all_addresses.update(tx_addresses)

    # 3. Addresses from block-level events (begin_block_events, end_block_events)
    for event_list_key in ["begin_block_events_parsed", "end_block_events_parsed"]:
        events = parsed_block_data.get(event_list_key, [])
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict):
                    attributes = event.get("attributes", {})
                    if isinstance(attributes, dict):
                        for attr_key, attr_value in attributes.items():
                            if isinstance(attr_value, str) and is_mayanode_address(attr_value):
                                all_addresses.add(attr_value)
                            if isinstance(attr_value, str) and ',' in attr_value: # Check for comma-separated list
                                possible_addrs = attr_value.split(',')
                                for pa in possible_addrs:
                                    if is_mayanode_address(pa.strip()):
                                        all_addresses.add(pa.strip())
    return all_addresses

if __name__ == '__main__':
    # Example Usage (requires a sample block JSON file)
    sample_block_file_path = "data/mayanode_blocks/block_11255511.json"
    try:
        with open(sample_block_file_path, 'r') as f:
            sample_block_data = json.load(f)
        
        print(f"--- Parsing block from {sample_block_file_path} ---")
        parsed_data = parse_confirmed_block(sample_block_data)

        if parsed_data:
            print(f"Successfully parsed block height: {parsed_data.get('block_height')}")
            print(f"Block time (str): {parsed_data.get('block_time_str')}")
            print(f"Block time (dt): {parsed_data.get('block_time_dt')}")
            print(f"Proposer: {parsed_data.get('proposer_address')}")
            print(f"Number of parsed transactions in block: {len(parsed_data.get('transactions', []))}")
            
            if parsed_data.get('transactions'):
                print("Sample of first parsed transaction object (keys):", list(parsed_data['transactions'][0].keys()))
                # print("Full first raw tx object:", json.dumps(parsed_data['transactions_raw'][0], indent=2))

            print(f"Number of raw begin_block_events: {len(parsed_data.get('begin_block_events_raw', []))}")
            print(f"Number of parsed begin_block_events: {len(parsed_data.get('begin_block_events_parsed', []))}")
            if parsed_data.get('begin_block_events_parsed'):
                print("Sample of first parsed begin_block_event:", json.dumps(parsed_data['begin_block_events_parsed'][0], indent=2))
            elif parsed_data.get('begin_block_events_raw'): # Print raw if parsed is empty but raw is not
                print("Sample of first raw begin_block_event (since parsed is empty):", json.dumps(parsed_data['begin_block_events_raw'][0], indent=2))

            print(f"Number of raw end_block_events: {len(parsed_data.get('end_block_events_raw', []))}")
            print(f"Number of parsed end_block_events: {len(parsed_data.get('end_block_events_parsed', []))}")
            if parsed_data.get('end_block_events_parsed'):
                print("Sample of first parsed end_block_event:", json.dumps(parsed_data['end_block_events_parsed'][0], indent=2))

            # Inspect first transaction (now parsed)
            if parsed_data.get('transactions'):
                first_parsed_tx = parsed_data['transactions'][0]
                print("\n--- Inspecting First Parsed Transaction --- ")
                print(f"Transaction Hash: {first_parsed_tx.get('hash')}")
                print("Transaction Content (JSON - showing keys):", list(first_parsed_tx.get('tx_content_json', {}).keys()))
                # print("Full Transaction Content (JSON):", json.dumps(first_parsed_tx.get('tx_content_json'), indent=2))
                print(f"Result Gas Used: {first_parsed_tx.get('result_gas_used')}")
                print(f"Number of parsed result events for first tx: {len(first_parsed_tx.get('result_events_parsed', []))}")
                if first_parsed_tx.get('result_events_parsed'):
                    print("Sample of first parsed result event for first tx:")
                    print(json.dumps(first_parsed_tx['result_events_parsed'][0], indent=2))
            
        else:
            print("Failed to parse block data.")

    except FileNotFoundError:
        print(f"Error: Sample block file not found at {sample_block_file_path}. Cannot run example.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {sample_block_file_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # TODO: Add example for parsing a 'next_block_template' if we create a parser for it
    # next_block_template_example = { # From api_connections.construct_next_block_template()
    #     "header": {"height": "PENDING", "chain_id": "mayachain-mainnet-v1", ...},
    #     "txs": ["BASE64_TX_STRING_1", "BASE64_TX_STRING_2", ...] # Note: txs is a list of strings here
    # }
    # # parsed_template = parse_block_template(next_block_template_example)
    # # print_parsed_data(parsed_template)

    print("\n--- Testing Address Extraction ---")
    sample_tx_message_with_signer = {
        "@type": "/types.MsgSend",
        "from_address": "maya1qqq...", # invalid for now
        "to_address": "maya1aaa...invalid", # invalid for now
        "signer": "maya12cljaw4j0fj795yzyq9q3e72zmsfwk7z3zy99v", # Valid
        "amount": [{"denom": "maya", "amount": "1000"}]
    }
    sample_tx_content_with_messages = {
        "body": {
            "messages": [sample_tx_message_with_signer, {"@type": "/types.MsgDeposit", "signer": "maya1zgw6t904sph0q0uwu93gqe0sfhdusymx9ycfy8"}]
        },
        "auth_info": {
            "fee": { "payer": "maya12cljaw4j0fj795yzyq9q3e72zmsfwk7z3zy99v"}
        }
    }
    sample_event_attributes = {
        "recipient": "maya1fpalp2vk9tstshs7k202vg3j26ggc5pqw2f9e5", # Valid
        "sender_list": "maya1zgw6t904sph0q0uwu93gqe0sfhdusymx9ycfy8, maya12cljaw4j0fj795yzyq9q3e72zmsfwk7z3zy99v"
    }
    sample_parsed_tx = {
        "hash": "TXHASH123",
        "tx_content_json": sample_tx_content_with_messages,
        "result_events_parsed": [
            {"type": "transfer", "attributes": sample_event_attributes}
        ]
    }

    extracted_addrs_tx = extract_addresses_from_parsed_tx(sample_parsed_tx)
    print(f"Extracted from sample tx: {extracted_addrs_tx}")
    # Expected: {'maya12cljaw4j0fj795yzyq9q3e72zmsfwk7z3zy99v', 
    #            'maya1zgw6t904sph0q0uwu93gqe0sfhdusymx9ycfy8', 
    #            'maya1fpalp2vk9tstshs7k202vg3j26ggc5pqw2f9e5'}

    sample_parsed_block_for_addr = {
        "proposer_address": "maya1mryx65dqz7df9vcvx3fhdgr5w2q5c2y5ks7n3r", # Valid
        "transactions": [sample_parsed_tx],
        "begin_block_events_parsed": [
            {"type": "validator_slash", "attributes": {"address": "maya1ua529fgcdey76304h57rftx350prn94fp5h7mq", "power": "100"}}
        ]
    }
    extracted_addrs_block = extract_addresses_from_parsed_block(sample_parsed_block_for_addr)
    print(f"Extracted from sample block: {extracted_addrs_block}")
    # Expected to include proposer, tx addresses, and event address.

    test_addr = "maya12cljaw4j0fj795yzyq9q3e72zmsfwk7z3zy99v"
    print(f"Is '{test_addr}' a Mayanode address? {is_mayanode_address(test_addr)}")
    test_addr_short = "maya12cljaw4j0"
    print(f"Is '{test_addr_short}' a Mayanode address? {is_mayanode_address(test_addr_short)}")
    test_addr_long = "maya12cljaw4j0fj795yzyq9q3e72zmsfwk7z3zy99vasdfasdfasdf"
    print(f"Is '{test_addr_long}' a Mayanode address? {is_mayanode_address(test_addr_long)}")
    test_addr_wrong_prefix = "thor12cljaw4j0fj795yzyq9q3e72zmsfwk7z3zy99v"
    print(f"Is '{test_addr_wrong_prefix}' a Mayanode address? {is_mayanode_address(test_addr_wrong_prefix)}")
    print(f"Is None an address? {is_mayanode_address(None)}")
    print(f"Is 123 an address? {is_mayanode_address(123)}")
