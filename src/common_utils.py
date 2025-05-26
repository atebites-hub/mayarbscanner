#!/usr/bin/env python3

import json
from datetime import datetime
import base64 # Added for event attribute decoding
import sys # For sys.path manipulation if needed for common_utils standalone test
import os # For path joining
import hashlib # For address derivation
import bech32 # For address derivation
import re # For camel_to_snake function
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import betterproto

# --- Adjust sys.path for protobuf imports (similar to api_connections.py) ---
# This ensures that common_utils can find pb_stubs if run directly or tested in isolation.
_current_script_dir_common = os.path.dirname(os.path.abspath(__file__))
# Project root is one level up from 'src' where common_utils.py is
_project_root_common = os.path.dirname(_current_script_dir_common) 
# Path to the parent of pb_stubs, i.e., 'proto/generated/'
_proto_generated_path_common = os.path.join(_project_root_common, "proto", "generated")

if _proto_generated_path_common not in sys.path:
    sys.path.insert(0, _proto_generated_path_common)
    # print(f"[common_utils.py] Prepended to sys.path: {_proto_generated_path_common}") # Optional: for debugging

# Attempt to import generated protobuf message types
PROTO_TYPES_AVAILABLE = False
CosmosTx = None
Secp256k1PubKey = None
MsgNetworkFee = None 
MsgTssKeysignFail = None
MsgConsolidate = None
MsgOutboundTx = None
MsgObservedTxIn = None # Initialize new type variable
# Add other specific type vars here, initialized to None

try:
    # Cosmos specific, widely used
    from pb_stubs.cosmos.tx.v1beta1 import Tx as CosmosTx_imported # Main Tx wrapper
    from pb_stubs.cosmos.crypto.secp256k1 import PubKey as Secp256k1PubKey_imported # For public keys
    # Mayanode specific types that are often packed in Any
    from pb_stubs.types import (
        MsgNetworkFee as MsgNetworkFee_imported,
        MsgTssKeysignFail as MsgTssKeysignFail_imported,
        MsgConsolidate as MsgConsolidate_imported,
        MsgOutboundTx as MsgOutboundTx_imported,
        MsgObservedTxIn as MsgObservedTxIn_imported, # Import MsgObservedTxIn
    )
    
    # Assign to global-like vars
    CosmosTx = CosmosTx_imported
    Secp256k1PubKey = Secp256k1PubKey_imported
    MsgNetworkFee = MsgNetworkFee_imported
    MsgTssKeysignFail = MsgTssKeysignFail_imported
    MsgConsolidate = MsgConsolidate_imported
    MsgOutboundTx = MsgOutboundTx_imported
    MsgObservedTxIn = MsgObservedTxIn_imported # Assign to global var
    
    PROTO_TYPES_AVAILABLE = True
    # print("[common_utils.py] Successfully imported betterproto CosmosTx, Secp256k1PubKey, and specific Mayanode types (MsgNetworkFee, MsgTssKeysignFail, MsgConsolidate, MsgOutboundTx, MsgObservedTxIn).") # Updated print
except ImportError as e:
    print(f"[common_utils.py] Warning: Protobuf types not fully available during import. Error: {e}. Advanced decoding might be limited.")
    # Vars remain None as initialized

# Mapping from Protobuf Any.type_url to the @type string Mayanode API uses
# This map helps in adding the correct "@type" string after decoding an Any.
TYPE_URL_TO_MAYANODE_TYPE = {
    # Cosmos SDK General
    "/cosmos.bank.v1beta1.MsgSend": "cosmos-sdk/MsgSend",
    "/cosmos.staking.v1beta1.MsgDelegate": "cosmos-sdk/MsgDelegate",
    "/cosmos.staking.v1beta1.MsgBeginRedelegate": "cosmos-sdk/MsgBeginRedelegate",
    "/cosmos.staking.v1beta1.MsgUndelegate": "cosmos-sdk/MsgUndelegate",
    "/cosmos.distribution.v1beta1.MsgWithdrawDelegatorReward": "cosmos-sdk/MsgWithdrawDelegatorReward",
    "/cosmos.gov.v1beta1.MsgVote": "cosmos-sdk/MsgVote",
    "/ibc.applications.transfer.v1.MsgTransfer": "cosmos-sdk/MsgTransfer",
    
    # THORChain/Mayanode specific types from /types.*
    # Adjusted to match Mayanode API's direct use of /types.* for @type in this block
    "/types.MsgSend": "/types.MsgSend", # Was "thorchain/MsgSend"
    "/types.MsgDeposit": "/types.MsgDeposit", # Was "thorchain/MsgDeposit"
    "/types.MsgOutboundTx": "/types.MsgOutboundTx", # Was "thorchain/MsgOutboundTx"
    "/types.MsgRefundTx": "/types.MsgRefundTx", # Was "thorchain/MsgRefundTx"
    "/types.MsgSwitch": "/types.MsgSwitch", # Was "thorchain/MsgSwitch"
    "/types.MsgSetVersion": "/types.MsgSetVersion", # Was "thorchain/MsgSetVersion"
    "/types.MsgConsolidate": "/types.MsgConsolidate", # Was "thorchain/MsgConsolidate"
    "/types.MsgYggdrasil": "/types.MsgYggdrasil", # Was "thorchain/MsgYggdrasil"
    "/types.MsgReserveContributor": "/types.MsgReserveContributor", # Was "thorchain/MsgReserveContributor"
    "/types.MsgBond": "/types.MsgBond", # Was "thorchain/MsgBond"
    "/types.MsgUnBond": "/types.MsgUnBond", # Was "thorchain/MsgUnBond"
    "/types.MsgLeave": "/types.MsgLeave", # Was "thorchain/MsgLeave"
    "/types.MsgDonate": "/types.MsgDonate", # Was "thorchain/MsgDonate"
    "/types.MsgNetworkFee": "/types.MsgNetworkFee", # Was "thorchain/MsgNetworkFee"
    "/types.MsgTssKeysignFail": "/types.MsgTssKeysignFail", # Was "thorchain/MsgTssKeysignFail"
    "/types.MsgObservedTxIn": "/types.MsgObservedTxIn", # Add mapping for MsgObservedTxIn
    # Add other /types.* mappings as needed if they appear in blocks

    # PubKey type - Mayanode API for this block uses the direct proto URL
    "/cosmos.crypto.secp256k1.PubKey": "/cosmos.crypto.secp256k1.PubKey", # Was "tendermint/PubKeySecp256k1"
}

# Dictionary mapping type URLs to their actual Python classes for decoding Any messages.
KNOWN_ANY_TYPES = {}
if PROTO_TYPES_AVAILABLE:
    # Populate only if imports were successful
    if Secp256k1PubKey: KNOWN_ANY_TYPES["/cosmos.crypto.secp256k1.PubKey"] = Secp256k1PubKey
    if MsgNetworkFee: KNOWN_ANY_TYPES["/types.MsgNetworkFee"] = MsgNetworkFee
    if MsgTssKeysignFail: KNOWN_ANY_TYPES["/types.MsgTssKeysignFail"] = MsgTssKeysignFail
    if MsgConsolidate: KNOWN_ANY_TYPES["/types.MsgConsolidate"] = MsgConsolidate
    if MsgOutboundTx: KNOWN_ANY_TYPES["/types.MsgOutboundTx"] = MsgOutboundTx
    if MsgObservedTxIn: KNOWN_ANY_TYPES["/types.MsgObservedTxIn"] = MsgObservedTxIn # Add to KNOWN_ANY_TYPES
    # TODO: Add other common /types.Msg... from Mayanode as they are identified and imported
else:
    print("[common_utils.py] KNOWN_ANY_TYPES is empty due to protobuf import errors.")

# Map chain identifiers to their Human-Readable Part (HRP) - from compare_block_data.py
CHAIN_TO_HRP = {
    "MAYA": "maya", "THOR": "thor", "COSMOS": "cosmos", "GAIA": "cosmos",
    "BTC": "bc", "ETH": None, "TERRA": "terra", "XRD": None,
}

# --- Helper Functions for Advanced Decoding (adapted from compare_block_data.py) ---

def try_bech32_decode(hrp, data_bytes):
    try:
        if hrp is None: return None
        converted_bits = bech32.convertbits(data_bytes, 8, 5)
        if converted_bits is None: return None
        return bech32.bech32_encode(hrp, converted_bits)
    except Exception:
        return None

def derive_cosmos_address_from_pubkey_bytes(pubkey_bytes, hrp):
    if not pubkey_bytes or hrp is None: return None
    try:
        sha256_hash = hashlib.sha256(pubkey_bytes).digest()
        ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
        return try_bech32_decode(hrp, ripemd160_hash)
    except ValueError: # ripemd160 not available (should be with Python 3.6+)
        print("Warning: ripemd160 hash not available in hashlib. Cannot derive address.")
        return None
    except Exception:
        return None

def deep_decode_any_messages(current_obj, tx_level_hrp=None):
    if isinstance(current_obj, dict):
        if current_obj.get("typeUrl") and "value" in current_obj and isinstance(current_obj.get("value"), str):
            type_url = current_obj["typeUrl"]
            encoded_value_b64 = current_obj["value"]
            try:
                decoded_bytes = base64.b64decode(encoded_value_b64)
                specific_msg_obj = KNOWN_ANY_TYPES[type_url]() # Get class constructor
                specific_msg_obj.parse(decoded_bytes)
                
                # Convert to dict, ensuring snake_case keys and all fields
                decoded_dict_content = specific_msg_obj.to_dict(include_default_values=True, casing=betterproto.Casing.SNAKE)
                
                current_obj.clear() # Clear the original Any structure {typeUrl, value}
                current_obj.update(decoded_dict_content) # Replace with decoded and snake_cased content

                # Inject @type using the mapped value
                mayanode_specific_type = TYPE_URL_TO_MAYANODE_TYPE.get(type_url, type_url) # Fallback to type_url if no map
                if mayanode_specific_type:
                    current_obj["@type"] = mayanode_specific_type
                
                # Special handling for PubKey remains, but @type is now part of current_obj
                if type_url == "/cosmos.crypto.secp256k1.PubKey" and tx_level_hrp:
                    pubkey_b64_for_addr = current_obj.get("key") # Get key from the now updated current_obj
                    if pubkey_b64_for_addr and isinstance(pubkey_b64_for_addr, str):
                        try:
                            actual_pubkey_bytes_for_addr = base64.b64decode(pubkey_b64_for_addr)
                            derived_address = derive_cosmos_address_from_pubkey_bytes(actual_pubkey_bytes_for_addr, tx_level_hrp)
                            if derived_address:
                                current_obj["derivedAddress"] = derived_address
                                # print(f"[DEBUG deep_decode] Added derivedAddress: {derived_address} to PubKey Any for typeUrl: {type_url}")
                        except Exception: 
                            pass # nosemgrep: general-exception-pass
                return current_obj
            except Exception: 
                pass # nosemgrep: general-exception-pass -- Safely suppress errors during optional deep decoding; original b64 value is kept
            return current_obj # Return original if any error or no specific decoder

        local_hrp_for_field = tx_level_hrp # Default to HRP of the transaction's main chain
        # Check if the current dictionary has a "chain" field to override HRP locally
        if "chain" in current_obj and isinstance(current_obj["chain"], str) and current_obj["chain"] in CHAIN_TO_HRP:
            if CHAIN_TO_HRP[current_obj["chain"]] is not None:
                 local_hrp_for_field = CHAIN_TO_HRP[current_obj["chain"]]

        for key, value in list(current_obj.items()): # Iterate over a copy if modifying dict
            if key == "signer" and isinstance(value, str) and local_hrp_for_field:
                # This was an attempt to decode signer fields if they were base64 bytes of an address.
                # However, 'signer' in Cosmos messages is typically already a Bech32 string.
                # Pubkeys are in signer_infos.public_key.
                # Keeping for now if some custom messages have base64 signer bytes.
                try:
                    signer_bytes = base64.b64decode(value)
                    bech32_addr = try_bech32_decode(local_hrp_for_field, signer_bytes)
                    if bech32_addr:
                        current_obj[key] = bech32_addr
                except Exception:
                    pass # nosemgrep: general-exception-pass -- Original value kept if not decodable as b64 to bech32
            current_obj[key] = deep_decode_any_messages(value, tx_level_hrp) # Pass the potentially updated local_hrp
        return current_obj
    elif isinstance(current_obj, list):
        for i in range(len(current_obj)):
            current_obj[i] = deep_decode_any_messages(current_obj[i], tx_level_hrp)
        return current_obj
    else: # Primitive type
        return current_obj

def decode_cosmos_tx_string_to_dict(tx_b64_string: str, default_hrp: str = "maya") -> dict | str:
    """
    Decodes a base64 encoded Cosmos transaction string into a dictionary.
    Performs deep decoding of known Any types and attempts to derive addresses.
    Args:
        tx_b64_string: The base64 encoded transaction string.
        default_hrp: The default Human-Readable Part for address derivation (e.g., "maya").
                     This HRP is used for deriving addresses from public keys found within
                     the transaction, particularly in `signer_infos` and `Any` messages
                     like `cosmos.crypto.secp256k1.PubKey`.
    Returns:
        A dictionary representing the fully decoded and processed transaction if successful.
        The dictionary structure aims to be detailed, with nested `Any` messages resolved
        to their specific types where possible, and derived Bech32 addresses added.
        Returns the original `tx_b64_string` if decoding fails or if the necessary
        Protobuf message definitions (e.g., `CosmosTx`) were not available during import.
    """
    # print(f"\n<<<<< [decode_cosmos_tx_string_to_dict] CALLED with tx_b64_string (first 50 chars): {tx_b64_string[:50]} >>>>>\n") # DEBUG Reverted

    if not CosmosTx: # Check if CosmosTx was successfully imported
        print("[common_utils.decode_cosmos_tx_string_to_dict] CosmosTx type not available. Cannot decode.")
        return tx_b64_string 
    try:
        tx_bytes = base64.b64decode(tx_b64_string)
        decoded_tx_proto = CosmosTx()
        decoded_tx_proto.parse(tx_bytes) # Use betterproto .parse() method
        
        decoded_tx_dict = decoded_tx_proto.to_dict(include_default_values=True)
        
        tx_hrp = CHAIN_TO_HRP.get(default_hrp.upper(), default_hrp)
        fully_decoded_tx_dict = deep_decode_any_messages(decoded_tx_dict, tx_hrp)

        if 'authInfo' in fully_decoded_tx_dict and isinstance(fully_decoded_tx_dict.get('authInfo'), dict):
            auth_info_processed = fully_decoded_tx_dict['authInfo']
            top_level_signatures_list = fully_decoded_tx_dict.get('signatures', []) # These are base64 strings

            if 'signerInfos' in auth_info_processed and isinstance(auth_info_processed.get('signerInfos'), list):
                for i, signer_info_item in enumerate(auth_info_processed['signerInfos']):
                    if isinstance(signer_info_item, dict): 
                        actual_signature_b64 = "N/A" # Default if not found
                        if isinstance(top_level_signatures_list, list) and i < len(top_level_signatures_list):
                            current_sig_b64 = top_level_signatures_list[i]
                            if isinstance(current_sig_b64, str): # Signatures are base64 strings
                                actual_signature_b64 = current_sig_b64
                        signer_info_item['associated_signature_b64'] = actual_signature_b64
                        
                        # Derive address from pubkey in signer_info.public_key if it's a cosmos.crypto.secp256k1.PubKey
                        pub_key_info = signer_info_item.get('publicKey')
                        if isinstance(pub_key_info, dict) and pub_key_info.get('typeUrl') == "/cosmos.crypto.secp256k1.PubKey":
                            # The deep_decode_any_messages should have already processed this if Secp256k1PubKey was available.
                            # It would replace the 'publicKey' field with the decoded PubKey content.
                            # If it has 'key' (base64 bytes of pubkey) and potentially 'derivedAddress'
                            if 'key' in pub_key_info and not pub_key_info.get('derivedAddress'): # If address wasn't derived yet
                                pk_b64 = pub_key_info['key']
                                try:
                                    pk_bytes = base64.b64decode(pk_b64)
                                    derived_addr = derive_cosmos_address_from_pubkey_bytes(pk_bytes, tx_hrp)
                                    if derived_addr:
                                        pub_key_info['derivedAddress'] = derived_addr
                                        print(f"[DEBUG common_utils auth_info_loop] Added derivedAddress: {derived_addr} to pub_key_info.")
                                except Exception:
                                    pass # nosemgrep: general-exception-pass
            
            # Collect all derived_signer addresses into a top-level list
            derived_signers_list = []
            if 'signerInfos' in auth_info_processed and isinstance(auth_info_processed.get('signerInfos'), list):
                for signer_info_item in auth_info_processed['signerInfos']:
                    if isinstance(signer_info_item, dict):
                        pub_key_info = signer_info_item.get('publicKey')
                        if isinstance(pub_key_info, dict) and pub_key_info.get('derivedAddress'):
                            derived_signers_list.append(pub_key_info['derivedAddress'])
            fully_decoded_tx_dict['derived_signers'] = derived_signers_list
            # Ensure 'derived_signers' key exists even if empty, for consistent schema
            if 'derived_signers' not in fully_decoded_tx_dict:
                 fully_decoded_tx_dict['derived_signers'] = []

        messages = fully_decoded_tx_dict.get('body', {}).get('messages', [])
        signer_infos_list = fully_decoded_tx_dict.get('authInfo', {}).get('signerInfos', [])

        if isinstance(messages, list) and isinstance(signer_infos_list, list):
            for i, msg_item in enumerate(messages):
                if isinstance(msg_item, dict) and i < len(signer_infos_list): 
                    current_msg_original_signer = msg_item.get('signer') 
                    signer_info_item = signer_infos_list[i]

                    public_key_data = None
                    derived_address_for_msg = None

                    if isinstance(signer_info_item.get("publicKey"), dict):
                        public_key_data = signer_info_item["publicKey"]
                        derived_address_for_msg = public_key_data.get("derivedAddress") 
                    
                    if derived_address_for_msg:
                        msg_item["signer"] = derived_address_for_msg
            
        return fully_decoded_tx_dict
    except Exception as e:
        print(f"  [common_utils.decode_cosmos_tx_string_to_dict] Error decoding transaction: {e}. Returning original base64 string.")
        return tx_b64_string

def parse_iso_datetime(datetime_str: str):
    """Safely parses an ISO 8601 datetime string, handling potential missing milliseconds or Z."""
    if not datetime_str:
        return None
    try:
        # datetime.fromisoformat handles 'Z' correctly.
        # It also handles varying fractional seconds.
        # Truncate microseconds if more than 6 digits before parsing
        if '.' in datetime_str:
            parts = datetime_str.split('.')
            if len(parts) > 1:
                # Split by timezone indicators first to isolate microseconds part
                micros_and_tz = parts[1]
                tz_char = None
                if '+' in micros_and_tz:
                    tz_char = '+'
                elif '-' in micros_and_tz:
                    tz_char = '-'
                
                micros_part_only = micros_and_tz
                tz_suffix = ""

                if tz_char:
                    micros_part_only = micros_and_tz.split(tz_char)[0]
                    tz_suffix = tz_char + micros_and_tz.split(tz_char)[1]
                elif micros_and_tz.endswith('Z'):
                    micros_part_only = micros_and_tz[:-1]
                    tz_suffix = 'Z'

                if len(micros_part_only) > 6:
                    datetime_str = parts[0] + '.' + micros_part_only[:6] + tz_suffix
        
        return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
    except ValueError:
        # Fallback for other formats if fromisoformat fails
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z", 
            "%Y-%m-%dT%H:%M:%S%z" 
        ):
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue
        print(f"Warning: Could not parse datetime string: {datetime_str} with common ISO formats.")
        return None
    except Exception as e:
        print(f"Error parsing datetime string '{datetime_str}': {e}")
        return None

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

def parse_confirmed_block(
    block_json_data: dict, 
    source_type: str = "mayanode_api", # Default to Mayanode API
    tm_block_raw_data: Optional[dict] = None, # For Tendermint /block response
    tm_block_results_data: Optional[dict] = None # For Tendermint /block_results response
) -> dict:
    """
    Parses a raw block JSON payload into a standardized dictionary format.
    Handles data from Mayanode API directly or constructs it from Tendermint RPC responses.

    Args:
        block_json_data: The primary block data. 
                         For Mayanode, this is the direct API response.
                         For Tendermint, this can be a dictionary constructed from /block and /block_results
                         OR it can be tm_block_raw_data if source_type is 'tendermint_rpc'.
        source_type: "mayanode_api" or "tendermint_rpc".
        tm_block_raw_data: The JSON response from Tendermint's /block endpoint.
                           Used when source_type is "tendermint_rpc".
        tm_block_results_data: The JSON response from Tendermint's /block_results endpoint.
                               Used when source_type is "tendermint_rpc".
    Returns:
        A dictionary containing the parsed block data.
    """
    # print(f"[DEBUG parse_confirmed_block] Called with source_type: {source_type}") # DEBUG REMOVED

    parsed_block = {
        "block_height_str": None, "block_height_int": None, "block_hash": None,
        "block_time_str": None, "block_time_dt": None, "chain_id": None,
        "proposer_address": None, "last_block_hash": None,
        "data_hash": None, "validators_hash": None, "next_validators_hash": None,
        "consensus_hash": None, "app_hash": None, "last_results_hash": None,
        "evidence_hash": None,
        "begin_block_events_raw": [], "begin_block_events_parsed": [],
        "end_block_events_raw": [], "end_block_events_parsed": [],
        "transactions_raw_base64": [], # Specific to Tendermint source before decoding
        "transactions_source": source_type, # Track where the tx data originated
        "transactions": [],
        "is_template": False # Default, can be overridden if it's a constructed template
    }

    header_data = {}
    block_id_data = {}
    data_field_for_txs = {} # This is block.data for Tendermint, or block for Mayanode direct
    
    hrp_for_addresses = "maya" # Default

    if source_type == "mayanode_api":
        # Mayanode API gives a fairly flat structure or a JSON-RPC like structure
        # Handle if it's the JSON-RPC like structure from fetch_mayanode_block reconstruction
        if "result" in block_json_data and "block" in block_json_data["result"]: # As from Tendermint /block via Mayanode proxy
            # print("[DEBUG parse_confirmed_block] Parsing Mayanode block (Tendermint RPC structure via Mayanode proxy)") # DEBUG REMOVED
            actual_block_content = block_json_data["result"]["block"]
            block_id_data = block_json_data["result"].get("block_id", {})
            header_data = actual_block_content.get("header", {})
            data_field_for_txs = actual_block_content.get("data", {}) # txs are in data.txs (base64)
            parsed_block["begin_block_events_raw"] = block_json_data["result"].get("begin_block_events", [])
            parsed_block["end_block_events_raw"] = block_json_data["result"].get("end_block_events", [])
            # If txs are base64, they need decoding. This path assumes they might be.
            # However, fetch_mayanode_block tries to return a more direct structure if possible.
            # This specific path is more for when Mayanode acts like a Tendermint RPC passthrough.
            if isinstance(data_field_for_txs.get("txs"), list) and \
               all(isinstance(tx, str) for tx in data_field_for_txs.get("txs", [])):
                parsed_block["transactions_source"] = "mayanode_api_tendermint_proxied_b64_txs"
                parsed_block["transactions_raw_base64"] = data_field_for_txs.get("txs", [])
                 # Need tm_block_results_data if we are to fully process these b64 txs with results
                if tm_block_results_data:
                     # print("[DEBUG parse_confirmed_block] Mayanode proxied b64 txs found, and tm_block_results_data is available.") # DEBUG REMOVED
                     pass
                else:
                     # print("[DEBUG parse_confirmed_block] Mayanode proxied b64 txs found, but tm_block_results_data is MISSING. Full tx processing will be limited.") # DEBUG REMOVED
                     pass

        elif "id" in block_json_data and "header" in block_json_data: # Direct structure from Mayanode /mayachain/block
            # print("[DEBUG parse_confirmed_block] Parsing Mayanode block (direct API structure)") # DEBUG REMOVED
            header_data = block_json_data.get("header", {})
            block_id_data = block_json_data.get("id", {}) # This 'id' from mayanode /block is block_id
            # For direct Mayanode /mayachain/block, 'txs' are at the top level of block_json_data
            # and are already decoded JSON objects.
            data_field_for_txs = {"txs": block_json_data.get("txs", [])} # Simulate data field for txs
            parsed_block["begin_block_events_raw"] = block_json_data.get("begin_block_events", [])
            parsed_block["end_block_events_raw"] = block_json_data.get("end_block_events", [])
            parsed_block["transactions_source"] = "mayanode_api_direct_json_txs"
        else:
            print(f"Warning: Mayanode block data does not match expected structures. Keys: {list(block_json_data.keys())}")
            # Attempt to find header and id heuristically if top structure is unknown
            header_data = block_json_data.get("header", block_json_data.get("block", {}).get("header", {}))
            block_id_data = block_json_data.get("id", block_json_data.get("block_id", {}))
            # Search for 'txs' list
            if "txs" in block_json_data and isinstance(block_json_data["txs"], list):
                data_field_for_txs = {"txs": block_json_data.get("txs", [])}
            elif "data" in block_json_data and "txs" in block_json_data["data"] and isinstance(block_json_data["data"]["txs"], list):
                data_field_for_txs = {"txs": block_json_data.get("data",{}).get("txs", [])}
            else: # last resort
                data_field_for_txs = {"txs": []}


    elif source_type == "tendermint_rpc":
        # print("[DEBUG parse_confirmed_block] Parsing Tendermint RPC block") # DEBUG REMOVED
        if not tm_block_raw_data or not tm_block_results_data:
            print("Error: For Tendermint RPC source, tm_block_raw_data and tm_block_results_data must be provided.")
            return parsed_block # Return empty shell essentially

        # tm_block_raw_data is from /block endpoint
        # tm_block_results_data is from /block_results endpoint
        header_data = tm_block_raw_data.get("block", {}).get("header", {})
        block_id_data = tm_block_raw_data.get("block_id", {})
        data_field_for_txs = tm_block_raw_data.get("block", {}).get("data", {}) # txs are in data.txs (base64)
        
        parsed_block["begin_block_events_raw"] = tm_block_results_data.get("begin_block_events", []) # Initialize to [] if None
        if parsed_block["begin_block_events_raw"] is None: parsed_block["begin_block_events_raw"] = []
            
        parsed_block["end_block_events_raw"] = tm_block_results_data.get("end_block_events", []) # Initialize to [] if None
        if parsed_block["end_block_events_raw"] is None: parsed_block["end_block_events_raw"] = []
            
        parsed_block["transactions_raw_base64"] = data_field_for_txs.get("txs", [])
        parsed_block["transactions_source"] = "tendermint_rpc_b64_txs_with_results"
        
        # Debug prints for received data
        print(f"  [TM_RPC_DEBUG] Received tm_block_results_data. Keys: {list(tm_block_results_data.keys()) if tm_block_results_data else 'None'}")
        if tm_block_results_data and "txs_results" in tm_block_results_data:
            print(f"  [TM_RPC_DEBUG] tm_block_results_data['txs_results'] type: {type(tm_block_results_data['txs_results'])}, len: {len(tm_block_results_data['txs_results']) if isinstance(tm_block_results_data['txs_results'], list) else 'N/A'}")
        else:
            print(f"  [TM_RPC_DEBUG] 'txs_results' not in tm_block_results_data or tm_block_results_data is None.")

    else: # Unknown source_type
        print(f"Error: Unknown source_type '{source_type}' in parse_confirmed_block.")
        return parsed_block

    # Common header parsing logic
    parsed_block["chain_id"] = header_data.get("chain_id")
    if parsed_block["chain_id"]:
        hrp_for_addresses = CHAIN_TO_HRP.get(parsed_block["chain_id"].split('-')[0].upper(), "maya")


    block_height_value = header_data.get("height")
    if block_height_value is not None:
        parsed_block["block_height_str"] = str(block_height_value) # Ensure it's a string
        try:
            parsed_block["block_height_int"] = int(block_height_value)
        except ValueError:
            print(f"Warning: Could not parse block height '{block_height_value}' to int.")
    else:
        parsed_block["block_height_str"] = None # Explicitly None if not found
        parsed_block["block_height_int"] = None
    
    parsed_block["block_hash"] = block_id_data.get("hash")
    parsed_block["block_time_str"] = header_data.get("time")
    if parsed_block["block_time_str"]:
        parsed_block["block_time_dt"] = parse_iso_datetime(parsed_block["block_time_str"])

    parsed_block["proposer_address"] = header_data.get("proposer_address")
    last_block_id_obj = header_data.get("last_block_id", {})
    if isinstance(last_block_id_obj, dict): # Ensure it's a dict before .get()
        parsed_block["last_block_hash"] = last_block_id_obj.get("hash")
    
    parsed_block["data_hash"] = header_data.get("data_hash")
    parsed_block["validators_hash"] = header_data.get("validators_hash")
    parsed_block["next_validators_hash"] = header_data.get("next_validators_hash")
    parsed_block["consensus_hash"] = header_data.get("consensus_hash")
    parsed_block["app_hash"] = header_data.get("app_hash")
    parsed_block["last_results_hash"] = header_data.get("last_results_hash")
    evidence_data = header_data.get("evidence_hash")
    if isinstance(evidence_data, dict):
        parsed_block["evidence_hash"] = evidence_data.get("hash")
    elif isinstance(evidence_data, str):
        parsed_block["evidence_hash"] = evidence_data if evidence_data else None # Use string if non-empty, else None
    else: # None or other types
        parsed_block["evidence_hash"] = None

    # Transaction Processing
    final_tx_list_to_process = []

    if source_type == "mayanode_api":
        # If direct Mayanode API, txs are already dicts
        if parsed_block["transactions_source"] == "mayanode_api_direct_json_txs":
            tx_list_from_data_field = data_field_for_txs.get("txs") # Get it, could be None or [] or list of dicts
            
            if isinstance(tx_list_from_data_field, list):
                if not tx_list_from_data_field: # Empty list is valid, no transactions
                    pass # Do nothing, transactions list remains empty
                elif all(isinstance(tx_obj, dict) for tx_obj in tx_list_from_data_field):
                    # List of dicts, process them
                    for i, tx_obj_dict in enumerate(tx_list_from_data_field):
                        parsed_block["transactions"].append(parse_transaction_data(tx_obj_dict, tx_index_in_block=i))
                else:
                    # List, but not all elements are dicts - this is an unexpected format
                    print("Warning: Mayanode direct txs list contains non-dictionary elements.")
            elif tx_list_from_data_field is None: # Explicitly None is also valid (no transactions)
                 pass # Do nothing, transactions list remains empty
            else:
                # It's not a list and not None - this is an unexpected format
                print(f"Warning: Expected list or None for Mayanode direct txs, but found {type(tx_list_from_data_field)}.")

        # If Mayanode API proxied Tendermint b64 txs, and we have results data
        elif parsed_block["transactions_source"] == "mayanode_api_tendermint_proxied_b64_txs" and tm_block_results_data:
            # This path is similar to "tendermint_rpc" path below
            txs_base64_list = parsed_block["transactions_raw_base64"]
            txs_results_list = tm_block_results_data.get("txs_results", [])
            if not isinstance(txs_results_list, list): txs_results_list = []

            if len(txs_base64_list) != len(txs_results_list) and txs_results_list : # only warn if results were expected
                print(f"Warning (Mayanode proxied): Mismatch in length of txs_b64 ({len(txs_base64_list)}) and txs_results ({len(txs_results_list)}).")

            for i, tx_b64_string in enumerate(txs_base64_list):
                tx_hash = "TBD_HASH_" + str(i)
                decoded_content = {}
                tx_result = {}
                try:
                    tx_bytes = base64.b64decode(tx_b64_string)
                    tx_hash = hashlib.sha256(tx_bytes).hexdigest().upper()
                    decoded_content = decode_cosmos_tx_string_to_dict(tx_b64_string, hrp_for_addresses)
                    if not isinstance(decoded_content, dict): # If decoding failed
                        decoded_content = {"error": "failed to decode b64 tx", "raw_b64": tx_b64_string}
                    if i < len(txs_results_list) and txs_results_list[i] is not None:
                        tx_result = txs_results_list[i]
                except Exception as e:
                    decoded_content = {"error": f"exception during b64 tx processing: {str(e)}", "raw_b64": tx_b64_string}
                
                parsed_block["transactions"].append({
                    "hash": tx_hash, 
                    "tx_index_in_block": i, # Added field
                    "tx_content_json": decoded_content,
                    "result_log": tx_result.get("log"), "result_gas_wanted": tx_result.get("gas_wanted"),
                    "result_gas_used": tx_result.get("gas_used"),
                    "result_events_raw": tx_result.get("events", []),
                    "result_events_parsed": _parse_tendermint_tx_events(tx_result.get("events", []))
                })
        else: # Mayanode proxied but no results, or other unexpected Mayanode scenario
             for i, tx_b64_string in enumerate(parsed_block.get("transactions_raw_base64", [])):
                tx_hash = "TBD_HASH_" + str(i)
                decoded_content = decode_cosmos_tx_string_to_dict(tx_b64_string, hrp_for_addresses)
                if not isinstance(decoded_content, dict): decoded_content = {"error": "failed to decode b64 tx", "raw_b64": tx_b64_string}
                try:
                    tx_bytes = base64.b64decode(tx_b64_string)
                    tx_hash = hashlib.sha256(tx_bytes).hexdigest().upper()
                except: pass # nosemgrep
                parsed_block["transactions"].append({
                    "hash": tx_hash, 
                    "tx_index_in_block": i, # Added field
                    "tx_content_json": decoded_content,
                    "result_log": None, "result_gas_wanted": None, "result_gas_used": None,
                    "result_events_raw": [], "result_events_parsed": []
                })


    elif source_type == "tendermint_rpc":
        txs_base64_list = parsed_block["transactions_raw_base64"]
        txs_results_list = tm_block_results_data.get("txs_results", []) # Should be populated by now
        if not isinstance(txs_results_list, list): txs_results_list = []


        if len(txs_base64_list) != len(txs_results_list):
             print(f"  [TM_RPC_DEBUG] Warning: Mismatch in tx_count ({len(txs_base64_list)}) and tx_results_count ({len(txs_results_list)})")

        for i, tx_b64_string in enumerate(txs_base64_list):
            tx_content_dict = {}
            tx_result_data = {}
            tx_hash_calculated = "PROTO_DECODED_NO_HASH_INDEX_" + str(i) 

            try:
                # 1. Decode the base64 transaction string
                decoded_tx_content = decode_cosmos_tx_string_to_dict(tx_b64_string, hrp_for_addresses)
                
                if isinstance(decoded_tx_content, dict):
                    # CRITICAL FIX: Apply transformation for Tendermint source
                    transformed_tx_for_block = transform_decoded_tm_tx_to_mayanode_format(decoded_tx_content)
                    tx_content_dict = transformed_tx_for_block
                    # 2. Calculate Tendermint Tx Hash
                    try:
                        tx_bytes_for_hash = base64.b64decode(tx_b64_string)
                        tx_hash_calculated = hashlib.sha256(tx_bytes_for_hash).hexdigest().upper()
                        if i == 0: print(f"  [TM_RPC_DEBUG] Tx 0 calculated_hash: {tx_hash_calculated}")
                    except Exception as e_hash:
                        print(f"Error calculating Tendermint hash for tx index {i}: {e_hash}")
                else:
                    # Decoding failed, store the raw string as content for audit
                    # Apply transformation to an error structure if needed, or keep simple
                    error_content = {"error": "Failed to decode b64 tx string", "raw_b64_tx": tx_b64_string}
                    tx_content_dict = transform_decoded_tm_tx_to_mayanode_format(error_content) # transform even error structure

                # 3. Get corresponding transaction result
                tx_events_parsed_for_this_tx = [] # Initialize
                if i < len(txs_results_list) and txs_results_list[i] is not None:
                    tx_result_data = txs_results_list[i]
                    tx_events_parsed_for_this_tx = _parse_tendermint_tx_events(tx_result_data.get("events", []))

                    # Attempt to populate fee from AuthInfo if null in 'tx' event for Tendermint source
                    for event_item in tx_events_parsed_for_this_tx:
                        if event_item.get("type") == "tx":
                            attributes = event_item.get("attributes", {})
                            current_fee_val = attributes.get("fee")
                            if i == 0: # Debug for first tx only
                                print(f"  [TM_RPC_DEBUG] Tx 0: Checking 'tx' event. Current fee attribute value: {current_fee_val} (type: {type(current_fee_val)})")

                            if current_fee_val is None:
                                # For some chains/transactions (e.g., Mayanode MsgNetworkFee),
                                # the Tendermint /block_results tx event may have a null fee.
                                # We attempt to populate it from AuthInfo.fee.amount.
                                # However, AuthInfo.fee.amount might also be empty for such transactions.
                                # In such cases, the fee will correctly remain None, reflecting the source data.
                                auth_info = tx_content_dict.get("auth_info", {})
                                fee_detail = auth_info.get("fee", {})
                                fee_amount_list = fee_detail.get("amount", [])
                                if i == 0: # Debug for first tx only
                                    print(f"  [TM_RPC_DEBUG] Tx 0: fee was None. AuthInfo fee_amount_list: {fee_amount_list}")

                                if fee_amount_list and isinstance(fee_amount_list, list):
                                    fee_strings = []
                                    for coin_obj in fee_amount_list:
                                        if isinstance(coin_obj, dict) and "amount" in coin_obj and "denom" in coin_obj:
                                            fee_strings.append(f"{coin_obj['amount']}{coin_obj['denom']}")
                                    if fee_strings:
                                        attributes["fee"] = ",".join(fee_strings)
                                        # event_item["attributes"] = attributes # attributes is a reference, modification is enough
                                        if i == 0: # Debug print for the first transaction only
                                            print(f"  [TM_RPC_DEBUG] Tx 0: Populated null 'fee' in 'tx' event from AuthInfo: {attributes['fee']}")
                            break # Process only the first 'tx' event type if multiple exist (shouldn't typically)
                    
                    if i == 0: 
                        # print(f"  [TM_RPC_DEBUG] Tx 0 tx_result_data (keys): {list(tx_result_data.keys()) if isinstance(tx_result_data, dict) else 'Not a dict'}") # DEBUG REMOVED
                        # raw_events = tx_result_data.get('events', [])
                        # print(f"  [TM_RPC_DEBUG] Tx 0 tx_result_data.get('events', []) (type): {type(raw_events)}, (len): {len(raw_events) if isinstance(raw_events,list) else 'N/A'}") # DEBUG REMOVED
                        pass
                else:
                    tx_result_data = {"warning": "No tx_result data found or result was null."}
                    # if i == 0: print(f"  [TM_RPC_DEBUG] Tx 0: No result data or result was None.") # DEBUG REMOVED
                    pass
                
                parsed_block["transactions"].append({
                    "hash": tx_hash_calculated,
                    "tx_index_in_block": i, # Added field
                    "tx_content_json": tx_content_dict,
                    "result_log": tx_result_data.get("log"),
                    "result_gas_wanted": tx_result_data.get("gas_wanted"),
                    "result_gas_used": tx_result_data.get("gas_used"),
                    "result_events_raw": tx_result_data.get("events", []),
                    "result_events_parsed": tx_events_parsed_for_this_tx # Use the potentially modified list
                })
            except Exception as e_tx_proc:
                print(f"Error processing Tendermint Tx index {i} for block {parsed_block['block_height_str']}: {e_tx_proc}")
                parsed_block["transactions"].append({
                    "hash": "ERROR_PROCESSING_TX_INDEX_" + str(i),
                    "tx_index_in_block": i, # Added field, even for error case
                    "tx_content_json": {"error": str(e_tx_proc), "raw_b64_tx": tx_b64_string},
                    "result_log": None, "result_gas_wanted": None, "result_gas_used": None,
                    "result_events_raw": [], "result_events_parsed": []
                })
    
    # Common event parsing for block-level events (already populated in raw fields)
    # (This needs to be adapted for Mayanode vs Tendermint event structures if different)
    parsed_block["begin_block_events_parsed"] = _parse_block_events_common(parsed_block["begin_block_events_raw"] if parsed_block["begin_block_events_raw"] is not None else [])
    parsed_block["end_block_events_parsed"] = _parse_block_events_common(parsed_block["end_block_events_raw"] if parsed_block["end_block_events_raw"] is not None else [])
    
    return parsed_block

def _parse_block_events_common(raw_events_list: list) -> list:
    """Helper to parse block-level events which might have base64 encoded values."""
    parsed_event_list = []
    if not raw_events_list: return parsed_event_list
    for raw_event in raw_events_list:
        if isinstance(raw_event, dict) and 'type' in raw_event:
            parsed_attributes = {}
            event_type = raw_event["type"] # Store type first

            # Check for Tendermint style: raw_event['attributes'] is a list of {'key':k, 'value':v}
            if 'attributes' in raw_event and isinstance(raw_event['attributes'], list) and \
               all(isinstance(attr, dict) and 'key' in attr and 'value' in attr for attr in raw_event['attributes']):
                # Tendermint style from /block_results
                for attr_dict in raw_event['attributes']:
                    key = _try_decode_base64(attr_dict['key']) 
                    value = _try_decode_base64(attr_dict['value'])
                    parsed_attributes[key] = value
            # Check for Mayanode style with a nested 'attributes' dictionary
            elif 'attributes' in raw_event and isinstance(raw_event['attributes'], dict):
                for k, v_raw in raw_event['attributes'].items():
                    # Keys from Mayanode are usually not base64, values might be.
                    parsed_attributes[k] = _try_decode_base64(v_raw)
            # Handle Mayanode style where attributes are direct keys on the raw_event object itself
            else: 
                # Iterate over all keys in raw_event, skipping 'type'
                temp_attrs_from_direct_keys = {}
                for key, value_raw in raw_event.items():
                    if key != 'type': # Don't re-add 'type' into attributes
                        temp_attrs_from_direct_keys[key] = _try_decode_base64(value_raw)
                
                if temp_attrs_from_direct_keys: # If we found any direct attributes
                    parsed_attributes = temp_attrs_from_direct_keys
                else: # If no 'attributes' field and no direct keys other than 'type'
                    # This path might be hit if raw_event is just {'type': 'some_type'}
                    # print(f"Event type {event_type} has no 'attributes' field and no other direct keys.")
                    parsed_attributes = {} # Ensure it's an empty dict, not unhandled

            parsed_event_list.append({
                "type": event_type,
                "attributes": parsed_attributes
            })
        # else: print(f"Skipping malformed block event: {raw_event}")
    return parsed_event_list

def _parse_tendermint_tx_events(raw_tx_events_list: list) -> list:
    """
    Parses transaction events specifically from Tendermint's tx_result.events.
    Attributes are {key, value, index: bool}, and key/value are base64 encoded.
    """
    parsed_event_list = []
    if not raw_tx_events_list: return parsed_event_list
    for raw_event in raw_tx_events_list:
        if isinstance(raw_event, dict) and 'type' in raw_event:
            parsed_attributes = {}
            if isinstance(raw_event['attributes'], list):
                for attr_dict in raw_event['attributes']:
                    if isinstance(attr_dict, dict) and 'key' in attr_dict and 'value' in attr_dict:
                        key = _try_decode_base64(attr_dict['key'])
                        value = _try_decode_base64(attr_dict['value'])
                        parsed_attributes[key] = value 
            parsed_event_list.append({
                "type": raw_event["type"], # Type is usually not base64
                "attributes": parsed_attributes
            })
        # else: print(f"Skipping malformed Tendermint tx event: {raw_event}")
    return parsed_event_list


def parse_transaction_data(tx_obj: dict, tx_index_in_block: Optional[int] = None) -> dict:
    """
    Parses a single raw transaction object from a block.
    - Assumes tx_obj["tx"] is already a JSON-like dict for confirmed blocks.
    - Parses the transaction result events.
    Args:
        tx_obj: A dictionary representing a single transaction from block_json_data["txs"]
        tx_index_in_block: The 0-based index of this transaction within its block.
    Returns:
        A dictionary containing the parsed transaction data.
    """
    parsed_tx = {
        "hash": tx_obj.get("hash"),
        "tx_index_in_block": tx_index_in_block, # Added field
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

# --- Transformation Utilities: Tendermint Decoded JSON to Mayanode-like JSON ---

def camel_to_snake(name: str) -> str:
    # Handles simple CamelCase and more complex cases like Mayanode's "BlockHeight" -> "block_height"
    # and also preserves existing snake_case like "tx_hash"
    if '_' in name: # Already snake_case or similar
        return name.lower() # Ensure it's all lowercase if mixed
    
    # Insert underscore before uppercase letters (but not at the start)
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert underscore before uppercase letter followed by another uppercase letter (e.g. BlockID -> Block_ID)
    # and then convert to lowercase
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'_', s1)
    # Additional check for all-caps acronyms at the beginning or after an underscore
    s3 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'_', s2)
    # Handle cases like "TXID" -> "txid" (if it wasn't caught by previous) or "ABCDef" -> "abc_def"
    s4 = re.sub(r'([A-Z0-9]+)([A-Z][a-z0-9]*)', r'_', s3)

    final_snake = s4.lower()
    
    # Prevent double underscores that might have been introduced
    final_snake = re.sub(r'_+', '_', final_snake)
    
    # print(f"camel_to_snake: {name} -> {final_snake}") # Debug
    return final_snake

def transform_decoded_tm_tx_to_mayanode_format(input_data: any, depth=0) -> any:
    # print(f"{'  ' * depth}Transforming: {type(input_data)} {str(input_data)[:100]}") # General debug

    if isinstance(input_data, dict):
        new_dict = {}
        # First, apply camel_to_snake to all keys and recursively transform values
        temp_snake_cased_dict = {}
        for key, value in input_data.items():
            new_key = camel_to_snake(key)
            temp_snake_cased_dict[new_key] = transform_decoded_tm_tx_to_mayanode_format(value, depth + 1)

        # Now, iterate through the snake_cased_dict to apply specific transformations
        for new_key, transformed_value in temp_snake_cased_dict.items():
            current_value_to_assign = transformed_value

            # Handle blame_data and blame_signature: '' -> None
            if new_key in ("blame_data", "blame_signature") and current_value_to_assign == "":
                current_value_to_assign = None
            
            # Handle aggregator_target_limit: "" or "0" -> None (within ObservedTx items)
            if new_key == "aggregator_target_limit" and current_value_to_assign in (None, "", "0"):
                 current_value_to_assign = None

            # Handle asset dictionary to string conversion for Coin-like structures
            if new_key == "asset" and isinstance(current_value_to_assign, dict) and \
               all(k in current_value_to_assign for k in ["chain", "symbol"]):
                current_value_to_assign = f"{current_value_to_assign['chain']}.{current_value_to_assign['symbol']}"
            
            # Special handling for 'mode' field if its value is 'DIRECT' (from older proto versions perhaps)
            if new_key == 'mode' and current_value_to_assign == 'DIRECT':
                 current_value_to_assign = "SIGN_MODE_DIRECT"

            new_dict[new_key] = current_value_to_assign
        
        # After all keys in the current dictionary are processed and snake_cased:
        # Ensure 'txs': [] for MsgObservedTxIn if not present and @type indicates it.
        # This must happen *after* new_dict is populated with its @type.
        if new_dict.get("@type") == "/types.MsgObservedTxIn" and "txs" not in new_dict:
            new_dict["txs"] = []
        
        # Special handling for PubKey that might have been decoded from Any
        # If the original key was 'public_key' and it contained a typeUrl for Secp256k1PubKey
        if 'public_key' in input_data and isinstance(input_data['public_key'], dict) and \
           input_data['public_key'].get('typeUrl') == "/cosmos.crypto.secp256k1.PubKey":
            # The content of new_dict['public_key'] is already transformed by recursive call.
            # We just need to ensure @type is correctly set based on our mapping.
            if 'public_key' in new_dict and isinstance(new_dict['public_key'], dict):
                mayanode_pk_type = TYPE_URL_TO_MAYANODE_TYPE.get("/cosmos.crypto.secp256k1.PubKey")
                if mayanode_pk_type:
                    new_dict['public_key']['@type'] = mayanode_pk_type


        # --- Remove superfluous default fields if present and empty/default to match Mayanode API ---
        # This should run after all keys are snake_cased and values are processed.

        # Handle 'body' fields specifically related to 'timeout_timestamp' and 'unordered'
        if "body" in new_dict and isinstance(new_dict["body"], dict):
            body_dict = new_dict["body"]

            # Remove 'timeout_timestamp' if it's the default epoch value or zero
            if "timeout_timestamp" in body_dict:
                ts_val = body_dict['timeout_timestamp']
                if ts_val == '1970-01-01T00:00:00Z' or ts_val == 0 or ts_val == "0":
                    body_dict.pop("timeout_timestamp")
            
            # Remove 'unordered' if false
            if "unordered" in body_dict and body_dict['unordered'] is False:
                body_dict.pop("unordered")
            
            # Remove 'extension_options' and 'non_critical_extension_options' if they are empty lists
            if "extension_options" in body_dict and body_dict["extension_options"] == []:
                body_dict.pop("extension_options")
            if "non_critical_extension_options" in body_dict and body_dict["non_critical_extension_options"] == []:
                body_dict.pop("non_critical_extension_options")


            if not body_dict: 
                new_dict.pop("body")

        # Handle 'auth_info' fields specifically for 'tip' and 'signer_infos.mode_info.multi'
        if "auth_info" in new_dict and isinstance(new_dict["auth_info"], dict):
            auth_info_dict = new_dict["auth_info"]

            if "tip" in auth_info_dict:
                # Check for the specific empty structure of Tip
                if auth_info_dict['tip'] == {'amount': [], 'tipper': ''}:
                    auth_info_dict.pop("tip")

            if "signer_infos" in auth_info_dict and isinstance(auth_info_dict["signer_infos"], list) and auth_info_dict["signer_infos"]:
                first_signer_info = auth_info_dict["signer_infos"][0]
                if isinstance(first_signer_info, dict):
                    # Remove associated_signature_b64 if it exists
                    if "associated_signature_b64" in first_signer_info: # This key is added by our decoding, not from proto
                        first_signer_info.pop("associated_signature_b64")
                    
                    # Remove derived_address from public_key if it exists (added by our decoding)
                    if "public_key" in first_signer_info and isinstance(first_signer_info["public_key"], dict):
                        if "derived_address" in first_signer_info["public_key"]: # This key is added by our decoding
                            first_signer_info["public_key"].pop("derived_address")
                        # If public_key itself becomes empty or only contains "@type" after this,
                        # and the Mayanode API doesn't show it for this context, it could be removed.
                        # Current Mayanode API examples for MsgSend include public_key as null.
                        # For MsgSetIPAddress, public_key contains key and @type.
                        # Let's evaluate based on comparison: if Mayanode has null, make it null.
                        # If Mayanode omits, omit.
                        
                        # If after removing derived_address, public_key is empty or just has {'@type': '...'}
                        # and this is not typical for Mayanode, this is where we would adjust.
                        # For now, the logic is to keep 'key' and '@type' if they exist.

                    if "mode_info" in first_signer_info and isinstance(first_signer_info["mode_info"], dict):
                        mode_info_dict = first_signer_info["mode_info"]
                        # print(f"DEBUG REMOVAL: Checking 'mode_info' dict: {mode_info_dict}") # Keep for one more run if needed
                        if "multi" in mode_info_dict:
                            # Check for the specific empty structure of ModeInfoMulti
                            if mode_info_dict['multi'] == {'bitarray': {'extra_bits_stored': 0, 'elems': ''}, 'mode_infos': []}:
                                mode_info_dict.pop("multi")
                        
                        if not mode_info_dict: 
                            first_signer_info.pop("mode_info")
            
            if not auth_info_dict: 
                new_dict.pop("auth_info")
        
        return new_dict
    elif isinstance(input_data, list):
        return [transform_decoded_tm_tx_to_mayanode_format(item, depth + 1) for item in input_data]
    else:
        return input_data

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
        },
        "derived_signers": ["maya12cljaw4j0fj795yzyq9q3e72zmsfwk7z3zy99v", "maya1zgw6t904sph0q0uwu93gqe0sfhdusymx9ycfy8"] # Added for testing
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

    print("\n--- Testing camel_to_snake ---")
    test_cases_camel = {
        "simpleCase": "simple_case",
        "anotherSimpleCase": "another_simple_case",
        "URLAddress": "url_address",
        "TXID": "txid",
        "BlockHeight": "block_height",
        "Already_Snake_Case": "already_snake_case",
        "mixed_Case_With_Underscore": "mixed_case_with_underscore",
        "SHA256Sum": "sha256_sum",
        "authInfo": "auth_info",
        "signerInfos": "signer_infos",
        "publicKey": "public_key",
        "typeUrl": "type_url",
        "feeAmount": "fee_amount",
        "gasLimit": "gas_limit",
        "sequenceNumber": "sequence_number",
        "modeInfo": "mode_info",
        "single": {"mode": "single"}, # Test dict value
        "multi": {"modes": ["DIRECT", "LEGACY_AMINO_JSON"]}, # Test list value
        "nonCriticalExtensionOptions": "non_critical_extension_options",
        "observedTxs": "observed_txs",
        "txId": "tx_id",
        "outTxs": "out_txs",
    }

    for camel_key, expected_snake_key in test_cases_camel.items():
        actual_snake = camel_to_snake(camel_key)
        print(f"'{camel_key}' -> '{actual_snake}' (Expected: '{expected_snake_key}') {'PASS' if actual_snake == expected_snake_key else 'FAIL'}")

    print("\n--- Testing transform_decoded_tm_tx_to_mayanode_format ---")
    sample_tm_decoded_tx = { # Simulating output from decode_cosmos_tx_string_to_dict
        "body": {
            "messages": [{
                "@type": "/types.MsgSend", # Already has @type from deep_decode
                "fromAddress": "tmaya1sender", # Camel case
                "toAddress": "tmaya1receiver",   # Camel case
                "amount": [{"denom": "thor", "amount": "100"}] 
            }],
            "memo": "Test memo",
            "timeoutHeight": "0", # Should be kept if not default "0" for block height related timeouts
            "timeoutTimestamp": "1970-01-01T00:00:00Z", # Should be removed
            "extensionOptions": [], # Should be removed
            "nonCriticalExtensionOptions": [] # Should be removed
        },
        "authInfo": {
            "signerInfos": [{
                "publicKey": { # This would have been decoded by deep_decode_any_messages
                    "@type": "/cosmos.crypto.secp256k1.PubKey", # Added by deep_decode
                    "key": "A123KeyBytes", # Base64 pubkey bytes
                    "derivedAddress": "tmaya1signerfrompk" # Added by deep_decode
                },
                "modeInfo": {
                    "single": {"mode": "SIGN_MODE_DIRECT"}, # Mode as string from betterproto
                    "multi": {"bitarray": {"extraBitsStored": 0, "elems": ""}, "modeInfos": []} # Should be removed
                },
                "sequence": "10",
                "associated_signature_b64": "sig123" # Added by our decode function, should be removed
            }],
            "fee": {
                "amount": [{"denom": "maya", "amount": "10"}],
                "gasLimit": "200000",
                "payer": "", # Should be kept as is if empty string
                "granter": ""  # Should be kept as is if empty string
            },
            "tip": {"amount": [], "tipper": ""} # Should be removed
        },
        "signatures": ["base64sig1"],
        "derived_signers": ["tmaya1signerfrompk"] # Added by our decode, should be removed from final Mayanode structure
    }

    expected_mayanode_like_tx = {
        "body": {
            "messages": [{
                "@type": "/types.MsgSend",
                "from_address": "tmaya1sender",
                "to_address": "tmaya1receiver",
                "amount": [{"denom": "thor", "amount": "100"}]
            }],
            "memo": "Test memo",
            "timeout_height": "0"
        },
        "auth_info": {
            "signer_infos": [{
                "public_key": {
                    "@type": "/cosmos.crypto.secp256k1.PubKey", # Kept, as Mayanode also uses this for PubKey structure sometimes
                    "key": "A123KeyBytes"
                     # derivedAddress removed
                },
                "mode_info": { # multi removed
                    "single": {"mode": "SIGN_MODE_DIRECT"}
                },
                "sequence": "10"
                # associated_signature_b64 removed
            }],
            "fee": {
                "amount": [{"denom": "maya", "amount": "10"}],
                "gas_limit": "200000",
                "payer": "",
                "granter": ""
            }
            # tip removed
        },
        "signatures": ["base64sig1"]
        # derived_signers removed by not explicitly carrying it over unless it's a Mayanode top-level field.
        # The transformation works on the input dict; if 'derived_signers' isn't part of the Mayanode 'tx' structure,
        # it naturally wouldn't be in the output unless we added specific logic to map it, which we don't want.
    }
    
    transformed_tx = transform_decoded_tm_tx_to_mayanode_format(sample_tm_decoded_tx)
    print("\nTransformed Tendermint Decoded TX:")
    print(json.dumps(transformed_tx, indent=2))
    print("\nExpected Mayanode-like TX:")
    print(json.dumps(expected_mayanode_like_tx, indent=2))

    # Basic check for equality (will likely fail due to order, but good for visual diff)
    if json.dumps(transformed_tx, sort_keys=True) == json.dumps(expected_mayanode_like_tx, sort_keys=True):
        print("\nTransformation Test: PASS")
    else:
        print("\nTransformation Test: FAIL (check differences above)")

    # Test a MsgObservedTxIn case (from Tendermint decoding)
    sample_tm_msg_observed_tx_in = {
        "body": {
            "messages": [{
                "@type": "/types.MsgObservedTxIn", # This would be set by deep_decode_any
                "txs": None, # Protobuf might set it to None if not present
                "signer": "tmaya1signer",
                "memo": "some memo",
                "observedPubKey": "thorpub1add...",
                "aggregator": "thorpub1agg...",
                "aggregatorTargetAsset": {"chain": "ETH", "symbol": "USDT", "ticker": "USDT", "synth": False},
                "aggregatorTargetLimit": None, # Or "0" or ""
                "action": "SWAP",
                "externalObservedHeight": "12345",
                "finaliseHeight": "12390",
                "keysignMs": "5000"
            }]
        },
        # ... other tx parts ...
    }
    expected_mayanode_msg_observed_tx_in_message_part = {
        "@type": "/types.MsgObservedTxIn",
        "txs": [], # Should be defaulted to []
        "signer": "tmaya1signer",
        "memo": "some memo",
        "observed_pub_key": "thorpub1add...",
        "aggregator": "thorpub1agg...",
        "aggregator_target_asset": "ETH.USDT", # Transformed
        "aggregator_target_limit": None, # Transformed to None
        "action": "SWAP",
        "external_observed_height": "12345",
        "finalise_height": "12390",
        "keysign_ms": "5000"
    }

    transformed_observed_tx = transform_decoded_tm_tx_to_mayanode_format(sample_tm_msg_observed_tx_in)
    print("\nTransformed MsgObservedTxIn (full tx structure):")
    print(json.dumps(transformed_observed_tx, indent=2))
    
    if transformed_observed_tx["body"]["messages"][0] == expected_mayanode_msg_observed_tx_in_message_part:
        print("\nMsgObservedTxIn Transformation Test (message part): PASS")
    else:
        print("\nMsgObservedTxIn Transformation Test (message part): FAIL")
        print("Expected message part:")
        print(json.dumps(expected_mayanode_msg_observed_tx_in_message_part, indent=2))
