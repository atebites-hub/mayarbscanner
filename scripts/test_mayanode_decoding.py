"""
Script to fetch a block from Mayanode (Tendermint RPC) and decode transactions.
"""
import sys
import os
import requests
import base64
import json
from pathlib import Path
import bech32 # Added for address conversion
import hashlib # For SHA256
# Print the path of the imported bech32 module to verify which one is being used
print(f"Using bech32 module from: {bech32.__file__}")
# We might need a RIPEMD160 implementation if not readily available
# For simplicity, this example might omit RIPEMD160 if it requires an external library not easily added.
# However, a proper implementation for Cosmos SDK addresses usually requires RIPEMD160.

# Add a global variable for a specific block height for testing
BLOCK_HEIGHT = 11255442 # MsgSend and MsgDeposit (older block with known txs)
# BLOCK_HEIGHT = 14868841 # MsgSwap example (currently causes 500 error on public node)

# Define the parent directory of the generated protobuf package
PROTO_GEN_PARENT_DIR = str(Path(__file__).parent.parent / "proto" / "generated")
# Define the project root directory for output files
PROJECT_ROOT_DIR = Path(__file__).parent.parent

# Add the parent directory of our 'pb_stubs' package to sys.path
if PROTO_GEN_PARENT_DIR not in sys.path:
    sys.path.insert(0, PROTO_GEN_PARENT_DIR)

# Import after sys.path modification, using the root_package name
try:
    from pb_stubs.cosmos.tx.v1beta1 import Tx as CosmosTx
    # We might need more types if we want to deeply inspect messages
    # e.g. from pb_stubs.cosmos.bank.v1beta1 import MsgSend
    # e.g. from pb_stubs.mayachain.v1.x.mayachain.types import MsgDeposit (path needs verification)
    from pb_stubs.types import MsgNetworkFee  # Added for deep decoding
    from pb_stubs.cosmos.crypto.secp256k1 import PubKey as Secp256k1PubKey # Added for deep decoding
    print("Successfully imported generated protobuf modules via pb_stubs package.")
except ImportError as e:
    print(f"Error importing generated protobuf modules: {e}")
    print(f"Please ensure that '{PROTO_GEN_PARENT_DIR}' contains the 'pb_stubs' package and is in sys.path.")
    print(f"Current sys.path: {sys.path}")
    CosmosTx = None
    sys.exit(1)

# Configuration for Mayanode Tendermint RPC endpoint
TENDERMINT_RPC_URL = "https://tendermint.mayachain.info"

# Map chain identifiers to their Human-Readable Part (HRP) for Bech32 addresses
# This map will need to be expanded as we encounter more chains.
# Note: Some chains might use prefixes for mainnet, testnet, etc. This is simplified.
CHAIN_TO_HRP = {
    "MAYA": "maya",
    "THOR": "thor",
    "COSMOS": "cosmos",
    "GAIA": "cosmos", # Gaia is often used for Cosmos Hub
    "BTC": "bc",       # For SegWit addresses, if we ever decode those specific parts
    "ETH": None,       # Ethereum uses hex, not Bech32
    "TERRA": "terra",
    # "XRD" - Radix. Radix uses its own address format, not typically Bech32 in the same way.
    # For now, we'll leave XRD out or assign None. If its addresses *are* Bech32 with a known HRP,
    # we can add it. From the example, `signer` was Base64 for XRD.
    "XRD": None, # Placeholder, Radix addresses are complex. Assuming raw bytes if not specifically Bech32.
}

def fetch_block_from_tendermint_rpc(height: int) -> list[str]:
    """
    Fetches a block from the Tendermint RPC and returns a list of
    Base64 encoded transaction strings.
    """
    endpoint = f"{TENDERMINT_RPC_URL}/block?height={height}"
    print(f"Fetching block from Tendermint RPC: {endpoint}")
    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        block_data = response.json()
        if "result" in block_data and "block" in block_data["result"] and "data" in block_data["result"]["block"] and \
           "txs" in block_data["result"]["block"]["data"]:
            txs_b64 = block_data["result"]["block"]["data"]["txs"]
            if txs_b64:
                print(f"Found {len(txs_b64)} transaction(s) in block {height} from Tendermint RPC.")
                return txs_b64
            else:
                print(f"No transactions found in block {height} from Tendermint RPC.")
                return []
        else:
            print(f"Error: Unexpected Tendermint RPC response structure for block {height}.")
            print(json.dumps(block_data, indent=2))
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching block from Tendermint RPC: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Tendermint RPC: {e}")
        return []

# Function to attempt Bech32 conversion
def try_bech32_decode(hrp, data_bytes):
    try:
        if hrp is None: # Cannot encode if HRP is None
            print(f"      try_bech32_decode: HRP is None. Skipping for data (len {len(data_bytes)}): {base64.b64encode(data_bytes).decode()[:30]}...")
            return None
        
        converted_bits = bech32.convertbits(data_bytes, 8, 5)
        print(f"      try_bech32_decode: HRP='{hrp}', data_bytes (len {len(data_bytes)}) input: {base64.b64encode(data_bytes).decode()[:30]}..., converted_bits for encode: {converted_bits}")
        
        if converted_bits is None: # convertbits can return None on failure
            print(f"      try_bech32_decode: bech32.convertbits returned None for HRP '{hrp}'. Cannot encode.")
            return None
            
        # Call the generic bech32_encode function, not the segwit-specific encode function
        encoded_address = bech32.bech32_encode(hrp, converted_bits) 
        print(f"      try_bech32_decode: Successfully encoded to: {encoded_address}")
        return encoded_address
    except Exception as e:
        print(f"      try_bech32_decode: Bech32 conversion failed for hrp '{hrp}': {e}")
        return None

# Function to derive address from public key bytes (Cosmos SDK style)
def derive_cosmos_address_from_pubkey_bytes(pubkey_bytes, hrp):
    if not pubkey_bytes or hrp is None:
        print(f"    derive_address: Called with empty pubkey_bytes or None HRP ({hrp}). Skipping.")
        return None
    print(f"    derive_address: Attempting for HRP '{hrp}' with pubkey_bytes (len {len(pubkey_bytes)}): {base64.b64encode(pubkey_bytes).decode()[:30]}...")
    try:
        # 1. SHA256
        sha256_hash = hashlib.sha256(pubkey_bytes).digest()
        # 2. RIPEMD160
        try:
            ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
            print(f"    derive_address: RIPEMD160 hash (len {len(ripemd160_hash)}): {base64.b64encode(ripemd160_hash).decode()}")
        except ValueError:
            print(f"    derive_address: RIPEMD160 not available in hashlib for HRP '{hrp}'. Cannot derive address.")
            return None

        # 3. Bech32 encode
        derived_addr = try_bech32_decode(hrp, ripemd160_hash)
        print(f"    derive_address: Bech32 result for HRP '{hrp}': {derived_addr}")
        return derived_addr
    except Exception as e:
        print(f"    derive_address: Error deriving address for HRP '{hrp}': {e}")
        return None

# Helper function to recursively decode 'Any' messages and byte-addresses
def deep_decode_any_messages(current_obj, tx_level_hrp=None):
    if isinstance(current_obj, dict):
        # Handle Any messages first by replacing them with their decoded content
        if current_obj.get("typeUrl") and "value" in current_obj and isinstance(current_obj.get("value"), str):
            type_url = current_obj["typeUrl"]
            encoded_value_b64 = current_obj["value"]
            # print(f"  Processing Any: {type_url}")
            try:
                decoded_bytes = base64.b64decode(encoded_value_b64)
                specific_msg_obj = None
                is_pubkey_any = False

                if type_url == "/types.MsgNetworkFee":
                    specific_msg_obj = MsgNetworkFee().parse(decoded_bytes)
                elif type_url == "/cosmos.crypto.secp256k1.PubKey":
                    specific_msg_obj = Secp256k1PubKey().parse(decoded_bytes)
                    is_pubkey_any = True
                
                if specific_msg_obj:
                    # Convert the parsed proto object to a dictionary
                    decoded_dict_content = specific_msg_obj.to_dict(include_default_values=False)
                    
                    # If it was a PubKey, try to derive and add its address immediately
                    if is_pubkey_any and tx_level_hrp:
                        pubkey_b64_for_addr = decoded_dict_content.get("key") # This is Base64 from Secp256k1PubKey.key
                        if pubkey_b64_for_addr and isinstance(pubkey_b64_for_addr, str):
                            print(f"    PubKey found: key='{pubkey_b64_for_addr[:30]}...', attempting address derivation with HRP: {tx_level_hrp}")
                            try:
                                actual_pubkey_bytes_for_addr = base64.b64decode(pubkey_b64_for_addr)
                                derived_address = derive_cosmos_address_from_pubkey_bytes(actual_pubkey_bytes_for_addr, tx_level_hrp)
                                if derived_address:
                                    decoded_dict_content["derivedAddress"] = derived_address
                                    print(f"    Successfully added derivedAddress: {derived_address}")
                                else:
                                    print(f"    Address derivation returned None for PubKey with HRP {tx_level_hrp}.")
                            except Exception as e_derive_addr_inline:
                                print(f"    Error during inline address derivation from PubKey: {e_derive_addr_inline}")
                    
                    current_obj.clear()
                    current_obj.update(decoded_dict_content)
                    # The object is now the decoded_dict_content. We need to recurse into *this new content*.
                    return deep_decode_any_messages(current_obj, tx_level_hrp)
                
            except Exception as e_deep_decode:
                print(f"    Error during initial deep decoding of Any {type_url}: {e_deep_decode}")
            # If Any processing failed or it wasn't a target type, return the original Any dict without iterating its children like "typeUrl".
            return current_obj 

        # If not an Any message (or an Any that was already processed and replaced), iterate its children.
        local_hrp_for_field_specific_conversion = tx_level_hrp # Default for fields like 'signer'
        if "chain" in current_obj and current_obj["chain"] in CHAIN_TO_HRP:
            # Use HRP from message's own 'chain' field for its 'signer' if valid
            if CHAIN_TO_HRP[current_obj["chain"]] is not None:
                 local_hrp_for_field_specific_conversion = CHAIN_TO_HRP[current_obj["chain"]]

        for key, value in list(current_obj.items()):
            if key == "signer" and isinstance(value, str):
                if local_hrp_for_field_specific_conversion:
                    # print(f"  Processing signer field with HRP: {local_hrp_for_field_specific_conversion}")
                    try:
                        signer_bytes = base64.b64decode(value)
                        bech32_addr = try_bech32_decode(local_hrp_for_field_specific_conversion, signer_bytes)
                        if bech32_addr:
                            current_obj[key] = bech32_addr
                    except Exception as e_signer_decode:
                        print(f"    Error decoding signer field '{value}' with HRP '{local_hrp_for_field_specific_conversion}': {e_signer_decode}")
            
            # publicKey.key would have been handled when the PubKey Any was initially decoded and replaced.
            # No special handling for "key" needed here anymore if the above PubKey processing is correct.

            current_obj[key] = deep_decode_any_messages(value, tx_level_hrp)
        return current_obj

    elif isinstance(current_obj, list):
        for i in range(len(current_obj)):
            current_obj[i] = deep_decode_any_messages(current_obj[i], tx_level_hrp)
        return current_obj
    else:
        return current_obj

def decode_transaction_and_print_info(tx_b64: str, tx_index: int, block_height: int):
    """
    Decodes a Base64 encoded transaction string using the generated protobuf Tx type
    and prints information about its messages. Also saves encoded and decoded tx to files.
    """
    if not CosmosTx:
        print("CosmosTx type not available, skipping decoding.")
        return

    # Define file paths for outputs
    encoded_tx_filename = PROJECT_ROOT_DIR / f"block_{block_height}_tx_{tx_index + 1}_encoded.json"
    decoded_tx_filename = PROJECT_ROOT_DIR / f"block_{block_height}_tx_{tx_index + 1}_decoded.json"

    print(f"--- Decoding Tendermint RPC transaction {tx_index + 1} for block {block_height} ---")
    
    # Save the original Base64 encoded transaction
    try:
        with open(encoded_tx_filename, 'w') as f:
            json.dump({"block_height": block_height, "tx_index": tx_index + 1, "encoded_tx_b64": tx_b64}, f, indent=2)
        print(f"  Saved encoded transaction to: {encoded_tx_filename}")
    except IOError as e:
        print(f"  Error saving encoded transaction: {e}")

    try:
        tx_bytes = base64.b64decode(tx_b64)
        decoded_tx_proto = CosmosTx()
        decoded_tx_proto.parse(tx_bytes)
        
        print(f"  Successfully parsed Tx. Number of messages: {len(decoded_tx_proto.body.messages)}")
        for j, msg_any in enumerate(decoded_tx_proto.body.messages):
            print(f"    Msg {j+1}: type_url='{msg_any.type_url}', value_len={len(msg_any.value)}")
            # Further unpacking would require importing the specific message types
            # based on msg_any.type_url and then parsing msg_any.value with them.
            # Example (conceptual, actual types and paths need to be verified and imported):
            # if msg_any.type_url == "/cosmos.bank.v1beta1.MsgSend":
            #     try:
            #         from pb_stubs.cosmos.bank.v1beta1 import MsgSend
            #         msg_send = MsgSend()
            #         msg_send.parse(msg_any.value)
            #         print(f"      Parsed MsgSend: From={msg_send.from_address}, To={msg_send.to_address}, Amount={msg_send.amount}")
            #     except Exception as e_unpack:
            #         print(f"      Error unpacking {msg_any.type_url}: {e_unpack}")
            # elif msg_any.type_url == "/mayachain.MsgDeposit": # Actual type_url for Mayanode MsgDeposit
            #     try:
            #         from pb_stubs.mayachain import MsgDeposit # Assuming this path
            #         msg_deposit = MsgDeposit()
            #         msg_deposit.parse(msg_any.value)
            #         print(f"      Parsed MsgDeposit: Signer={msg_deposit.signer}, Memo={msg_deposit.memo}, Asset={msg_deposit.asset}")
            #     except Exception as e_unpack:
            #         print(f"      Error unpacking {msg_any.type_url}: {e_unpack}")
            # else:
            #     print(f"      Further unpacking for type '{msg_any.type_url}' not implemented in this example.")

        print(f"  Memo: '{decoded_tx_proto.body.memo}'")

        # Save the decoded transaction as JSON
        try:
            # Convert protobuf to a dictionary for JSON serialization
            # betterproto messages have a to_dict() method
            print(f"  Type of decoded_tx_proto.to_dict: {type(decoded_tx_proto.to_dict)}")
            # print(f"  Attributes of decoded_tx_proto: {dir(decoded_tx_proto)}") # Already confirmed this is fine

            print("  Attempting to convert body to dict...")
            if hasattr(decoded_tx_proto, 'body') and decoded_tx_proto.body and hasattr(decoded_tx_proto.body, 'to_dict'):
                try:
                    body_dict = decoded_tx_proto.body.to_dict(include_default_values=False)
                    print(f"    Body_dict (first 200 chars): {str(body_dict)[:200]}")
                except Exception as e_body:
                    print(f"    Error converting body to dict: {e_body}")
            else:
                print("    Body or body.to_dict not available.")

            print("  Attempting to convert auth_info to dict...")
            if hasattr(decoded_tx_proto, 'auth_info') and decoded_tx_proto.auth_info and hasattr(decoded_tx_proto.auth_info, 'to_dict'):
                try:
                    auth_info_dict = decoded_tx_proto.auth_info.to_dict(include_default_values=False)
                    print(f"    Auth_info_dict (first 200 chars): {str(auth_info_dict)[:200]}")
                except Exception as e_auth:
                    print(f"    Error converting auth_info to dict: {e_auth}")
            else:
                print("    Auth_info or auth_info.to_dict not available.")
            
            print("  Attempting to convert full decoded_tx_proto to dict...")
            decoded_tx_dict = decoded_tx_proto.to_dict(include_default_values=False)
            
            print("  Performing deep decoding of Any messages...")
            # Determine the HRP for the overall transaction (e.g., for AuthInfo PubKeys).
            # For Mayanode transactions, this should consistently be "maya".
            tx_hrp_for_auth_info = CHAIN_TO_HRP.get("MAYA") 

            try:
                fully_decoded_tx_dict = deep_decode_any_messages(decoded_tx_dict, tx_hrp_for_auth_info) 
            except Exception as e_deep_decode_main:
                print(f"    Error during main deep_decode_any_messages call: {e_deep_decode_main}")
                fully_decoded_tx_dict = decoded_tx_dict # Fallback to partially decoded

            # Modified section to print signer details and augment signer_infos in the dict for JSON
            if 'authInfo' in fully_decoded_tx_dict and fully_decoded_tx_dict['authInfo']:
                auth_info_processed = fully_decoded_tx_dict['authInfo']
                top_level_signatures_list = fully_decoded_tx_dict.get('signatures', [])

                if 'signerInfos' in auth_info_processed and isinstance(auth_info_processed['signerInfos'], list):
                    print("  Signer Info and Associated Signatures (linking in JSON):")
                    for i, signer_info_dict in enumerate(auth_info_processed['signerInfos']):
                        derived_addr = "N/A"
                        
                        pub_key_details = {}
                        try:
                            pub_key_details = signer_info_dict['publicKey'] # Direct access
                        except KeyError:
                            print(f"    DEBUG: Signer {i+1}, 'publicKey' key NOT FOUND in signer_info_dict. Keys: {list(signer_info_dict.keys())}")
                        except Exception as e_access:
                            print(f"    DEBUG: Signer {i+1}, Error accessing signer_info_dict['publicKey']: {e_access}")
                        
                        if isinstance(pub_key_details, dict): # After deep_decode, it should be a dict
                            derived_addr = pub_key_details.get('derivedAddress', "Error/Not Found")
                        
                        actual_signature_b64 = "N/A"
                        signature_b64_snippet_for_console = "N/A"

                        if isinstance(top_level_signatures_list, list) and i < len(top_level_signatures_list):
                            current_sig = top_level_signatures_list[i]
                            if isinstance(current_sig, str):
                                actual_signature_b64 = current_sig
                                signature_b64_snippet_for_console = f"{actual_signature_b64[:30]}..."
                            else:
                                signature_b64_snippet_for_console = f"Non-string signature data (type: {type(current_sig)})"
                        
                        # Augment the signer_info dictionary for the JSON output
                        signer_info_dict['associated_signature'] = actual_signature_b64
                            
                        print(f"    Signer {i+1}:")
                        print(f"      Derived Address: {derived_addr}")
                        print(f"      Associated Signature (snippet for console): {signature_b64_snippet_for_console}")

            with open(decoded_tx_filename, 'w') as f:
                json.dump(fully_decoded_tx_dict, f, indent=2)
            print(f"  Saved decoded transaction to: {decoded_tx_filename}")
        except AttributeError: # If to_dict is not available or fails
             print(f"  Could not convert decoded transaction to dict. The to_dict() method might be missing or failed.")
        except IOError as e:
            print(f"  Error saving decoded transaction: {e}")
        except Exception as e_dict:
            print(f"  Error converting full protobuf to dict or saving: {e_dict}")
            import traceback
            traceback.print_exc()

    except base64.binascii.Error as e:
        print(f"  Base64 decoding error: {e}")
    except Exception as e:
        print(f"  Protobuf decoding error: {e}")
        # print(f"  Failed raw bytes (first 100): {tx_bytes[:100]}")
    finally:
        print("--------------------------------------------------")

def main():
    print(f"Attempting to decode Mayanode transactions from Tendermint RPC for block: {BLOCK_HEIGHT}\n")

    if not CosmosTx:
        print("CosmosTx protobuf type not loaded due to import error. Cannot proceed.")
        return

    raw_txs_b64 = fetch_block_from_tendermint_rpc(BLOCK_HEIGHT)
    if not raw_txs_b64:
        print("No transactions fetched from Tendermint RPC. Exiting.")
        return
        
    print(f"\nFetched {len(raw_txs_b64)} Base64 transaction string(s). Details below:\n")

    for i, tx_b64 in enumerate(raw_txs_b64):
        decode_transaction_and_print_info(tx_b64, i, BLOCK_HEIGHT)

if __name__ == "__main__":
    main() 