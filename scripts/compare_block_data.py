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
import bech32
import hashlib

# Configuration
BLOCK_HEIGHT = 11255442  # Example block height
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
    from src.api_connections import fetch_mayanode_block # Import from src
    print("Successfully imported generated protobuf modules and api_connections.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    CosmosTx = None
    MsgNetworkFee = None
    Secp256k1PubKey = None
    fetch_mayanode_block = None # Ensure it's defined for checks

# Map chain identifiers to their Human-Readable Part (HRP)
CHAIN_TO_HRP = {
    "MAYA": "maya", "THOR": "thor", "COSMOS": "cosmos", "GAIA": "cosmos",
    "BTC": "bc", "ETH": None, "TERRA": "terra", "XRD": None,
}

# --- Helper Functions (adapted from test_mayanode_decoding.py) ---

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
    except ValueError: # ripemd160 not available
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
                specific_msg_obj = None
                is_pubkey_any = False

                if type_url == "/types.MsgNetworkFee" and MsgNetworkFee:
                    specific_msg_obj = MsgNetworkFee().parse(decoded_bytes)
                elif type_url == "/cosmos.crypto.secp256k1.PubKey" and Secp256k1PubKey:
                    specific_msg_obj = Secp256k1PubKey().parse(decoded_bytes)
                    is_pubkey_any = True
                
                if specific_msg_obj:
                    decoded_dict_content = specific_msg_obj.to_dict(include_default_values=False)
                    if is_pubkey_any and tx_level_hrp:
                        pubkey_b64_for_addr = decoded_dict_content.get("key")
                        if pubkey_b64_for_addr and isinstance(pubkey_b64_for_addr, str):
                            try:
                                actual_pubkey_bytes_for_addr = base64.b64decode(pubkey_b64_for_addr)
                                derived_address = derive_cosmos_address_from_pubkey_bytes(actual_pubkey_bytes_for_addr, tx_level_hrp)
                                if derived_address:
                                    decoded_dict_content["derivedAddress"] = derived_address
                            except Exception: 
                                pass
                    current_obj.clear()
                    current_obj.update(decoded_dict_content)
                    return deep_decode_any_messages(current_obj, tx_level_hrp)
            except Exception: 
                pass 
            return current_obj

        local_hrp_for_field = tx_level_hrp
        if "chain" in current_obj and current_obj["chain"] in CHAIN_TO_HRP:
            if CHAIN_TO_HRP[current_obj["chain"]] is not None:
                 local_hrp_for_field = CHAIN_TO_HRP[current_obj["chain"]]

        for key, value in list(current_obj.items()):
            if key == "signer" and isinstance(value, str) and local_hrp_for_field:
                try:
                    signer_bytes = base64.b64decode(value)
                    bech32_addr = try_bech32_decode(local_hrp_for_field, signer_bytes)
                    if bech32_addr:
                        current_obj[key] = bech32_addr
                except Exception:
                    pass 
            current_obj[key] = deep_decode_any_messages(value, tx_level_hrp)
        return current_obj
    elif isinstance(current_obj, list):
        for i in range(len(current_obj)):
            current_obj[i] = deep_decode_any_messages(current_obj[i], tx_level_hrp)
        return current_obj
    else:
        return current_obj

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

def decode_tm_transaction_for_block(tx_b64: str) -> dict | str:
    if not CosmosTx:
        print("CosmosTx type not available, cannot decode transaction.")
        return tx_b64 
    try:
        tx_bytes = base64.b64decode(tx_b64)
        decoded_tx_proto = CosmosTx()
        decoded_tx_proto.parse(tx_bytes)
        
        decoded_tx_dict = decoded_tx_proto.to_dict(include_default_values=False)
        
        tx_hrp_for_auth_info = CHAIN_TO_HRP.get("MAYA") 
        fully_decoded_tx_dict = deep_decode_any_messages(decoded_tx_dict, tx_hrp_for_auth_info)

        if 'authInfo' in fully_decoded_tx_dict and fully_decoded_tx_dict.get('authInfo'):
            auth_info_processed = fully_decoded_tx_dict['authInfo']
            top_level_signatures_list = fully_decoded_tx_dict.get('signatures', [])

            if 'signerInfos' in auth_info_processed and isinstance(auth_info_processed.get('signerInfos'), list):
                for i, signer_info_item in enumerate(auth_info_processed['signerInfos']):
                    if isinstance(signer_info_item, dict): 
                        actual_signature_b64 = "N/A"
                        if isinstance(top_level_signatures_list, list) and i < len(top_level_signatures_list):
                            current_sig = top_level_signatures_list[i]
                            if isinstance(current_sig, str):
                                actual_signature_b64 = current_sig
                        signer_info_item['associated_signature'] = actual_signature_b64
        return fully_decoded_tx_dict
    except Exception as e:
        print(f"  Error decoding transaction: {e}. Returning original base64 string.")
        return tx_b64


def main():
    if not CosmosTx or not fetch_mayanode_block: # Check if all essential imports loaded
        print("Critical definitions (CosmosTx or fetch_mayanode_block) not loaded. Exiting.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--- Processing block {BLOCK_HEIGHT} ---")
    
    # 1. Fetch Tendermint block
    tm_block_raw_tendermint_structure = fetch_block_from_tendermint_rpc_full(BLOCK_HEIGHT)
    if not tm_block_raw_tendermint_structure:
        print("Failed to fetch Tendermint block. Exiting.")
        return

    tm_block_decoded_struct = json.loads(json.dumps(tm_block_raw_tendermint_structure))

    if "data" in tm_block_decoded_struct and "txs" in tm_block_decoded_struct["data"] and \
       isinstance(tm_block_decoded_struct["data"]["txs"], list) and tm_block_decoded_struct["data"]["txs"]:
        num_txs = len(tm_block_decoded_struct["data"]["txs"])
        print(f"Decoding {num_txs} transaction(s) from Tendermint block...")
        decoded_txs_list = []
        for i, tx_b64_item in enumerate(tm_block_decoded_struct["data"]["txs"]):
            if isinstance(tx_b64_item, str): 
                print(f"  Decoding transaction {i + 1}/{num_txs}...")
                decoded_tx = decode_tm_transaction_for_block(tx_b64_item)
                decoded_txs_list.append(decoded_tx)
            else:
                decoded_txs_list.append(tx_b64_item) 
        tm_block_decoded_struct["data"]["txs"] = decoded_txs_list
    else:
        print("No transactions found in Tendermint block data or 'txs' is not a list.")

    tm_output_path = OUTPUT_DIR / f"tendermint_block_{BLOCK_HEIGHT}_decoded.json"
    try:
        with open(tm_output_path, 'w') as f:
            json.dump(tm_block_decoded_struct, f, indent=2)
        print(f"Saved decoded Tendermint block to: {tm_output_path}")
    except IOError as e:
        print(f"Error saving decoded Tendermint block: {e}")

    # 3. Fetch and save RAW Mayanode API block data
    mayanode_raw_api_data = None
    mayanode_api_direct_url = f"https://mayanode.mayachain.info/mayachain/block?height={BLOCK_HEIGHT}"
    print(f"Fetching RAW Mayanode API block data from: {mayanode_api_direct_url}")
    try:
        response = requests.get(mayanode_api_direct_url, timeout=30)
        response.raise_for_status()
        mayanode_raw_api_data = response.json()
        mn_raw_output_path = OUTPUT_DIR / f"mayanode_api_block_{BLOCK_HEIGHT}_raw.json"
        with open(mn_raw_output_path, 'w') as f:
            json.dump(mayanode_raw_api_data, f, indent=2)
        print(f"Saved RAW Mayanode API block data to: {mn_raw_output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching RAW Mayanode API block data: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from RAW Mayanode API response: {e}")
    except IOError as e:
        print(f"Error saving RAW Mayanode API block data: {e}")


    # 4. Fetch RECONSTRUCTED Mayanode API block using imported function
    # The imported fetch_mayanode_block uses the /mayachain/block?height= endpoint
    # and reconstructs the output to have "id" and "header" at the top level.
    print(f"Fetching RECONSTRUCTED Mayanode block {BLOCK_HEIGHT} via src.api_connections...")
    mayanode_block_reconstructed = fetch_mayanode_block(height=BLOCK_HEIGHT) # This function already prints its own status
    
    if not mayanode_block_reconstructed:
        print("Failed to fetch Mayanode API block via src.api_connections. Comparison will be limited.")
    else:
        mn_output_path = OUTPUT_DIR / f"mayanode_api_block_{BLOCK_HEIGHT}_reconstructed.json"
        try:
            with open(mn_output_path, 'w') as f:
                json.dump(mayanode_block_reconstructed, f, indent=2)
            print(f"Saved reconstructed Mayanode block to: {mn_output_path}")
        except IOError as e:
            print(f"Error saving reconstructed Mayanode block: {e}")
            
    # 5. Basic Comparison Summary
    print("\n--- Comparison Summary ---")
    print(f"Block Height: {BLOCK_HEIGHT}")

    if "header" in tm_block_decoded_struct:
        print(f"Tendermint (Decoded) - Time: {tm_block_decoded_struct['header'].get('time')}, "
              f"App Hash: {tm_block_decoded_struct['header'].get('app_hash')}")
        num_tm_txs = len(tm_block_decoded_struct.get("data", {}).get("txs", []))
        print(f"Tendermint (Decoded) - Transactions: {num_tm_txs}")

    if mayanode_block_reconstructed:
        # The fetch_mayanode_block from api_connections.py reconstructs the response
        # to have 'header' and 'id' at the top level.
        # Transactions are under 'data.txs'.
        header = mayanode_block_reconstructed.get("header", {})
        if header:
            print(f"Mayanode API (Reconstructed) - Time: {header.get('time')}, "
                  f"App Hash: {header.get('app_hash')}")
        else:
            print("Mayanode API (Reconstructed) - Header not found.")

        num_mn_txs = len(mayanode_block_reconstructed.get("data", {}).get("txs", []))
        # Note: The 'txs' in the reconstructed data from fetch_mayanode_block 
        # are the base64 encoded strings from the original Tendermint-like structure within that API response.
        # They are NOT the already-decoded JSON transactions seen in the /mayachain/block web example directly.
        # This is because fetch_mayanode_block in api_connections.py was designed to mimic the Tendermint structure.
        print(f"Mayanode API (Reconstructed) - Transactions (base64 from original struct): {num_mn_txs}")
    else:
        print("Mayanode API (Reconstructed) - Block data not available for summary.")
    
    if mayanode_raw_api_data:
        # Attempt to find a transaction count in the raw data for informational purposes
        # This depends on the exact structure of the /mayachain/block response
        raw_mn_tx_count = 0
        if isinstance(mayanode_raw_api_data.get("txs"), list): # Path seen in some examples
            raw_mn_tx_count = len(mayanode_raw_api_data["txs"])
        elif isinstance(mayanode_raw_api_data.get("block", {}).get("data", {}).get("txs"), list) and \
             mayanode_raw_api_data["block"]["data"]["txs"] is not None : # Tendermint-like path if populated
             raw_mn_tx_count = len(mayanode_raw_api_data["block"]["data"]["txs"])
        elif isinstance(mayanode_raw_api_data.get("tx_results"), list): # Another common path for results
            raw_mn_tx_count = len(mayanode_raw_api_data["tx_results"])
        
        print(f"Mayanode API (RAW) - Transactions (approx count from common paths): {raw_mn_tx_count}")
    else:
        print("Mayanode API (RAW) - Block data not available for summary.")

    print("\nDetailed comparison can be done by inspecting the saved JSON files in the "
          f"'{OUTPUT_DIR.relative_to(PROJECT_ROOT_DIR)}' directory.")

if __name__ == "__main__":
    if CosmosTx and MsgNetworkFee and Secp256k1PubKey and fetch_mayanode_block:
        main()
    else:
        print("Exiting due to missing critical protobuf message definitions or api_connections import.") 