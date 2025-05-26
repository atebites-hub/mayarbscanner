import sys
import os
import json
from pathlib import Path

# --- Adjust sys.path for local imports ---
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = _SCRIPT_DIR # Assuming this script is in the project root
PROTO_GEN_PARENT_DIR = PROJECT_ROOT_DIR / "proto" / "generated"

# Add project root to sys.path to allow importing from src
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# Add the parent directory of our 'pb_stubs' package to sys.path
if str(PROTO_GEN_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_GEN_PARENT_DIR))

print(f"TEMP_TEST: Current sys.path: {sys.path}")

try:
    print("TEMP_TEST: Attempting to import from src.common_utils...")
    from src.common_utils import decode_cosmos_tx_string_to_dict, transform_decoded_tm_tx_to_mayanode_format, PROTOBUF_AVAILABLE
    print(f"TEMP_TEST: Successfully imported from src.common_utils. PROTOBUF_AVAILABLE={PROTOBUF_AVAILABLE}")
except Exception as e:
    print(f"TEMP_TEST: Error importing from src.common_utils: {e}")
    sys.exit(1)

# The problematic base64 encoded transaction string from Tendermint block
# This was the original one we suspected might have issues or was not being processed correctly by betterproto's to_dict()
# ORIGINAL_TX_B64_STRING_TM = "Co0BCoEBCiEvY29zbW9zLmJhbmsudjFiZXRhMS5Nc2dTZW5kEmIKK21heWExZmVoZmFncGV5ZGFkZWNncWY3NTN0emN0aHZqMmhrbjN6M2U5dngSLG1heWExbmwzd3Nja3hzdmh6eHQydHp2aDZ4dnNtdTJwZXI2cWM0a3NocRIQCgVjYWNhbxIBMGgBkewBCkQKInR5cGVzL0Nvc21vc1R4AAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKZNBL8BEkAKRgoFL2Nvc21vc19wcm90by9Db3Ntb3MSIQohQ4aGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABiEKBOgwgSEAoGChBiYXRjaBACY2FjYW8SBAgwYRICClMKRgofL2Nvc21vcy5jcnlwdG8uc2VjcDI1NmsxLlB1YktleRIjCiECgJ58U1pAqnKvODgVdR87u5pM4YEGhztO8x7pY+7f+KkSBAoCCAEYWBhAUGgQCRw0GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOA1Co0BCoEBCiEvY29zbW9zLmJhbmsudjFiZXRhMS5Nc2dTZW5kEmIKK21heWExZmVoZmFncGV5ZGFkZWNncWY3NTN0emN0aHZqMmhrbjN6M2U5dngSLG1heWExbmwzd3Nja3hzdmh6eHQydHp2aDZ4dnNtdTJwZXI2cWM0a3NocRIQCgVjYWNhbxIBMGgBkewBCkQKInR5cGVzL0Nvc21vc1R4AAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKZNBL8BEkAKRgoFL2Nvc21vc19wcm90by9Db3Ntb3MSIQohQ4aGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABiEKBOgwgSEAoGChBiYXRjaBACY2FjYW8SBAgwYRICClMKRgofL2Nvc21vcy5jcnlwdG8uc2VjcDI1NmsxLlB1YktleRIjCiECgJ58U1pAqnKvODgVdR87u5pM4YEGhztO8x7pY+7f+KkSBAoCCAEYWBhAUGgQCRw0GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOA1"

# NEW FOUND TX_B64_STRING FROM Tendermint Block 11320570 (Transaction 0)
# This transaction was confirmed by find_test_tx_with_authinfo.py to have populated auth_info.signer_infos
# when parsed with CosmosTx().parse()
TX_B64_STRING_TM = "CkEKPwoUL3R5cGVzLk1zZ05ldHdvcmtGZWUSJwj2nJEKEgRUSE9SGAEggIl6KhSIWVEghD0+LrZSsRhI73sfvvIYvxJdClMKRgofL2Nvc21vcy5jcnlwdG8uc2VjcDI1NmsxLlB1YktleRIjCiEDoF33+f4omZvU8JCuuV4Z67F+n7cXaBQm+D6l5rOLv9ESBAoCCAEY99T6AhIGEIDQrPMOGkCNDXS6x/bZCUbnYINwZP2Rkpn18Qk86c44RvnR+Jb0Ow8bzcNv0DAtoGiOPaUnI+5rAxSM3YSelDVB3o04yyir"

print(f"\nTEMP_TEST: Using Tendermint Tx String (len {len(TX_B64_STRING_TM)}): {TX_B64_STRING_TM[:100]}...")

print("\nTEMP_TEST: Calling decode_cosmos_tx_string_to_dict...")
decoded_dict = decode_cosmos_tx_string_to_dict(TX_B64_STRING_TM)

if isinstance(decoded_dict, dict):
    print("TEMP_TEST: decode_cosmos_tx_string_to_dict returned a DICT.")
    # Check the signer field directly from the output of decode_cosmos_tx_string_to_dict
    messages = decoded_dict.get('body', {}).get('messages', [])
    if messages and isinstance(messages[0], dict):
        signer_after_decode = messages[0].get('signer')
        print(f"TEMP_TEST: Signer of first message AFTER decode_cosmos_tx_string_to_dict: '{signer_after_decode}'")
    else:
        print("TEMP_TEST: No messages found or first message not a dict after decode_cosmos_tx_string_to_dict.")
    
    # print("\nTEMP_TEST: Full output of decode_cosmos_tx_string_to_dict:")
    # print(json.dumps(decoded_dict, indent=2))

    print("\nTEMP_TEST: Calling transform_decoded_tm_tx_to_mayanode_format...")
    transformed_dict = transform_decoded_tm_tx_to_mayanode_format(decoded_dict)

    if isinstance(transformed_dict, dict):
        print("TEMP_TEST: transform_decoded_tm_tx_to_mayanode_format returned a DICT.")
        messages_after_transform = transformed_dict.get('body', {}).get('messages', [])
        if messages_after_transform and isinstance(messages_after_transform[0], dict):
            signer_after_transform = messages_after_transform[0].get('signer')
            print(f"TEMP_TEST: Signer of first message AFTER transform_decoded_tm_tx_to_mayanode_format: '{signer_after_transform}'")
            
            # Also check key casing for auth_info
            auth_info_after_transform = transformed_dict.get('auth_info')
            if auth_info_after_transform is not None:
                print("TEMP_TEST: 'auth_info' key IS PRESENT after transform.")
            else:
                print("TEMP_TEST: 'auth_info' key IS MISSING after transform. Checking for 'authInfo'...")
                authInfo_after_transform = transformed_dict.get('authInfo')
                if authInfo_after_transform is not None:
                    print("TEMP_TEST: 'authInfo' (camelCase) key IS PRESENT after transform.")


        else:
            print("TEMP_TEST: No messages found or first message not a dict after transform_decoded_tm_tx_to_mayanode_format.")
        
        # print("\nTEMP_TEST: Full output of transform_decoded_tm_tx_to_mayanode_format:")
        # print(json.dumps(transformed_dict, indent=2))
    else:
        print(f"TEMP_TEST: transform_decoded_tm_tx_to_mayanode_format did NOT return a dict, got: {type(transformed_dict)}")

else:
    print(f"TEMP_TEST: decode_cosmos_tx_string_to_dict did NOT return a dict, got: {type(decoded_dict)}")

print("\nTEMP_TEST: Script finished.") 