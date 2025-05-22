# Python Protobuf Decoding Guide for Mayanode & Cosmos SDK Messages

This guide details the methods to successfully deserialize Protobuf messages encountered when interacting with Mayanode, particularly for `cosmos.tx.v1beta1.Tx` (used in Tendermint RPC) and the more complex Mayanode-specific types like `types.MsgObservedTxOut`.

Achieving reliable Protobuf decoding for Mayanode, with its use of gogoproto extensions, can be challenging. This document outlines two approaches:

1.  **Recommended: Native Python with `betterproto`**: For standard Cosmos SDK messages like `cosmos.tx.v1beta1.Tx`. This method provides clean, native Python objects.
2.  **Fallback/Alternative: `protoc --decode` via `subprocess`**: For highly specific or complex Mayanode types (e.g., `types.MsgObservedTxOut`) where native Python libraries might struggle, or as a diagnostic tool.

## Part 1: Native Python Decoding with `betterproto` (Recommended)

This method uses `betterproto` to generate Python stubs from `.proto` files, allowing for direct, type-hinted deserialization in Python.

### Prerequisites

*   **Python**: Version 3.9+ recommended.
*   **`pip`**: For installing Python packages.
*   **`git`**: For cloning repositories (if fetching fresh `.proto` files).
*   **`protoc` Compiler**: `betterproto`'s compiler plugin works with `protoc` provided by `grpcio-tools`. No separate global `protoc` installation is strictly necessary if using this method.

### Step 1: Prepare Source `.proto` Definitions

The key to successful compilation is having all necessary `.proto` files and their dependencies correctly structured.

1.  **Project Directory Structure for Protos**:
    Create a root directory for your source `.proto` files. In this project, we use `proto/src/`.
    ```
    mayarbscanner/
    ├── proto/
    │   ├── src/                # All source .proto files go here
    │   │   ├── cosmos/
    │   │   │   ├── tx/
    │   │   │   │   └── v1beta1/
    │   │   │   │       └── tx.proto
    │   │   │   │       └── service.proto (if needed for services)
    │   │   │   └── base/
    │   │   │       └── v1beta1/
    │   │   │           └── coin.proto
    │   │   │       └── query/
    │   │   │           └── v1beta1/
    │   │   │               └── pagination.proto
    │   │   │   └── crypto/
    │   │   │       └── secp256k1/
    │   │   │           └── keys.proto
    │   │   └── gogoproto/
    │   │       └── gogo.proto
    │   │   └── cosmos_proto/
    │   │       └── cosmos.proto
    │   │   └── google/
    │   │       └── api/
    │   │           └── annotations.proto
    │   │           └── http.proto
    │   │       └── protobuf/
    │   │           └── any.proto (and other standard Google types)
    │   │   └── mayachain/      # Mayanode specific protos
    │   │       └── v1/
    │   │           └── common/
    │   │               └── common.proto
    │   │           └── x/mayachain/types/
    │   │               └── msg_observed_txout.proto 
    │   │               └── ... (other Mayanode types)
    │   │   └── tendermint/     # Tendermint protos
    │   │       └── ...
    │   └── generated/
    │       └── pb_stubs/       # Output directory for generated Python files
    └── ... (rest of your project)
    ```

2.  **Obtain `.proto` files**:
    *   **Cosmos SDK & Third-Party**:
        *   `gogoproto/gogo.proto`: From [github.com/gogo/protobuf](https://github.com/gogo/protobuf/blob/master/gogoproto/gogo.proto).
        *   `cosmos_proto/cosmos.proto`: From [github.com/cosmos/cosmos-proto](https://github.com/cosmos/cosmos-proto/blob/main/proto/cosmos_proto/cosmos.proto).
        *   Standard Cosmos SDK protos (like `cosmos.tx.v1beta1.Tx`, `cosmos.base.v1beta1.Coin`): From a specific version of the Cosmos SDK repository that aligns with Mayanode's usage (e.g., [github.com/cosmos/cosmos-sdk](https://github.com/cosmos/cosmos-sdk)). Mayanode typically forks or uses a specific version.
        *   Google Protos (`google/api/...`, `google/protobuf/...`): These are often included with `protoc` distributions or `grpcio-tools`. If not, get them from [github.com/googleapis/googleapis](https://github.com/googleapis/googleapis).
    *   **Mayanode-Specific**:
        *   Clone the Mayanode repository: `git clone https://gitlab.com/mayachain/mayanode.git temp_mayanode_protos`
        *   Navigate to its `proto` directory (`temp_mayanode_protos/proto/`) and copy the contents (e.g., `mayachain/`, `tendermint/`, etc.) into your project's `proto/src/` directory, maintaining the structure.
        *   Ensure that any imports within these `.proto` files correctly resolve based on the `proto/src/` root. For example, an import `import "gogoproto/gogo.proto";` should find `proto/src/gogoproto/gogo.proto`.

    *In this project, `proto/src/` is committed to Git, containing the necessary collection of pre-vetted `.proto` files.*

3.  **Manual Edits (Generally NOT needed for `betterproto` with `CosmosTx`):**
    *   The `Tx.chain` to `bytes` fix mentioned in historical logs (`mayachain/v1/common/common.proto`) was primarily for `protoc --decode types.MsgObservedTxOut` and is **not required** when using `betterproto` to generate stubs for `cosmos.tx.v1beta1.Tx`. `betterproto` handles standard types well.

### Step 2: Install Python Dependencies

Ensure your project's virtual environment has the following:

```bash
pip install protobuf==4.21.12
# (4.21.12 is known to work with betterproto 2.0.0b7 and was pulled in by cosmospy-protobuf)
# Adjust if betterproto explicitly requires a different version, but this is a stable choice.

pip install betterproto[compiler]==2.0.0b7
# Installs betterproto and the necessary grpcio-tools for its compiler.

pip install grpclib
# Often used with betterproto, good to have.
```
Update your `requirements.txt` accordingly.

### Step 3: Compile `.proto` Files using `betterproto`

From the root of your project (e.g., `mayarbscanner/`):

```bash
python -m grpc_tools.protoc \
    -I./proto/src \
    --python_betterproto_out=./proto/generated/pb_stubs \
    $(find ./proto/src -name '*.proto' -print)
```

**Explanation:**
*   `python -m grpc_tools.protoc`: Invokes the `protoc` compiler bundled with `grpcio-tools`. This ensures compatibility.
*   `-I./proto/src`: Sets the import path for `protoc`. All `.proto` file `import` statements will be resolved relative to this directory.
*   `--python_betterproto_out=./proto/generated/pb_stubs`: Tells `protoc` to use the `betterproto` plugin and specifies the output directory for the generated Python files. `betterproto` creates a package-like structure here mirroring your `.proto` file paths.
*   `$(find ./proto/src -name '*.proto' -print)`: This command finds all `.proto` files within your `proto/src/` directory and passes them to `protoc` for compilation. Ensure this command works in your shell or list the files manually if needed.

This will populate `proto/generated/pb_stubs/` with Python files (e.g., `proto/generated/pb_stubs/cosmos/tx/v1beta1/tx.py`).

### Step 4: Using Generated Stubs in Python

1.  **Add Generated Stubs to `sys.path`**:
    Your Python scripts need to be able to find the generated modules.

    ```python
    import sys
    from pathlib import Path

    # Assuming your script is in a directory like 'scripts/' or 'src/'
    # Adjust the relative path to PROJECT_ROOT if necessary.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent 
    PROTO_GEN_PB_STUBS_DIR = str(PROJECT_ROOT / "proto" / "generated" / "pb_stubs")

    if PROTO_GEN_PB_STUBS_DIR not in sys.path:
        sys.path.insert(0, PROTO_GEN_PB_STUBS_DIR)
    ```

2.  **Import and Deserialize**:

    ```python
    import base64
    # The import path mirrors the .proto file path within 'proto/src'
    from cosmos.tx.v1beta1 import Tx as CosmosTx 
    # For other messages, e.g., from mayachain:
    # from mayachain.v1.common import Coin as MayachainCoin

    # Example base64 encoded Tendermint transaction string
    # (This is a sample Tx, replace with your actual data)
    sample_base64_tx = "CoUCIgokaHR0cHM6Ly9hcGkuZXhhbXBsZS5jb20vcHJvamVjdHMVNWMyYzJlMzAtMDY2Yi0wMTNlLTM0ZDUtMmVhNzNkMGUzZDA0IpHZChwvY29zbW9zLmJhbmsudjFiZXRhMS5Nc2dTZW5kEoABCitjb3Ntb3MxNWY5amZ4cDQwdHY0dXdhdHVncDJzcnNwcXNkaGJ1eGNkaHQ4MnMSK2Nvc21vczE4c3AycXNwbjZ4NTh2cDZ5cHI4eG40dzhscjR0enNnMmdxaGQaDAoFdWNvc20SAzEwMBJoClAKRgofL2Nvc21vcy5jcnlwdG8uc2VjcDI1NmsxLlB1YktleRIjCiEDPjYnXPfN/r40zaLKFwms8kR88qgzdpACxSAhJq0jBfUSBAoCCAESFwoRCgV1Y29zbRIGMTAwMDAwEMCaDBpAK66PqLG0Ady0x5x30DMRQZfF5qj8yYgXj0dH4HkL7p5sQhA4hI4rN3Q7nSm05W7x3ZM+Z+YhhtX9nkWgKOYnIg=="
    tx_bytes = base64.b64decode(sample_base64_tx)

    try:
        decoded_tx_proto = CosmosTx().parse(tx_bytes)
        
        print("Successfully decoded CosmosTx!")
        # You can now access fields natively:
        # print(f"Memo: {decoded_tx_proto.body.memo}")
        # for msg in decoded_tx_proto.body.messages:
        #     if msg.type_url == "/cosmos.bank.v1beta1.MsgSend":
        #         # Need to parse the 'Any' message's value for specific types
        #         # from cosmos.bank.v1beta1 import MsgSend
        #         # msg_send = MsgSend().parse(msg.value)
        #         # print(f"  MsgSend from: {msg_send.from_address} to: {msg_send.to_address}")
        #         pass # Further parsing of Any needed for message content

        # For easier inspection, convert to a dictionary:
        # decoded_tx_dict = decoded_tx_proto.to_dict(include_default_values=True,
        #                                           casing_transform=betterproto.Casing.SNAKE) # or CAMEL
        # print(json.dumps(decoded_tx_dict, indent=2))

    except Exception as e:
        print(f"Error decoding CosmosTx: {e}")
        # import traceback
        # traceback.print_exc()

    ```

### Step 5: Committing Generated Stubs (Strongly Recommended)

*   **Rationale**: Typically, generated code is not committed to version control. However, due to the significant challenges and specific environment (versions of `protoc`, `betterproto`, source `.proto` files) required for successful compilation with `betterproto` for these complex protos, committing the generated Python stubs is **strongly recommended**.
*   **Benefits**:
    *   Ensures any developer cloning the repository can run the code without needing to perfectly replicate the `protoc` compilation environment.
    *   Preserves the exact working version of the stubs.
    *   Simplifies the setup process for new contributors or for yourself on a new machine.
*   **Action**:
    *   If `proto/generated/pb_stubs/` is in your `.gitignore` file, remove that line.
    *   Commit the entire `proto/generated/pb_stubs/` directory to your Git repository.

## Part 2: Fallback/Alternative - `protoc --decode` via `subprocess`

This method is useful for:
*   Debugging raw byte streams.
*   Handling extremely complex or Mayanode-specific messages if `betterproto` (or other native libraries) struggle.
*   Cases where you need a quick human-readable text output of a Protobuf message.

The script `scripts/decode_local_block_for_comparison.py` in this project demonstrates this for `types.MsgObservedTxOut`.

### Prerequisites

*   A compatible `protoc` compiler accessible in your PATH or via a direct path. Version `3.20.1` was known to work for `MsgObservedTxOut` with the specific protos mentioned below.
*   The exact `.proto` definition file for the message type you are decoding.

### Steps (Example for `types.MsgObservedTxOut`)

1.  **Obtain Specific Mayanode Protos**:
    For `types.MsgObservedTxOut`, it was found that protos from a specific Mayanode commit were necessary:
    *   Clone Mayanode: `git clone https://gitlab.com/mayachain/mayanode.git temp_mayanode_for_decode`
    *   Checkout commit: `cd temp_mayanode_for_decode && git checkout 59743feb12940328336888b94f852e09786d797a && cd ..`
    *   You'll need `temp_mayanode_for_decode/proto/mayachain/v1/x/mayachain/types/msg_observed_txout.proto` and all its dependencies (like `common.proto`, `gogo.proto` etc.) from this commit. Structure them in a temporary directory (e.g., `temp_decode_protos/`) maintaining relative paths for imports.

2.  **Crucial Manual Edit for `MsgObservedTxOut`**:
    In `temp_decode_protos/mayachain/v1/common/common.proto` (from the commit above), a manual edit was required for `Tx.chain`:
    ```diff
    // In message Tx:
    // string id = 1 [ (gogoproto.casttype) = "gitlab.com/mayachain/mayanode/common.TxID", (gogoproto.customname) = "ID" ];
    // -  string chain = 2 [ (gogoproto.casttype) = "gitlab.com/mayachain/mayanode/common.Chain" ];
    // +  bytes chain = 2 [ (gogoproto.customname) = "ChainBytes" ]; // Changed from string to bytes to handle non-UTF8 chain identifiers
    // string from_address = 3 [ (gogoproto.casttype) = "gitlab.com/mayachain/mayanode/common.Address" ];
    ```
    *(This specific path and field might vary; the principle is that some `string` fields might actually contain non-UTF8 bytes and need to be `bytes` in the .proto for `protoc --decode` to work.)*

3.  **`protoc --decode` Command**:
    ```bash
    /path/to/your/protoc_3.20.1_or_compatible/bin/protoc \
        --decode=types.MsgObservedTxOut \
        -I./temp_decode_protos \
        ./temp_decode_protos/mayachain/v1/x/mayachain/types/msg_observed_txout.proto \
        < /path/to/your/binary_message_data.bin
    ```
    This command prints the decoded message in a text format to standard output.

4.  **Using from Python via `subprocess`**:
    The script `scripts/decode_local_block_for_comparison.py` shows how to:
    *   Take a base64 encoded string of `MsgObservedTxOut`.
    *   Decode it to bytes.
    *   Pass these bytes to `protoc --decode` via `subprocess.run()`.
    *   Capture the text output and parse it (e.g., using regex or simple string manipulation) into a Python dictionary.

    This approach is less ideal than native parsing but can be a robust fallback for specific, troublesome messages.

## Conclusion

For most common tasks involving Cosmos SDK messages from Mayanode (like transactions from Tendermint RPC), the **`betterproto` method is highly recommended** due to its native Python integration, type safety, and ease of use once set up. The `subprocess` method should be reserved for specific complex types where native solutions fail or for diagnostic purposes.

By committing the `proto/src/` (source .proto files) and `proto/generated/pb_stubs/` (Python stubs generated by betterproto) directories, this project aims to make Protobuf handling as straightforward as possible for anyone working with the codebase. 