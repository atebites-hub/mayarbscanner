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

## Troubleshooting Common Issues

### 1. `ImportError: cannot import name 'GenericAlias' from partially initialized module 'types'` (or similar for `MappingProxyType`)

*   **Cause:** This error typically occurs if Python's import system, during its early startup phase, encounters a user-defined module or package named `types` (e.g., `your_protobuf_output_dir/types/__init__.py`) *before* it fully initializes the standard library `types` module. This can happen if `your_protobuf_output_dir/` (which contains the `types/` subdirectory) is added too broadly to `sys.path` or `PYTHONPATH`, especially if prepended.
*   **Solution for `betterproto` with `pb_stubs` structure:**
    *   Ensure your `betterproto` output directory (e.g., `proto/generated/pb_stubs/`) contains the compiled modules (`cosmos/`, `gogoproto/`, `google/`, `mayachain/`, `types/` etc., as subdirectories which are Python packages).
    *   Instead of adding `proto/generated/pb_stubs/` directly to `sys.path`, add its **parent directory**, `proto/generated/`, to `sys.path`.
    *   Then, in your Python code, import the generated stubs by referencing the `pb_stubs` package explicitly.
    *   **Example `sys.path` modification (e.g., in `src/api_connections.py` or `src/common_utils.py`):**
        ```python
        import sys
        import os
        _current_script_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_current_script_dir) # If script is in src/
        # Path to the PARENT of pb_stubs, i.e., 'proto/generated/'
        _proto_generated_path = os.path.join(_project_root, "proto", "generated")

        if _proto_generated_path not in sys.path:
            sys.path.insert(0, _proto_generated_path) # Prepend to allow finding pb_stubs
            # print(f"[script_name.py] Prepended to sys.path: {_proto_generated_path}")
        ```
    *   **Example import statement:**
        ```python
        # Import Tx from the __init__.py of pb_stubs.cosmos.tx.v1beta1
        from pb_stubs.cosmos.tx.v1beta1 import Tx as CosmosTx
        ```
    *   **Rationale:** By adding `proto/generated/` to the path, `pb_stubs` becomes a top-level package that Python can find. Imports like `from pb_stubs.cosmos...` then correctly navigate *within* this package structure. The `pb_stubs/types/` module is then correctly treated as `pb_stubs.types` and does not conflict with the standard library `types` module during interpreter initialization.
*   **If using `PYTHONPATH` environment variable:** Set `PYTHONPATH` to include `proto/generated/` (the parent of `pb_stubs`), not `proto/generated/pb_stubs/` itself.

### 2. `ModuleNotFoundError: No module named 'pb_stubs.cosmos...'` (or similar)

*   **Cause:** This means Python cannot find the `pb_stubs` package or the subsequent modules even after `sys.path` or `PYTHONPATH` modifications.
*   **Checks:**
    1.  Verify that `proto/generated/` (the directory *containing* `pb_stubs/`) is indeed on `sys.path` (print `sys.path` to confirm).
    2.  Ensure that `proto/generated/pb_stubs/` actually exists and is populated with the compiled `.py` files and that it has an `__init__.py` file (it should, if `betterproto` ran correctly).
    3.  Ensure that the subsequent directories like `proto/generated/pb_stubs/cosmos/`, `proto/generated/pb_stubs/cosmos/tx/`, `proto/generated/pb_stubs/cosmos/tx/v1beta1/` all exist and contain `__init__.py` files. `betterproto` should create these.
    4.  Double-check your import statement for typos against the actual directory and file structure within `pb_stubs`. For instance, if `Tx` is defined in `proto/generated/pb_stubs/cosmos/tx/v1beta1/__init__.py`, then `from pb_stubs.cosmos.tx.v1beta1 import Tx` is correct.

### 3. General Python Caching / Stale Code Execution

*   **Issue:** You modify a Python file (e.g., a utility module like `common_utils.py` that might consume these Protobuf stubs or perform other parsing), but when you run a script that imports this module, your changes don't seem to take effect (e.g., new print statements don't appear, old behavior persists).
*   **Cause:** This can be due to Python's caching mechanisms, which can be more persistent than just `__pycache__` directories. The Python interpreter might hold an old version of the module in memory, or your IDE (like VS Code or Cursor) might have its own caching or fail to pick up changes immediately.
*   **Troubleshooting Steps:**
    1.  **Save All Files:** Ensure all modified files are saved in your IDE.
    2.  **Clear `__pycache__`:** First, try removing all `__pycache__` directories from your project. You can do this from your terminal in the project root:
        ```bash
        find . -type d -name '__pycache__' -exec rm -rf {} +
        find . -type f -name '*.pyc' -delete
        ```
    3.  **IDE-Specific Actions:**
        *   **Restart Python Kernel/Session:** If you are using an interactive environment (like Jupyter notebooks or an IDE's Python console), try restarting the kernel or Python session.
        *   **Reload Window (VS Code/Cursor):** Sometimes, simply reloading the IDE window can help. In VS Code or Cursor, open the Command Palette (Cmd/Ctrl+Shift+P) and search for "Developer: Reload Window".
    4.  **Full Environment Restart (More Drastic):** If the problem continues, a full environment restart is highly recommended:
        *   Close your IDE completely.
        *   Close all terminal sessions related to the project.
        *   Re-open the project in your IDE.
        *   Start a new terminal session.
        *   Re-activate your Python virtual environment (e.g., `source .venv/bin/activate`).
        *   Try running your script again. This often resolves stubborn caching issues.
    5.  **Verify File Being Run:** Double-check that you are indeed running the script you *think* you are running and that it's importing the correct version of the module you modified. Temporary, unique print statements at the very top of the module and the script can help confirm this.

## Conclusion

For most common tasks involving Cosmos SDK messages from Mayanode (like transactions from Tendermint RPC), the **`betterproto` method is highly recommended** due to its native Python integration, type safety, and ease of use once set up. The `subprocess` method should be reserved for specific complex types where native solutions fail or for diagnostic purposes.

By committing the `proto/src/` (source .proto files) and `proto/generated/pb_stubs/` (Python stubs generated by betterproto) directories, this project aims to make Protobuf handling as straightforward as possible for anyone working with the codebase. 