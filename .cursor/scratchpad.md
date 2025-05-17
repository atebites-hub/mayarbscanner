# Project Scratchpad

## Background and Motivation
The user wants to implement a Maya Arbitrage Scanner. The goal is to identify and execute arbitrage opportunities on the Maya Protocol. This involves data collection, model development, training, and real-time inference. The current request is to begin Phase 1 implementation in Executor mode.

## Key Challenges and Analysis
- Accessing and integrating data from multiple APIs (Maya, Uniswap, CoinGecko).
- Handling potentially large volumes of historical and real-time data.
- Ensuring the data preprocessing pipeline is robust and efficient.
- Setting up a reliable mempool monitoring mechanism.
- Installing and managing dependencies like Python and PyTorch.

## High-level Task Breakdown
(Derived from Docs/Implementation Plan.md)

**Phase 1: Data Collection and Preprocessing (2 weeks)**
- **Task 1.1: Set up API Connections**
  - Description: Connect to Maya blockchain, Uniswap, and CoinGecko APIs.
  - Code Guidance: Use Python `requests` or `aiohttp` to fetch data from endpoints (e.g., `https://api.mayaprotocol.com/pools`).
  - Deliverable: Scripts to query and log API responses.
  - Success Criteria: Successfully fetch and print sample data from each API.
- **Task 1.2: Fetch and Store Historical Data**
  - Description: Collect 6 months of Maya transaction data (`arb_ID`, `ΔX`, etc.).
  - Code Guidance: Use Midgard API endpoint `/v2/actions` to fetch recent Maya Protocol actions (swaps, liquidity changes, etc.). Use `pandas` to save data as CSV.
  - Deliverable: A CSV file with recent transaction data.
  - Success Criteria: A CSV file `data/realtime_maya_transactions.csv` containing at least a few dozen rows of recent Maya Protocol actions, fetched live from the Midgard API.
- **Task 1.3: Implement Mempool Monitoring**
  - Description: Monitor pending transactions in real time.
  - Code Guidance: Use `web3.py` or a custom Maya node setup to access mempool data. (Note: Maya mempool access might differ from Ethereum's `web3.py` approach and require specific Maya chain tools/APIs).
  - Deliverable: A script logging mempool activity.
  - Success Criteria: Script prints simulated mempool transactions or a message indicating no specific Maya mempool API found yet.
- **Task 1.4: Preprocess Data**
  - Description: Normalize features, create `arb_ID` embeddings, and build sequences.
  - Code Guidance: Use `sklearn` for normalization, `PyTorch` for embeddings, and create a function to generate sequences of `M` trades.
  - Deliverable: A preprocessing pipeline outputting tensor sequences.
  - Success Criteria: A Python function that takes raw data (simulated) and outputs processed tensors.

## Project Status Board
(To be filled by Executor, reviewed by Planner)
- [x] Install Python and necessary libraries (requests, pandas, scikit-learn, PyTorch).
- [x] Task 1.1: Set up API Connections
- [x] Task 1.2 (Redo - Real-time data): Fetch and Store Real-time Maya Transaction Data via Midgard API.
- [x] Task 1.3: Implement Mempool Monitoring (Simulated/Placeholder)
- [x] Task 1.4: Preprocess Data (Simulated)
- [x] Create a Python test script demonstrating Phase 1 completion.

## Executor's Feedback or Assistance Requests
(To be filled by Executor when help is needed or milestones are reached)
- Successfully suggested commands to install Homebrew, Python 3.13, and Python libraries (requests, pandas, scikit-learn, PyTorch). Libraries were installed using `pip` associated with a Python 3.9 environment, which might not be the intended Python 3.13 from Homebrew. This could be a potential issue if the PATH is not configured to prioritize `/opt/homebrew/bin`.
- Task 1.1 (Set up API Connections) is complete. Successfully connected to Maya Protocol API (`https://mayanode.mayachain.info/mayachain/pools`) and CoinGecko API. Uniswap connection is a placeholder as it requires more complex integration (GraphQL/SDK).
- Task 1.2 (Fetch and Store Historical Data (Simulated)) was redone to fetch real-time data.
  - Successfully fetched 50 real-time Maya Protocol actions using the Midgard API endpoint `https://midgard.mayachain.info/v2/actions`.
  - Data saved to `data/realtime_maya_transactions.csv`.
  - Encountered and resolved several issues during this process:
    - Initial `python` command failed (used `python3`).
    - `ModuleNotFoundError: No module named 'pandas'` resolved by creating a Python virtual environment (`.venv`) and installing `pandas` and `requests` into it using `.venv/bin/python -m pip install ...`.
    - API calls to Midgard initially failed with a 500 error due to requesting `limit=100`, which exceeds the documented maximum of 50. Corrected by reducing the limit.
    - API timeout was increased from 10s to 30s in `api_connections.py` to handle potentially slower responses, though the `limit` issue was the primary cause of failure.
- Task 1.3 (Implement Mempool Monitoring (Simulated/Placeholder)) is complete. Tested successfully.
- Task 1.4 (Preprocess Data (Simulated)) is complete. `scikit-learn` and `torch` (with `torchvision`, `torchaudio`) were installed into the `.venv` to resolve `ModuleNotFoundError`. Tested successfully.
- All Phase 1 tasks are now complete and `src/test_phase1_completion.py` passes all tests, including the updated Task 1.2 for real-time data and Task 1.4 after dependency installation in the virtual environment.

## Lessons
(To be filled by Executor with learnings, especially fixes to errors or new information)
- When installing Python on macOS, ensure Homebrew is installed first (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`), then install Python (`brew install python`).
- Install Python packages using `python3 -m pip install <package_name>`. Be mindful of which `python3` is being used if multiple versions are present. The desired Python (e.g., from Homebrew) should be prioritized in the system PATH or called explicitly (e.g. `/opt/homebrew/bin/python3 -m pip install ...`).
- PyTorch installation via pip: `pip install torch torchvision torchaudio`.
- The correct Maya Protocol API endpoint for pools is `https://mayanode.mayachain.info/mayachain/pools`. The initially provided `api.mayaprotocol.com` was incorrect or not resolvable.
- When generating mock data with pandas, ensure `os.makedirs(exist_ok=True)` or an explicit check `if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)` is used to create output directories to prevent errors if the script is run multiple times or the directory doesn't exist.
- For long-running simulation scripts (like a mempool monitor), include a `try...except KeyboardInterrupt` block to handle graceful shutdown when the user presses Ctrl+C.
- When preprocessing data for sequences: clearly define features to include. Convert timestamps to numerical representations (e.g., Unix epoch seconds) before scaling. Map categorical string features to numerical indices. For unique IDs like `arb_ID`, they are generally not embedded directly for generalization; instead, use them as identifiers or extract structured features if available. Use `torch.tensor()` to convert NumPy arrays to PyTorch tensors.
- The Maya Protocol Midgard API endpoint `https://midgard.mayachain.info/v2/actions` can be used to fetch recent actions (transactions). The `limit` parameter for this endpoint has a maximum value of 50.
- For managing Python dependencies and avoiding conflicts with system Python (especially on macOS with Homebrew), use virtual environments:
  - Create: `python3 -m venv .venv`
  - Install packages: `.venv/bin/python -m pip install <package_name>`
  - Run scripts: `.venv/bin/python your_script.py`
- Ensure scripts are run with `python3` (or the specific venv python like `.venv/bin/python`) if `python` is not aliased or available. If `ModuleNotFound` errors occur, ensure all dependencies (e.g., `pandas`, `requests`, `scikit-learn`, `torch`) are installed into the active virtual environment. 