import time
import random
from datetime import datetime
import requests
import json

# --- Configuration ---
MONITORING_INTERVAL_SECONDS = 10  # How often to check Midgard
MIDGARD_API_URL = "https://midgard.mayachain.info/v2"  # Corrected Maya Protocol Midgard API endpoint
MAX_ACTIONS_LIMIT = 50 # Max number of actions to fetch per call

# ASSETS = ["BTC.BTC", "ETH.ETH", "MAYA.CACAO", "USDC.USDC-0XA0B86991C6218B36C1D19D4A2E9EB0CE3606EB48", "KUJI.KUJI"] # No longer needed for generation

# --- Helper Function ---
# def generate_simulated_pending_tx(): # Removed
#     """Generates a dictionary representing a simulated pending transaction.""" # Removed
#     tx_hash = f"0x{random.randbytes(32).hex()}" # Removed
#     asset_in = random.choice(ASSETS) # Removed
#     asset_out = random.choice([a for a in ASSETS if a != asset_in]) # Removed
#     amount_in = round(random.uniform(0.001, 5.0), 8) # Removed
#     # Simulate a slightly different amount out for a pending tx # Removed
#     amount_out_expected = round(amount_in * random.uniform(0.9, 1.1) * random.uniform(10, 50000), 8) # Removed
# # Removed
#     return { # Removed
#         "simulated_tx_hash": tx_hash, # Removed
#         "timestamp_detected": datetime.now().isoformat(), # Removed
#         "asset_in": asset_in, # Removed
#         "amount_in": amount_in, # Removed
#         "asset_out": asset_out, # Removed
#         "amount_out_expected": amount_out_expected, # Removed
#         "status": "pending" # Removed
#     } # Removed

def fetch_pending_maya_transactions():
    """Fetches pending transactions from the Midgard API."""
    endpoint = f"{MIDGARD_API_URL}/actions"
    params = {
        "limit": MAX_ACTIONS_LIMIT, # Reverted to MAX_ACTIONS_LIMIT
        "offset": 0,
        # "status": "pending", # Removed to avoid 400 error, will filter client-side
        "type": "swap" # Focusing on swaps for arbitrage
    }
    try:
        print(f"[{datetime.now().isoformat()}] Fetching from Midgard: {endpoint} with params {params}", flush=True)
        response = requests.get(endpoint, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        actions_data = response.json()
        
        pending_txs = []
        if isinstance(actions_data, dict) and "actions" in actions_data:
            for action in actions_data["actions"]:
                # We need to extract relevant details for an arbitrage bot
                # The structure of a 'swap' action:
                # action['type'] == 'swap'
                # action['status'] == 'pending' (or 'success' if we were looking at completed)
                # action['in'][0]['coins'][0]['asset'] & ['amount']
                # action['out'][0]['coins'][0]['asset'] & ['amount'] (this might be empty if truly pending)
                # action['date'] (timestamp of the action, ns)
                # action['height'] (block height)
                #
                # For a "pending" swap, the 'out' array might be empty or represent the expected out.
                # We need to be careful about what 'pending' means here.
                # If status is 'pending', it implies the swap is known but not fully completed (e.g., outbound tx not confirmed).

                # Client-side filter for pending status if not done by API
                if action.get('status') != 'pending': # Added client-side filter
                    continue # Skip if not pending

                tx_details = {
                    "action_id": f"{action.get('date')}-{action.get('in')[0].get('txID') if action.get('in') and action.get('in')[0] else 'N/A'}", # Create a somewhat unique ID
                    "timestamp_detected_milli": int(action.get('date', 0)) // 1000000, # Convert ns to ms
                    "status": action.get('status'),
                    "type": action.get('type'),
                    "metadata": action.get('metadata', {}).get('swap', {}) # Get swap specific metadata like target price
                }

                if action.get('in'):
                    for tx_item in action['in']:
                        if tx_item.get('coins'):
                            tx_details['in_tx_id'] = tx_item.get('txID')
                            tx_details['in_address'] = tx_item.get('address')
                            tx_details['in_coins'] = [{'asset': coin.get('asset'), 'amount': coin.get('amount')} for coin in tx_item['coins']]
                
                if action.get('out'):
                    for tx_item in action['out']:
                        if tx_item.get('coins'): # Out coins might not exist if it's truly "pending" an outcome
                            tx_details['out_tx_id'] = tx_item.get('txID') # This could be blank if not sent yet
                            tx_details['out_address'] = tx_item.get('address')
                            tx_details['out_coins'] = [{'asset': coin.get('asset'), 'amount': coin.get('amount')} for coin in tx_item['coins']]
                
                pending_txs.append(tx_details)
        else:
            print(f"[{datetime.now().isoformat()}] Unexpected response structure from Midgard: {actions_data}", flush=True)
            return []
            
        return pending_txs

    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now().isoformat()}] Error fetching from Midgard API: {e}", flush=True)
        return []
    except json.JSONDecodeError as e:
        print(f"[{datetime.now().isoformat()}] Error decoding JSON from Midgard API: {e}", flush=True)
        print(f"Response text: {response.text if 'response' in locals() else 'Response object not available'}", flush=True)
        return []


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Maya Protocol Mempool Monitor (Midgard API) ---", flush=True)
    print("This script monitors the Maya Protocol mempool for pending swap transactions using the Midgard API.", flush=True)
    print(f"Checking for pending swaps every {MONITORING_INTERVAL_SECONDS} seconds.", flush=True)
    print("Press Ctrl+C to stop monitoring.\n", flush=True)

    try:
        while True:
            print(f"[{datetime.now().isoformat()}] Monitoring for pending Maya swap actions via Midgard...", flush=True)
            
            pending_actions = fetch_pending_maya_transactions()
            
            if pending_actions:
                print(f"[{datetime.now().isoformat()}] DETECTED {len(pending_actions)} PENDING SWAP ACTION(S) (or actions with pending components):", flush=True)
                for action in pending_actions:
                    # Pretty print the JSON
                    print(json.dumps(action, indent=2), flush=True)
                    print("---", flush=True)
            else:
                print(f"[{datetime.now().isoformat()}] No pending swap actions found or error in fetching.", flush=True)
            
            time.sleep(MONITORING_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.", flush=True)
    finally:
        print("Mempool monitor simulation finished.", flush=True) 