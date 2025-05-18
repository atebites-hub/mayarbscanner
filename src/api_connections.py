import requests
import json

# --- Configuration ---
MAYA_API_URL = "https://mayanode.mayachain.info/mayachain/pools"
COINGECKO_API_URL_PING = "https://api.coingecko.com/api/v3/ping"
COINGECKO_API_URL_COINLIST = "https://api.coingecko.com/api/v3/coins/list"
MAYA_MIDGARD_ACTIONS_URL = "https://midgard.mayachain.info/v2/actions"
MAYA_MIDGARD_HEALTH_URL = "https://midgard.mayachain.info/v2/health"
# Placeholder for Uniswap - direct REST API for general data is complex.
# Often requires The Graph or SDKs. For now, we'll simulate or use a general token data endpoint.
UNISWAP_PLACEHOLDER_INFO = "Uniswap V3 data is typically accessed via The Graph (GraphQL) or SDKs, not a simple REST API endpoint for general pool/transaction data. For specific token price, one might use a service that aggregates this."

# --- Helper Function ---
def fetch_api_data(url, api_name, params=None):
    """Fetches data from a given API URL and prints status."""
    print(f"--- Testing {api_name} API ---")
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        print(f"Successfully connected to {api_name} ({url}).")
        try:
            data = response.json()
            print(f"Sample data from {api_name}:")
            # Print a small part of the data to avoid flooding the console
            if isinstance(data, list):
                print(json.dumps(data[:2], indent=2)) # Print first 2 items if it's a list
            elif isinstance(data, dict):
                # Print a few key-value pairs if it's a dict
                print(json.dumps(dict(list(data.items())[:3]), indent=2))
            else:
                print(data)
            return data
        except requests.exceptions.JSONDecodeError:
            print("Response was not in JSON format, or content is empty.")
            print("Response text:", response.text[:200]) # Print first 200 chars of text
            return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to {api_name} ({url}): {e}")
        return None
    finally:
        print("-" * 30 + "\n")

# --- New Function to fetch Maya actions ---
def fetch_recent_maya_actions(limit=10, offset=0, height=None, action_type=None):
    """Fetches recent actions from the Maya Protocol Midgard API.
    Can be optionally filtered by a specific block height and action type."""
    print(f"--- Querying Maya Protocol Midgard for Actions (limit={limit}, offset={offset}, height={height or 'any'}, type={action_type or 'any'}) ---")
    
    params = {
        "limit": limit,
        "offset": offset
    }
    if height is not None:
        params["height"] = height # Add height to params if provided
    if action_type is not None:
        params["type"] = action_type # Add type to params if provided

    try:
        response = requests.get(MAYA_MIDGARD_ACTIONS_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        # print(f"Successfully fetched {len(data.get('actions', []))} actions.")
        # print(f"DEBUG: Raw actions data from Midgard (first 500 chars): {str(data)[:500]}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Maya Protocol Midgard Actions API: {e}")
        return None
    finally:
        print("-" * 30 + "\n")

# --- New Function to get last aggregated block height ---
def get_maya_last_aggregated_block_height():
    """Fetches the last aggregated block height from Maya Protocol's Midgard API."""
    health_data = fetch_api_data(MAYA_MIDGARD_HEALTH_URL, "Maya Protocol Midgard Health")
    if health_data and isinstance(health_data, dict) and "lastAggregated" in health_data:
        last_aggregated_height = health_data.get("lastAggregated", {}).get("height")
        if last_aggregated_height is not None:
            try:
                return int(last_aggregated_height)
            except ValueError:
                print(f"Error: lastAggregated.height ('{last_aggregated_height}') is not a valid integer.")
                return None
        else:
            print("Error: 'height' not found in lastAggregated data.")
            return None
    else:
        print("Failed to fetch health data or data is not in the expected format.")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting API connection tests...\n")

    # Test Maya Protocol API
    maya_data = fetch_api_data(MAYA_API_URL, "Maya Protocol")

    # Test CoinGecko API
    # First, ping to check connectivity
    coingecko_ping_data = fetch_api_data(COINGECKO_API_URL_PING, "CoinGecko (Ping)")
    if coingecko_ping_data: # If ping is successful, try fetching coin list
        coingecko_coins_data = fetch_api_data(COINGECKO_API_URL_COINLIST, "CoinGecko (Coin List)")

    # Placeholder for Uniswap
    print("--- Uniswap API Information ---")
    print(UNISWAP_PLACEHOLDER_INFO)
    print("For demonstration, we are not making a live call to a complex Uniswap data source in this initial script.")
    print("In a real scenario, one would use a library like `web3.py` with an Ethereum node,")
    print("or query a service like The Graph for Uniswap pool data.")
    print("-" * 30 + "\n")

    # Test fetching Maya actions
    maya_actions_data = fetch_recent_maya_actions(limit=5)
    if maya_actions_data and isinstance(maya_actions_data, dict) and 'actions' in maya_actions_data:
        print(f"Successfully fetched {len(maya_actions_data['actions'])} recent Maya actions for testing.")
    else:
        print("Could not fetch Maya actions or data is not in expected format.")

    # Test fetching actions of a specific type
    maya_swap_actions_data = fetch_recent_maya_actions(limit=3, action_type="swap")
    if maya_swap_actions_data and isinstance(maya_swap_actions_data, dict) and 'actions' in maya_swap_actions_data:
        print(f"Successfully fetched {len(maya_swap_actions_data['actions'])} recent Maya swap actions for testing.")
        # Further check if all returned actions are indeed swaps
        all_swaps = True
        for action in maya_swap_actions_data['actions']:
            if action.get('type') != 'swap':
                all_swaps = False
                break
        if all_swaps:
            print("All fetched actions are of type swap as expected.")
        else:
            print("Error: Not all fetched actions were of type swap when filtered.")
    else:
        print("Could not fetch Maya swap actions or data is not in expected format.")

    print("\n--- Testing Fetching Last Aggregated Block Height ---")
    last_block_height = get_maya_last_aggregated_block_height()
    if last_block_height is not None:
        print(f"Last aggregated block height: {last_block_height}")
    else:
        print("Failed to get last aggregated block height.")

    print("API connection tests finished.") 