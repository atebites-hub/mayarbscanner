import requests
import json

# --- Configuration ---
MAYA_API_URL = "https://mayanode.mayachain.info/mayachain/pools"
COINGECKO_API_URL_PING = "https://api.coingecko.com/api/v3/ping"
COINGECKO_API_URL_COINLIST = "https://api.coingecko.com/api/v3/coins/list"
MAYA_MIDGARD_ACTIONS_URL = "https://midgard.mayachain.info/v2/actions"
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
def fetch_recent_maya_actions(limit=10, offset=0):
    """Fetches recent actions from Maya Protocol's Midgard API."""
    params = {"limit": limit, "offset": offset}
    return fetch_api_data(MAYA_MIDGARD_ACTIONS_URL, "Maya Protocol Midgard Actions", params=params)

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

    print("API connection tests finished.") 