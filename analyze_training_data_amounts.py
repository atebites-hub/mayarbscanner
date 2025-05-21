import json
from collections import defaultdict
import numpy as np
import os
from decimal import Decimal, InvalidOperation

def get_decimal_places(s):
    """Helper function to count decimal places in a string representation of a number."""
    try:
        # Convert to Decimal to handle potential floating point inaccuracies if we used float()
        d = Decimal(s)
        # \'as_tuple()\' gives (sign, digits, exponent)
        # If exponent is negative, it represents the number of decimal places.
        if d.as_tuple().exponent < 0:
            return abs(d.as_tuple().exponent)
        return 0 # No decimal places or it\'s an integer
    except InvalidOperation:
        # Handle cases where s is not a valid number string (e.g., empty, "None", etc.)
        return 0

def analyze_amounts(data_path="data/transactions_data.json"):
    """
    Analyzes transaction amounts from a JSON file and infers asset precisions,
    also providing insights into whether amounts might already be in atomic units.
    """
    print(f"Analyzing transaction data from: {data_path} to infer asset precisions and format.")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {data_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    actions = data.get('actions', [])
    if not actions:
        print("No actions found in the data.")
        return

    asset_analysis = defaultdict(lambda: {
        "max_decimals": 0, 
        "count": 0, 
        "integer_strings": 0, 
        "decimal_strings": 0, 
        "samples": []
    })

    print("\n--- Analyzing Asset Amounts from In/Out Coins & Metadata ---")
    for action_idx, action in enumerate(actions):
        coins_to_process = []
        # Process 'in' coins
        for in_tx in action.get('in', []):
            for coin in in_tx.get('coins', []):
                coins_to_process.append(coin)
        
        # Process 'out' coins
        for out_tx in action.get('out', []):
            for coin in out_tx.get('coins', []):
                coins_to_process.append(coin)

        # Process metadata amounts
        metadata = action.get('metadata', {})
        swap_meta = metadata.get('swap', {})
        pools = action.get('pools', [])
        first_pool_asset = pools[0] if pools else None

        if swap_meta.get('liquidityFee') and first_pool_asset:
            coins_to_process.append({'asset': first_pool_asset, 'amount': swap_meta['liquidityFee']})
        
        for nf_coin in swap_meta.get('networkFees', []):
            coins_to_process.append(nf_coin)

        withdraw_meta = metadata.get('withdraw', {})
        ilp_amount_str = withdraw_meta.get('ilProtection')
        if ilp_amount_str: # Assuming ILP is in CACAO by default for precision analysis
            coins_to_process.append({'asset': 'MAYA.CACAO', 'amount': ilp_amount_str}) 
            
        # --- Iterate through all collected coins/amounts ---
        for coin_data in coins_to_process:
            asset = coin_data.get('asset')
            amount_str = coin_data.get('amount')

            if asset and amount_str and isinstance(amount_str, str):
                asset_stats = asset_analysis[asset]
                asset_stats['count'] += 1
                
                decimals = get_decimal_places(amount_str)
                if decimals > asset_stats['max_decimals']:
                    asset_stats['max_decimals'] = decimals
                
                if '.' in amount_str:
                    asset_stats['decimal_strings'] += 1
                else:
                    # Check if it's a valid integer string (catches empty strings or non-numeric)
                    try:
                        int(amount_str) 
                        asset_stats['integer_strings'] += 1
                    except ValueError:
                        pass # Not a simple integer string, could be None or other format

                if len(asset_stats['samples']) < 5: # Store a few samples
                    asset_stats['samples'].append(amount_str)
            elif asset and amount_str and (isinstance(amount_str, int) or isinstance(amount_str, float)):
                # Handle cases where amount might already be a number (less likely in raw JSON but good to cover)
                amount_str_conv = str(amount_str)
                asset_stats = asset_analysis[asset]
                asset_stats['count'] += 1
                decimals = get_decimal_places(amount_str_conv)
                if decimals > asset_stats['max_decimals']:
                    asset_stats['max_decimals'] = decimals
                if '.' in amount_str_conv:
                    asset_stats['decimal_strings'] += 1
                else:
                    asset_stats['integer_strings'] += 1
                if len(asset_stats['samples']) < 5:
                    asset_stats['samples'].append(amount_str_conv)


    print("\n--- Asset Amount Format Analysis & Inferred Precisions ---")
    sorted_assets = sorted(asset_analysis.keys())
    
    final_precisions_map_str = "ASSET_PRECISIONS = {\n"
    for asset in sorted_assets:
        stats = asset_analysis[asset]
        max_obs_decimals = stats['max_decimals']
        
        print(f"\nAsset: {asset}")
        print(f"  Total Entries: {stats['count']}")
        print(f"  Integer Strings: {stats['integer_strings']} ({stats['integer_strings']/stats['count']:.1%} if stats['count'] > 0 else 'N/A')")
        print(f"  Decimal Strings: {stats['decimal_strings']} ({stats['decimal_strings']/stats['count']:.1%} if stats['count'] > 0 else 'N/A')")
        print(f"  Max Observed Decimals: {max_obs_decimals}")
        print(f"  Sample Raw Amounts: {stats['samples']}")

        # Heuristic for final precision
        final_precision = max_obs_decimals
        asset_upper = asset.upper()

        # Apply known minimums/defaults, preferring them if observed is lower
        if asset_upper == "MAYA.CACAO": final_precision = max(max_obs_decimals, 10)
        elif asset_upper == "ETH.ETH": final_precision = max(max_obs_decimals, 18)
        elif asset_upper.endswith(('.USDT', '.USDC', '.USK')) and not asset_upper.startswith("ETH/") and not asset_upper.startswith("ARB/"): # Avoid pool names
             final_precision = max(max_obs_decimals, 6)
        elif "ETH" in asset_upper and not asset_upper.startswith("ETH/") and not asset_upper.startswith("ARB/"): # Other ETH-based tokens, default to 18 if observed is less
             final_precision = max(max_obs_decimals, 18)
        elif asset_upper == "BTC.BTC": final_precision = max(max_obs_decimals, 8)
        elif asset_upper == "THOR.RUNE": final_precision = max(max_obs_decimals, 8)
        elif asset_upper == "DASH.DASH": final_precision = max(max_obs_decimals, 8)
        # If still 0 after heuristics for common asset types, and all strings were integers,
        # it's possible they are atomic. But for safety, default to a common precision like 8 if it's not a stablecoin/ETH/CACAO.
        # However, if max_obs_decimals > 0, it's likely float strings.
        if final_precision == 0 and stats['decimal_strings'] == 0 and stats['integer_strings'] > 0:
            if not asset_upper.endswith(('.USDT', '.USDC', '.USK')) and not "ETH" in asset_upper and asset_upper != "MAYA.CACAO":
                # print(f"  Note: Asset {asset} has 0 observed decimals and all integer strings. Potentially atomic or uses default 8.")
                # Defaulting to 8 is a heuristic. The user should verify if these are truly atomic and precision should be 0 for preprocessing.
                final_precision = 8 # Fallback for non-major, non-stable, all-integer assets
        
        final_precisions_map_str += f'    "{asset}": {final_precision}, # ObservedMax: {max_obs_decimals}, IntStr: {stats["integer_strings"]}/{stats["count"]}, DecStr: {stats["decimal_strings"]}/{stats["count"]}\n'

    final_precisions_map_str += "}"
    print("\n--- Suggested ASSET_PRECISIONS map based on analysis: ---")
    print(final_precisions_map_str)
    print("\nNote: Review these inferred precisions and sample amounts carefully.")
    print("If 'Max Observed Decimals' is 0 and most strings are 'Integer Strings' (especially large ones), amounts might already be atomic.")
    print("In such a case, the preprocessing step should NOT multiply by 10^precision. The map would primarily be for decoding.")

if __name__ == "__main__":
    analyze_amounts() 