import pandas as pd

# Define DF_COLUMNS globally for consistent DataFrame structures
DF_COLUMNS = [
    "date", "height", "status", "type", "in_tx_count", "out_tx_count", "pools", 
    "first_in_tx_id", "first_out_tx_id", "in_asset", "in_amount", "in_address",
    "out_asset", "out_amount", "out_address", "swap_liquidity_fee", 
    "swap_slip_bps", "swap_target_asset", "swap_network_fee_asset", 
    "swap_network_fee_amount", "transaction_id",
    "memo_str", "affiliate_id"
]

def parse_action(action: dict) -> dict:
    """Parses a single action from Midgard API into a flat dictionary consistent with DF_COLUMNS."""
    parsed = {
        "date": action.get("date"), # This is a nanosecond timestamp string
        "height": action.get("height"),
        "status": action.get("status"),
        "type": action.get("type"),
        "in_tx_count": len(action.get("in", [])),
        "out_tx_count": len(action.get("out", [])),
        "pools": ",".join(action.get("pools", [])),
        "first_in_tx_id": "",
        "first_out_tx_id": "",
        "in_asset": None,
        "in_amount": None,
        "in_address": None,
        "out_asset": None,
        "out_amount": None,
        "out_address": None,
        "swap_liquidity_fee": None,
        "swap_slip_bps": None,
        "swap_target_asset": None,
        "swap_network_fee_asset": None,
        "swap_network_fee_amount": None,
        "transaction_id": None,
        "memo_str": None,
        "affiliate_id": None
    }

    in_tx = action.get("in", [])
    if in_tx:
        parsed["first_in_tx_id"] = in_tx[0].get("txID", "")
        if in_tx[0].get("coins"):
            first_in_coin = in_tx[0]["coins"][0]
            parsed["in_asset"] = first_in_coin.get("asset")
            parsed["in_amount"] = first_in_coin.get("amount")
        parsed["in_address"] = in_tx[0].get("address")
        parsed["memo_str"] = in_tx[0].get("memo")

        if parsed["memo_str"] and parsed["type"] == "swap":
            memo_parts = parsed["memo_str"].split(':')
            if memo_parts[0].upper() in ["SWAP", "S", "="] :
                affiliate_candidate = None
                if len(memo_parts) > 4 and memo_parts[4]:
                    affiliate_candidate = memo_parts[4]
                elif len(memo_parts) > 3 and not memo_parts[3] and len(memo_parts) > 4 and memo_parts[4]:
                    pass
                elif len(memo_parts) > 3 and memo_parts[3] and (memo_parts[3].startswith("maya1") or memo_parts[3].startswith("thor1") or memo_parts[3].startswith("tthor1") or len(memo_parts[3]) <= 20) :
                    if len(memo_parts) == 4:
                        affiliate_candidate = memo_parts[3]
                    elif len(memo_parts) > 4 and memo_parts[4]: 
                        if memo_parts[4]:
                            affiliate_candidate = memo_parts[4]
                        elif memo_parts[3]:
                            affiliate_candidate = memo_parts[3]

                if affiliate_candidate:
                    parsed["affiliate_id"] = affiliate_candidate.split('/')[0]

    out_tx = action.get("out", [])
    if out_tx: # 'out' can be empty for some action types or states
        parsed["first_out_tx_id"] = out_tx[0].get("txID", "")
        if out_tx[0].get("coins"): # Check if coins exist in the first out transaction
            first_out_coin = out_tx[0]["coins"][0]
            parsed["out_asset"] = first_out_coin.get("asset")
            parsed["out_amount"] = first_out_coin.get("amount")
        parsed["out_address"] = out_tx[0].get("address")

    if action.get("type") == "swap":
        # For swaps, use first_in_tx_id as the primary transaction_id if available
        parsed["transaction_id"] = parsed["first_in_tx_id"]
        swap_meta = action.get("metadata", {}).get("swap")
        if swap_meta:
            parsed["swap_liquidity_fee"] = swap_meta.get("liquidityFee")
            parsed["swap_slip_bps"] = swap_meta.get("swapSlip") 
            parsed["swap_target_asset"] = swap_meta.get("swapTarget") 
            network_fees = swap_meta.get("networkFees", [])
            if network_fees: # networkFees is a list
                parsed["swap_network_fee_asset"] = network_fees[0].get("asset")
                parsed["swap_network_fee_amount"] = network_fees[0].get("amount")
    elif action.get("type") in ["addLiquidity", "withdraw"]:
        # For liquidity actions, first_in_tx_id is usually the relevant identifier
        parsed["transaction_id"] = parsed["first_in_tx_id"]
    # Other types might need specific logic for transaction_id if first_in_tx_id is not appropriate

    # Ensure all keys from DF_COLUMNS are present, defaulting to None or empty string
    for col in DF_COLUMNS:
        if col not in parsed:
            # This case should ideally be minimized by comprehensive parsing above
            # print(f"DEBUG: Column {col} not explicitly parsed, defaulting to None/empty.")
            if col in ["first_in_tx_id", "first_out_tx_id", "pools"]:
                 parsed[col] = "" # Default empty strings for these
            else:
                 parsed[col] = None # Default None for others
    
    # Ensure all values are serializable (e.g. convert numeric types if they come as strings and should be numbers)
    # For now, amounts are kept as strings from API, which is fine for pandas.
    # Date and height are also strings initially.

    return parsed 