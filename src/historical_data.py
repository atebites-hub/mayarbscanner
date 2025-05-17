import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# --- Configuration ---
NUM_RECORDS = 1500  # Generate 1500 records to meet >1000 criteria
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "simulated_historical_maya_transactions.csv")

# Possible assets and pools (simplified)
ASSETS = ["BTC.BTC", "ETH.ETH", "MAYA.CACAO", "USDC.USDC-0XA0B86991C6218B36C1D19D4A2E9EB0CE3606EB48", "KUJI.KUJI"]
POOLS = [f"{a1}-{a2}" for i, a1 in enumerate(ASSETS) for a2 in ASSETS[i+1:]] # Generate some pair pools
ARB_ID_PREFIX = "ARB"

# --- Helper Functions ---
def generate_random_timestamp(start_date, end_date):
    """Generates a random timestamp between two dates."""
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)

def generate_mock_transaction(idx):
    """Generates a single mock Maya transaction record."""
    arb_id = f"{ARB_ID_PREFIX}_{idx:05d}_{random.randint(1000,9999)}"
    
    # Simulate 6 months of data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=180)
    timestamp = generate_random_timestamp(start_time, end_time)
    
    asset_in = random.choice(ASSETS)
    asset_out = random.choice([a for a in ASSETS if a != asset_in])
    
    # Ensure asset_in and asset_out are different for a valid pool
    # This simplified pool naming might not always match a real scenario but is for simulation
    pool_candidates = [p for p in POOLS if asset_in in p and asset_out in p]
    if not pool_candidates:
        # Fallback if a direct pair isn't in our simplified POOLS list (e.g. CACAO intermediary)
        source_pool = f"{asset_in}-MAYA.CACAO"
        target_pool = f"MAYA.CACAO-{asset_out}"
    else:
        chosen_pool = random.choice(pool_candidates)
        # This is a simplification; real routing is more complex
        source_pool = chosen_pool 
        target_pool = chosen_pool 

    amount_in = round(random.uniform(0.01, 10.0), 8) # e.g., 0.01 to 10 BTC/ETH
    # Simulate some exchange rate fluctuation for amount_out
    amount_out = round(amount_in * random.uniform(0.8, 1.2) * random.uniform(10, 50000), 8) # Simplified rate
    delta_x = round(amount_out - (amount_in * random.uniform(0.95, 1.05) * random.uniform(10, 50000)), 8) # Simplified profit/loss
    
    return {
        "arb_ID": arb_id,
        "timestamp": timestamp.isoformat(),
        "asset_in": asset_in,
        "amount_in": amount_in,
        "asset_out": asset_out,
        "amount_out": amount_out,
        "source_pool": source_pool,
        "target_pool": target_pool, # Simplified, could be same as source for direct or different for multi-hop
        "delta_X": delta_x # Profit/loss from this specific arb leg
    }

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting generation of {NUM_RECORDS} mock historical transactions...")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    mock_data = [generate_mock_transaction(i) for i in range(NUM_RECORDS)]
    
    df = pd.DataFrame(mock_data)
    
    # Ensure correct dtypes for columns that might be numeric but read as object
    df['amount_in'] = pd.to_numeric(df['amount_in'])
    df['amount_out'] = pd.to_numeric(df['amount_out'])
    df['delta_X'] = pd.to_numeric(df['delta_X'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully generated and saved {len(df)} records to {OUTPUT_FILE}")

    # Print some info and sample data
    print("\n--- Dataframe Info ---")
    df.info()
    print("\n--- Sample Data (first 5 rows) ---")
    print(df.head()) 