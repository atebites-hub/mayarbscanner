import time
import random
from datetime import datetime

# --- Configuration ---
MONITORING_INTERVAL_SECONDS = 10  # How often to simulate a check
SIMULATE_TX_PROBABILITY = 0.3   # Probability of generating a simulated tx each interval
ASSETS = ["BTC.BTC", "ETH.ETH", "MAYA.CACAO", "USDC.USDC-0XA0B86991C6218B36C1D19D4A2E9EB0CE3606EB48", "KUJI.KUJI"]

# --- Helper Function ---
def generate_simulated_pending_tx():
    """Generates a dictionary representing a simulated pending transaction."""
    tx_hash = f"0x{random.randbytes(32).hex()}"
    asset_in = random.choice(ASSETS)
    asset_out = random.choice([a for a in ASSETS if a != asset_in])
    amount_in = round(random.uniform(0.001, 5.0), 8)
    # Simulate a slightly different amount out for a pending tx
    amount_out_expected = round(amount_in * random.uniform(0.9, 1.1) * random.uniform(10, 50000), 8)

    return {
        "simulated_tx_hash": tx_hash,
        "timestamp_detected": datetime.now().isoformat(),
        "asset_in": asset_in,
        "amount_in": amount_in,
        "asset_out": asset_out,
        "amount_out_expected": amount_out_expected,
        "status": "pending"
    }

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Maya Protocol Mempool Monitor (Simulated) ---")
    print("This script simulates monitoring the mempool for pending transactions.")
    print("Real-time mempool access for Maya Protocol would require specific Maya node APIs or services,")
    print("which may differ from standard Ethereum tools like web3.py.")
    print("This simulation will periodically generate mock pending transactions.")
    print("Press Ctrl+C to stop monitoring.\n")

    try:
        while True:
            print(f"[{datetime.now().isoformat()}] Monitoring for pending Maya transactions...")
            
            if random.random() < SIMULATE_TX_PROBABILITY:
                pending_tx = generate_simulated_pending_tx()
                print(f"[{datetime.now().isoformat()}] DETECTED SIMULATED PENDING TRANSACTION:")
                for key, value in pending_tx.items():
                    print(f"  {key}: {value}")
                print("---")
            
            time.sleep(MONITORING_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    finally:
        print("Mempool monitor simulation finished.") 