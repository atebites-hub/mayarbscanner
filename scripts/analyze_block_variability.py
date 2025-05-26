import sqlite3
import json
import argparse
import sys
import os
from collections import Counter, defaultdict

# Adjust path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database_utils import get_db_connection
from src.common_utils import parse_iso_datetime # For potential date filtering/logging

DB_FILE = 'mayanode_blocks.db' # Assuming this is the DB file used by fetch_realtime_transactions.py

def get_blocks_for_analysis(conn, limit=1000, offset=0):
    """Fetches raw block JSON for a range of blocks from the database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT block_height, raw_block_json 
        FROM blocks 
        ORDER BY block_height ASC
        LIMIT ? OFFSET ?
    """, (limit, offset))
    return cursor.fetchall()

def analyze_blocks(blocks_data):
    """Analyzes transaction counts and basic content variability."""
    if not blocks_data:
        print("No block data to analyze.")
        return

    tx_counts_per_block = []
    msg_types_overall = Counter()
    events_per_block_count = []
    event_types_overall = Counter()

    print(f"Analyzing {len(blocks_data)} blocks...")

    for i, (block_height, raw_block_json_str) in enumerate(blocks_data):
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(blocks_data)} blocks...")
        
        try:
            block_data = json.loads(raw_block_json_str)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for block {block_height}. Skipping.")
            continue

        # Transaction counts
        transactions = block_data.get("txs", []) # txs is a list of already decoded tx JSON objects
        tx_counts_per_block.append(len(transactions))

        block_event_count_current_block = 0

        # Block-level events (begin_block_events, end_block_events)
        for event_list_key in ["begin_block_events", "end_block_events"]:
            block_level_events = block_data.get(event_list_key, [])
            block_event_count_current_block += len(block_level_events)
            for event in block_level_events:
                event_types_overall[event.get("type")] += 1        

        # Transaction content variability (message types and transaction events)
        for tx_obj in transactions:
            # Message types
            # The 'tx' field within each tx_obj contains the actual cosmos tx structure
            tx_content = tx_obj.get("tx", {})
            messages = tx_content.get("body", {}).get("messages", [])
            for msg in messages:
                msg_types_overall[msg.get("@type")] += 1
            
            # Transaction-level events
            tx_result_events = tx_obj.get("result", {}).get("events", [])
            block_event_count_current_block += len(tx_result_events)
            for event in tx_result_events:
                event_types_overall[event.get("type")] += 1
        
        events_per_block_count.append(block_event_count_current_block)

    # --- Print Statistics ---
    print("\n--- Block Analysis Results ---")
    if not tx_counts_per_block:
        print("No transactions found in the analyzed blocks.")
        return

    print("\nTransaction Counts per Block:")
    print(f"  Min Txs/Block: {min(tx_counts_per_block)}")
    print(f"  Max Txs/Block: {max(tx_counts_per_block)}")
    print(f"  Avg Txs/Block: {sum(tx_counts_per_block) / len(tx_counts_per_block):.2f}")
    # Distribution of tx counts (e.g., how many blocks have 0 txs, 1 tx, etc.)
    tx_count_distribution = Counter(tx_counts_per_block)
    print("  Tx Count Distribution:")
    for count, num_blocks in sorted(tx_count_distribution.items()):
        print(f"    {count} Txs: {num_blocks} blocks")

    print("\nOverall Message Types (@type field from tx.body.messages):")
    if msg_types_overall:
        for msg_type, count in msg_types_overall.most_common():
            print(f"  {msg_type}: {count}")
    else:
        print("  No messages found in transactions.")

    print("\nEvent Counts per Block (Block-level + Tx-level events combined):")
    if events_per_block_count:
        print(f"  Min Events/Block: {min(events_per_block_count)}")
        print(f"  Max Events/Block: {max(events_per_block_count)}")
        print(f"  Avg Events/Block: {sum(events_per_block_count) / len(events_per_block_count):.2f}")
        event_count_distribution = Counter(events_per_block_count)
        print("  Event Count Distribution (Top 10 most frequent counts):")
        for count, num_blocks in sorted(event_count_distribution.most_common(10)):
             print(f"    {count} Events: {num_blocks} blocks")
    else:
        print("  No events found.")

    print("\nOverall Event Types (from block-level and tx-level events):")
    if event_types_overall:
        for event_type, count in event_types_overall.most_common(20): # Show top 20 event types
            print(f"  {event_type}: {count}")
    else:
        print("  No event types found.")

def main():
    parser = argparse.ArgumentParser(description="Analyze block variability from the Mayanode database.")
    parser.add_argument("--db-file", type=str, default=DB_FILE, help=f"Path to the SQLite database file (default: {DB_FILE})")
    parser.add_argument("--limit", type=int, default=1000, help="Number of blocks to analyze (default: 1000). Use 0 for all.")
    parser.add_argument("--offset", type=int, default=0, help="Offset for querying blocks (default: 0).")

    args = parser.parse_args()

    if not os.path.exists(args.db_file):
        print(f"Error: Database file not found at {args.db_file}")
        print("Please run the data ingestion script (e.g., src/fetch_realtime_transactions.py) first to populate the database.")
        sys.exit(1)

    conn = None
    try:
        conn = get_db_connection(db_file=args.db_file)
        print(f"Connected to database: {args.db_file}")
        
        limit_query = args.limit if args.limit > 0 else -1 # SQLite uses -1 for no limit
        blocks_data = get_blocks_for_analysis(conn, limit=limit_query, offset=args.offset)
        
        if not blocks_data:
            print(f"No blocks found in the database with limit={args.limit} and offset={args.offset}. Ensure data has been ingested.")
            return

        analyze_blocks(blocks_data)

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main() 