#!/usr/bin/env python3

import sqlite3
import json # For storing complex attributes as JSON strings

# For address extraction during transaction insertion
from common_utils import extract_addresses_from_parsed_tx, is_mayanode_address
# For block-level address extraction
from common_utils import extract_addresses_from_parsed_block

# To test insertion, we might need to parse a sample block
# from ..src import common_utils # Can't do relative import like this in a script
# Instead, for testing, ensure common_utils.py is in PYTHONPATH or same dir,
# or use a more robust testing setup later.

DATABASE_FILE = "data/mayanode_chain.db" # Consider making this configurable

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row # Access columns by name
    return conn

def create_tables(conn):
    """Creates all necessary tables in the database if they don't already exist."""
    cursor = conn.cursor()

    # Blocks Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS blocks (
        block_height INTEGER PRIMARY KEY,
        block_hash TEXT UNIQUE NOT NULL,
        block_time_str TEXT,
        block_time_dt TEXT, 
        chain_id TEXT,
        proposer_address TEXT,
        data_hash TEXT,
        validators_hash TEXT,
        next_validators_hash TEXT,
        consensus_hash TEXT,
        app_hash TEXT,
        last_results_hash TEXT,
        evidence_hash TEXT,
        last_block_hash TEXT 
    )
    """)

    # Transactions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        tx_hash TEXT PRIMARY KEY,
        block_height INTEGER NOT NULL,
        tx_content_body_memo TEXT,
        tx_content_auth_info_json TEXT, -- Stores auth_info dict as JSON string
        tx_content_signatures_json TEXT, -- Stores signatures list as JSON string
        result_log TEXT,
        result_gas_wanted INTEGER,
        result_gas_used INTEGER,
        FOREIGN KEY (block_height) REFERENCES blocks(block_height)
    )
    """)

    # Transaction Messages Table
    # To store individual messages from a transaction's body
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transaction_messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tx_hash TEXT NOT NULL,
        message_index INTEGER NOT NULL, -- Order of message in the tx body
        message_type TEXT,              -- e.g., /types.MsgSend, /types.MsgDeposit
        message_body_json TEXT,         -- The actual message content as a JSON string
        UNIQUE (tx_hash, message_index),
        FOREIGN KEY (tx_hash) REFERENCES transactions(tx_hash)
    )
    """)

    # Block Events Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS block_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        block_height INTEGER NOT NULL,
        event_category TEXT NOT NULL, -- 'begin_block' or 'end_block'
        event_type TEXT,
        attributes_json TEXT,         -- Stores attributes dict as JSON string
        FOREIGN KEY (block_height) REFERENCES blocks(block_height)
    )
    """)

    # Transaction Events Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transaction_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tx_hash TEXT NOT NULL,
        event_type TEXT,
        attributes_json TEXT,         -- Stores attributes dict as JSON string
        FOREIGN KEY (tx_hash) REFERENCES transactions(tx_hash)
    )
    """)
    
    # Addresses Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS addresses (
        address TEXT PRIMARY KEY,
        first_seen_block_height INTEGER,
        last_seen_block_height INTEGER 
    )
    """)

    # Transaction-Address Link Table
    # Defines the role of an address in a transaction (e.g., sender, receiver, signer)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transaction_address_link (
        link_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tx_hash TEXT NOT NULL,
        address TEXT NOT NULL,
        role TEXT NOT NULL, -- e.g., 'signer', 'sender', 'receiver', 'from_event_attribute', etc.
        UNIQUE (tx_hash, address, role),
        FOREIGN KEY (tx_hash) REFERENCES transactions(tx_hash),
        FOREIGN KEY (address) REFERENCES addresses(address)
    )
    """)

    conn.commit()
    print("Database tables ensured.")

def check_if_block_exists(conn, block_height: int) -> bool:
    """Checks if a block with the given height already exists in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM blocks WHERE block_height = ?", (block_height,))
    return cursor.fetchone() is not None

def insert_block(conn, parsed_block_data: dict):
    """Inserts a parsed block into the database.
    Assumes parsed_block_data is the output of common_utils.parse_confirmed_block.
    """
    if not parsed_block_data or not parsed_block_data.get('block_height'):
        print("Error: Invalid or empty parsed_block_data for insertion.")
        return False

    cursor = conn.cursor()
    try:
        # Ensure block_time_dt is a string if it's a datetime object
        block_time_dt_str = parsed_block_data.get('block_time_dt')
        if hasattr(block_time_dt_str, 'isoformat'): # Check if it's a datetime object
            block_time_dt_str = block_time_dt_str.isoformat()

        cursor.execute("""
        INSERT OR IGNORE INTO blocks (
            block_height, block_hash, block_time_str, block_time_dt, 
            chain_id, proposer_address, data_hash, validators_hash, 
            next_validators_hash, consensus_hash, app_hash, 
            last_results_hash, evidence_hash, last_block_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            parsed_block_data.get('block_height'),
            parsed_block_data.get('block_hash'),
            parsed_block_data.get('block_time_str'),
            block_time_dt_str,
            parsed_block_data.get('chain_id'),
            parsed_block_data.get('proposer_address'),
            parsed_block_data.get('data_hash'),
            parsed_block_data.get('validators_hash'),
            parsed_block_data.get('next_validators_hash'),
            parsed_block_data.get('consensus_hash'),
            parsed_block_data.get('app_hash'),
            parsed_block_data.get('last_results_hash'),
            parsed_block_data.get('evidence_hash'),
            parsed_block_data.get('last_block_hash')
        ))
        
        # Extract and insert addresses found at the block level (e.g. in block events, proposer)
        # Note: proposer_address is already in the blocks table.
        # This ensures any other addresses from block events are in the addresses table.
        block_level_addresses = extract_addresses_from_parsed_block(parsed_block_data) # This also gets proposer and tx addrs
        # We only need to ensure they are in the address table, roles are more for tx links
        # insert_transaction will handle addresses specific to transactions including their roles.
        # Here we just make sure any address seen at block level (e.g. in block_events) is recorded.
        # extract_addresses_from_parsed_block includes addresses from transactions, 
        # so insert_address might be called multiple times for the same address if also in a tx,
        # but INSERT OR IGNORE and the update logic in insert_address handles this fine.
        for addr in block_level_addresses:
            insert_address(conn, addr, parsed_block_data.get('block_height'))

        # Insert block events
        if parsed_block_data.get('begin_block_events_parsed'):
            for bev in parsed_block_data['begin_block_events_parsed']:
                insert_block_event(conn, parsed_block_data['block_height'], 'begin_block', bev)
        if parsed_block_data.get('end_block_events_parsed'):
            for bev in parsed_block_data['end_block_events_parsed']:
                insert_block_event(conn, parsed_block_data['block_height'], 'end_block', bev)

        # Insert all transactions and their components
        if parsed_block_data.get('transactions'):
            for tx_data in parsed_block_data['transactions']:
                insert_transaction(conn, tx_data, parsed_block_data['block_height'])

        conn.commit() # Commit after the block and ALL its components are inserted
        print(f"Block {parsed_block_data['block_height']} and all its components processed and committed.")
        return True # Overall success if we reached here without a major error

    except sqlite3.Error as e:
        print(f"SQLite error during block insertion: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during block insertion: {e}")
        return False

def insert_transaction(conn, parsed_tx_data: dict, block_height: int):
    """Inserts a parsed transaction into the database.
    Assumes parsed_tx_data is an item from common_utils.parse_confirmed_block(...)'s 'transactions' list.
    """
    if not parsed_tx_data or not parsed_tx_data.get('hash'):
        print("Error: Invalid or empty parsed_tx_data for transaction insertion.")
        return False

    cursor = conn.cursor()
    try:
        tx_content = parsed_tx_data.get('tx_content_json', {})
        auth_info_json = json.dumps(tx_content.get('auth_info')) if tx_content.get('auth_info') else None
        signatures_json = json.dumps(tx_content.get('signatures')) if tx_content.get('signatures') else None
        body_memo = tx_content.get('body', {}).get('memo', '')

        # Insert the main transaction record first
        cursor.execute("""
        INSERT OR IGNORE INTO transactions (
            tx_hash, block_height, tx_content_body_memo, 
            tx_content_auth_info_json, tx_content_signatures_json, 
            result_log, result_gas_wanted, result_gas_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            parsed_tx_data.get('hash'),
            block_height,
            body_memo,
            auth_info_json,
            signatures_json,
            parsed_tx_data.get('result_log'),
            parsed_tx_data.get('result_gas_wanted'),
            parsed_tx_data.get('result_gas_used')
        ))
        
        tx_hash = parsed_tx_data.get('hash')

        # Now insert messages for this transaction
        messages = tx_content.get('body', {}).get('messages', [])
        for index, msg in enumerate(messages):
            insert_transaction_message(conn, tx_hash, index, msg)
        
        # Now insert transaction events for this transaction
        if parsed_tx_data.get('result_events_parsed'):
            for tev in parsed_tx_data['result_events_parsed']:
                insert_transaction_event(conn, tx_hash, tev)

        # Extract and insert addresses involved in this transaction
        # The extract_addresses_from_parsed_tx function gets a flat set of addresses.
        # We need to determine their roles.
        # For now, let's re-iterate parts of the transaction to assign roles.

        # Roles from messages
        if isinstance(tx_content, dict):
            messages = tx_content.get("body", {}).get("messages", [])
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        for key, value in msg.items():
                            if key in ["signer", "from_address", "to_address", "delegator_address", 
                                        "validator_address", "granter", "grantee"] and \
isinstance(value, str) and is_mayanode_address(value):
                                insert_address(conn, value, block_height)
                                link_transaction_to_address(conn, tx_hash, value, key) # Use field name as role
                            # Handle lists of addresses if a field is known to contain them (e.g. list of signers if any)
                            # This is a simplified approach; more specific message parsing might be needed for complex roles.
        
        # Roles from auth_info (fee payer/granter)
        if isinstance(tx_content, dict):
            auth_info = tx_content.get("auth_info", {})
            if isinstance(auth_info, dict):
                fee_info = auth_info.get("fee", {})
                if isinstance(fee_info, dict):
                    payer = fee_info.get("payer")
                    granter = fee_info.get("granter")
                    if isinstance(payer, str) and is_mayanode_address(payer):
                        insert_address(conn, payer, block_height)
                        link_transaction_to_address(conn, tx_hash, payer, "fee_payer")
                    if isinstance(granter, str) and is_mayanode_address(granter):
                        insert_address(conn, granter, block_height)
                        link_transaction_to_address(conn, tx_hash, granter, "fee_granter")

        # Roles from event attributes
        result_events = parsed_tx_data.get("result_events_parsed", [])
        if isinstance(result_events, list):
            for event in result_events:
                if isinstance(event, dict):
                    attributes = event.get("attributes", {})
                    event_type_str = event.get("type", "unknown_event")
                    if isinstance(attributes, dict):
                        for attr_key, attr_value in attributes.items():
                            if isinstance(attr_value, str) and is_mayanode_address(attr_value):
                                insert_address(conn, attr_value, block_height)
                                # Simple role: event_type.attribute_key
                                role = f"event_{event_type_str}_{attr_key}"
                                link_transaction_to_address(conn, tx_hash, attr_value, role)
                            elif isinstance(attr_value, str) and ',' in attr_value: # Comma-separated list
                                possible_addrs = attr_value.split(',')
                                for pa_raw in possible_addrs:
                                    pa = pa_raw.strip()
                                    if is_mayanode_address(pa):
                                        insert_address(conn, pa, block_height)
                                        role = f"event_{event_type_str}_{attr_key}_item"
                                        link_transaction_to_address(conn, tx_hash, pa, role)

        # DO NOT COMMIT HERE - commit is handled by insert_block after all transactions are done.
        return True
    except sqlite3.Error as e:
        print(f"SQLite error during transaction insertion for {parsed_tx_data.get('hash')}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during transaction insertion for {parsed_tx_data.get('hash')}: {e}")
        return False

def insert_transaction_message(conn, tx_hash: str, msg_index: int, msg_data: dict):
    """Inserts a single message from a transaction's body."""
    if not msg_data or not msg_data.get('@type'):
        print(f"Warning: Skipping empty or typeless message for tx {tx_hash} at index {msg_index}.")
        return False
    cursor = conn.cursor()
    try:
        message_type = msg_data.get('@type')
        message_body_json = json.dumps(msg_data) # Store the whole message dict as JSON

        cursor.execute("""
        INSERT OR IGNORE INTO transaction_messages (
            tx_hash, message_index, message_type, message_body_json
        ) VALUES (?, ?, ?, ?)
        """, (tx_hash, msg_index, message_type, message_body_json))
        # conn.commit() # Commit handled by caller
        # No need to print for every message, can get verbose
        return True
    except sqlite3.Error as e:
        print(f"SQLite error inserting message for tx {tx_hash}, index {msg_index}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error inserting message for tx {tx_hash}, index {msg_index}: {e}")
        return False

def insert_block_event(conn, block_height: int, event_category: str, parsed_event_data: dict):
    """Inserts a parsed block event (begin_block or end_block) into the database."""
    if not parsed_event_data or not parsed_event_data.get('type') or not event_category:
        print(f"Warning: Skipping block event due to missing data: category '{event_category}', event {parsed_event_data}")
        return False
    cursor = conn.cursor()
    try:
        event_type = parsed_event_data.get('type')
        attributes_json = json.dumps(parsed_event_data.get('attributes')) if parsed_event_data.get('attributes') else None

        cursor.execute("""
        INSERT INTO block_events (
            block_height, event_category, event_type, attributes_json
        ) VALUES (?, ?, ?, ?)
        """, (block_height, event_category, event_type, attributes_json))
        # conn.commit() # Commit handled by caller
        return True
    except sqlite3.Error as e:
        print(f"SQLite error inserting block event ({event_category} {event_type}) for block {block_height}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error inserting block event for block {block_height}: {e}")
        return False

def insert_transaction_event(conn, tx_hash: str, parsed_event_data: dict):
    """Inserts a parsed transaction result event into the database."""
    if not parsed_event_data or not parsed_event_data.get('type'):
        print(f"Warning: Skipping tx event due to missing data for tx {tx_hash}: {parsed_event_data}")
        return False
    cursor = conn.cursor()
    try:
        event_type = parsed_event_data.get('type')
        attributes_json = json.dumps(parsed_event_data.get('attributes')) if parsed_event_data.get('attributes') else None
        
        cursor.execute("""
        INSERT INTO transaction_events (
            tx_hash, event_type, attributes_json
        ) VALUES (?, ?, ?)
        """, (tx_hash, event_type, attributes_json))
        # conn.commit() # Commit handled by caller
        return True
    except sqlite3.Error as e:
        print(f"SQLite error inserting transaction event ({event_type}) for tx {tx_hash}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error inserting transaction event for tx {tx_hash}: {e}")
        return False

def insert_address(conn, address: str, block_height: int):
    """Inserts an address into the addresses table or updates its last_seen_block_height.
    The first_seen_block_height is only set if the address is new.
    """
    if not address:
        print("Warning: Attempted to insert an empty address.")
        return False
    cursor = conn.cursor()
    try:
        # Try to insert, if it's a new address, first_seen will be set.
        # If it exists, this part is ignored.
        cursor.execute("""
        INSERT OR IGNORE INTO addresses (address, first_seen_block_height, last_seen_block_height)
        VALUES (?, ?, ?)
        """, (address, block_height, block_height))
        
        # Always update last_seen_block_height for an existing address if current block is newer.
        # The WHERE clause handles the case where the address was just inserted (last_seen_block_height would be block_height)
        # or if it existed and current block_height is greater.
        cursor.execute("""
        UPDATE addresses 
        SET last_seen_block_height = ? 
        WHERE address = ? AND last_seen_block_height < ?
        """, (block_height, address, block_height))
        
        # No explicit commit here, will be handled by the calling function (e.g., insert_block)
        # or by the test script directly.
        return True
    except sqlite3.Error as e:
        print(f"SQLite error during address insertion for {address}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during address insertion for {address}: {e}")
        return False

def link_transaction_to_address(conn, tx_hash: str, address: str, role: str):
    """Links a transaction to an address with a specific role."""
    if not all([tx_hash, address, role]):
        print(f"Warning: Missing tx_hash, address, or role for linking. tx: {tx_hash}, addr: {address}, role: {role}")
        return False
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT OR IGNORE INTO transaction_address_link (tx_hash, address, role)
        VALUES (?, ?, ?)
        """, (tx_hash, address, role))
        # No explicit commit here.
        return True
    except sqlite3.Error as e:
        print(f"SQLite error linking tx {tx_hash} to address {address} with role {role}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error linking tx {tx_hash} to addr {address} (role {role}): {e}")
        return False

# --- Query Functions ---

def get_latest_block_height_from_db(conn) -> int | None:
    """Retrieves the highest block_height from the blocks table."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT MAX(block_height) FROM blocks")
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else None
    except sqlite3.Error as e:
        print(f"SQLite error getting latest block height: {e}")
        return None

def get_block_by_height(conn, block_height: int) -> dict | None:
    """Retrieves a block by its height, returning all columns as a dictionary."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM blocks WHERE block_height = ?", (block_height,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        print(f"SQLite error getting block by height {block_height}: {e}")
        return None

def get_transaction_by_hash(conn, tx_hash: str) -> dict | None:
    """Retrieves a transaction by its hash, returning all columns as a dictionary."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM transactions WHERE tx_hash = ?", (tx_hash,))
        row = cursor.fetchone()
        if not row:
            return None
        
        # Convert JSON strings back to dicts/lists for easier use
        tx_data = dict(row)
        if tx_data.get('tx_content_auth_info_json'):
            tx_data['tx_content_auth_info'] = json.loads(tx_data['tx_content_auth_info_json'])
        if tx_data.get('tx_content_signatures_json'):
            tx_data['tx_content_signatures'] = json.loads(tx_data['tx_content_signatures_json'])
        return tx_data
    except sqlite3.Error as e:
        print(f"SQLite error getting transaction by hash {tx_hash}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error for transaction {tx_hash}: {e}")
        return None # Or return partial data if preferred

def get_transactions_for_block(conn, block_height: int) -> list[dict]:
    """Retrieves all transactions for a given block_height."""
    cursor = conn.cursor()
    transactions = []
    try:
        cursor.execute("SELECT * FROM transactions WHERE block_height = ?", (block_height,))
        rows = cursor.fetchall()
        for row in rows:
            tx_data = dict(row)
            if tx_data.get('tx_content_auth_info_json'):
                try:
                    tx_data['tx_content_auth_info'] = json.loads(tx_data['tx_content_auth_info_json'])
                except json.JSONDecodeError:
                    tx_data['tx_content_auth_info'] = None # or some error indicator
            if tx_data.get('tx_content_signatures_json'):
                try:
                    tx_data['tx_content_signatures'] = json.loads(tx_data['tx_content_signatures_json'])
                except json.JSONDecodeError:
                     tx_data['tx_content_signatures'] = None # or some error indicator
            transactions.append(tx_data)
        return transactions
    except sqlite3.Error as e:
        print(f"SQLite error getting transactions for block {block_height}: {e}")
        return []

def get_address_details(conn, address: str) -> dict | None:
    """Retrieves details for a specific address from the addresses table."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM addresses WHERE address = ?", (address,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        print(f"SQLite error getting address details for {address}: {e}")
        return None

def get_all_block_heights_from_db(conn) -> set[int]:
    """Retrieves a set of all block_heights from the blocks table."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT block_height FROM blocks")
        rows = cursor.fetchall()
        return {row['block_height'] for row in rows}
    except sqlite3.Error as e:
        print(f"SQLite error getting all block heights: {e}")
        return set()

# --- Test Functions (Example Usage) ---

def _test_clear_specific_block_data(conn, block_height: int):
    # ... existing code ...
    pass

if __name__ == '__main__':
    # This import is tricky for direct script execution if common_utils is in parent src/
    # For testing, we can add the parent directory to sys.path temporarily
    import sys
    import os
    # Add the project root to sys.path to allow importing common_utils
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # This should be the 'mayarbscanner' directory
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Now we should be able to import common_utils correctly if structure is mayarbscanner/src/common_utils.py
    try:
        from src import common_utils
    except ImportError:
        print("Failed to import common_utils. Ensure PYTHONPATH is set or script is run from project root.")
        # Fallback for simpler structures or if common_utils.py is moved to the same directory for testing
        try:
            import common_utils
        except ImportError:
            print("Critical: common_utils.py not found. Aborting test.")
            sys.exit(1)

    print("--- Testing Database Utils (src/database_utils.py) ---")
    
    # Use a specific test database file to avoid interfering with a real one
    # For this test, we'll stick to the defined DATABASE_FILE but manage its state.
    if os.path.exists(DATABASE_FILE):
        print(f"Found existing database {DATABASE_FILE}. For a clean test, it might be deleted and recreated.")
        # For full idempotency, one might delete and recreate, but for now, we'll use OR IGNORE features.

    conn = get_db_connection()
    create_tables(conn)
    print("Database connection established and tables ensured.")

    # --- Test Data Setup ---
    # Load a sample block for testing (e.g., block_11255442.json)
    # Make sure this file exists in the specified path relative to the workspace root.
    sample_block_file_path = os.path.join(project_root, "data", "mayanode_blocks", "block_11255442.json")
    parsed_block_for_db = None
    
    # Define test_address and test_tx_hash early for use in cleanup message
    test_address = "maya1dhsycanx5dynhnx773qdx38t8kwr7lvkqzzzfq"
    test_tx_hash = "78B76695E4FE2FC0D15AEAD193D18FA104DA9FCF28BB784A7938C8FA61EABB96"

    if os.path.exists(sample_block_file_path):
        print(f"Loading sample block from: {sample_block_file_path}")
        with open(sample_block_file_path, 'r') as f:
            raw_block_json_from_file = json.load(f)
        
        # The file data/mayanode_blocks/block_11255442.json contains the RPC-wrapper.
        # common_utils.parse_confirmed_block expects the reconstructed data (like fetch_mayanode_block provides).
        # So, we need to reconstruct it here for the test.
        if "result" in raw_block_json_from_file and \
           isinstance(raw_block_json_from_file["result"], dict) and \
           "block" in raw_block_json_from_file["result"] and \
           isinstance(raw_block_json_from_file["result"]["block"], dict) and \
           "block_id" in raw_block_json_from_file["result"]:
            
            print("RPC-wrapped structure detected in file. Reconstructing for parser...")
            result_data = raw_block_json_from_file["result"]
            block_content = result_data["block"]
            block_id_info = result_data.get("block_id", {})

            input_to_parser = {
                "id": block_id_info,
                "header": block_content.get("header"),
                "data": block_content.get("data"),
                "evidence": block_content.get("evidence"),
                "last_commit": block_content.get("last_commit"),
                "begin_block_events": result_data.get("begin_block_events"),
                "end_block_events": result_data.get("end_block_events")
            }
            parsed_block_for_db = common_utils.parse_confirmed_block(input_to_parser)
        else:
            # This case assumes the file ALREADY contains the direct, reconstructed format.
            # For block_11255442.json, this branch will not be hit given its current content.
            print("File does not seem RPC-wrapped. Passing directly to parser...")
            parsed_block_for_db = common_utils.parse_confirmed_block(raw_block_json_from_file)

        if not parsed_block_for_db:
            print("Failed to parse the sample block. Check common_utils.parse_confirmed_block and file content.")
            conn.close()
            sys.exit(1)
        print(f"Sample block {parsed_block_for_db.get('block_height')} parsed successfully.")
        print(f"[DEBUG database_utils] Number of transactions in parsed_block_for_db: {len(parsed_block_for_db.get('transactions', []))}") # DEBUG
        print(f"[DEBUG database_utils] Transactions source: {parsed_block_for_db.get('transactions_source')}") # DEBUG
        if parsed_block_for_db.get('transactions_raw_base64'):
            print(f"[DEBUG database_utils] Found {len(parsed_block_for_db['transactions_raw_base64'])} raw base64 transactions.") # DEBUG

    else:
        print(f"Error: Sample block file not found at {sample_block_file_path}. Cannot run full DB insertion test.")
        # Create a minimal dummy parsed block for basic table checks if needed, or exit.
        # For now, we'll rely on the file existing for a full test.
        conn.close()
        sys.exit(1)

    target_block_height = int(parsed_block_for_db['block_height'])

    # --- Clean Slate for the Target Block & Test Address (for idempotent testing) ---
    print(f"\n--- Ensuring a clean slate for block {target_block_height} and test address {test_address} before insertion test ---")
    cursor = conn.cursor()
    # Must delete from child tables first due to foreign key constraints
    cursor.execute("DELETE FROM transaction_address_link WHERE tx_hash IN (SELECT tx_hash FROM transactions WHERE block_height = ?)", (target_block_height,))
    # Also specifically for the test_tx_hash if it spans blocks or for general cleanup
    cursor.execute("DELETE FROM transaction_address_link WHERE tx_hash = ?", (test_tx_hash,))
    
    cursor.execute("DELETE FROM transaction_events WHERE tx_hash IN (SELECT tx_hash FROM transactions WHERE block_height = ?)", (target_block_height,))
    cursor.execute("DELETE FROM transaction_messages WHERE tx_hash IN (SELECT tx_hash FROM transactions WHERE block_height = ?)", (target_block_height,))
    cursor.execute("DELETE FROM transactions WHERE block_height = ?", (target_block_height,))
    cursor.execute("DELETE FROM block_events WHERE block_height = ?", (target_block_height,))
    cursor.execute("DELETE FROM blocks WHERE block_height = ?", (target_block_height,))
    
    # Clean the specific test address to ensure its insertion is tested fresh
    cursor.execute("DELETE FROM addresses WHERE address = ?", (test_address,))

    conn.commit()
    print(f"Any existing data for block {target_block_height} and its children, and for address {test_address}, removed.")

    # --- Test Block Insertion ---
    print(f"\n--- Testing insertion of block {target_block_height} ---")
    insertion_success = insert_block(conn, parsed_block_for_db)
    if insertion_success:
        print(f"Block {target_block_height} insertion reported success.")
        if check_if_block_exists(conn, target_block_height):
            print(f"CONFIRMED: Block {target_block_height} exists in DB after insertion.")
        else:
            print(f"ERROR: Block {target_block_height} reported insertion success, but not found in DB.")
    else:
        print(f"Block {target_block_height} insertion reported FAILURE.")
        conn.close()
        sys.exit(1)

    # --- Verification Queries for Addresses and Links ---
    print(f"\n--- Verifying Address and Transaction-Address Link Table Data for Block {target_block_height} ---")
    
    # Example: Check a known address from block 11255442 (e.g., proposer or a tx participant)
    # Proposer: D9DF7A926FC92D4F028F4BDBFDAEF1C740E984FA (This is a validator consensus address, not maya1 format)
    # Let's find a maya1 address from a transaction in that block. Example tx hash: 78B76695E4FE2FC0D15AEAD193D18FA104DA9FCF28BB784A7938C8FA61EABB96
    # Signer/fee_payer in that tx: maya1dhsycanx5dynhnx773qdx38t8kwr7lvkqzzzfq
    # test_address = "maya1dhsycanx5dynhnx773qdx38t8kwr7lvkqzzzfq" # Moved up
    # test_tx_hash = "78B76695E4FE2FC0D15AEAD193D18FA104DA9FCF28BB784A7938C8FA61EABB96" # Moved up

    cursor.execute("SELECT * FROM addresses WHERE address = ?", (test_address,))
    address_row = cursor.fetchone()
    if address_row:
        print(f"Found address {test_address}: first_seen={address_row['first_seen_block_height']}, last_seen={address_row['last_seen_block_height']}")
        if address_row['first_seen_block_height'] == target_block_height and address_row['last_seen_block_height'] == target_block_height:
            print(f"Correct: first_seen and last_seen are {target_block_height} for {test_address}.")
        else:
            print(f"Warning: first_seen/last_seen mismatch for {test_address}. DB: {address_row['first_seen_block_height']}/{address_row['last_seen_block_height']}, Expected: {target_block_height}")
    else:
        print(f"ERROR: Test address {test_address} NOT found in addresses table.")

    cursor.execute("SELECT * FROM transaction_address_link WHERE tx_hash = ? AND address = ?", (test_tx_hash, test_address))
    links = cursor.fetchall()
    if links:
        print(f"Found {len(links)} link(s) for tx {test_tx_hash} and address {test_address}:")
        for link in links:
            print(f"  - Role: {link['role']}")
            # Expected roles could be 'signer' (from message) and/or 'fee_payer'
            if link['role'] in ['signer', 'fee_payer', 'event_tx_fee_payer']: # 'event_tx_fee_payer' if it came from an event
                print(f"    Role '{link['role']}' is plausible for this tx/address.")
            else:
                print(f"    Warning: Role '{link['role']}' might be unexpected for this context.")

    else:
        print(f"ERROR: No links found for tx {test_tx_hash} and address {test_address}.")

    # --- Test updating last_seen_block_height for an existing address ---
    # Simulate inserting the same address in a NEWER block
    newer_block_height = target_block_height + 100
    print(f"\n--- Testing update of last_seen for {test_address} with newer block {newer_block_height} ---")
    insert_address(conn, test_address, newer_block_height) # This only inserts address, no block context here
    conn.commit() # Commit this isolated address update test

    cursor.execute("SELECT * FROM addresses WHERE address = ?", (test_address,))
    address_row_updated = cursor.fetchone()
    if address_row_updated:
        print(f"Address {test_address} after update attempt: first_seen={address_row_updated['first_seen_block_height']}, last_seen={address_row_updated['last_seen_block_height']}")
        if address_row_updated['first_seen_block_height'] == target_block_height and address_row_updated['last_seen_block_height'] == newer_block_height:
            print(f"Correct: first_seen is still {target_block_height}, last_seen updated to {newer_block_height}.")
        else:
            print(f"ERROR: first_seen/last_seen incorrect after update. DB: {address_row_updated['first_seen_block_height']}/{address_row_updated['last_seen_block_height']}")
    else:
        print(f"CRITICAL ERROR: Test address {test_address} disappeared after update attempt!")

    conn.close()
    print("\nDatabase utils tests finished.") 