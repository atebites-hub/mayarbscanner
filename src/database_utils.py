#!/usr/bin/env python3

import sqlite3
import json # For storing complex attributes as JSON strings
import os # Added by previous edit, keep for DB_FILE path if needed
from datetime import datetime # For potential use in query functions or logging
import re # Added for regex matching
import base64 # Added for base64 encoding

# For address extraction during transaction insertion
from .common_utils import extract_addresses_from_parsed_tx, is_mayanode_address
# For block-level address extraction
from .common_utils import extract_addresses_from_parsed_block

# To test insertion, we might need to parse a sample block
# from ..src import common_utils # Can't do relative import like this in a script
# Instead, for testing, ensure common_utils.py is in PYTHONPATH or same dir,
# or use a more robust testing setup later.

# --- Global Configuration ---
# Determine project root dynamically to locate the database file
# Assuming database_utils.py is in mayarbscanner/src/
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_current_script_dir)  # This should be 'mayarbscanner'
DATABASE_FILE = os.path.join(PROJECT_ROOT, 'mayanode_blocks.db')

# --- Database Connection Utility ---
def get_db_connection(db_file_path=None):
    """Establishes a connection to the SQLite database.
    
    Args:
        db_file_path (str, optional): Path to the SQLite database file. 
                                      Defaults to DATABASE_FILE.
    
    Returns:
        sqlite3.Connection: Database connection object with row_factory set to sqlite3.Row.
    """
    path_to_use = db_file_path if db_file_path else DATABASE_FILE
    
    # Ensure the directory for the database exists if it's not in the root
    # This is more relevant if DATABASE_FILE is configured to be in a subfolder.
    db_dir = os.path.dirname(path_to_use)
    if db_dir and not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir)
            print(f"Created database directory: {db_dir}")
        except OSError as e:
            print(f"Error creating database directory {db_dir}: {e}")
            # Decide if to raise error or proceed assuming it might be created by connect
            
    try:
        conn = sqlite3.connect(path_to_use)
        conn.row_factory = sqlite3.Row  # Access columns by name
        conn.execute("PRAGMA foreign_keys = ON;") # Enforce foreign key constraints
        # print(f"Database connection established to: {path_to_use}") # Verbose
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database at {path_to_use}: {e}")
        raise # Re-raise the exception to make it clear connection failed

def create_tables(conn):
    """Creates all necessary tables in the database if they don't already exist.
    Schema is designed to be fully relational, avoiding JSON blobs for structured data.
    """
    cursor = conn.cursor()

    # Blocks Table: Core block header information
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS blocks (
        block_height INTEGER PRIMARY KEY,
        block_hash TEXT UNIQUE NOT NULL,
        block_time_str TEXT,              -- ISO format string from the source
        block_time_dt TEXT,               -- ISO format string (datetime.isoformat())
        chain_id TEXT,
        proposer_address TEXT,            -- Validator consensus address
        data_hash TEXT,                   -- Hash of block.Data
        validators_hash TEXT,
        next_validators_hash TEXT,
        consensus_hash TEXT,
        app_hash TEXT,                    -- State root after txs
        last_results_hash TEXT,           -- Hash of all results from previous block
        evidence_hash TEXT,
        last_block_hash TEXT,             -- Hash of the previous block
        block_data_source TEXT            -- e.g., 'mayanode_api', 'tendermint_rpc'
    )
    """)

    # Transactions Table: Core details of each transaction
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        tx_hash TEXT PRIMARY KEY,
        block_height INTEGER NOT NULL,
        tx_index_in_block INTEGER NOT NULL, -- The 0-based index of this transaction within its block
        success BOOLEAN,                    -- True if the transaction was successful (e.g., code == 0)
        code INTEGER,                       -- Result code, 0 for success
        log TEXT,                           -- Log message from the transaction execution
        gas_wanted INTEGER,
        gas_used INTEGER,
        memo TEXT,                          -- Memo from tx_body
        fee_amount TEXT,                    -- Store as string to handle multiple coins, e.g., "1000rune,50cacao" or just "100000000thor"
        fee_gas_limit INTEGER,
        fee_payer TEXT,                     -- Address of the fee payer
        fee_granter TEXT,                   -- Address of the fee granter, if any
        signer_pub_key_type TEXT,           -- e.g., /cosmos.crypto.secp256k1.PubKey (first signer)
        signer_pub_key_bytes BLOB,          -- Raw public key bytes (first signer)
        timeout_height INTEGER,
        UNIQUE (block_height, tx_index_in_block),
        FOREIGN KEY (block_height) REFERENCES blocks(block_height) ON DELETE CASCADE
    )
    """)

    # Transaction Messages Table: Individual messages within a transaction body
    # This table stores common information about a message.
    # Message-specific details will be in separate tables (e.g., msg_send_details)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transaction_messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tx_hash TEXT NOT NULL,
        message_index INTEGER NOT NULL,     -- Order of message in the transaction's body.messages list
        message_type TEXT NOT NULL,         -- e.g., /types.MsgSend, /cosmos.bank.v1beta1.MsgSend, /types.MsgSwap
                                            -- This type determines which specific detail table to join with.
        raw_message_content_json TEXT,      -- Store the original parsed message content as JSON for now,
                                            -- until all specific message types are fully decomposed into tables.
                                            -- This acts as a temporary measure to ensure no data loss during transition.
        UNIQUE (tx_hash, message_index),
        FOREIGN KEY (tx_hash) REFERENCES transactions(tx_hash) ON DELETE CASCADE
    )
    """)

    # Example: Details for MsgSend (cosmos.bank.v1beta1.MsgSend or types.MsgSend)
    # Similar tables would be created for MsgDeposit, MsgSwap, MsgOutboundTx, etc.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS msg_send_details (
        message_id INTEGER PRIMARY KEY,     -- Matches transaction_messages.message_id
        from_address TEXT,
        to_address TEXT,
        -- Amount is a list of coins. Store as JSON string for now, or normalize further if needed.
        -- For full relational, you'd have a msg_send_amounts table: (message_id, denom, amount_val)
        amount_json TEXT,                   -- e.g., '[{\"denom\": \"rune\", \"amount\": \"1000\"}]'
        FOREIGN KEY (message_id) REFERENCES transaction_messages(message_id) ON DELETE CASCADE
    )
    """)
    
    # Example: Details for MsgDeposit (types.MsgDeposit)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS msg_deposit_details (
        message_id INTEGER PRIMARY KEY,
        memo TEXT,
        signer TEXT, -- This is the 'signer' field directly from MsgDeposit, not tx signer
        -- Coins is a list. Store as JSON string or normalize to msg_deposit_coins table.
        coins_json TEXT,                    -- e.g., '[{\"asset\": \"MAYA.CACAO\", \"amount\": \"100000000\"}]'
        FOREIGN KEY (message_id) REFERENCES transaction_messages(message_id) ON DELETE CASCADE
    )
    """)

    # Details for MsgSwap (types.MsgSwap)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS msg_swap_details (
        message_id INTEGER PRIMARY KEY,     -- Matches transaction_messages.message_id
        from_address TEXT,                  -- Address performing the swap
        from_asset TEXT,                    -- Asset being swapped from (e.g., MAYA.CACAO)
        from_amount TEXT,                   -- Amount of from_asset (as string, represents full precision integer)
        to_asset TEXT,                      -- Asset being swapped to (e.g., ETH.USDT-0x...)
        to_amount_limit TEXT,               -- Minimum amount of to_asset expected (slippage protection)
        affiliate_address TEXT,             -- Optional: address for affiliate fee
        affiliate_basis_points TEXT,        -- Optional: basis points for affiliate fee (e.g., "10" for 0.1%)
        signer TEXT,                        -- Optional: signer of the MsgSwap itself, if different from tx signer
                                            -- Often this is the same as from_address or the tx signer.
        -- Note: A general 'memo' is usually part of the main transaction body.
        -- If MsgSwap has its own specific memo field distinct from the tx memo, it could be added here.
        FOREIGN KEY (message_id) REFERENCES transaction_messages(message_id) ON DELETE CASCADE
    )
    """)

    # Events Table: All events, whether from block results or transaction results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_db_id INTEGER PRIMARY KEY AUTOINCREMENT,
        block_height INTEGER NOT NULL,      -- Link to block
        tx_hash TEXT,                       -- Link to transaction (NULL if it's a block event)
        event_category TEXT NOT NULL,       -- 'begin_block', 'end_block', or 'transaction_result'
        event_index INTEGER NOT NULL,       -- Order of this event within its category/source
        event_type TEXT,                    -- Type of the event, e.g., 'transfer', 'message', 'rewards'
        UNIQUE (block_height, tx_hash, event_category, event_index), -- Ensures uniqueness
        FOREIGN KEY (block_height) REFERENCES blocks(block_height) ON DELETE CASCADE,
        FOREIGN KEY (tx_hash) REFERENCES transactions(tx_hash) ON DELETE CASCADE
    )
    """)

    # Event Attributes Table: Key-value pairs for each event
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS event_attributes (
        attribute_id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_db_id INTEGER NOT NULL,       -- Link to the parent event in the 'events' table
        attribute_key TEXT NOT NULL,
        attribute_value TEXT,
        attribute_index INTEGER NOT NULL,   -- Order of attribute within the event
        UNIQUE (event_db_id, attribute_key, attribute_index), -- Allowing same key if it appears multiple times (e.g. multiple 'recipient' in a complex event)
        FOREIGN KEY (event_db_id) REFERENCES events(event_db_id) ON DELETE CASCADE
    )
    """)

    # Addresses Table: Unique addresses encountered
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS addresses (
        address TEXT PRIMARY KEY,
        first_seen_block_height INTEGER,  -- Block height when this address was first recorded
        last_seen_block_height INTEGER    -- Last block height this address was observed in
    )
    """)

    # Transaction-Address Link Table: Links addresses to transactions with specific roles
    # This table helps understand an address's involvement in a transaction beyond generic event attributes.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transaction_address_roles (
        link_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tx_hash TEXT NOT NULL,
        address TEXT NOT NULL,
        role TEXT NOT NULL,               -- e.g., 'signer_tx', 'fee_payer_tx', 'msg_sender', 'msg_recipient', 
                                          -- 'event_transfer_sender', 'event_transfer_recipient', etc.
                                          -- Make roles specific to their origin for clarity.
        message_id INTEGER,               -- Optional: if role is message-specific (e.g. msg_sender)
        event_db_id INTEGER,              -- Optional: if role is event-specific (e.g. event_transfer_sender)
        -- UNIQUE (tx_hash, address, role) -- This might be too restrictive if an address plays same role via multiple messages/events
        FOREIGN KEY (tx_hash) REFERENCES transactions(tx_hash) ON DELETE CASCADE,
        FOREIGN KEY (address) REFERENCES addresses(address) ON DELETE CASCADE,
        FOREIGN KEY (message_id) REFERENCES transaction_messages(message_id) ON DELETE CASCADE,
        FOREIGN KEY (event_db_id) REFERENCES events(event_db_id) ON DELETE CASCADE
    )
    """)
    # Removed UNIQUE (tx_hash, address, role) to allow an address to appear multiple times with the same role if it's from different messages or events.
    # Consider if a more granular unique constraint is needed, e.g., (tx_hash, address, role, message_id, event_db_id) if that makes sense.

    conn.commit()
    print("Database tables ensured (created or verified existence with new relational schema).")

def check_if_block_exists(conn, block_height: int) -> bool:
    """Checks if a block with the given height already exists in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM blocks WHERE block_height = ? LIMIT 1", (block_height,))
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
            last_results_hash, evidence_hash, last_block_hash,
            block_data_source, raw_tx_list_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            parsed_block_data.get('last_block_hash'),
            parsed_block_data.get('transactions_source'), # New field
            json.dumps(parsed_block_data.get('transactions_raw_base64')) if parsed_block_data.get('transactions_raw_base64') else None # New field
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
        # Extract relevant parts from parsed_tx_data.tx_content_json
        tx_content = parsed_tx_data.get("tx_content_json", {})
        body_content = tx_content.get("body", {})
        auth_info_content = tx_content.get("auth_info", {})
        signatures_list = tx_content.get("signatures", [])

        # Prepare body_details (excluding memo and messages)
        body_details_to_store = { 
            k: v for k, v in body_content.items() 
            if k not in ["messages", "memo"]
        }

        # Insert the main transaction record first
        cursor.execute("""
        INSERT OR IGNORE INTO transactions (
            tx_hash, block_height, tx_index_in_block, success, code, log, gas_wanted, gas_used, memo, fee_amount, fee_gas_limit, fee_payer, fee_granter, signer_pub_key_type, signer_pub_key_bytes, timeout_height
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            parsed_tx_data.get("hash"),
            block_height,
            parsed_tx_data.get('tx_index_in_block'),
            parsed_tx_data.get('success'),
            parsed_tx_data.get('code'),
            parsed_tx_data.get('log'),
            parsed_tx_data.get('gas_wanted'),
            parsed_tx_data.get('gas_used'),
            parsed_tx_data.get('memo'),
            parsed_tx_data.get('fee_amount'),
            parsed_tx_data.get('fee_gas_limit'),
            parsed_tx_data.get('fee_payer'),
            parsed_tx_data.get('fee_granter'),
            parsed_tx_data.get('signer_pub_key_type'),
            parsed_tx_data.get('signer_pub_key_bytes'),
            parsed_tx_data.get('timeout_height')
        ))
        
        tx_hash = parsed_tx_data.get('hash')

        # Now insert messages for this transaction
        messages = body_content.get('messages', [])
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
            tx_hash, message_index, message_type, raw_message_content_json
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
        INSERT INTO events (
            block_height, tx_hash, event_category, event_index, event_type, attributes_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (block_height, None, event_category, 0, event_type, attributes_json))
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
        INSERT INTO events (
            block_height, tx_hash, event_category, event_index, event_type, attributes_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (None, tx_hash, 'transaction_result', 0, event_type, attributes_json))
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

def link_transaction_to_address(conn, tx_hash: str, address: str, role: str, message_id: int = None, event_db_id: int = None):
    """Links a transaction to an address with a specific role, optionally linking to a message or event."""
    if not all([tx_hash, address, role]):
        print(f"Warning: Missing tx_hash, address, or role for linking. tx: {tx_hash}, addr: {address}, role: {role}")
        return False
    cursor = conn.cursor()
    try:
        # Check if this specific link already exists to prevent redundant unique key generations if we add one later
        # For now, INSERT OR IGNORE handles redundancy at the DB level if a strict UNIQUE constraint is on the table.
        # Since we removed the broad UNIQUE constraint, this check is less critical for IGNORE but good for avoiding multiple inserts if this func is called repeatedly with same data before commit.
        
        # Simplified: just try to insert. If a more complex unique constraint is added to transaction_address_roles later,
        # this might need pre-checking.
        cursor.execute("""
        INSERT INTO transaction_address_roles (tx_hash, address, role, message_id, event_db_id)
        VALUES (?, ?, ?, ?, ?)
        """, (tx_hash, address, role, message_id, event_db_id))
        return True
    except sqlite3.IntegrityError: # Catches violation if a UNIQUE constraint was defined and hit
        # This means the exact link already exists, which is fine.
        return True 
    except sqlite3.Error as e:
        print(f"SQLite error linking tx {tx_hash} to address {address} with role {role} (msg_id {message_id}, evt_id {event_db_id}): {e}")
        return False
    except Exception as e_gen:
        print(f"Unexpected error linking tx {tx_hash} to addr {address} (role {role}): {e_gen}")
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
        if tx_data.get('tx_object_json'):
            tx_data['tx_content_json'] = json.loads(tx_data['tx_object_json'])
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
            if tx_data.get('tx_object_json'):
                try:
                    tx_data['tx_content_json'] = json.loads(tx_data['tx_object_json'])
                except json.JSONDecodeError:
                    tx_data['tx_content_json'] = None # or some error indicator
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

# --- Helper for Event Reconstruction ---
def _get_formatted_events_internal(conn, block_height: int = None, tx_hash: str = None, event_category: str = None) -> list[dict]:
    """Internal helper to fetch and format events and their attributes.
    Can fetch by block_height + category (for block events) OR by tx_hash + category (for tx events).
    """
    if not ((block_height and event_category) or (tx_hash and event_category)):
        raise ValueError("Must provide (block_height and event_category) or (tx_hash and event_category)")

    cursor = conn.cursor()
    formatted_events = []
    
    base_query = "SELECT event_db_id, event_type FROM events WHERE "
    params = []

    if block_height is not None and event_category:
        base_query += "block_height = ? AND event_category = ? "
        params.extend([block_height, event_category])
    elif tx_hash is not None and event_category: # Implicitly block_height will be NULL for these if correctly inserted
        base_query += "tx_hash = ? AND event_category = ? "
        params.extend([tx_hash, event_category])
    
    base_query += "ORDER BY event_index" # Crucial for maintaining order

    try:
        cursor.execute(base_query, tuple(params))
        event_rows = cursor.fetchall()

        for event_row in event_rows:
            event_db_id = event_row['event_db_id']
            event_type = event_row['event_type']
            
            attributes_cursor = conn.cursor()
            attributes_cursor.execute(
                "SELECT attribute_key, attribute_value FROM event_attributes WHERE event_db_id = ? ORDER BY attribute_index",
                (event_db_id,)
            )
            attrs_dict = {attr_row['attribute_key']: attr_row['attribute_value'] for attr_row in attributes_cursor.fetchall()}
            
            formatted_events.append({
                "type": event_type,
                "attributes": attrs_dict
            })
        return formatted_events
    except sqlite3.Error as e:
        err_context = f"block {block_height}" if block_height else f"tx {tx_hash}"
        print(f"SQLite error getting formatted events for {err_context}, category {event_category}: {e}")
        return []

# --- Helper for Transaction Message Reconstruction ---
def _get_formatted_messages_for_tx(conn, tx_hash: str) -> list[dict]:
    """Fetches and formats messages for a given transaction hash.
    Currently, it loads the raw_message_content_json as a shortcut.
    Long-term, this should reconstruct messages from their specific relational tables.
    """
    messages = []
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT raw_message_content_json, message_type, message_index FROM transaction_messages WHERE tx_hash = ? ORDER BY message_index", (tx_hash,))
        rows = cursor.fetchall()
        for row in rows:
            if row['raw_message_content_json']:
                try:
                    # Load the stored JSON content of the message
                    msg_content = json.loads(row['raw_message_content_json'])
                    # Ensure @type is present, using the stored message_type if not in the JSON itself
                    # (though common_utils.decode_cosmos_tx_string_to_dict should add @type)
                    if '@type' not in msg_content and row['message_type']:
                        msg_content['@type'] = row['message_type']
                    messages.append(msg_content)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for a message in tx {tx_hash} (index {row['message_index']}): {e}. Skipping message.")
        return messages
    except sqlite3.Error as e:
        print(f"SQLite error getting messages for tx {tx_hash}: {e}")
        return []

# --- Helper for Full Transaction Reconstruction ---
def _get_formatted_transactions_for_block(conn, block_height: int) -> list[dict]:
    """Fetches and reconstructs all transactions for a given block_height in Mayanode API format."""
    formatted_transactions = []
    cursor = conn.cursor()
    try:
        # Fetch core transaction data
        cursor.execute("""
            SELECT 
                tx_hash, tx_index_in_block, success, code, log, gas_wanted, gas_used, memo, 
                fee_amount, fee_gas_limit, fee_payer, fee_granter, 
                signer_pub_key_type, signer_pub_key_bytes, timeout_height
            FROM transactions 
            WHERE block_height = ? 
            ORDER BY tx_index_in_block
        """, (block_height,))
        tx_rows = cursor.fetchall()

        for tx_row_data in tx_rows:
            tx_hash = tx_row_data['tx_hash']
            
            # 1. Reconstruct 'tx' (tx_content_json) part
            tx_content_json = {
                "body": {
                    "messages": _get_formatted_messages_for_tx(conn, tx_hash),
                    "memo": tx_row_data['memo'],
                    "timeout_height": str(tx_row_data['timeout_height']) if tx_row_data['timeout_height'] else "0" # API expects string
                    # non_critical_tx_info, extension_options, non_critical_extension_options are usually empty/omitted
                },
                "auth_info": {
                    "signer_infos": [], # Placeholder - see note below
                    "fee": {
                        "amount": [], # Placeholder - see note below
                        "gas_limit": str(tx_row_data['fee_gas_limit']) if tx_row_data['fee_gas_limit'] is not None else "0",
                        "payer": tx_row_data['fee_payer'],
                        "granter": tx_row_data['fee_granter']
                    }
                },
                "signatures": [] # Placeholder - Signatures are raw bytes, not typically reconstructed in this detail
            }

            # Reconstruct fee.amount (e.g., "1000rune,50cacao" -> [{'denom': 'rune', 'amount': '1000'}])
            if tx_row_data['fee_amount']:
                fee_amount_parts = tx_row_data['fee_amount'].split(',')
                for part in fee_amount_parts:
                    # Find the split point between amount and denom
                    # Assumes denom is alphabetic and at the end
                    match = re.match(r"(\\d+)([a-zA-Z]+.*)", part)
                    if match:
                        amount_val, denom_val = match.groups()
                        tx_content_json["auth_info"]["fee"]["amount"].append({"denom": denom_val, "amount": amount_val})
                    else: # Fallback if format is unexpected, store as is if simple
                        if not any(c.isdigit() for c in part) or not any(c.isalpha() for c in part):
                             # If it's purely numeric or purely alphabetic, it might be a single coin without explicit split
                             # This case is less common for multi-coin fees but can happen for single coin like "100000rune"
                             # For simplicity, we'll skip complex parsing here; common_utils should store it more structured if needed.
                             pass # Or add a default unknown coin if critical
            
            # Reconstruct auth_info.signer_infos (Simplified)
            # We stored only the first signer's pubkey type and bytes.
            # Full reconstruction would require storing all signer_infos relationally.
            if tx_row_data['signer_pub_key_type'] and tx_row_data['signer_pub_key_bytes']:
                signer_info = {
                    "public_key": {
                        "@type": tx_row_data['signer_pub_key_type'],
                        "key": base64.b64encode(tx_row_data['signer_pub_key_bytes']).decode('utf-8')
                        # sequence and mode_info are harder to get from current DB schema
                    },
                    "mode_info": {"single": {"mode": "SIGN_MODE_DIRECT"}}, # Common default
                    "sequence": "0" # Placeholder, actual sequence not stored this way
                }
                tx_content_json["auth_info"]["signer_infos"].append(signer_info)

            # 2. Reconstruct 'result' part
            tx_events = _get_formatted_events_internal(conn, tx_hash=tx_hash, event_category='transaction_result')
            tx_result = {
                "log": tx_row_data['log'],
                "gas_wanted": str(tx_row_data['gas_wanted']) if tx_row_data['gas_wanted'] is not None else "0",
                "gas_used": str(tx_row_data['gas_used']) if tx_row_data['gas_used'] is not None else "0",
                "events": tx_events
                # Mayanode API also includes 'code' if non-zero. Our 'transactions.code' can be used.
            }
            if tx_row_data['code'] is not None and tx_row_data['code'] != 0:
                tx_result['code'] = tx_row_data['code']

            # 3. Assemble final transaction object for the block
            final_tx_object = {
                "hash": tx_hash,
                "height": str(block_height), # Mayanode API includes height in each tx object
                "tx": tx_content_json,
                "result": tx_result
            }
            formatted_transactions.append(final_tx_object)
            
        return formatted_transactions
    except sqlite3.Error as e:
        print(f"SQLite error getting formatted transactions for block {block_height}: {e}")
        return []
    except Exception as ex:
        print(f"Unexpected error getting formatted transactions for block {block_height}: {ex}") # General exception
        return []

# --- Test Functions (Example Usage) ---

def _test_clear_specific_block_data(conn, block_height: int):
    # ... existing code ...
    pass

# --- Dividend Related Queries ---

# Known Mayanode Reserve/Rewards module address(es) that distribute CACAO
# TODO: Make this configurable or fetch dynamically if possible
RESERVE_MODULE_ADDRESSES = ["maya1dheycdevq39qlkxs2a6wuuzyn4aqxhve4hc8sm"]

def get_dividend_receiving_addresses(conn) -> list[str]:
    """
    Queries the database for unique wallet addresses that have received CACAO dividends.
    Dividends are identified as 'transfer' events in 'end_block_events' originating
    from known Reserve/Rewards module addresses.
    """
    cursor = conn.cursor()
    addresses = set()

    # The attributes_json stores a dictionary of attributes for an event.
    # We need to parse this JSON and look for sender, recipient, and amount.
    # SQLite's JSON1 extension (json_extract) is very helpful here.
    # We are looking for events where:
    # - event_category = 'end_block'
    # - event_type = 'transfer'
    # - attributes_json contains a 'sender' attribute matching one of RESERVE_MODULE_ADDRESSES
    # - attributes_json contains an 'amount' attribute ending with 'cacao'
    # - We want to extract the 'recipient' attribute.

    query = """
    SELECT attributes_json
    FROM events
    WHERE event_category = 'end_block' AND event_type = 'transfer';
    """
    
    try:
        cursor.execute(query)
        for row in cursor.fetchall():
            attributes_str = row['attributes_json']
            if not attributes_str:
                continue
            
            try:
                attributes = json.loads(attributes_str)
                sender = attributes.get('sender')
                recipient = attributes.get('recipient')
                amount_str = attributes.get('amount', '')

                if sender in RESERVE_MODULE_ADDRESSES and \
                   recipient and \
                   is_mayanode_address(recipient) and \
                   amount_str.endswith('cacao') and \
                   not amount_str.startswith('0cacao'): # Ensure non-zero amount

                    addresses.add(recipient)
            except json.JSONDecodeError:
                # print(f"Warning: Could not decode attributes_json: {attributes_str}")
                continue # Skip malformed JSON
            except Exception as e:
                # print(f"Warning: Error processing event attributes: {e}")
                continue
                
    except sqlite3.Error as e:
        print(f"SQLite error in get_dividend_receiving_addresses: {e}")
        return [] # Return empty list on error

    return sorted(list(addresses))

def get_dividends_for_address(conn, wallet_address: str) -> list[dict]:
    """
    Queries the database for all CACAO dividend transactions for a specific wallet address.
    Returns a list of dictionaries, each containing block_height, block_time_dt, and amount.
    """
    if not is_mayanode_address(wallet_address):
        print(f"Error: Invalid Mayanode address format for query: {wallet_address}")
        return []

    cursor = conn.cursor()
    dividends = []

    # We need to join events with blocks table to get block_time_dt
    # The filtering logic for sender, recipient, and amount remains similar.
    query = """
    SELECT e.attributes_json, b.block_height, b.block_time_dt
    FROM events e
    JOIN blocks b ON e.block_height = b.block_height
    WHERE e.event_category = 'end_block' AND e.event_type = 'transfer';
    """

    try:
        cursor.execute(query)
        for row in cursor.fetchall():
            attributes_str = row['attributes_json']
            block_h = row['block_height']
            block_time = row['block_time_dt'] # This is already a string from DB
            
            if not attributes_str:
                continue
            
            try:
                attributes = json.loads(attributes_str)
                sender = attributes.get('sender')
                recipient = attributes.get('recipient')
                amount_str = attributes.get('amount', '')

                if sender in RESERVE_MODULE_ADDRESSES and \
                   recipient == wallet_address and \
                   amount_str.endswith('cacao') and \
                   not amount_str.startswith('0cacao'):
                    
                    # Extract numeric part of amount and convert to integer (or float if necessary)
                    # Amount is like "12345cacao", so remove "cacao" suffix.
                    try:
                        cacao_amount_raw = int(amount_str[:-len('cacao')]) # Assumes 8 decimal places are implicit
                        # For display, you might want to format this, e.g., divide by 10**8
                    except ValueError:
                        # print(f"Warning: Could not parse amount '{amount_str}' to int for block {block_h}")
                        continue # Skip if amount is not a valid number

                    dividends.append({
                        "block_height": block_h,
                        "block_time_dt": block_time, # Already a string in ISO format
                        "amount_raw_cacao": cacao_amount_raw, # Store raw integer amount
                        "amount_str_cacao": amount_str # Store original string for reference
                    })
            except json.JSONDecodeError:
                # print(f"Warning: Could not decode attributes_json: {attributes_str} for block {block_h}")
                continue
            except Exception as e:
                # print(f"Warning: Error processing event attributes for block {block_h}: {e}")
                continue

    except sqlite3.Error as e:
        print(f"SQLite error in get_dividends_for_address: {e}")
        return []
    
    # Sort by block height descending (most recent first)
    return sorted(dividends, key=lambda d: d['block_height'], reverse=True)

# --- Insertion Helper Functions (internal, generally not called directly from outside) ---

def _insert_transaction_message_internal(cursor, tx_hash: str, msg_index: int, msg_data: dict, block_height: int, conn):
    """Internal helper to insert a single message and its specific details. Assumes cursor is provided."""
    if not msg_data or not isinstance(msg_data, dict):
        return False
    
    message_type = msg_data.get('@type')
    raw_message_content_json_str = json.dumps(msg_data) # For temporary storage

    try:
        cursor.execute("""
        INSERT OR IGNORE INTO transaction_messages (
            tx_hash, message_index, message_type, raw_message_content_json
        ) VALUES (?, ?, ?, ?)
        """, (tx_hash, msg_index, message_type, raw_message_content_json_str))
        
        message_id = cursor.lastrowid
        if not message_id: # Message might have already existed
            cursor.execute("SELECT message_id FROM transaction_messages WHERE tx_hash = ? AND message_index = ?", 
                           (tx_hash, msg_index))
            row = cursor.fetchone()
            if row: message_id = row['message_id']
            else: 
                print(f"Warning: Could not get message_id for tx {tx_hash}, msg_idx {msg_index}")
                return False # Cannot insert details without message_id

        # Call specific handlers based on message_type
        if message_type == '/cosmos.bank.v1beta1.MsgSend' or message_type == '/types.MsgSend':
            _insert_msg_send_details_internal(cursor, message_id, msg_data, tx_hash, block_height, conn)
        elif message_type == '/types.MsgDeposit':
            _insert_msg_deposit_details_internal(cursor, message_id, msg_data, tx_hash, block_height, conn)
        elif message_type == '/types.MsgSwap':
            _insert_msg_swap_details_internal(cursor, message_id, msg_data, tx_hash, block_height, conn)
        # Add other elif conditions for other message types here
        # e.g., /types.MsgOutboundTx, etc.

        return True
    except sqlite3.Error as e:
        print(f"SQLite error inserting message (tx {tx_hash}, index {msg_index}, type {message_type}): {e}")
        return False

# --- Specific Message Detail Inserters ---
def _insert_msg_send_details_internal(cursor, message_id: int, msg_data: dict, tx_hash: str, block_height: int, conn):
    from_address = msg_data.get('from_address')
    to_address = msg_data.get('to_address')
    amount_list = msg_data.get('amount', []) # list of coin dicts
    amount_json_str = json.dumps(amount_list)

    try:
        cursor.execute("""
        INSERT OR IGNORE INTO msg_send_details (message_id, from_address, to_address, amount_json)
        VALUES (?, ?, ?, ?)
        """, (message_id, from_address, to_address, amount_json_str))
        # Link addresses
        if from_address and is_mayanode_address(from_address):
            insert_address(conn, from_address, block_height)
            link_transaction_to_address(conn, tx_hash, from_address, 'msg_send_from', message_id=message_id)
        if to_address and is_mayanode_address(to_address):
            insert_address(conn, to_address, block_height)
            link_transaction_to_address(conn, tx_hash, to_address, 'msg_send_to', message_id=message_id)
    except sqlite3.Error as e:
        print(f"SQLite error inserting MsgSend details for message_id {message_id}: {e}")

def _insert_msg_deposit_details_internal(cursor, message_id: int, msg_data: dict, tx_hash: str, block_height: int, conn):
    memo = msg_data.get('memo')
    signer = msg_data.get('signer') # This is the msg-level signer
    coins_list = msg_data.get('coins', []) # list of asset-amount dicts
    coins_json_str = json.dumps(coins_list)
    try:
        cursor.execute("""
        INSERT OR IGNORE INTO msg_deposit_details (message_id, memo, signer, coins_json)
        VALUES (?, ?, ?, ?)
        """, (message_id, memo, signer, coins_json_str))
        if signer and is_mayanode_address(signer):
            insert_address(conn, signer, block_height)
            link_transaction_to_address(conn, tx_hash, signer, 'msg_deposit_signer', message_id=message_id)
        # Potentially extract asset addresses from coins if they are ever actual addresses
    except sqlite3.Error as e:
        print(f"SQLite error inserting MsgDeposit details for message_id {message_id}: {e}")

def _insert_msg_swap_details_internal(cursor, message_id: int, msg_data: dict, tx_hash: str, block_height: int, conn):
    from_address = msg_data.get('from_address')
    from_asset = msg_data.get('from_asset')
    from_amount = msg_data.get('from_amount')
    to_asset = msg_data.get('to_asset')
    to_amount_limit = msg_data.get('to_amount_limit')
    affiliate_address = msg_data.get('affiliate_address')
    affiliate_basis_points = msg_data.get('affiliate_basis_points')
    signer = msg_data.get('signer') # MsgSwap specific signer

    try:
        cursor.execute("""
        INSERT OR IGNORE INTO msg_swap_details (
            message_id, from_address, from_asset, from_amount, to_asset, 
            to_amount_limit, affiliate_address, affiliate_basis_points, signer
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (message_id, from_address, from_asset, from_amount, to_asset, 
                  to_amount_limit, affiliate_address, affiliate_basis_points, signer))
        
        # Link addresses involved in the swap
        if from_address and is_mayanode_address(from_address):
            insert_address(conn, from_address, block_height)
            link_transaction_to_address(conn, tx_hash, from_address, 'msg_swap_from', message_id=message_id)
        
        if affiliate_address and is_mayanode_address(affiliate_address):
            insert_address(conn, affiliate_address, block_height)
            link_transaction_to_address(conn, tx_hash, affiliate_address, 'msg_swap_affiliate', message_id=message_id)

        if signer and is_mayanode_address(signer):
            insert_address(conn, signer, block_height)
            # If signer is same as from_address, this link might be redundant but harmless if roles are distinct
            link_transaction_to_address(conn, tx_hash, signer, 'msg_swap_signer', message_id=message_id)
            
    except sqlite3.Error as e:
        print(f"SQLite error inserting MsgSwap details for message_id {message_id}: {e}")

# --- Block/Transaction Event Insertion Helpers (Internal) ---

def _insert_block_event_internal(cursor, block_height: int, event_category: str, event_index: int, event_data: dict):
    """Internal helper to insert a single block event and its attributes. Assumes cursor is provided."""
    if not event_data or not isinstance(event_data, dict) or not event_data.get('type'):
        # print(f"DEBUG [_insert_block_event_internal]: Skipping event for block {block_height}, category {event_category}, index {event_index}. Data: {str(event_data)[:200]}")
        return False
    
    event_type = event_data.get('type')
    
    try:
        cursor.execute("""
        INSERT OR IGNORE INTO events (
            block_height, tx_hash, event_category, event_index, event_type
        ) VALUES (?, ?, ?, ?, ?)
        """, (block_height, None, event_category, event_index, event_type))
        
        event_db_id = cursor.lastrowid
        if not event_db_id: # Event might have already existed if UNIQUE constraint hit
            # Try to fetch existing event_db_id
            cursor.execute("""SELECT event_db_id FROM events 
                            WHERE block_height = ? AND tx_hash IS NULL 
                            AND event_category = ? AND event_index = ?""", 
                           (block_height, event_category, event_index))
            row = cursor.fetchone()
            if row: event_db_id = row['event_db_id']
            else: 
                print(f"Warning: Could not get event_db_id for block event: Blk {block_height}, Cat {event_category}, Idx {event_index}")
                return False # Cannot insert attributes without event_db_id

        # event_data['attributes'] is expected to be a dictionary from _parse_block_events_common
        attributes_dict = event_data.get('attributes', {}) 
        if isinstance(attributes_dict, dict):
             for attr_idx, (attr_key, attr_value) in enumerate(attributes_dict.items()):
                # Ensure key/value are strings
                if isinstance(attr_key, bytes): attr_key = attr_key.decode('utf-8', 'replace')
                # Value from _parse_block_events_common should already be decoded string or None
                attr_value_str = str(attr_value) if attr_value is not None else None

                cursor.execute("""
                INSERT OR IGNORE INTO event_attributes (
                    event_db_id, attribute_key, attribute_value, attribute_index
                ) VALUES (?, ?, ?, ?)
                """, (event_db_id, attr_key, attr_value_str, attr_idx))
        return True

    except sqlite3.Error as e:
        print(f"SQLite error inserting block event (block {block_height}, type {event_type}, category {event_category}, index {event_index}): {e}")
        return False
    except Exception as ex: # General exception for safety
        print(f"Unexpected error inserting block event (block {block_height}, type {event_type}, category {event_category}, index {event_index}): {ex}")
        return False


def _insert_transaction_event_internal(cursor, tx_hash: str, block_height: int, event_index: int, event_data: dict, conn):
    """Internal helper to insert a single transaction event and its attributes. Assumes cursor is provided."""
    if not event_data or not isinstance(event_data, dict) or not event_data.get('type'):
        return False

    event_type = event_data.get('type')
    
    try:
        cursor.execute("""
        INSERT OR IGNORE INTO events (
            block_height, tx_hash, event_category, event_index, event_type
        ) VALUES (?, ?, ?, ?, ?)
        """, (block_height, tx_hash, 'transaction_result', event_index, event_type))
        
        event_db_id = cursor.lastrowid
        if not event_db_id: 
            cursor.execute("""
            SELECT event_db_id FROM events 
            WHERE tx_hash = ? AND event_category = 'transaction_result' AND event_index = ?
            """, (tx_hash, event_index))
            row = cursor.fetchone()
            if row: event_db_id = row['event_db_id']
            else: 
                print(f"Warning: Could not get event_db_id for tx event: Tx{tx_hash}, Idx:{event_index}")
                return False

        attributes = event_data.get('attributes', []) # common_utils parses to list of dicts
        if isinstance(attributes, list):
            for attr_idx, attr_pair in enumerate(attributes):
                if isinstance(attr_pair, dict) and 'key' in attr_pair:
                    attr_key = attr_pair.get('key')
                    attr_value = attr_pair.get('value')
                    # Ensure key/value are strings
                    if isinstance(attr_key, bytes): attr_key = attr_key.decode('utf-8', 'replace')
                    if isinstance(attr_value, bytes): attr_value = attr_value.decode('utf-8', 'replace')
                    else: attr_value = str(attr_value)

                    cursor.execute("""
                    INSERT OR IGNORE INTO event_attributes (
                        event_db_id, attribute_key, attribute_value, attribute_index
                    ) VALUES (?, ?, ?, ?)
                    """, (event_db_id, attr_key, attr_value, attr_idx))
                    
                    # Link address if attribute value is an address
                    if is_mayanode_address(attr_value):
                        insert_address(conn, attr_value, block_height)
                        role = f"event_{event_type}_{attr_key}"
                        link_transaction_to_address(conn, tx_hash, attr_value, role, event_db_id=event_db_id)

        elif isinstance(attributes, dict): # Fallback for older flat dict attribute parsing
             for attr_idx, (attr_key, attr_value) in enumerate(attributes.items()):
                if isinstance(attr_key, bytes): attr_key = attr_key.decode('utf-8', 'replace')
                if isinstance(attr_value, bytes): attr_value = attr_value.decode('utf-8', 'replace')
                else: attr_value = str(attr_value)

                cursor.execute("""
                INSERT OR IGNORE INTO event_attributes (
                    event_db_id, attribute_key, attribute_value, attribute_index
                ) VALUES (?, ?, ?, ?)
                """, (event_db_id, attr_key, attr_value, attr_idx))
                if is_mayanode_address(attr_value):
                    insert_address(conn, attr_value, block_height)
                    role = f"event_{event_type}_{attr_key}"
                    link_transaction_to_address(conn, tx_hash, attr_value, role, event_db_id=event_db_id)
        return True

    except sqlite3.Error as e:
        print(f"SQLite error inserting transaction event (tx {tx_hash}, type {event_type}, index {event_index}): {e}")
        return False
    except Exception as ex:
        print(f"Unexpected error inserting transaction event (tx {tx_hash}, type {event_type}, index {event_index}): {ex}")
        return False

# --- Main Insertion Functions ---

def insert_transaction_and_components(conn, cursor, block_height: int, tx_data_parsed: dict):
    """Processes and inserts a single transaction and its components (messages, events, address links).
    This function DOES NOT commit; commit is handled by the calling block insertion logic.

    Args:
        conn (sqlite3.Connection): Active database connection (used by insert_address, link_transaction_to_address).
        cursor (sqlite3.Cursor): Active database cursor for main transaction data operations.
        block_height (int): The height of the block this transaction belongs to.
        tx_data_parsed (dict): The parsed transaction data object, expected to be one item 
                               from the 'transactions' list of common_utils.parse_confirmed_block output.
                               This dict should contain 'hash', 'tx_content_json', 'result_events_parsed', etc.
    Returns:
        bool: True if successful, False otherwise.
    """
    tx_hash = tx_data_parsed.get('hash')
    if not tx_hash:
        print(f"Error: Transaction data missing hash for block {block_height}. Data: {str(tx_data_parsed)[:200]}")
        return False

    # The `tx_data_parsed` from `common_utils.parse_confirmed_block` for a transaction already 
    # has the structure: {\"hash\": \"...\", \"tx_content_json\": {...}, \"result_log\": ..., etc.}
    # For the `transactions` table, we want to store this entire `tx_data_parsed` object as JSON
    # IF it\'s from Mayanode API. If from Tendermint, common_utils already transformed it to this structure.
    # The key is that common_utils.parse_confirmed_block is the single source of `tx_data_parsed` structure.
    
    # Ensure `height` field is in the stored tx_object_json, matching Mayanode API blocks
    # The `block_height` is passed as an argument, so we add it to the object being stored.
    # tx_object_to_store = tx_data_parsed.copy() # Make a copy to avoid modifying the input dict
    # tx_object_to_store[\'height\'] = str(block_height) # Mayanode API typically has height as string in tx object
    # tx_object_json_str = json.dumps(tx_object_to_store) # REMOVED: We are not storing the full JSON object

    try:
        fee_amount_str = tx_data_parsed.get('fee_amount')
        if isinstance(fee_amount_str, list):
            fee_amount_str = ",".join([f"{c.get('amount', '0')}{c.get('denom', '')}" for c in fee_amount_str])
        
        # Simplified for linter fix - will need to restore proper data extraction
        values_tuple = (
            tx_hash,
            block_height,
            tx_data_parsed.get('tx_index_in_block', 0), 
            True if tx_data_parsed.get('code', -1) == 0 else False, 
            tx_data_parsed.get('code', -1),
            tx_data_parsed.get('log', ''), 
            tx_data_parsed.get('gas_wanted', 0),
            tx_data_parsed.get('gas_used', 0),
            tx_data_parsed.get('tx_content_json', {}).get('body', {}).get('memo', ''), 
            fee_amount_str,
            tx_data_parsed.get('tx_content_json', {}).get('auth_info', {}).get('fee', {}).get('gas_limit', 0),
            tx_data_parsed.get('tx_content_json', {}).get('auth_info', {}).get('fee', {}).get('payer', ''),
            tx_data_parsed.get('tx_content_json', {}).get('auth_info', {}).get('fee', {}).get('granter', ''),
            tx_data_parsed.get('signer_pub_key_type', ''), 
            tx_data_parsed.get('signer_pub_key_bytes', b''),
            tx_data_parsed.get('tx_content_json', {}).get('body', {}).get('timeout_height', 0)
        )

        cursor.execute("""
        INSERT OR IGNORE INTO transactions (
            tx_hash, block_height, tx_index_in_block, success, code, log, gas_wanted, gas_used, memo, 
            fee_amount, fee_gas_limit, fee_payer, fee_granter, 
            signer_pub_key_type, signer_pub_key_bytes, timeout_height
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, values_tuple)

    except sqlite3.Error as e:
        print(f"SQLite error inserting main transaction record for {tx_hash}: {e}")
        return False
    # Removed the bare try, the except above covers this block.

    # --- Process and Insert Components --- 
    # 1. Messages (from tx_content_json.body.messages)
    tx_content = tx_data_parsed.get('tx_content_json', {})
    body = tx_content.get('body', {})
    messages = body.get('messages', [])
    if isinstance(messages, list):
        for i, msg_dict in enumerate(messages):
            # Pass conn for insert_address and link_transaction_to_address within specific message handlers
            _insert_transaction_message_internal(cursor, tx_hash, i, msg_dict, block_height, conn)
            # Address linking is now handled within _insert_msg_XXX_details_internal for message-specific roles.
    
    # 2. Events (from result_events_parsed)
    result_events = tx_data_parsed.get('result_events_parsed', [])
    if isinstance(result_events, list):
        for i, event_dict in enumerate(result_events):
            # Pass block_height and conn for address insertion/linking within event attribute handling
            _insert_transaction_event_internal(cursor, tx_hash, block_height, i, event_dict, conn)
            # Address linking from event attributes is now handled within _insert_transaction_event_internal.

    # 3. Signer addresses from AuthInfo (fee payer, granter, signer pubkey derived addresses)
    auth_info = tx_content.get('auth_info', {})
    fee_info = auth_info.get('fee', {})
    if isinstance(fee_info, dict):
        payer = fee_info.get('payer')
        granter = fee_info.get('granter')
        if payer and is_mayanode_address(payer):
            if insert_address(conn, payer, block_height):
                link_transaction_to_address(conn, tx_hash, payer, "tx_fee_payer")
        if granter and is_mayanode_address(granter):
            if insert_address(conn, granter, block_height):
                link_transaction_to_address(conn, tx_hash, granter, "tx_fee_granter")

    # Signer infos (derived addresses)
    # common_utils should populate 'derived_signers' at the top level of tx_data_parsed if successful
    derived_signers = tx_data_parsed.get('derived_signers', [])
    if isinstance(derived_signers, list):
        for signer_address in derived_signers:
            if is_mayanode_address(signer_address):
                if insert_address(conn, signer_address, block_height):
                    link_transaction_to_address(conn, tx_hash, signer_address, "tx_auth_info_signer")
    return True

def insert_block_and_components(conn, parsed_block_data: dict):
    """Inserts a fully parsed block and all its components (transactions, events, addresses) 
    into the database. This function handles its own transaction commit/rollback.

    Args:
        conn (sqlite3.Connection): Active database connection.
        parsed_block_data (dict): The output of common_utils.parse_confirmed_block.

    Returns:
        bool: True if the block and all components were successfully inserted, False otherwise.
    """
    block_height_int_val = parsed_block_data.get('block_height_int') # From common_utils
    block_height_str_val = parsed_block_data.get('block_height') or parsed_block_data.get('block_height_str') # str versions

    if block_height_int_val is None:
        try:
            block_height_int_val = int(block_height_str_val) if block_height_str_val else None
        except (ValueError, TypeError):
            print(f"Error: Block data missing valid integer block_height. Found str: {block_height_str_val}. Aborting insertion.")
            return False
    
    if block_height_int_val is None: # Final check
        print(f"Error: Block data missing any parsable block_height. Aborting insertion.")
        return False

    if check_if_block_exists(conn, block_height_int_val):
        return True 

    cursor = conn.cursor()
    try:
        block_time_dt_val = parsed_block_data.get('block_time_dt')
        block_time_dt_to_store = None
        if isinstance(block_time_dt_val, datetime):
            block_time_dt_to_store = block_time_dt_val.isoformat()
        elif isinstance(block_time_dt_val, str): 
            block_time_dt_to_store = block_time_dt_val
        
        # raw_tx_list = parsed_block_data.get('transactions_raw_base64') # No longer storing raw_tx_list_json
        # raw_tx_list_json_to_store = json.dumps(raw_tx_list) if raw_tx_list is not None else None

        cursor.execute("""
        INSERT INTO blocks (
            block_height, block_hash, block_time_str, block_time_dt, 
            chain_id, proposer_address, data_hash, validators_hash, 
            next_validators_hash, consensus_hash, app_hash, 
            last_results_hash, evidence_hash, last_block_hash,
            block_data_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            block_height_int_val,
            parsed_block_data.get('block_hash'),
            parsed_block_data.get('block_time_str'),
            block_time_dt_to_store,
            parsed_block_data.get('chain_id'),
            parsed_block_data.get('proposer_address'), # This is often the validator's consensus address (hex)
            parsed_block_data.get('data_hash'),
            parsed_block_data.get('validators_hash'),
            parsed_block_data.get('next_validators_hash'),
            parsed_block_data.get('consensus_hash'),
            parsed_block_data.get('app_hash'),
            parsed_block_data.get('last_results_hash'),
            parsed_block_data.get('evidence_hash'),
            parsed_block_data.get('last_block_hash'),
            parsed_block_data.get('transactions_source') # e.g., 'mayanode_api', 'tendermint_rpc'
            # raw_tx_list_json_to_store was removed
        ))

        # Insert block-level events
        begin_events = parsed_block_data.get('begin_block_events_parsed', [])
        if isinstance(begin_events, list):
            for i, event_data in enumerate(begin_events):
                if not _insert_block_event_internal(cursor, block_height_int_val, 'begin_block', i, event_data):
                    # Log error or decide if this should be a critical failure
                    print(f"Warning: Failed to insert a begin_block event for block {block_height_int_val}, index {i}")

        end_events = parsed_block_data.get('end_block_events_parsed', [])
        if isinstance(end_events, list):
            for i, event_data in enumerate(end_events):
                if not _insert_block_event_internal(cursor, block_height_int_val, 'end_block', i, event_data):
                    # Log error or decide if this should be a critical failure
                    print(f"Warning: Failed to insert an end_block event for block {block_height_int_val}, index {i}")
        
        # Extract and insert addresses found at the block level (e.g., proposer, addresses in block events)
        # The `extract_addresses_from_parsed_block` function should be reviewed/updated
        # to correctly find addresses from the new event attribute structure if necessary.
        # For now, we rely on it finding addresses from the parsed structure.
        block_level_addresses = extract_addresses_from_parsed_block(parsed_block_data)
        for addr in block_level_addresses:
            if is_mayanode_address(addr): # Ensure it's a valid format before inserting
                insert_address(conn, addr, block_height_int_val) 
                # Roles for block-level addresses are generally not stored in transaction_address_roles,
                # unless they are also part of a tx. Proposer is in `blocks` table.
                # Event-related addresses from block events could be linked if a specific need arises.

        # Insert all transactions and their components for this block
        transactions_data = parsed_block_data.get('transactions', []) # This comes from common_utils.parse_confirmed_block
        if isinstance(transactions_data, list):
            for tx_parsed in transactions_data:
                 # Add tx_index_in_block to tx_parsed if not already present by common_utils
                if 'tx_index_in_block' not in tx_parsed: # common_utils should be adding this
                    # This is a fallback, ideally common_utils.py guarantees this field
                    tx_hex_hash_for_index = tx_parsed.get('hash_str_tendermint_style') # from tendermint source
                    if tx_hex_hash_for_index and parsed_block_data.get('original_block_data',{}).get('block',{}).get('data',{}).get('txs'):
                        original_tm_txs = parsed_block_data['original_block_data']['block']['data']['txs']
                        try:
                            tx_parsed['tx_index_in_block'] = original_tm_txs.index(tx_hex_hash_for_index)
                        except ValueError:
                             # If not found, or if structure is different, assign a sequential index as last resort
                            tx_parsed['tx_index_in_block'] = transactions_data.index(tx_parsed)
                    else: # Fallback if original Tendermint tx list isn't available for indexing
                        tx_parsed['tx_index_in_block'] = transactions_data.index(tx_parsed)


                if not insert_transaction_and_components(conn, cursor, block_height_int_val, tx_parsed):
                    raise sqlite3.Error(f"Failed to insert transaction {tx_parsed.get('hash')} and its components for block {block_height_int_val}.")
        
        conn.commit() 
        return True

    except sqlite3.Error as e:
        print(f"SQLite transaction error during insertion of block {block_height_int_val}: {e}. Rolling back.")
        try:
            conn.rollback()
        except sqlite3.Error as rb_err:
            print(f"Rollback failed: {rb_err}")
        return False
    except Exception as e_gen:
        print(f"Unexpected error during insertion of block {block_height_int_val}: {e_gen}. Rolling back.")
        try:
            conn.rollback()
        except sqlite3.Error as rb_err:
            print(f"Rollback failed: {rb_err}")
        return False

# --- Detailed Query Functions ---

def get_block_header_data(conn, block_height: int) -> dict | None:
    """Retrieves the header data for a specific block_height."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM blocks WHERE block_height = ? LIMIT 1", (block_height,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        print(f"SQLite error getting block header for height {block_height}: {e}")
        return None

def get_transaction_object_json(conn, tx_hash: str) -> dict | None:
    """Retrieves the stored tx_object_json for a given transaction hash and parses it."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT tx_object_json FROM transactions WHERE tx_hash = ? LIMIT 1", (tx_hash,))
        row = cursor.fetchone()
        if row and row['tx_object_json']:
            return json.loads(row['tx_object_json'])
        return None
    except sqlite3.Error as e:
        print(f"SQLite error getting transaction object for hash {tx_hash}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error for transaction object (hash {tx_hash}): {e}")
        return None

def get_block_transactions_objects(conn, block_height: int) -> list[dict]:
    """Retrieves all transaction objects (as dicts) for a given block_height."""
    transactions_data = []
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT tx_object_json FROM transactions WHERE block_height = ? ORDER BY rowid", (block_height,))
        rows = cursor.fetchall()
        for row in rows:
            if row['tx_object_json']:
                try:
                    transactions_data.append(json.loads(row['tx_object_json']))
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for a transaction in block {block_height}: {e}. Skipping this transaction.")
        return transactions_data
    except sqlite3.Error as e:
        print(f"SQLite error getting transaction objects for block {block_height}: {e}")
        return []

def get_transaction_messages_parsed(conn, tx_hash: str) -> list[dict]:
    """Retrieves and parses all messages for a given transaction hash."""
    messages = []
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT raw_message_content_json FROM transaction_messages WHERE tx_hash = ? ORDER BY message_index", (tx_hash,))
        rows = cursor.fetchall()
        for row in rows:
            if row['raw_message_content_json']:
                try:
                    messages.append(json.loads(row['raw_message_content_json']))
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for a message in tx {tx_hash}: {e}. Skipping this message.")
        return messages
    except sqlite3.Error as e:
        print(f"SQLite error getting messages for tx {tx_hash}: {e}")
        return []

def get_transaction_events_parsed(conn, tx_hash: str) -> list[dict]:
    """Retrieves and parses all events for a given transaction hash."""
    events = []
    cursor = conn.cursor()
    try:
        # This query needs to be updated to use the new event_attributes table
        # For now, this function will be made obsolete by reconstruction logic.
        # cursor.execute("SELECT event_type, attributes_json FROM events WHERE tx_hash = ? ORDER BY event_index", (tx_hash,))
        # rows = cursor.fetchall()
        # for row in rows:
        #     attributes = {}
        #     if row['attributes_json']:
        #         try:
        #             attributes = json.loads(row['attributes_json'])
        #         except json.JSONDecodeError as e:
        #             print(f"JSON decode error for event attributes in tx {tx_hash}: {e}. Using empty attributes.")
        #     events.append({"type": row['event_type'], "attributes": attributes})
        print(f"Warning: get_transaction_events_parsed for {tx_hash} is being called but is slated for removal/replacement by new reconstruction logic.")
        return _get_formatted_events_internal(conn, tx_hash=tx_hash, event_category='transaction_result')
    except sqlite3.Error as e:
        print(f"SQLite error getting events for tx {tx_hash}: {e}")
        return []

def get_block_begin_events_parsed(conn, block_height: int) -> list[dict]:
    """Retrieves and parses all begin_block events for a given block_height."""
    # return _get_block_events_by_category_parsed(conn, block_height, 'begin_block') # Old helper
    print(f"Warning: get_block_begin_events_parsed for {block_height} is being called but is slated for removal/replacement.")
    return _get_formatted_events_internal(conn, block_height=block_height, event_category='begin_block')

def get_block_end_events_parsed(conn, block_height: int) -> list[dict]:
    """Retrieves and parses all end_block events for a given block_height."""
    # return _get_block_events_by_category_parsed(conn, block_height, 'end_block') # Old helper
    print(f"Warning: get_block_end_events_parsed for {block_height} is being called but is slated for removal/replacement.")
    return _get_formatted_events_internal(conn, block_height=block_height, event_category='end_block')

def get_address_info(conn, address: str) -> dict | None:
    """Retrieves all information for a specific address."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM addresses WHERE address = ? LIMIT 1", (address,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        print(f"SQLite error getting info for address {address}: {e}")
        return None

def get_transactions_for_address(conn, address: str, limit: int = 100, offset: int = 0) -> list[str]:
    """Retrieves a list of transaction hashes involving a specific address, with pagination."""
    tx_hashes = []
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT DISTINCT tx_hash 
            FROM transaction_address_roles 
            WHERE address = ? 
            ORDER BY tx_hash DESC -- Or by block_height via join, then tx_hash
            LIMIT ? OFFSET ?
        """, (address, limit, offset))
        rows = cursor.fetchall()
        return [row['tx_hash'] for row in rows]
    except sqlite3.Error as e:
        print(f"SQLite error getting transactions for address {address}: {e}")
        return []

# --- Block Reconstruction Function ---

def reconstruct_block_as_mayanode_api_json(conn, block_height: int) -> dict | None:
    """Reconstructs a full block from the database into a JSON structure that mimics 
    the Mayanode API '/mayachain/block/{height}' response as closely as possible.

    Args:
        conn (sqlite3.Connection): Active database connection.
        block_height (int): The height of the block to reconstruct.

    Returns:
        dict | None: A dictionary representing the block, or None if the block is not found 
                      or an error occurs.
    """
    
    block_header = get_block_header_data(conn, block_height)
    if not block_header:
        print(f"Block {block_height} not found in database for reconstruction.")
        return None

    # Start building the Mayanode API-like structure
    # Based on observed Mayanode API structure for /mayachain/block/{height}
    # This might differ slightly from Tendermint's /block or /block_results structure.
    
    # The Mayanode API has a top-level structure like:
    # {
    #   "id": { "hash": "...", "parts": { "total": ..., "hash": "..." } }, // This is block_id in Tendermint /block
    #   "header": { "version": ..., "chain_id": ..., "height": ..., ... },
    #   "data": { "txs": [...] }, // This part is more complex; Mayanode /block gives decoded txs directly.
    #                               // If our DB stores raw Tendermint txs, this part would be raw b64 strings.
    #                               // However, our `transactions` table stores `tx_object_json` which is already structured.
    #   "evidence": { "evidence": [...] },
    #   "last_commit": { ... }, // Not fully stored in our current `blocks` table in this detail.
    #   "begin_block_events": [ ... ],
    #   "end_block_events": [ ... ],
    #   "txs": [ ... ] // CRITICAL: Mayanode API returns fully formed tx objects HERE at top level.
    # }
    # Our `transactions` table stores these tx objects (via `tx_object_json`).

    # The direct Mayanode API (/mayachain/block/{height}) for a specific block seems to be simpler:
    # It often has: `id`, `
    reconstructed_block = {}

    # 1. Block ID (simplified - Mayanode API often has parts, but hash is key)
    reconstructed_block['id'] = {
        "hash": block_header.get('block_hash')
        # 'parts' is usually from Tendermint direct /block. Mayanode /mayachain/block often omits it or has null.
        # We don't store parts, so we'll omit it to match a potentially simpler Mayanode API view.
    }

    # 2. Header (map fields from our `blocks` table)
    # Mayanode API header includes a nested 'version': { 'block': '...', 'app': '...' }
    # We don't store these version sub-fields directly. We'll populate what we have.
    header_map = {
        "chain_id": block_header.get('chain_id'),
        "height": str(block_header.get('block_height')), # Mayanode API uses string for height here
        "time": block_header.get('block_time_dt') or block_header.get('block_time_str'), # Prefer dt if available
        "last_block_id": { # Mayanode API has last_block_id.hash
            "hash": block_header.get('last_block_hash')
            # "parts": {} // Again, omitting parts as we don't store them
        },
        "last_commit_hash": None, # We don't store this directly from `blocks` table.
                                # Mayanode /mayachain/block often has this as null or omits it.
                                # Tendermint /block has it as block.last_commit.hash.
        "data_hash": block_header.get('data_hash'),
        "validators_hash": block_header.get('validators_hash'),
        "next_validators_hash": block_header.get('next_validators_hash'),
        "consensus_hash": block_header.get('consensus_hash'),
        "app_hash": block_header.get('app_hash'),
        "last_results_hash": block_header.get('last_results_hash'),
        "evidence_hash": block_header.get('evidence_hash'), # This is already just the hash string
        "proposer_address": block_header.get('proposer_address')
        # Mayanode API might have 'version':{'block': '...', 'app': '...'} - we don't store these currently.
    }
    reconstructed_block['header'] = {k: v for k, v in header_map.items() if v is not None}

    # 3. Transactions (fetch and reconstruct)
    reconstructed_block['txs'] = _get_formatted_transactions_for_block(conn, block_height)

    # 4. Begin Block Events (fetch and format using new helper)
    reconstructed_block['begin_block_events'] = _get_formatted_events_internal(conn, block_height=block_height, event_category='begin_block')

    # 5. End Block Events (fetch and format using new helper)
    reconstructed_block['end_block_events'] = _get_formatted_events_internal(conn, block_height=block_height, event_category='end_block')
    
    # Optional fields often seen in Mayanode API but not fully captured or structured the same way in our DB:
    # - `data`: { "txs": null } (if txs are top-level, this is often null or just an empty list if raw txs were there)
    #           Our `blocks.raw_tx_list_json` could populate this if we were mimicking Tendermint /block more directly.
    #           But since we aim for `/mayachain/block/{height}` where `txs` is top-level and decoded, this is less relevant.
    # - `evidence`: { "evidence": [] } (we store only evidence_hash, Mayanode API also usually has just the hash in header)
    # - `last_commit`: { ... detailed commit info ... } (we only store last_commit_hash indirectly via last_block_id structure,
    #                                                  Mayanode API often has this as null or omits it at this top block level)

    return reconstructed_block

if __name__ == '__main__':
    # Basic test for the new database utils
    print("--- Basic Test for Re-written database_utils.py ---")
    db_conn = None
    try:
        db_conn = get_db_connection()
        print(f"Connected to database: {DATABASE_FILE}")
        create_tables(db_conn)
        print("Tables ensured.")

        latest_height = get_latest_block_height_from_db(db_conn)
        print(f"Current latest block height in DB: {latest_height if latest_height is not None else 'No blocks found'}")

        if latest_height:
            print(f"Attempting to reconstruct block {latest_height} as Mayanode API JSON...")
            reconstructed_json = reconstruct_block_as_mayanode_api_json(db_conn, latest_height)
            if reconstructed_json:
                print(f"Successfully reconstructed block {latest_height}. Outputting to reconstructed_block_sample.json")
                # Save to a file for inspection
                sample_output_path = os.path.join(PROJECT_ROOT, "reconstructed_block_sample.json")
                with open(sample_output_path, 'w') as f:
                    json.dump(reconstructed_json, f, indent=2)
                print(f"Sample output saved to: {sample_output_path}")
                
                # Example: Print number of transactions in reconstructed block
                if 'txs' in reconstructed_json and isinstance(reconstructed_json['txs'], list):
                    print(f"Number of transactions in reconstructed block: {len(reconstructed_json['txs'])}")
                else:
                    print("No 'txs' list found in reconstructed block or not a list.")
                
                if 'begin_block_events' in reconstructed_json:
                    print(f"Number of begin_block_events: {len(reconstructed_json['begin_block_events'])}")
                if 'end_block_events' in reconstructed_json:
                    print(f"Number of end_block_events: {len(reconstructed_json['end_block_events'])}")

            else:
                print(f"Failed to reconstruct block {latest_height}.")
        else:
            print("No blocks in DB to test reconstruction.")

    except sqlite3.Error as e:
        print(f"Database test error: {e}")
    except Exception as e_gen:
        print(f"General test error: {e_gen}")
    finally:
        if db_conn:
            db_conn.close()
            print("Database connection closed.")
    print("--- Basic Test Finished ---") 