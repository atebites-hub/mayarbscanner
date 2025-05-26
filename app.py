from flask import Flask, render_template, g
import sqlite3
import os

# Assuming database_utils.py is in a 'src' subdirectory from where app.py is run
# and common_utils.py is also in 'src'
import sys
_project_root = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_project_root, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Now import from src
try:
    from database_utils import get_db_connection, get_dividend_receiving_addresses, get_dividends_for_address, DATABASE_FILE
except ImportError as e:
    print(f"Error importing from database_utils: {e}")
    # Define placeholders if import fails, so app can partially run or show an error
    DATABASE_FILE = "mayanode_blocks.db" # Fallback
    def get_db_connection(db_file=None): raise RuntimeError("Database utils not loaded")
    def get_dividend_receiving_addresses(conn): raise RuntimeError("Database utils not loaded")
    def get_dividends_for_address(conn, addr): raise RuntimeError("Database utils not loaded")

app = Flask(__name__)

# Database setup
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        # Use DATABASE_FILE imported from database_utils
        db_path = os.path.join(_project_root, DATABASE_FILE)
        # Check if the database file exists, provide a helpful error if not
        if not os.path.exists(db_path):
            print(f"ERROR: Database file not found at {db_path}")
            print("Please ensure the database has been created and populated, e.g., by running src/fetch_realtime_transactions.py.")
            # Optionally, could raise an error here or return a specific response in routes
        
        db = g._database = get_db_connection(db_file=db_path)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def index():
    conn = get_db()
    try:
        addresses = get_dividend_receiving_addresses(conn)
        # Create a dummy error for testing if addresses is empty
        error_message = None
        if not addresses:
            error_message = "No dividend-receiving addresses found in the database. Ensure data has been fetched and processed."
        return render_template('index.html', addresses=addresses, error_message=error_message)
    except RuntimeError as e: # Catch if db_utils failed to load
        return f"Application Error: {str(e)}. Please check server logs.", 500
    except sqlite3.Error as e:
        # This might happen if the table doesn't exist or query fails
        return f"Database Error: {str(e)}. Ensure the database is correctly set up and populated.", 500

@app.route('/address/<wallet_address>')
def address_detail(wallet_address):
    conn = get_db()
    try:
        dividends = get_dividends_for_address(conn, wallet_address)
        error_message = None
        if not dividends:
            # Check if the address itself is valid but just has no dividends,
            # vs. the address not being found in general (though our current DB doesn't distinguish this specific case for dividends)
            error_message = f"No CACAO dividends found for address: {wallet_address}."
        return render_template('address_detail.html', address=wallet_address, dividends=dividends, error_message=error_message)
    except RuntimeError as e: # Catch if db_utils failed to load
        return f"Application Error: {str(e)}. Please check server logs.", 500
    except sqlite3.Error as e:
        return f"Database Error while fetching details for {wallet_address}: {str(e)}. Ensure the database is correctly set up and populated.", 500

# TODO: Create templates/index.html and templates/address_detail.html

if __name__ == '__main__':
    # Ensure the templates directory exists
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        print(f"Created directory: {templates_dir}")
        # Create dummy template files if they don't exist
        with open(os.path.join(templates_dir, "index.html"), "w") as f:
            f.write("<h1>Addresses</h1><ul>{% for addr in addresses %}<li><a href=\"{{ url_for('address_detail', wallet_address=addr) }}\">{{ addr }}</a></li>{% else %}<li>No addresses found.</li>{% endfor %}</ul>{% if error_message %}<p style=\"color:red;\">{{ error_message }}</p>{% endif %}")
        with open(os.path.join(templates_dir, "address_detail.html"), "w") as f:
            f.write("<h1>Dividends for {{ address }}</h1><table border=\"1\"><tr><th>Block Height</th><th>Time (UTC)</th><th>Amount (Raw CACAO)</th><th>Amount (Formatted CACAO)</th></tr>{% for div in dividends %}<tr><td>{{ div.block_height }}</td><td>{{ div.block_time_dt }}</td><td>{{ div.amount_raw_cacao }}</td><td>{{ '%.8f' | format(div.amount_raw_cacao / (10**8)) }} CACAO</td></tr>{% else %}<tr><td colspan=\"4\">No dividends found for this address.</td></tr>{% endfor %}</table><p><a href=\"{{ url_for('index') }}\">Back to list</a></p>")

    app.run(debug=True) 