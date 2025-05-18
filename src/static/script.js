// Placeholder for JavaScript logic to fetch and display data.
// This will be implemented in Task 1.4.5.

document.addEventListener('DOMContentLoaded', () => {
    console.log('Maya Scanner UI Initialized');

    const historicalContainer = document.getElementById('historical-table-container');
    const liveConfirmedContainer = document.getElementById('live-confirmed-table-container');
    const livePendingContainer = document.getElementById('live-pending-table-container');
    const fetchHistoricalBtn = document.getElementById('fetch-historical-btn');

    const API_BASE_URL = '/api';
    const LIVE_UPDATE_INTERVAL = 5000; // 5 seconds

    // Columns to display for each type of data
    // These should ideally match some of the more important DF_COLUMNS from Python
    const HISTORICAL_COLUMNS = ['date', 'type', 'status', 'in_asset', 'in_amount', 'out_asset', 'out_amount', 'transaction_id'];
    const LIVE_COLUMNS = ['date', 'type', 'status', 'in_asset', 'in_amount', 'out_asset', 'out_amount', 'transaction_id']; // Can be same or different

    function createTable(data, columns) {
        if (!data || data.length === 0) {
            const p = document.createElement('p');
            p.textContent = 'No data available.';
            return p; // Return a P Node
        }
        const table = document.createElement('table');
        const thead = document.createElement('thead');
        const tbody = document.createElement('tbody');
        const headerRow = document.createElement('tr');

        columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col.replace(/_/g, ' ').toUpperCase(); // Format header
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        data.forEach(item => {
            const tr = document.createElement('tr');
            columns.forEach(col => {
                const td = document.createElement('td');
                let value = item[col];
                if (col === 'date' && value) {
                    // Convert nanosecond timestamp string to a more readable date
                    try {
                        value = new Date(parseInt(value) / 1000000).toLocaleString();
                    } catch (e) { /* ignore, use original value */ }
                }
                td.textContent = value !== null && value !== undefined ? value : '-';
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        return table;
    }

    async function fetchData(endpoint, container, tableColumns) {
        try {
            container.innerHTML = '<p>Loading data...</p>'; // Show loading message
            const response = await fetch(`${API_BASE_URL}${endpoint}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            container.innerHTML = ''; // Clear loading message
            container.appendChild(createTable(data, tableColumns));
        } catch (error) {
            console.error(`Error fetching ${endpoint}:`, error);
            container.innerHTML = `<p>Error loading data from ${endpoint}. Check console for details.</p>`;
        }
    }

    // Fetch historical data on button click
    if (fetchHistoricalBtn) {
        fetchHistoricalBtn.addEventListener('click', () => {
            fetchData('/historical-24hr', historicalContainer, HISTORICAL_COLUMNS);
        });
    }

    // Fetch live data periodically
    function fetchLiveConfirmed() {
        fetchData('/live-confirmed', liveConfirmedContainer, LIVE_COLUMNS);
    }

    function fetchLivePending() {
        fetchData('/live-pending', livePendingContainer, LIVE_COLUMNS);
    }

    // Initial fetch for live data and set intervals
    if (liveConfirmedContainer) fetchLiveConfirmed();
    if (livePendingContainer) fetchLivePending();

    setInterval(() => {
        if (liveConfirmedContainer) fetchLiveConfirmed();
    }, LIVE_UPDATE_INTERVAL);

    setInterval(() => {
        if (livePendingContainer) fetchLivePending();
    }, LIVE_UPDATE_INTERVAL);

    // Optional: Initial load for historical if desired, or leave to button click
    // if (historicalContainer) fetchData('/historical-24hr', historicalContainer, HISTORICAL_COLUMNS);
}); 