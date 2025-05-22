# Technology Resources Document

This document lists the key technologies used in the project, along with links to their documentation for reference.

| **Aspect**              | **Technology/Library**             | **Documentation URL**                                                                 |
|-------------------------|------------------------------------|--------------------------------------------------------------------------------------|
| **Core Language**       | Python                             | [Python Documentation](https://docs.python.org/3/)                                   |
| **Data Handling**       | Pandas                             | [Pandas Documentation](https://pandas.pydata.org/docs/)                               |
|                         | NumPy                              | [NumPy Documentation](https://numpy.org/doc/)                                         |
| **Web Framework**       | Flask                              | [Flask Documentation](https://flask.palletsprojects.com/)                             |
| **Machine Learning**    | PyTorch                            | [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)                   |
|                         | Scikit-learn (for scaling)         | [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)      |
| **Maya Protocol API**   | Midgard API (via HTTP requests)    | [Midgard Public API](https://midgard.mayachain.info/v2/doc)                             |
| **External Price API**  | CoinGecko API (via `pycoingecko`)  | [CoinGecko API](https://www.coingecko.com/en/api/documentation), [`pycoingecko` GitHub](https://github.com/man-c/pycoingecko) |
| **HTTP Requests**       | `requests` library                 | [Requests Documentation](https://requests.readthedocs.io/en/latest/)                  |
| **Local Database**      | SQLite (via Python's `sqlite3`)   | [sqlite3 Documentation](https://docs.python.org/3/library/sqlite3.html)             |

## Notes:

*   **Uniswap:** Previously considered for external price data, but the project has shifted to using CoinGecko.
*   **Tendermint RPC & Mayanode API:** The project now uses the Mayanode REST API (`https://mayanode.mayachain.info/mayachain`) for fetching confirmed block data and the Tendermint RPC (`https://tendermint.mayachain.info`) for mempool/unconfirmed transaction data. Midgard is no longer the primary source.
*   **Database Choice:** SQLite is chosen for the initial implementation of the local data store (Task 10.2.B) due to its simplicity and ease of local development. PostgreSQL (potentially via Supabase) is a candidate for future upgrades if advanced JSON querying, realtime features, or greater scalability are required.
*   **Cloud Services:** May be considered for future, more intensive model training or deployment, but current development is focused on local execution.