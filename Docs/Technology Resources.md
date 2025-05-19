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

## Notes:

*   **Uniswap:** Previously considered for external price data, but the project has shifted to using CoinGecko.
*   **Tendermint RPC:** While part of the broader MayaChain ecosystem, direct interaction with Tendermint RPC is not currently a core part of this project's data fetching for AI model inputs (Midgard is the primary source).
*   **Cloud Services:** May be considered for future, more intensive model training or deployment, but current development is focused on local execution.