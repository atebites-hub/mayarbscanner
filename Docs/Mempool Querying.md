Key Points
It seems likely that querying Maya Protocol's mempool involves using the Tendermint RPC endpoint of your node.
Research suggests the /unconfirmed_txs endpoint is used to get unconfirmed transactions, essential for block construction.
The evidence leans toward connecting to your node's Tendermint RPC at port 27147 for MAINNET, using a host like localhost if running locally.
Connecting to the Mempool
To query Maya Protocol's mempool, connect to your node's Tendermint RPC endpoint. For MAINNET, use port 27147, and the endpoint is typically [http://localhost:27147](http://localhost:27147) if running locally. Use the /unconfirmed_txs endpoint to retrieve unconfirmed transactions, which you can then use to construct the next block.
Finding Public Nodes
If you need a public node, check the official documentation at Maya Protocol Docs or community resources, as public RPC endpoints may not be widely listed for security reasons.
Additional Considerations
Ensure your node is properly set up and running, as constructing the next block requires node operator access. For detailed guidance, refer to the Connecting to MAYAChain section in the documentation.
Survey Note: Detailed Analysis of Querying Maya Protocol's Mempool
This survey note provides a comprehensive exploration of querying Maya Protocol's mempool, particularly for the purpose of constructing the next block, based on available documentation and related blockchain technologies. The analysis is grounded in the understanding that Maya Protocol is a cross-chain decentralized exchange (DEX) and a fork of THORChain, utilizing the Tendermint consensus engine and Cosmos-SDK state machine.
Background and Context
Maya Protocol, launched with its mainnet on March 7, 2023, is designed to facilitate non-custodial, permissionless cross-chain swaps without relying on wrapped or pegged assets. It employs on-chain vaults and economic security, managed through the Tendermint consensus engine, Cosmos-SDK state machine, and GG20 Threshold Signature Scheme (TSS). Given the user's intent to construct the next block, it is inferred that they are likely a node operator or validator, requiring access to the mempool to select unconfirmed transactions for inclusion.
The mempool, in blockchain terminology, is a collection of unconfirmed transactions waiting to be included in a block. For Tendermint-based chains like Maya Protocol, the mempool is managed at the consensus layer, making the Tendermint RPC a likely interface for querying it.
Technical Approach to Querying the Mempool
Research into Maya Protocol's documentation, particularly the "Connecting to MAYAChain" section at Maya Protocol Docs, reveals that the network supports multiple RPC interfaces, including Tendermint RPC for consensus-related information. The documentation specifies the following ports for different network environments:
Service
Network
Port
RPC Guide/Endpoints Guide
Midgard
Mainnet
8080
http://:8080/v2/doc
MAYANode
Mainnet
1317
http://:1317/mayachain/doc/
Cosmos RPC
-
-
https://v1.cosmos.network/rpc/v0.45.1, Example URL: https://stagenet.mayanode.mayachain.info/cosmos/bank/v1beta1/balances/smaya18z343fsdlav47chtkyp0aawqt6sgxsh3ctcu6u
Tendermint RPC
Mainnet
27147
https://docs.tendermint.com/master/rpc/#/
Tendermint RPC
Stagenet
26657
https://docs.tendermint.com/master/rpc/#/
Tendermint RPC
Testnet
26657
https://docs.tendermint.com/master/rpc/#/
P2P
Mainnet
27146
https://docs.tendermint.com/master/spec/p2p/
P2P
Stagenet
27146
https://docs.tendermint.com/master/spec/p2p/
For MAINNET, the Tendermint RPC operates on port 27147, which is critical for querying consensus-related data, including the mempool. The evidence leans toward using the /unconfirmed_txs endpoint, a standard Tendermint RPC method for retrieving unconfirmed transactions, as confirmed by examining Tendermint's RPC specifications at Tendermint RPC Docs. An example usage might look like making an HTTP GET request to [http://localhost:27147/unconfirmed_txs](http://localhost:27147/unconfirmed_txs) if running a local node.
Node Operation and Public Endpoints
Given the user's role in constructing the next block, it is likely they are operating a MAYANode, which consists of multiple independent servers in a cluster, as detailed in the "MAYANode Overview" at Maya Protocol Docs. Each node runs full-nodes for linked chains and includes components like mayanode, which provides a RESTful API, and bifrost for cross-chain connections. The documentation notes that public endpoints, such as those for Midgard and MAYANode, are run by specific organizations, implying a cost and potential limitation for heavy querying, suggesting users might need to run their own nodes for such operations.
However, the documentation does not explicitly list public Tendermint RPC endpoints, and attempts to find such information on the MayaScan block explorer at MayaScan did not yield specific API or RPC details. This suggests that for security and decentralization, Maya Protocol may not widely publicize public RPC nodes, and users are encouraged to run their own nodes, especially for validator activities like block construction.
Comparison with THORChain
Given Maya Protocol's origin as a fork of THORChain, a comparison is insightful. THORChain, as seen in its documentation at THORChain Docs, also uses Tendermint RPC for consensus-related queries, including mempool information, with similar endpoints like /unconfirmed_txs. This reinforces the approach for Maya Protocol, as both share the same underlying technology stack, including Tendermint consensus and Cosmos-SDK.
Practical Steps for the User
For the user, the process involves:
Ensuring their MAYANode is operational, as detailed in guides like "Bare Metal Node Guides" at Maya Protocol Docs.
Connecting to the Tendermint RPC endpoint, likely at [http://localhost:27147](http://localhost:27147) for MAINNET if local, and querying /unconfirmed_txs to retrieve the mempool.
Using the retrieved transactions to construct the next block, following the network's consensus rules and economic security measures, such as those outlined in the "Technology" section at Maya Protocol Docs.
If public nodes are needed, the user can explore community resources or the official Discord, as mentioned in Maya Protocol's ecosystem links, though specific public RPC hosts were not found in the documentation reviewed.
Challenges and Considerations
One challenge is the potential overload of public nodes if querying heavily, as noted in the documentation, which advises running personal nodes for such purposes. Additionally, the lack of explicit public RPC endpoints underscores the decentralized nature, requiring node operators to manage their infrastructure. The user, as a block constructor, must ensure high uptime and security, given the risks and costs associated with node operation, as highlighted in the "MAYANode Overview."
Conclusion
In summary, querying Maya Protocol's mempool for block construction involves leveraging the Tendermint RPC endpoint at port 27147 for MAINNET, using the /unconfirmed_txs endpoint to retrieve unconfirmed transactions. The user, likely a node operator, should connect to their local node's RPC, ensuring proper setup and operation as per the documentation. For public node access, further exploration of community resources may be necessary, given the limited public endpoint information.
Key Citations
Connecting to MAYAChain Maya Protocol One-Stop-Shop
Maya Protocol One-Stop-Shop Documentation
Tendermint RPC Documentation
Maya Protocol Block Explorer MayaScan
THORChain Documentation Introduction
Maya Protocol Node Docs MAYANode Overview
Maya Protocol Deep Dive How It Works Technology
Maya Protocol Node Docs Bare Metal Node Guides


DEV FINDINGS:
The devs just got back to me, this is the tendermint endpoint:

@http://tendermint.mayachain.info/
 http://tendermint.mayachain.info/
 if we need to, port is 443