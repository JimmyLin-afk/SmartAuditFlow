# Teller Protocol

The Teller Protocol operates as decentralized software, enabling unsecured DeFi digital asset lending and borrowing through an open order-book model.
Through the protocol, borrowers can bridge off-chain data onto on-chain loan requests. Those requesting assets propose a loan request, and those supplying assets commit those assets to loan requests of their choosing. Lenders who agree to loan terms requested by borrowers, based on the data provided or required, transact directly.
Information appended to a loan request is at the borrower's discretion. This may include details from a borrower’s financial stature, social status, identity, or other relevant data. The Teller Protocol is data agnostic and does not have an opinion on the user. The user can gather any type of data from third parties.

Users
- Market Owners 
  - Entrepreneurs, developers, and creators, who desire to launch a new lending market, can easily initiate their own lending book within the protocol to become a market owner. They can build out a simple frontend interface for their borrowers who can then connect the required data to support the market’s specific loans.
- Lenders 
  - Lenders within a market can fill any open asset request. This can be an individual, entity, protocol, DAO, or other asset suppliers. Likely, lenders will create their own criteria for committing assets, based on rules and filtered by asset request data. Lending can be automated by individuals, companies, protocols, DAOs, and other asset suppliers.
- Borrowers 
  - Borrowers who request assets through the Teller Protocol can be an individual, entity, or protocol. Consumers, creators, businesses, DAO’s, protocols, lending markets, and more can all submit loan requests to the market of their choosing directly through the Teller Protocol.

## Contracts
Details on a few of the core contracts within the Teller Protocol

**TellerV2**
The TellerV2 contract lies at the heart of the protocol and allows Pool owners to open markets and, lenders and borrowers to participate in those markets.

Borrowers (and developers building DApps for users) can submit bids by calling:
```solidity
/**
* @notice Function for a borrower to create a bid for a loan.
* @param _lendingToken The lending token asset requested to be borrowed.
* @param _principal The principal amount of the loan bid.
* @param _duration The length of time, in seconds, the loan will remain active.
* @param _APY The proposed interest rate for the loan bid.
* @param _paymentCycle The recurrent length of time before which a payment is due.
* @param _metadataURI The URI for additional borrower loan information as part of loan bid.
*/
function submitBid(
  address _lendingToken,
  uint256 _marketplaceId,
  uint256 _principal,
  uint32 _duration,
  uint16 _APY,
  uint32 _paymentCycle,
  bytes32 _metadataURI,
  address _receiver
);
``` 
The _metadataURI submitted by a borrower along with the proposed loan terms (_APY, _principal, _lendingToken, etc), needs to correspond with the metadata required by the market. Depending on the market, this could be in the form of a verified credit report from the relative credit bureau, identity, or other relevant data.

Lenders (and developers building DApps for users) can accept bids by calling:
```solidity
/**
* @notice Function for a lender to accept a proposed loan bid.
* @param _bidId The id of the loan bid to accept.
*/
function lenderAcceptBid(uint256 _bidId);
```

**MarketRegistry**
Market owners interested in opening Pools on the Teller Protocol can do so through the MarketRegistry contract by calling:
```solidity
Create a market
/**
 * @notice Creates a new market.
 * @param _initialOwner Address who will initially own the market.
 * @param _paymentCycleDuration Length of time in seconds before a bid's next payment is required to be made.
 * @param _paymentDefaultDuration Length of time in seconds before a loan is considered in default for non-payment.
 * @param _bidExpirationTime Length of time in seconds before pending bids expire.
 * @param _requireLenderAttestation Boolean that indicates if lenders require attestation to join market, meaning
            if the market owner wants the lender to be KYC'ed then we enable this as True else it will be False
 * @param _requireBorrowerAttestation Boolean that indicates if borrowers require attestation to join market, meaning
            if the market owner wants the borrower to be KYC'ed then we enable this as True else it will be False
 * @param _uri URI string to get metadata details about the market.
 */
function createMarket(
   address _initialOwner,
   uint32 _paymentCycleDuration,
   uint32 _paymentDefaultDuration,
   uint32 _bidExpirationTime,
   uint16 _feePercent,
   bool _requireLenderAttestation,
   bool _requireBorrowerAttestation,
   string calldata _uri
);
```
With the _requireLenderAttestation and _requireBorrowerAttestation, market owners have the option to require their users to be verified through an attestation service, before being allowed to interact with the market. It is then up to the market owner's discretion to include them in their market.

## Running Tests 
To set up the repo locally, you will need `yarn` and `node v16`:
- `git clone https://github.com/teller-protocol/teller-protocol-v2`
- Run `yarn install`

Run tests locally:
- Run `yarn contracts test`

## Additional Documentation
Get more info here: https://teller.gitbook.io/teller-v2/