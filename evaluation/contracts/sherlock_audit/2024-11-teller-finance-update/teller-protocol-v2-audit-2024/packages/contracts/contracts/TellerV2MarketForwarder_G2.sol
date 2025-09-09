pragma solidity >=0.8.0 <0.9.0;
// SPDX-License-Identifier: MIT

import "./interfaces/ITellerV2.sol";

import "./interfaces/IMarketRegistry.sol";
import "./interfaces/ITellerV2MarketForwarder.sol";

import "@openzeppelin/contracts-upgradeable/utils/AddressUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/IERC20Upgradeable.sol";

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";

import "@openzeppelin/contracts-upgradeable/utils/ContextUpgradeable.sol";

/**
 * @dev Simple helper contract to forward an encoded function call to the TellerV2 contract. See {TellerV2Context}
 */
abstract contract TellerV2MarketForwarder_G2 is
    Initializable,
    ContextUpgradeable,
    ITellerV2MarketForwarder
{
    using AddressUpgradeable for address;

    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    address public immutable _tellerV2;
    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    address public immutable _marketRegistry;

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor(address _protocolAddress, address _marketRegistryAddress) {
        _tellerV2 = _protocolAddress;
        _marketRegistry = _marketRegistryAddress;
    }

    function getTellerV2() public view returns (address) {
        return _tellerV2;
    }

    function getMarketRegistry() public view returns (address) {
        return _marketRegistry;
    }

    function getTellerV2MarketOwner(uint256 marketId) public returns (address) {
        return IMarketRegistry(getMarketRegistry()).getMarketOwner(marketId);
    }

    /**
     * @dev Performs function call to the TellerV2 contract by appending an address to the calldata.
     * @param _data The encoded function calldata on TellerV2.
     * @param _msgSender The address that should be treated as the underlying function caller.
     * @return The encoded response from the called function.
     *
     * Requirements:
     *  - The {_msgSender} address must set an approval on TellerV2 for this forwarder contract __before__ making this call.
     */
    function _forwardCall(bytes memory _data, address _msgSender)
        internal
        returns (bytes memory)
    {
        return
            address(_tellerV2).functionCall(
                abi.encodePacked(_data, _msgSender)
            );
    }

     
    /**
     * @notice Creates a new loan using the TellerV2 lending protocol.
     * @param _createLoanArgs Details describing the loan agreement.]
     * @param _borrower The borrower address for the new loan.
     */
    function _submitBidWithCollateral(
        CreateLoanArgs memory _createLoanArgs,
        address _borrower
    ) internal virtual returns (uint256 bidId) {
        bytes memory responseData;

        responseData = _forwardCall(
            abi.encodeWithSignature(
                "submitBid(address,uint256,uint256,uint32,uint16,string,address,(uint8,uint256,uint256,address)[])",
                _createLoanArgs.lendingToken,
                _createLoanArgs.marketId,
                _createLoanArgs.principal,
                _createLoanArgs.duration,
                _createLoanArgs.interestRate,
                _createLoanArgs.metadataURI,
                _createLoanArgs.recipient,
                _createLoanArgs.collateral
            ),
            _borrower
        );

        return abi.decode(responseData, (uint256));
    }

    /**
     * @notice Accepts a new loan using the TellerV2 lending protocol.
     * @param _bidId The id of the new loan.
     * @param _lender The address of the lender who will provide funds for the new loan.
     */
    function _acceptBid(uint256 _bidId, address _lender)
        internal
        virtual
        returns (bool)
    {
        // Approve the borrower's loan
        _forwardCall(
            abi.encodeWithSelector(ITellerV2.lenderAcceptBid.selector, _bidId),
            _lender
        );

        return true;
    }

    /**
     * @dev This empty reserved space is put in place to allow future versions to add new
     * variables without shifting down storage in the inheritance chain.
     * See https://docs.openzeppelin.com/contracts/4.x/upgradeable#storage_gaps
     */
    uint256[50] private __gap;
}
