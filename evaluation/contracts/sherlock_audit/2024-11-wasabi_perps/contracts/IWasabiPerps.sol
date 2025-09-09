// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "./vaults/IWasabiVault.sol";

interface IWasabiPerps {

    error LiquidationThresholdNotReached(); // 0xc4d82e43
    error InvalidSignature(); // 0x8baa579f
    error PositionAlreadyTaken(); // 0xe168e4db
    error SwapFunctionNeeded(); // 0xac8da8e3
    error OrderExpired(); // 0xc56873ba
    error InvalidOrder(); // 0xaf610693
    error PriceTargetNotReached(); // 0x5d5ce003
    error InvalidCurrency(); // 0xf5993428
    error InvalidTargetCurrency(); // 0x0415b9ce
    error InsufficientAmountProvided(); // 0xf948951e
    error PrincipalTooHigh(); // 0xd7cdb444
    error InsufficientPrincipalUsed(); // 0xb1084a42
    error InsufficientPrincipalRepaid(); // 0xb0f8fc9b
    error InsufficientCollateralReceived(); // 0x406220a9
    error InsufficientInterest(); // 0x0ffe80f0
    error TooMuchCollateralSpent(); // 0x1cbf0b89
    error SenderNotTrader(); // 0x79184208
    error InvalidPosition(); // 0xce7e065e
    error EthTransferFailed(uint256 amount, address _target); // 0xf733a609
    error InvalidVault(); // 0xd03a6320
    error VaultAlreadyExists(); // 0x04aabf33
    error ValueDeviatedTooMuch(); // 0x604e9173
    error EthReceivedForNonEthCurrency(); // 0x94427663
    error InvalidInterestAmount(); // 0xe749867e
    error InvalidInput(); // 0xb4fa3fb3

    event PositionOpened(
        uint256 positionId,
        address trader,
        address currency,
        address collateralCurrency,
        uint256 downPayment,
        uint256 principal,
        uint256 collateralAmount,
        uint256 feesToBePaid
    );

    event PositionClosed(
        uint256 id,
        address trader,
        uint256 payout,
        uint256 principalRepaid,
        uint256 interestPaid,
        uint256 feeAmount
    );

    event PositionClosedWithOrder(
        uint256 id,
        address trader,
        uint8 orderType,
        uint256 payout,
        uint256 principalRepaid,
        uint256 interestPaid,
        uint256 feeAmount
    );

    event PositionLiquidated(
        uint256 id,
        address trader,
        uint256 payout,
        uint256 principalRepaid,
        uint256 interestPaid,
        uint256 feeAmount
    );

    event PositionIncreased(
        uint256 id,
        address trader,
        uint256 downPaymentAdded,
        uint256 principalAdded,
        uint256 collateralAdded,
        uint256 feesAdded
    );

    event PositionDecreased(
        uint256 id,
        address trader,
        uint256 payout,
        uint256 principalRepaid,
        uint256 interestPaid,
        uint256 closeFee,
        uint256 pastFees,
        uint256 collateralReduced,
        uint256 downPaymentReduced
    );

    event PositionDecreasedWithOrder(
        uint256 id,
        address trader,
        uint8 orderType,
        uint256 payout,
        uint256 principalRepaid,
        uint256 interestPaid,
        uint256 closeFee,
        uint256 pastFees,
        uint256 collateralReduced,
        uint256 downPaymentReduced
    );

    event CollateralAdded(
        uint256 id,
        address trader,
        uint256 downPaymentAdded,
        uint256 collateralAdded,
        uint256 principalReduced,
        uint256 interestPaid
    );

    event CollateralRemoved(
        uint256 id,
        address trader,
        uint256 downPaymentReduced,
        uint256 collateralReduced,
        uint256 principalAdded
    );

    event NativeYieldClaimed(
        address vault,
        address token,
        uint256 amount
    );

    event InterestPaid(
        uint256 id,
        uint256 interestPaid,
        uint256 principalAdded,
        uint256 collateralReduced,
        uint256 downPaymentReduced
    );

    /// @dev Emitted when a new vault is created
    event NewVault(address indexed pool, address indexed asset, address vault);

    /// @dev Flag specifying whether to send WETH to the trader, send ETH to the trader, or deposit WETH to the vault
    enum PayoutType {
        WRAPPED,
        UNWRAPPED,
        VAULT_DEPOSIT
    }

    /// @dev Defines a function call
    struct FunctionCallData {
        address to;
        uint256 value;
        bytes data;
    }

    /// @dev Defines a position
    /// @param id The unique identifier for the position.
    /// @param trader The address of the trader who opened the position.
    /// @param currency The address of the currency to be paid for the position.
    /// @param collateralCurrency The address of the currency to be received for the position.
    /// @param lastFundingTimestamp The timestamp of the last funding payment.
    /// @param downPayment The initial down payment amount required to open the position (is in `currency` for long, `collateralCurrency` for short positions)
    /// @param principal The total principal amount to be borrowed for the position (is in `currency`)
    /// @param collateralAmount The total collateral amount to be received for the position (is in `collateralCurrency`)
    /// @param feesToBePaid The total fees to be paid for the position (is in `currency`)
    struct Position {
        uint256 id;
        address trader;
        address currency;
        address collateralCurrency;
        uint256 lastFundingTimestamp;
        uint256 downPayment;
        uint256 principal;
        uint256 collateralAmount;
        uint256 feesToBePaid;
    }

    /// @dev Defines a request to open a position.
    /// @param id The unique identifier for the position.
    /// @param currency The address of the currency to be paid for the position.
    /// @param targetCurrency The address of the currency to be received for the position.
    /// @param downPayment The initial down payment amount required to open the position (is in `currency` for long, `collateralCurrency` for short positions)
    /// @param principal The total principal amount to be borrowed for the position.
    /// @param minTargetAmount The minimum amount of target currency to be received for the position to be valid.
    /// @param expiration The timestamp when this position request expires.
    /// @param fee The fee to be paid for the position
    /// @param functionCallDataList A list of FunctionCallData structures representing functions to call to open the position.
    /// @param existingPosition The existing position to be increased, or an empty position if a new position is to be opened.
    /// @param referrer The address of the partner that referred the trader
    struct OpenPositionRequest {
        uint256 id;
        address currency;
        address targetCurrency;
        uint256 downPayment;
        uint256 principal;
        uint256 minTargetAmount;
        uint256 expiration;
        uint256 fee;
        FunctionCallData[] functionCallDataList;
        Position existingPosition;
        address referrer;
    }

    /// @dev Defines a request to add collateral to a position.
    /// @param amount The amount of collateral to add.
    /// @param interest The interest to be paid for the position.
    /// @param expiration The timestamp when this request expires.
    /// @param position The position to add collateral to.
    struct AddCollateralRequest {
        uint256 amount;
        uint256 interest;
        uint256 expiration;
        Position position;
    }

    /// @dev Defines a request to remove collateral from a position.
    /// @param amount The amount of collateral to remove.
    /// @param expiration The timestamp when this request expires.
    /// @param position The position to remove collateral from.
    struct RemoveCollateralRequest {
        uint256 amount;
        uint256 expiration;
        Position position;
    }

    /// @dev Defines the amounts to be paid when closing a position.
    /// @param payout The amount to be paid to the trader.
    /// @param collateralSold The amount of the collateral used to swap for principal.
    /// @param principalRepaid The amount of the principal to be repaid.
    /// @param interestPaid The amount of the interest to be paid.
    /// @param pastFees The amount of past fees to be paid.
    /// @param closeFee The amount of the close fee to be paid.
    /// @param liquidationFee The amount of the liquidation fee to be paid.
    /// @param downPaymentReduced The amount by which the down payment was reduced.
    /// @param collateralReduced The total amount by which the collateral was reduced. Not the same as `collateralSold` for shorts.
    struct CloseAmounts {
        uint256 payout;
        uint256 collateralSold;
        uint256 principalRepaid;
        uint256 interestPaid;
        uint256 pastFees;
        uint256 closeFee;
        uint256 liquidationFee;
        uint256 downPaymentReduced;
        uint256 collateralReduced;
    }

    /// @dev Defines an order to close a position.
    /// @param orderType The type of the order (0 = Take Profit, 1 = Stop Loss)
    /// @param positionId The unique identifier for the position.
    /// @param createdAt The timestamp when this order was created.
    /// @param expiration The timestamp when this order expires.
    /// @param makerAmount The amount that will be sold from the position (is in `position.collateralCurrency`)
    /// @param takerAmount The amount that will be bought to close the position (is in `position.currency`)
    /// @param executionFee The amount of the execution fee to be paid. (gas)
    struct ClosePositionOrder {
        uint8 orderType;
        uint256 positionId;
        uint256 createdAt;
        uint256 expiration;
        uint256 makerAmount;
        uint256 takerAmount;
        uint256 executionFee;
    }

    /// @dev Defines a request to close a position.
    /// @param expiration The timestamp when this position request expires.
    /// @param interest The interest to be paid for the position.
    /// @param amount The amount of collateral to sell (for longs) or amount of principal to buy back (for shorts), or 0 to fully close the position.
    /// @param position The position to be closed.
    /// @param functionCallDataList A list of FunctionCallData structures representing functions to call to close the position.
    /// @param referrer The address of the partner that referred the trader
    struct ClosePositionRequest {
        uint256 expiration;
        uint256 interest;
        uint256 amount;
        Position position;
        FunctionCallData[] functionCallDataList;
        address referrer;
    }

    /// @dev Defines the arguments needed for the internal close position function.
    /// @param _interest the interest amount to be paid
    /// @param _amount the amount of collateral to sell (for longs) or amount of principal to buy back (for shorts), or 0 to fully close the position.
    /// @param _executionFee the execution fee
    /// @param _payoutType whether to send WETH to the trader, send ETH, or deposit WETH to the vault
    /// @param _isLiquidation flag indicating if the close is a liquidation
    /// @param _referrer the address of the partner that referred the trader
    struct ClosePositionInternalArgs {
        uint256 _interest;
        uint256 _amount;
        uint256 _executionFee;
        PayoutType _payoutType;
        bool _isLiquidation;
        address _referrer;
    }

    /// @dev Defines a signature
    struct Signature {
        uint8 v;
        bytes32 r;
        bytes32 s;
    }

    /// @dev Opens a position
    /// @param _request the request to open a position
    /// @param _signature the signature of the request
    function openPosition(
        OpenPositionRequest calldata _request,
        Signature calldata _signature
    ) external payable returns (Position memory);

    /// @dev Opens a position on behalf of a user
    /// @param _request the request to open a position
    /// @param _signature the signature of the request
    /// @param _trader the address of the user for whom the position is opened
    function openPositionFor(
        OpenPositionRequest calldata _request,
        Signature calldata _signature,
        address _trader
    ) external payable returns (Position memory);

    /// @dev Adds collateral to a position
    /// @param _request the request to add collateral
    /// @param _signature the signature of the request
    function addCollateral(
        AddCollateralRequest calldata _request,
        Signature calldata _signature
    ) external payable returns (Position memory);

    /// @dev Removes collateral from a position
    /// @param _request the request to remove collateral
    /// @param _signature the signature of the request
    function removeCollateral(
        RemoveCollateralRequest calldata _request,
        Signature calldata _signature
    ) external payable returns (Position memory);

    /// @dev Closes a position
    /// @param _payoutType whether to send WETH to the trader, send ETH, or deposit WETH to the vault
    /// @param _request the request to close a position
    /// @param _signature the signature of the request
    function closePosition(
        PayoutType _payoutType,
        ClosePositionRequest calldata _request,
        Signature calldata _signature
    ) external payable;

    /// @dev Closes a position
    /// @param _payoutType whether to send WETH to the trader, send ETH, or deposit WETH to the vault
    /// @param _request the request to close a position
    /// @param _signature the signature of the request, signed by the ORDER_SIGNER_ROLE
    /// @param _order the order to close the position
    /// @param _orderSignature the signature of the order, signed by the owner of the position
    function closePosition(
        PayoutType _payoutType,
        ClosePositionRequest calldata _request,
        Signature calldata _signature,
        ClosePositionOrder calldata _order,
        Signature calldata _orderSignature
    ) external payable;

    /// @dev Liquidates a position
    /// @param _payoutType whether to send WETH to the trader, send ETH, or deposit WETH to the vault
    /// @param _interest the interest to be paid
    /// @param _position the position to liquidate
    /// @param _swapFunctions the swap functions to use to liquidate the position
    /// @param _referrer the address of the partner that referred the trader
    function liquidatePosition(
        PayoutType _payoutType,
        uint256 _interest,
        Position calldata _position,
        FunctionCallData[] calldata _swapFunctions,
        address _referrer
    ) external payable;

    /// @dev Records interest for a position and updates the position
    /// @notice Only callable by the InterestRecorder contract
    /// @param _positions the positions to record interest for
    /// @param _interests the interests to record
    /// @param _swapFunctions the swap functions to use for short positions
    function recordInterest(Position[] calldata _positions, uint256[] calldata _interests, FunctionCallData[] calldata _swapFunctions) external;

    /// @dev Returns the vault used for the given asset
    function getVault(address _asset) external view returns (IWasabiVault);

    /// @dev Adds a new vault
    function addVault(IWasabiVault _vault) external;

    /// @dev Adds a new quote token
    function addQuoteToken(address _token) external;
}