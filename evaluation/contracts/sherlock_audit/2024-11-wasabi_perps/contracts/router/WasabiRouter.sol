// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/cryptography/EIP712Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/Address.sol";

import "./IWasabiRouter.sol";
import "../IWasabiPerps.sol";
import "../vaults/IWasabiVault.sol";
import "../admin/PerpManager.sol";
import "../admin/Roles.sol";
import "../weth/IWETH.sol";
import "../Hash.sol";

contract WasabiRouter is
    IWasabiRouter,
    UUPSUpgradeable,
    ReentrancyGuardUpgradeable,
    EIP712Upgradeable,
    OwnableUpgradeable
{
    using Hash for IWasabiPerps.OpenPositionRequest;
    using Hash for IWasabiPerps.AddCollateralRequest;
    using SafeERC20 for IERC20;
    using Address for address;
    using Address for address payable;

    /// @dev Wasabi long pool contract
    IWasabiPerps public longPool;
    /// @dev Wasabi short pool contract
    IWasabiPerps public shortPool;
    /// @dev The address of the swap router (i.e., Uniswap/Thruster)
    address public swapRouter;
    /// @dev The Wrapped ETH contract
    IWETH public weth;
    /// @dev The fee to be charged on vault withdrawals if no swap is performed (in bips)
    uint256 public withdrawFeeBips;
    /// @dev The address to receive withdrawal fees
    address public feeReceiver;

    /**
     * @dev Checks if the caller has the correct role
     */
    modifier onlyRole(uint64 roleId) {
        _getManager().checkRole(roleId, msg.sender);
        _;
    }

    /**
     * @dev Checks if the caller is an admin
     */
    modifier onlyAdmin() {
        _getManager().isAdmin(msg.sender);
        _;
    }

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    receive() external payable {
        if (msg.sender != swapRouter && msg.sender != address(weth)) 
            revert InvalidETHReceived();
    }

    /// @dev Initializes the router as per UUPSUpgradeable
    /// @param _longPool The long pool address
    /// @param _shortPool The short pool address
    /// @param _weth The WETH address
    /// @param _manager The PerpManager address
    /// @param _swapRouter The swap router address
    /// @param _feeReceiver The address to receive withdrawal fees
    /// @param _withdrawFeeBips The fee to be charged on vault withdrawals if no swap is performed (in bips)
    function initialize(
        IWasabiPerps _longPool,
        IWasabiPerps _shortPool,
        IWETH _weth,
        PerpManager _manager,
        address _swapRouter,
        address _feeReceiver,
        uint256 _withdrawFeeBips
    ) public virtual initializer {
        __WasabiRouter_init(_longPool, _shortPool, _weth, _manager, _swapRouter, _feeReceiver, _withdrawFeeBips);
    }

    function __WasabiRouter_init(
        IWasabiPerps _longPool,
        IWasabiPerps _shortPool,
        IWETH _weth,
        PerpManager _manager,
        address _swapRouter,
        address _feeReceiver,
        uint256 _withdrawFeeBips
    ) public onlyInitializing {
        __UUPSUpgradeable_init();
        __ReentrancyGuard_init();
        __EIP712_init("WasabiRouter", "1");
        __Ownable_init(address(_manager));

        longPool = _longPool;
        shortPool = _shortPool;
        weth = _weth;
        swapRouter = _swapRouter;
        feeReceiver = _feeReceiver;
        withdrawFeeBips = _withdrawFeeBips;
    }

    /// @inheritdoc IWasabiRouter
    function openPosition(
        IWasabiPerps _pool,
        IWasabiPerps.OpenPositionRequest calldata _request,
        IWasabiPerps.Signature calldata _signature
    ) external nonReentrant {
        _openPositionInternal(_pool, _request, _signature, msg.sender, 0);
    }

    /// @inheritdoc IWasabiRouter
    function openPosition(
        IWasabiPerps _pool,
        IWasabiPerps.OpenPositionRequest calldata _request,
        IWasabiPerps.Signature calldata _signature,
        IWasabiPerps.Signature calldata _traderSignature,
        uint256 _executionFee
    ) external onlyRole(Roles.ORDER_EXECUTOR_ROLE) nonReentrant {
        IWasabiPerps.OpenPositionRequest memory traderRequest = IWasabiPerps
            .OpenPositionRequest(
                _request.id,
                _request.currency,
                _request.targetCurrency,
                _request.downPayment,
                _request.principal,
                _request.minTargetAmount,
                _request.expiration,
                _request.fee,
                new IWasabiPerps.FunctionCallData[](0),
                _request.existingPosition,
                _request.referrer
            );
        address trader = _recoverSigner(traderRequest.hash(), _traderSignature);
        _openPositionInternal(_pool, _request, _signature, trader, _executionFee);
    }

    /// @inheritdoc IWasabiRouter
    function addCollateral(
        IWasabiPerps _pool,
        IWasabiPerps.AddCollateralRequest calldata _request,
        IWasabiPerps.Signature calldata _signature
    ) external nonReentrant {
        _addCollateralInternal(_pool, _request, _signature, msg.sender, 0);
    }

    /// @inheritdoc IWasabiRouter
    function swapVaultToVault(
        uint256 _amount,
        address _tokenIn,
        address _tokenOut,
        bytes calldata _swapCalldata
    ) external nonReentrant {
        // Withdraw tokenIn from vault on user's behalf
        _withdrawFromVault(_tokenIn, _amount);

        // Perform the swap
        _swapInternal(_tokenIn, _amount, _swapCalldata);

        // Deposit tokenOut into vault on user's behalf
        _depositToVault(_tokenOut, IERC20(_tokenOut).balanceOf(address(this)));

        // If full amount of tokenIn was not used, return it to the vault
        _depositToVault(_tokenIn, IERC20(_tokenIn).balanceOf(address(this)));
    }

    /// @inheritdoc IWasabiRouter
    function swapVaultToToken(
        uint256 _amount,
        address _tokenIn,
        address _tokenOut,
        bytes calldata _swapCalldata
    ) external nonReentrant {
        // Withdraw tokenIn from vault on user's behalf
        _withdrawFromVault(_tokenIn, _amount);

        if (_tokenIn != _tokenOut) {
            if (_tokenOut == address(0) && _tokenIn == address(weth)) {
                // Unwrap WETH to ETH
                weth.withdraw(_amount);
                _takeWithdrawFee(_tokenOut, _amount);
            } else {
                // Perform the swap (should send tokenOut directly to user)
                _swapInternal(_tokenIn, _amount, _swapCalldata);
            }
        } else {
            // Transfer the withdrawn assets to user (minus withdraw fee)
            _takeWithdrawFee(_tokenOut, _amount);
        }

        // If full amount of tokenIn was not used, return it to the vault
        _depositToVault(_tokenIn, IERC20(_tokenIn).balanceOf(address(this)));
    }

    /// @inheritdoc IWasabiRouter
    function swapTokenToVault(
        uint256 _amount,
        address _tokenIn,
        address _tokenOut,
        bytes calldata _swapCalldata
    ) external payable nonReentrant {
        // Check if paying in native ETH
        bool isETHSwap = msg.value != 0;

        // Transfer tokenIn from the user (unless paying in ETH)
        if (isETHSwap) {
            if (_tokenIn != address(weth)) revert InvalidETHReceived();
        } else {
            IERC20(_tokenIn).safeTransferFrom(msg.sender, address(this), _amount);
        }

        if (_tokenIn != _tokenOut) {
            // Perform the swap
            _swapInternal(_tokenIn, _amount, _swapCalldata);
        } else if (isETHSwap) {
            // Wrap the ETH received before depositing to the WETH vault
            weth.deposit{value: msg.value}();
        }

        // Deposit tokenOut into vault on user's behalf
        _depositToVault(_tokenOut, IERC20(_tokenOut).balanceOf(address(this)));

        // If full amount of tokenIn was not used, return it to the user
        if (isETHSwap) {
            uint256 amountRemaining = address(this).balance;
            if (amountRemaining != 0) {
                payable(msg.sender).sendValue(amountRemaining);
            }
        } else {
            uint256 amountRemaining = IERC20(_tokenIn).balanceOf(address(this));
            if (amountRemaining != 0) {
                IERC20(_tokenIn).safeTransfer(msg.sender, amountRemaining);
            }
        }
    }

    /// @inheritdoc IWasabiRouter
    function sweepToken(address _token) external onlyAdmin {
        if (_token == address(0)) {
            payable(msg.sender).sendValue(address(this).balance);
        } else {
            IERC20(_token).safeTransfer(msg.sender, IERC20(_token).balanceOf(address(this)));
        }
    }

    /// @inheritdoc IWasabiRouter
    function setSwapRouter(
        address _newSwapRouter
    ) external onlyAdmin {
        emit SwapRouterUpdated(swapRouter, _newSwapRouter);
        swapRouter = _newSwapRouter;
    }

    /// @inheritdoc IWasabiRouter
    function setWETH(
        IWETH _newWETH
    ) external onlyAdmin {
        weth = _newWETH;
    }

    /// @inheritdoc IWasabiRouter
    function setFeeReceiver(
        address _newFeeReceiver
    ) external onlyAdmin {
        feeReceiver = _newFeeReceiver;
    }

    /// @inheritdoc IWasabiRouter
    function setWithdrawFeeBips(
        uint256 _feeBips
    ) external onlyAdmin {
        if (_feeBips > 10000) revert InvalidFeeBips();
        emit WithdrawFeeUpdated(withdrawFeeBips, _feeBips);
        withdrawFeeBips = _feeBips;
    }

    function _openPositionInternal(
        IWasabiPerps _pool,
        IWasabiPerps.OpenPositionRequest calldata _request,
        IWasabiPerps.Signature calldata _signature,
        address _trader,
        uint256 _executionFee
    ) internal {
        if (_pool != longPool) {
            // Nested checks save a little gas over && operator
            if (_pool != shortPool) revert InvalidPool();
        }

        // Currency to withdraw from vault for payment - always the quote currency
        address currency = _pool == longPool
            ? _request.currency
            : _request.targetCurrency;
        uint256 amount = _request.downPayment + _request.fee + _executionFee;

        // Vault to withdraw from
        IWasabiVault vault = _pool.getVault(currency);
        vault.withdraw(amount, address(this), _trader);

        // If the pool is not approved to transfer the currency from the router, approve it
        if (
            IERC20(currency).allowance(address(this), address(_pool)) == 0
        ) {
            IERC20(currency).forceApprove(address(_pool), type(uint256).max);
        }

        // Open the position on behalf of the trader
        _pool.openPositionFor(_request, _signature, _trader);

        // Transfer the execution fee
        if (_executionFee != 0) {
            IERC20(currency).safeTransfer(
                msg.sender,
                _executionFee
            );
        }
    }

    function _addCollateralInternal(
        IWasabiPerps _pool,
        IWasabiPerps.AddCollateralRequest calldata _request,
        IWasabiPerps.Signature calldata _signature,
        address _trader,
        uint256 _executionFee
    ) internal {
        if (_pool != longPool) {
            // Nested checks save a little gas over && operator
            if (_pool != shortPool) revert InvalidPool();
        }

        // Currency to withdraw from vault for payment - always the quote currency
        address currency = _pool == longPool
            ? _request.position.currency
            : _request.position.collateralCurrency;
        uint256 amount = _request.amount + _executionFee;

        // Vault to withdraw from
        IWasabiVault vault = _pool.getVault(currency);
        vault.withdraw(amount, address(this), _trader);

        // If the pool is not approved to transfer the currency from the router, approve it
        if (
            IERC20(currency).allowance(address(this), address(_pool)) == 0
        ) {
            IERC20(currency).forceApprove(address(_pool), type(uint256).max);
        }

        // Add collateral to the position
        _pool.addCollateral(_request, _signature);

        // Transfer the execution fee
        if (_executionFee != 0) {
            IERC20(currency).safeTransfer(
                msg.sender,
                _executionFee
            );
        }
    }

    function _swapInternal(
        address _tokenIn,
        uint256 _amount,
        bytes calldata _swapCalldata
    ) internal {
        if (msg.value == 0) {
            IERC20 token = IERC20(_tokenIn);
            token.forceApprove(swapRouter, _amount);
        }
        swapRouter.functionCallWithValue(_swapCalldata, msg.value);
    }

    function _withdrawFromVault(
        address _asset,
        uint256 _amount
    ) internal {
        IWasabiVault vault = shortPool.getVault(_asset);
        vault.withdraw(_amount, address(this), msg.sender);
    }

    function _depositToVault(
        address _asset,
        uint256 _amount
    ) internal {
        if (_amount > 0) {
            IWasabiVault vault = shortPool.getVault(_asset);
            IERC20(_asset).forceApprove(address(vault), _amount);
            vault.deposit(_amount, msg.sender);
        }
    }

    function _takeWithdrawFee(
        address _tokenOut,
        uint256 _amount
    ) internal {
        if (feeReceiver == address(0)) {
            revert FeeReceiverNotSet();
        }
        uint256 feeAmount = _amount * withdrawFeeBips / 10000;
        if (_tokenOut == address(0)) {
            if (feeAmount != 0) {
                payable(feeReceiver).sendValue(feeAmount);
            }
            payable(msg.sender).sendValue(_amount - feeAmount);
        } else {
            if (feeAmount != 0) {
                IERC20(_tokenOut).safeTransfer(feeReceiver, feeAmount);
            }
            IERC20(_tokenOut).safeTransfer(msg.sender, _amount - feeAmount);
        }
    }

    /// @dev Checks if the signer for the given structHash and signature is the expected signer
    /// @param _structHash the struct hash
    /// @param _signature the signature
    function _recoverSigner(
        bytes32 _structHash,
        IWasabiPerps.Signature calldata _signature
    ) internal view returns (address signer) {
        bytes32 typedDataHash = _hashTypedDataV4(_structHash);
        signer = ecrecover(
            typedDataHash,
            _signature.v,
            _signature.r,
            _signature.s
        );

        if (signer == address(0)) {
            revert InvalidSignature();
        }
    }

    /// @inheritdoc UUPSUpgradeable
    function _authorizeUpgrade(address) internal view override onlyAdmin {}

    /// @dev returns the manager of the contract
    function _getManager() internal view returns (PerpManager) {
        return PerpManager(owner());
    }
}
