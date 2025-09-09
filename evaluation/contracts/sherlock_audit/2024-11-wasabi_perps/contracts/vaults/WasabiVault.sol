// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/extensions/ERC4626Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts/utils/StorageSlot.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

import "./IWasabiVault.sol";
import "../IWasabiPerps.sol";
import "../PerpUtils.sol";
import "../addressProvider/IAddressProvider.sol";
import "../admin/PerpManager.sol";
import "../admin/Roles.sol";
import "../strategy/IStrategy.sol";
import "../weth/IWETH.sol";

contract WasabiVault is 
    IWasabiVault, UUPSUpgradeable, OwnableUpgradeable, ERC4626Upgradeable, ReentrancyGuardUpgradeable 
{
    using SafeERC20 for IERC20;

    /// @custom:oz-renamed-from pool
    IWasabiPerps public _deprecated_pool;
    /// @dev The total value of the assets deposited, including assets borrowed by the pools and admin
    uint256 public totalAssetValue;
    /// @dev The address provider
    IAddressProvider public addressProvider;
    /// @dev The Wasabi long pool
    IWasabiPerps public longPool;
    /// @dev The Wasabi short pool
    IWasabiPerps public shortPool;
    /// @dev Mapping from strategy address to the amount owed to the vault for the strategy
    mapping(address => uint256) public strategyDebt;
    /// @dev The fee charged on interest in basis points
    uint256 public interestFeeBips;

    uint256 private constant LEVERAGE_DENOMINATOR = 100;
    uint256 private constant BPS_DENOMINATOR = 10000;
    uint256 private constant MAX_INTEREST_FEE_BIPS = 2000; // 20%

    // @notice The slot where the deposit cap is stored, if set
    // @dev This equals bytes32(uint256(keccak256("wasabi.vault.max_deposit")) - 1)
    bytes32 private constant DEPOSIT_CAP_SLOT = 0x5f64ef5afc66734d661a0e9d6aa10a8d47dcf2c1c681696cce952f8ef9115384;

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /// @dev Initializer for proxy
    /// @notice This function should only be called to initialize a new vault
    /// @param _longPool The WasabiLongPool contract
    /// @param _shortPool The WasabiShortPool contract
    /// @param _addressProvider The address provider
    /// @param _manager The PerpManager contract that will own this vault
    /// @param _asset The asset
    /// @param name The name of the vault
    /// @param symbol The symbol of the vault
    function initialize(
        IWasabiPerps _longPool,
        IWasabiPerps _shortPool,
        IAddressProvider _addressProvider,
        PerpManager _manager,
        IERC20 _asset,
        string memory name,
        string memory symbol
    ) public virtual initializer {
        __WasabiVault_init(_longPool, _shortPool, _addressProvider, _manager, _asset, name, symbol);
    }

    function __WasabiVault_init(
        IWasabiPerps _longPool, 
        IWasabiPerps _shortPool, 
        IAddressProvider _addressProvider, 
        PerpManager _manager, 
        IERC20 _asset, 
        string memory name, 
        string memory symbol
    ) public onlyInitializing {
        __ERC20_init(name, symbol);
        __Ownable_init(address(_manager));
        __ERC4626_init(_asset);
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        addressProvider = _addressProvider;
        longPool = _longPool;
        shortPool = _shortPool;
        interestFeeBips = 1000;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                         MODIFIERS                          */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Checks if the caller is one of the pool contracts
    modifier onlyPool() {
        if (msg.sender != address(shortPool)) {
            // Nested checks save a little gas compared to using &&
            if (msg.sender != address(longPool)) revert CallerNotPool();
        }
        _;
    }

    /// @dev Checks if the caller is an admin
    modifier onlyAdmin() {
        _getManager().isAdmin(msg.sender);
        _;
    }

    /// @dev Checks if the caller has the correct role
    modifier onlyRole(uint64 roleId) {
        _getManager().checkRole(roleId, msg.sender);
        _;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          GETTERS                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc ERC4626Upgradeable
    function totalAssets() public view override(ERC4626Upgradeable, IERC4626) returns (uint256) {
        return totalAssetValue;
    }

    /// @inheritdoc ERC4626Upgradeable
    function maxDeposit(address) public view override(ERC4626Upgradeable, IERC4626) returns (uint256) {
        uint256 depositCap = _getDepositCap();
        if (depositCap == type(uint256).max) {
            return type(uint256).max;
        }
        if (totalAssetValue >= depositCap) {
            return 0;
        }
        return depositCap - totalAssetValue;
    }

    /// @inheritdoc IWasabiVault
    function getDepositCap() external view returns (uint256) {
        return _getDepositCap();
    }

    /// @inheritdoc IWasabiVault
    function getPoolAddress(bool _long) external view returns (address) {
        return _long ? address(longPool) : address(shortPool);
    }

    /// @inheritdoc IWasabiVault
    function checkMaxLeverage(uint256 _downPayment, uint256 _total) external view {
        if (_total * LEVERAGE_DENOMINATOR > _getDebtController().maxLeverage() * _downPayment) {
            revert PrincipalTooHigh();
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                           WRITES                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /** @dev See {IERC4626-deposit}. */
    function depositEth(address receiver) public payable virtual nonReentrant returns (uint256) {
        address wethAddress = addressProvider.getWethAddress();
        if (asset() != wethAddress) revert CannotDepositEth();

        uint256 assets = msg.value;
        if (assets == 0) revert InvalidEthAmount();

        uint256 maxAssets = maxDeposit(receiver);
        if (assets > maxAssets) {
            revert ERC4626ExceededMaxDeposit(receiver, assets, maxAssets);
        }

        uint256 shares = previewDeposit(assets);

        IWETH(wethAddress).deposit{value: assets}();

        _mint(receiver, shares);
        totalAssetValue += assets;
        emit Deposit(msg.sender, receiver, assets, shares);

        return shares;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                        POOL FUNCTIONS                      */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc IWasabiVault
    function borrow(uint256 _amount) external onlyPool {
        _borrow(msg.sender, _amount);
    }

    /// @inheritdoc IWasabiVault
    function recordRepayment(uint256 _totalRepaid, uint256 _principal, bool _isLiquidation) external onlyPool {
        if (_totalRepaid < _principal) {
            // Only liquidations can cause bad debt
            if (!_isLiquidation) revert InsufficientPrincipalRepaid();
            uint256 loss = _principal - _totalRepaid;
            totalAssetValue -= loss;
        } else {
            uint256 interestPaid = _totalRepaid - _principal;
            // Mint interest fee shares to the fee receiver
            _handleInterestFee(interestPaid);
            totalAssetValue += interestPaid;
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                       ADMIN FUNCTIONS                      */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc IWasabiVault
    function strategyDeposit(address _strategy, uint256 _depositAmount) external onlyAdmin {
        // Claim any interest earned on previous deposits first
        _strategyClaim(_strategy);
        // Send the deposit amount to the strategy contract
        _borrow(_strategy, _depositAmount);
        // Update the strategy debt
        strategyDebt[_strategy] += _depositAmount;
        // Trigger the deposit logic in the strategy contract and get the resulting collateral asset and amount
        (address collateral, uint256 collateralAmount) = IStrategy(_strategy).deposit(_depositAmount);
        // Emit the event
        emit StrategyDeposit(_strategy, collateral, _depositAmount, collateralAmount);
    }

    /// @inheritdoc IWasabiVault
    function strategyWithdraw(address _strategy, uint256 _withdrawAmount) external onlyAdmin {
        // Claim any interest earned on deposits first
        _strategyClaim(_strategy);
        // Check the withdraw amount - 0 is used to withdraw all deposits
        if (_withdrawAmount == 0) {
            _withdrawAmount = strategyDebt[_strategy];
        }
        if (_withdrawAmount > strategyDebt[_strategy]) {
            revert AmountExceedsDebt();
        }
        // Withdraw the assets from the strategy contract
        (address collateral, uint256 collateralSold) = IStrategy(_strategy).withdraw(_withdrawAmount);
        // Update the strategy debt
        strategyDebt[_strategy] -= _withdrawAmount;
        // Emit the event
        emit StrategyWithdraw(_strategy, collateral, _withdrawAmount, collateralSold);
    }

    /// @inheritdoc IWasabiVault
    function strategyClaim(address _strategy) external onlyAdmin {
        _strategyClaim(_strategy);
    }

    /// @inheritdoc IWasabiVault
    function donate(uint256 _amount) external onlyRole(Roles.VAULT_ADMIN_ROLE) {
        if (_amount == 0) revert InvalidAmount();
        IERC20(asset()).safeTransferFrom(msg.sender, address(this), _amount);
        totalAssetValue += _amount;
        emit NativeYieldClaimed(asset(), _amount);
    }

    /// @inheritdoc IWasabiVault
    function cleanDust() external onlyRole(Roles.VAULT_ADMIN_ROLE) {
        if (totalSupply() == 0 && totalAssetValue > 0) {
            uint256 assets = totalAssetValue;
            totalAssetValue = 0;
            IERC20(asset()).safeTransfer(msg.sender, assets);
        } else {
            revert NoDustToClean();
        }
    }

    /// @inheritdoc IWasabiVault
    function setDepositCap(uint256 _newDepositCap) external onlyRole(Roles.VAULT_ADMIN_ROLE) {
        StorageSlot.getUint256Slot(DEPOSIT_CAP_SLOT).value = _newDepositCap;
        emit DepositCapUpdated(_newDepositCap);
    }

    /// @inheritdoc IWasabiVault
    function setInterestFeeBips(uint256 _newInterestFeeBips) external onlyRole(Roles.VAULT_ADMIN_ROLE) {
        if (_newInterestFeeBips > MAX_INTEREST_FEE_BIPS) {
            revert InterestFeeTooHigh();
        }
        interestFeeBips = _newInterestFeeBips;
        emit InterestFeeBipsUpdated(_newInterestFeeBips);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      INTERNAL FUNCTIONS                    */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc UUPSUpgradeable
    function _authorizeUpgrade(address) internal override onlyAdmin {}

    /// @inheritdoc ERC4626Upgradeable
    function _deposit(
        address caller,
        address receiver,
        uint256 assets,
        uint256 shares
    ) internal virtual override nonReentrant {
        if (assets == 0 || shares == 0) revert InvalidAmount();

        IERC20(asset()).safeTransferFrom(caller, address(this), assets);

        _mint(receiver, shares);
        totalAssetValue += assets;
        emit Deposit(caller, receiver, assets, shares);
    }

    /// @inheritdoc ERC4626Upgradeable
    function _withdraw(
        address caller,
        address receiver,
        address owner,
        uint256 assets,
        uint256 shares
    ) internal virtual override nonReentrant {
        if (assets == 0 || shares == 0) revert InvalidAmount();

        if (caller != owner) {
            if (caller != address(addressProvider.getWasabiRouter())) {
                _spendAllowance(owner, caller, shares);
            }
        }

        _burn(owner, shares);

        totalAssetValue -= assets;

        if (totalSupply() == 0) {
            // Clean dust if no shares are left
            assets += totalAssetValue;
            totalAssetValue = 0;
        }

        IERC20(asset()).safeTransfer(receiver, assets);

        emit Withdraw(caller, receiver, owner, assets, shares);
    }

    function _borrow(address _receiver, uint256 _amount) internal {
        IERC20 assetToken = IERC20(asset());
        if (assetToken.balanceOf(address(this)) < _amount) {
            revert InsufficientAvailablePrincipal();
        }
        assetToken.safeTransfer(_receiver, _amount);
    }

    function _strategyClaim(address _strategy) internal {
        // Get the interest earned since the last observed amount
        uint256 interestReceived = IStrategy(_strategy).getNewInterest(strategyDebt[_strategy]);
        if (interestReceived == 0) return;
        // Get the collateral asset from the strategy
        address collateral = IStrategy(_strategy).collateralAsset();
        // Increment both the totalAssetValue and strategyDebt, since interest was earned but not paid yet
        totalAssetValue += interestReceived;
        strategyDebt[_strategy] += interestReceived;
        // Mint interest fee shares to the fee receiver
        _handleInterestFee(interestReceived);
        // Emit the event
        emit StrategyClaim(_strategy, collateral, interestReceived);
    }

    function _handleInterestFee(uint256 _interestAmount) internal {
        if (interestFeeBips != 0 && _interestAmount != 0) {
            address feeReceiver = _getFeeReceiver();
            uint256 interestFeeShares = _convertToShares(_interestAmount * interestFeeBips / BPS_DENOMINATOR, Math.Rounding.Floor);
            if (interestFeeShares != 0) {
                _mint(feeReceiver, interestFeeShares);
            }
            emit InterestReceived(_interestAmount, interestFeeShares, feeReceiver);
        }
    }

    /// @dev returns the manager of the contract
    function _getManager() internal view returns (PerpManager) {
        return PerpManager(owner());
    }

    /// @dev returns the WETH address
    function _getWethAddress() internal view returns (address) {
        return addressProvider.getWethAddress();
    }

    /// @dev returns the debt controller
    function _getDebtController() internal view returns (IDebtController) {
        return addressProvider.getDebtController();
    }

    /// @dev returns the fee receiver
    function _getFeeReceiver() internal view returns (address) {
        return addressProvider.getFeeReceiver();
    }

    /// @dev returns the deposit cap
    function _getDepositCap() internal view returns (uint256) {
        uint256 depositCap = StorageSlot.getUint256Slot(DEPOSIT_CAP_SLOT).value;
        return depositCap == 0 ? type(uint256).max : depositCap;
    }
}