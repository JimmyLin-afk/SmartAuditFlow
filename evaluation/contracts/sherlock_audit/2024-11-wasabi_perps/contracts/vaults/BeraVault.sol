// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "./WasabiVault.sol";
import {IBeraVault} from "./IBeraVault.sol";
import {IInfrared} from "../bera/IInfrared.sol";
import {IRewardVault, IInfraredVault} from "../bera/IInfraredVault.sol";
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";

contract BeraVault is WasabiVault, IBeraVault {
    using SafeERC20 for IERC20;
    using Math for uint256;

    struct RewardStorage {
        IRewardVault rewardVault;
        uint256 rewardFeeBips;
        mapping(address => uint256) rewardFeeUserBalance;
        IInfraredVault infraredVault;
    }

    // @notice The slot where the RewardStorage struct is stored
    // @dev This equals bytes32(uint256(keccak256("wasabi.vault.reward_storage")) - 1)
    bytes32 private constant REWARD_STORAGE_SLOT = 0x3a98d39551c291449e156f1efe80f323dad9e74efefffe75144eae654edcfd08;

    IInfrared public constant INFRARED = IInfrared(0xb71b3DaEA39012Fb0f2B14D2a9C86da9292fC126);

    uint256 private constant HUNDRED_PERCENT = 10000;

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
    ) public override virtual initializer {
        __WasabiVault_init(_longPool, _shortPool, _addressProvider, _manager, _asset, name, symbol);
    }

    /// @notice Initialize the reward vaults
    function initializeRewardVaults() external onlyAdmin {
        RewardStorage storage rs = _getRewardStorage();
        rs.infraredVault = INFRARED.registerVault(address(this));
        rs.rewardVault = rs.infraredVault.rewardsVault();
        rs.rewardFeeBips = 1000; // 10%

        _approve(address(this), address(rs.infraredVault), type(uint256).max);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          GETTERS                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc IERC4626
    function maxWithdraw(address owner) public view override(ERC4626Upgradeable, IERC4626) returns (uint256) {
        return _convertToAssets(maxRedeem(owner), Math.Rounding.Floor);
    }

    /// @inheritdoc IERC4626
    function maxRedeem(address owner) public view override(ERC4626Upgradeable, IERC4626) returns (uint256) {
        uint256 balance = balanceOf(owner);
        if (balance == 0) return 0;

        uint256 partialFees = _computePartialFee(owner, balance);
        return partialFees + balance;
    }

    /// @inheritdoc IBeraVault
    function cumulativeBalanceOf(address account) public view returns (uint256) {
        RewardStorage storage rs = _getRewardStorage();
        uint256 unstakedBalance = balanceOf(account); // ERC4626 shares held by the user
        uint256 directStakedBalance = rs.rewardVault.balanceOf(account); // Shares staked by the user via RewardVault.stake
        uint256 infraredStakedBalance = rs.infraredVault.balanceOf(account); // Shares staked by the user via InfraredVault.stake
        uint256 rewardFeeBalance = rs.rewardFeeUserBalance[account]; // Shares staked by this vault via InfraredVault.stake
        return unstakedBalance + directStakedBalance + infraredStakedBalance + rewardFeeBalance;
    }

    /// @inheritdoc IBeraVault
    function getRewardFeeBips() public view returns (uint256) {
        return _getRewardStorage().rewardFeeBips;
    }

    /// @inheritdoc IBeraVault
    function getRewardVault() public view returns (IRewardVault) {
        return _getRewardStorage().rewardVault;
    }

    /// @inheritdoc IBeraVault
    function getInfraredVault() public view returns (IInfraredVault) {
        return _getRewardStorage().infraredVault;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                       ADMIN FUNCTIONS                      */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc IBeraVault
    function claimRewardFees(address _receiver) external onlyAdmin returns (uint256[] memory) {
        IInfraredVault infraredVault = getInfraredVault();

        // Get a list of all reward tokens from the InfraredVault
        address[] memory rewardTokens = infraredVault.getAllRewardTokens();
        uint256 rewardTokensLength = rewardTokens.length;
        uint256[] memory amounts = new uint256[](rewardTokensLength);

        // Claim all rewards from the InfraredVault
        infraredVault.getReward();

        // Record amounts received and transfer all rewards to the receiver
        for (uint256 i; i < rewardTokensLength; ) {
            IERC20 rewardToken = IERC20(rewardTokens[i]);
            uint256 amount = rewardToken.balanceOf(address(this));
            amounts[i] = amount;
            if (amount != 0) {
                rewardToken.safeTransfer(_receiver, amount);
            }
            unchecked {
                ++i;
            }
        }
        return amounts;
    }

    /// @inheritdoc IBeraVault
    function setRewardFeeBips(uint256 _rewardFeeBips) external onlyAdmin {
        if (_rewardFeeBips > 1000) revert InvalidFeeBips();
        RewardStorage storage rs = _getRewardStorage();
        emit RewardFeeBipsUpdated(rs.rewardFeeBips, _rewardFeeBips);
        rs.rewardFeeBips = _rewardFeeBips;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          WRITES                            */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc IERC20
    function transfer(address to, uint256 value) public override(ERC20Upgradeable, IERC20) returns (bool) {
        address sender = _msgSender();
        RewardStorage storage rs = _getRewardStorage();
        // If the sender is the InfraredVault or RewardVault, this must be an unstaking tx, so skip fee transfer
        // Otherwise, this is a share transfer, and we need to update the reward fee user balances
        if (sender != address(rs.rewardVault) && sender != address(rs.infraredVault)) {
            if (to == address(rs.infraredVault) || to == address(rs.rewardVault) || to == address(this)) {
                // Transferring tokens directly to any of the vaults will not stake them correctly
                revert ERC20InvalidReceiver(to);
            }
            uint256 partialFee = _computePartialFee(sender, value);
            rs.rewardFeeUserBalance[sender] -= partialFee;
            rs.rewardFeeUserBalance[to] += partialFee;
        }
        _transfer(sender, to, value);
        return true;
    }

    /// @inheritdoc IERC20
    function transferFrom(address from, address to, uint256 value) public override(ERC20Upgradeable, IERC20) returns (bool) {
        address spender = _msgSender();
        RewardStorage storage rs = _getRewardStorage();
        _spendAllowance(from, spender, value);
        // If the recipient is the InfraredVault or RewardVault, this must be a staking tx, so skip fee transfer
        // Otherwise, this is a share transfer, and we need to update the reward fee user balances
        if (to != address(rs.rewardVault) && to != address(rs.infraredVault)) {
            uint256 partialFee = _computePartialFee(from, value);
            rs.rewardFeeUserBalance[from] -= partialFee;
            rs.rewardFeeUserBalance[to] += partialFee;
        }
        _transfer(from, to, value);
        return true;
    }

    /// @inheritdoc WasabiVault
    /// @dev Actually BERA and WBERA, not ETH and WETH
    function depositEth(address receiver) public payable override(IWasabiVault, WasabiVault) nonReentrant returns (uint256) {
        address wberaAddress = addressProvider.getWethAddress();
        if (asset() != wberaAddress) revert CannotDepositEth();

        uint256 assets = msg.value;
        if (assets == 0) revert InvalidEthAmount();

        uint256 maxAssets = maxDeposit(receiver);
        if (assets > maxAssets) {
            revert ERC4626ExceededMaxDeposit(receiver, assets, maxAssets);
        }

        uint256 shares = previewDeposit(assets);

        IWETH(wberaAddress).deposit{value: assets}();

        // Mint shares to the user, minus a portion of the shares that will accrue the reward fee to the vault
        // The user needs to stake their shares in the InfraredVault on their own to earn iBGT
        RewardStorage storage rs = _getRewardStorage();
        IInfraredVault infraredVault = rs.infraredVault;
        uint256 rewardFee = shares.mulDiv(rs.rewardFeeBips, HUNDRED_PERCENT, Math.Rounding.Floor);
        _mint(receiver, shares - rewardFee);
        if (rewardFee != 0) {
            _mint(address(this), rewardFee);
            infraredVault.stake(rewardFee);
            rs.rewardFeeUserBalance[receiver] += rewardFee;
        }
        
        totalAssetValue += assets;
        emit Deposit(msg.sender, receiver, assets, shares);

        return shares;
    }

    /// @inheritdoc IBeraVault
    function unstakeShares() external nonReentrant {
        // Check if the caller has any shares staked on their behalf by this vault
        IRewardVault rewardVault = getRewardVault();
        uint256 stakedBalance = rewardVault.getDelegateStake(msg.sender, address(this));
        if (stakedBalance == 0) revert NoSharesToUnstake();
        // Withdraw the shares from the reward vault on their behalf
        rewardVault.delegateWithdraw(msg.sender, stakedBalance);
        // Transfer the shares to the caller
        _transfer(address(this), msg.sender, stakedBalance);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                        INTERNAL FUNCTIONS                  */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc ERC4626Upgradeable
    function _deposit(
        address caller,
        address receiver,
        uint256 assets,
        uint256 shares
    ) internal virtual override nonReentrant {
        if (assets == 0 || shares == 0) revert InvalidAmount();

        IERC20(asset()).safeTransferFrom(caller, address(this), assets);

        // Mint shares to the user, minus a portion of the shares that will accrue the reward fee to the vault
        // The user needs to stake their shares in the InfraredVault on their own to earn iBGT
        RewardStorage storage rs = _getRewardStorage();
        IInfraredVault infraredVault = rs.infraredVault;
        uint256 rewardFee = shares.mulDiv(rs.rewardFeeBips, HUNDRED_PERCENT, Math.Rounding.Floor);
        _mint(receiver, shares - rewardFee);
        if (rewardFee != 0) {
            _mint(address(this), rewardFee);
            infraredVault.stake(rewardFee);
            rs.rewardFeeUserBalance[receiver] += rewardFee;
        }

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

        uint256 feeWithdrawAmount = _getFeeWithdrawAmount(owner, shares);
        RewardStorage storage rs = _getRewardStorage();
        IInfraredVault infraredVault = rs.infraredVault;
        if (feeWithdrawAmount != 0) {
            infraredVault.withdraw(feeWithdrawAmount);
            rs.rewardFeeUserBalance[owner] -= feeWithdrawAmount;
            _burn(address(this), feeWithdrawAmount);
        }
        _burn(owner, shares - feeWithdrawAmount);

        totalAssetValue -= assets;

        if (totalSupply() == 0) {
            // Clean dust if no shares are left
            assets += totalAssetValue;
            totalAssetValue = 0;
        }

        IERC20(asset()).safeTransfer(receiver, assets);

        emit Withdraw(caller, receiver, owner, assets, shares);
    }

    // @notice Determine the portion of the fee shares to withdraw, given the cumulative shares the user wants to redeem
    function _getFeeWithdrawAmount(address owner, uint256 shares) internal view returns (uint256) {
        RewardStorage storage rs = _getRewardStorage();
        uint256 totalBalance = cumulativeBalanceOf(owner);
        uint256 totalFeeStake = rs.rewardFeeUserBalance[owner];
        if (totalBalance == shares) {
            // Handle full withdrawal
            return totalFeeStake;
        }
        uint256 feeWithdrawAmount = totalFeeStake.mulDiv(shares, totalBalance, Math.Rounding.Floor);
        if (feeWithdrawAmount > totalFeeStake) {
            // Handle edge case where rounding error causes feeWithdrawAmount to exceed totalFeeStake
            feeWithdrawAmount = totalFeeStake;
        }
        return feeWithdrawAmount;
    }

    /// @notice Determine the portion of the user's fee shares to transfer, given the non-fee shares the user wants to transfer
    function _computePartialFee(address owner, uint256 shares) internal view returns (uint256) {
        uint256 cumulativeBalance = cumulativeBalanceOf(owner);
        uint256 feeBalance = _getRewardStorage().rewardFeeUserBalance[owner];
        if (shares + feeBalance == cumulativeBalance) {
            return feeBalance;
        }

        uint256 totalNonFeeBalance = cumulativeBalance - feeBalance;
        return feeBalance.mulDiv(shares, totalNonFeeBalance, Math.Rounding.Floor);
    }

    function _getRewardStorage() internal pure returns (RewardStorage storage $) {
        assembly {
            $.slot := REWARD_STORAGE_SLOT
        }
    }
}