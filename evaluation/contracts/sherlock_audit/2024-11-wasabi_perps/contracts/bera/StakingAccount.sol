// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

import "./IStakingAccount.sol";
import "./IStakingAccountFactory.sol";
import "./IInfraredVault.sol";
import "../admin/PerpManager.sol";

contract StakingAccount is IStakingAccount, OwnableUpgradeable, ReentrancyGuardUpgradeable {
    using SafeERC20 for IERC20;

    address public accountHolder;
    IStakingAccountFactory public factory;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                         MODIFIERS                          */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Checks if the caller is an admin
    modifier onlyAdmin() {
        _getManager().isAdmin(msg.sender);
        _;
    }

    /// @dev Checks if the position trader is the account holder
    modifier onlyAccountHolder(address _trader) {
        if (_trader != accountHolder) revert TraderNotAccountHolder();
        _;
    }

    /// @dev Checks if the caller is the StakingAccountFactory
    modifier onlyFactory() {
        if (msg.sender != address(factory)) revert CallerNotFactory();
        _;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                        INITIALIZER                         */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(PerpManager _manager, address _accountHolder) public initializer {
        __Ownable_init(address(_manager));
        __ReentrancyGuard_init();
        factory = IStakingAccountFactory(msg.sender);
        accountHolder = _accountHolder;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                     FACTORY FUNCTIONS                      */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc IStakingAccount
    function stakePosition(
        IWasabiPerps.Position memory _position,
        IWasabiPerps.Position memory _existingPosition,
        StakingContract memory _stakingContract
    ) external onlyFactory onlyAccountHolder(_position.trader) {
        IERC20 collateralToken = IERC20(_position.collateralCurrency);
        uint256 stakeAmount = _position.collateralAmount - _existingPosition.collateralAmount;
        collateralToken.forceApprove(_stakingContract.contractAddress, stakeAmount);

        if (_stakingContract.stakingType == StakingType.INFRARED) {
            if (stakeAmount > 0) {
                IInfraredVault(_stakingContract.contractAddress).stake(stakeAmount);
            }
        } else {
            revert StakingTypeNotSupported();
        }
    }

    /// @inheritdoc IStakingAccount
    function unstakePosition(
        IWasabiPerps.Position memory _position,
        StakingContract memory _stakingContract,
        address _pool,
        uint256 _amount
    ) external onlyFactory onlyAccountHolder(_position.trader) {
        if (_amount == 0) {
            _amount = _position.collateralAmount;
        }
        if (_stakingContract.stakingType == StakingType.INFRARED) {
            IInfraredVault(_stakingContract.contractAddress).withdraw(_amount);
        } else {
            revert StakingTypeNotSupported();
        }

        IERC20 collateralToken = IERC20(_position.collateralCurrency);
        collateralToken.safeTransfer(_pool, _amount);
    }

    /// @inheritdoc IStakingAccount
    function claimRewards(StakingContract memory _stakingContract) external onlyFactory returns (IERC20[] memory, uint256[] memory) {
        address[] memory allRewardTokens;
        if (_stakingContract.stakingType == StakingType.INFRARED) {
            allRewardTokens = IInfraredVault(_stakingContract.contractAddress).getAllRewardTokens();
            IInfraredVault(_stakingContract.contractAddress).getReward();
        } else {
            revert StakingTypeNotSupported();
        }
        
        // Use dynamic arrays to store tokens and amounts
        IERC20[] memory tempTokens = new IERC20[](allRewardTokens.length);
        uint256[] memory tempAmounts = new uint256[](allRewardTokens.length);
        uint256 numRewards;

        for (uint256 i; i < allRewardTokens.length; ) {
            IERC20 token = IERC20(allRewardTokens[i]);
            uint256 amount = token.balanceOf(address(this));
            if (amount > 0) {
                tempTokens[numRewards] = token;
                tempAmounts[numRewards] = amount;
                token.safeTransfer(accountHolder, amount);
                numRewards++;
            }
            unchecked {
                i++;
            }
        }

        // Create fixed-size arrays with the correct size
        IERC20[] memory tokens = new IERC20[](numRewards);
        uint256[] memory amounts = new uint256[](numRewards);

        // Copy the data to the fixed-size arrays
        for (uint256 i; i < numRewards; ) {
            tokens[i] = tempTokens[i];
            amounts[i] = tempAmounts[i];
            unchecked {
                i++;
            }
        }

        return (tokens, amounts);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      INTERNAL FUNCTIONS                    */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev returns the manager of the contract
    function _getManager() internal view returns (PerpManager) {
        return PerpManager(owner());
    }
}