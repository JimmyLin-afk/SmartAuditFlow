// SPDX-License-Identifier: BUSL-1.1
pragma solidity 0.8.26;

import "@berachain/pol-contracts/src/pol/interfaces/IDistributor.sol";
import "@berachain/pol-contracts/src/pol/interfaces/IBeraChef.sol";
import "@berachain/pol-contracts/src/pol/interfaces/IBlockRewardController.sol";
import "@berachain/pol-contracts/src/pol/interfaces/IRewardVault.sol";
import "@berachain/pol-contracts/src/libraries/Utils.sol";
import { AccessControlUpgradeable } from "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import { UUPSUpgradeable } from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import { ReentrancyGuardUpgradeable } from "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import { SafeERC20, IERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import { FixedPointMathLib } from "solady/src/utils/FixedPointMathLib.sol";

contract BerachainDistributorMock is
    IDistributor,
    ReentrancyGuardUpgradeable,
    AccessControlUpgradeable,
    UUPSUpgradeable
{
    using Utils for bytes4;
    using Utils for address;
    using SafeERC20 for IERC20;

    /// @notice The MANAGER role.
    bytes32 public constant MANAGER_ROLE = keccak256("MANAGER_ROLE");

    /// @dev Represents 100%. Chosen to be less granular.
    uint96 internal constant ONE_HUNDRED_PERCENT = 1e4;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          STORAGE                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @notice The BeraChef contract that we are getting the reward allocation from.
    IBeraChef public beraChef;

    /// @notice The rewards controller contract that we are getting the rewards rate from.
    /// @dev And is responsible for minting the BGT token.
    IBlockRewardController public blockRewardController;

    /// @notice The BGT token contract that we are distributing to the reward allocation receivers.
    address public bgt;

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(
        address _berachef,
        address _bgt,
        address _blockRewardController,
        address _governance
    )
        external
        initializer
    {
        __AccessControl_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        _grantRole(DEFAULT_ADMIN_ROLE, _governance);
        beraChef = IBeraChef(_berachef);
        bgt = _bgt;
        blockRewardController = IBlockRewardController(_blockRewardController);
    }

    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) { }

    function distributeFor(
        uint64 nextTimestamp,
        uint64,
        bytes calldata pubkey,
        bytes32[] calldata,
        bytes32[] calldata
    ) external {
        distributeFor(nextTimestamp, pubkey);
    }

    function distributeFor(
        uint64 nextTimestamp,
        bytes calldata pubkey
    ) public nonReentrant() {
        // Process the rewards with the block rewards controller for the specified block number.
        // Its dependent on the beraChef being ready, if not it will return zero rewards for the current block.
        uint256 rewardRate = blockRewardController.processRewards(pubkey, nextTimestamp, beraChef.isReady());
        if (rewardRate == 0) {
            // If berachef is not ready (genesis) or there aren't rewards to distribute, skip. This will skip since
            // there is no default reward allocation.
            return;
        }

        // Activate the queued reward allocation if it is ready.
        beraChef.activateReadyQueuedRewardAllocation(pubkey);

        // Get the active reward allocation for the validator.
        // This will return the default reward allocation if the validator does not have an active reward allocation.
        IBeraChef.RewardAllocation memory ra = beraChef.getActiveRewardAllocation(pubkey);
        uint256 totalRewardDistributed;

        IBeraChef.Weight[] memory weights = ra.weights;
        uint256 length = weights.length;
        for (uint256 i; i < length;) {
            IBeraChef.Weight memory weight = weights[i];
            address receiver = weight.receiver;

            uint256 rewardAmount;
            if (i == length - 1) {
                rewardAmount = rewardRate - totalRewardDistributed;
            } else {
                // Calculate the reward for the receiver: (rewards * weightPercentage / ONE_HUNDRED_PERCENT).
                rewardAmount =
                    FixedPointMathLib.fullMulDiv(rewardRate, weight.percentageNumerator, ONE_HUNDRED_PERCENT);
                totalRewardDistributed += rewardAmount;
            }

            // The reward vault will pull the rewards from this contract so we can keep the approvals for the
            // soul bound token BGT clean.
            IERC20(bgt).safeIncreaseAllowance(receiver, rewardAmount);

            // Notify the receiver of the reward.
            IRewardVault(receiver).notifyRewardAmount(pubkey, rewardAmount);

            emit Distributed(pubkey, nextTimestamp, receiver, rewardAmount);

            unchecked {
                ++i;
            }
        }
    }
}