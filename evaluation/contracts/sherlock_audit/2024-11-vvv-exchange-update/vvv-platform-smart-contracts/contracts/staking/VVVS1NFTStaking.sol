///SPDX-License-Identifier: MIT
pragma solidity 0.8.23;

import { IERC721 } from "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import { IERC721Receiver } from "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";

///@notice Handles staking for S1NFT
contract VVVS1NFTStaking is IERC721Receiver {
    IERC721 public s1nft;

    ///@notice enum to represent the duration options of a stake
    enum StakeDuration {
        DAYS_180,
        DAYS_360
    }

    /**
        @notice struct to store the stake data
        @param tokenId the staked token ID
        @param startTime the timestamp at which the stake began
        @param duration the StakeDuration enum entry that corresponds to the duration of the stake
    */
    struct StakeData {
        uint256 tokenId;
        uint256 startTime;
        StakeDuration duration;
    }

    /// @notice mapping of stake duration to the number of seconds it lasts
    mapping(StakeDuration => uint256) public stakeDurations;

    ///@notice mapping of user to their stakes
    mapping(address => StakeData[]) public stakes;

    ///@notice mapping of tokenId to the address that staked it
    mapping(uint256 => address) public stakerOfId;

    ///@notice emitted when a S1 NFT is staked
    event Stake(address indexed user, uint256 indexed tokenId, StakeDuration indexed stakingDuration);

    ///@notice emitted when a S1 NFT is unstaked
    event Unstake(address indexed user, uint256 indexed tokenId);

    ///@notice thrown when a caller for stake or unstake is not the owner of the token
    error NotTokenOwner();

    ///@notice thrown when a user attempts to unstake before the stake duration has elapsed
    error StakeLocked();

    constructor(address _s1nft) {
        s1nft = IERC721(_s1nft);
        stakeDurations[StakeDuration.DAYS_180] = 180 days;
        stakeDurations[StakeDuration.DAYS_360] = 360 days;
    }

    ///@notice stakes an S1 NFT. setApprovalForAll must be called before this function
    function stake(uint256 _tokenId, StakeDuration _stakingDuration) external {
        stakes[msg.sender].push(StakeData(_tokenId, block.timestamp, _stakingDuration));
        stakerOfId[_tokenId] = msg.sender;

        s1nft.safeTransferFrom(msg.sender, address(this), _tokenId);

        emit Stake(msg.sender, _tokenId, _stakingDuration);
    }

    ///@notice unstakes an S1 NFT
    function unstake(uint256 _tokenId) external {
        if (stakerOfId[_tokenId] != msg.sender) revert NotTokenOwner();
        StakeData[] storage thisStakes = stakes[msg.sender];

        //check if the stake is locked, if not remove the stake from the user's array and set the stakerOfId to address(0)
        for (uint256 i = 0; i < thisStakes.length; i++) {
            StakeData storage thisStake = thisStakes[i];
            if (thisStake.tokenId == _tokenId) {
                if (block.timestamp < thisStake.startTime + stakeDurations[thisStake.duration]) {
                    revert StakeLocked();
                }
                thisStake = thisStakes[thisStakes.length - 1];
                thisStakes.pop();
                stakerOfId[_tokenId] = address(0);
                break;
            }
        }

        s1nft.safeTransferFrom(address(this), msg.sender, _tokenId);

        emit Unstake(msg.sender, _tokenId);
    }

    ///@notice returns an array of the user's current stakes
    function getStakes(address _user) external view returns (StakeData[] memory) {
        return stakes[_user];
    }

    ///@notice Handles the receipt of an NFT
    function onERC721Received(
        address operator,
        address from,
        uint256 tokenId,
        bytes calldata data
    ) external override returns (bytes4) {
        return IERC721Receiver.onERC721Received.selector;
    }
}
