//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { ERC721 } from "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { SafeERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import { VVVAuthorizationRegistryChecker } from "contracts/auth/VVVAuthorizationRegistryChecker.sol";

contract VVVNodes is ERC721, VVVAuthorizationRegistryChecker {
    using SafeERC20 for IERC20;

    ///@notice Additional data for each token
    struct TokenData {
        uint256 unvestedAmount; //Remaining tokens to be vested, starts at 60% of $VVV initially locked in each node
        uint256 vestingSince; //timestamp of most recent token activation or claim
        uint256 lockedTransactionProcessingYield; //Remaining transaction yield to be unlocked, starts at 40% of the $VVV initially locked in each node.
        uint256 claimableAmount; //claimable $VVV across vesting, transaction, and launchpad yield sources
        uint256 amountToVestPerSecond; //amount of $VVV to vest per second
        uint256 stakedAmount; //total staked $VVV for the node
    }

    ///@notice The total number of nodes that can be minted
    uint256 public constant TOTAL_SUPPLY = 5000;

    ///@notice The vesting duration in seconds (2 years)
    uint256 public constant VESTING_DURATION = 2 * 365 * 24 * 60 * 60;

    ///@notice The address of the authorization registry
    address public authorizationRegistry;

    ///@notice Flag for whether nodes are soulbound
    bool public soulbound = true;

    ///@notice The baseURI for the token metadata
    string public baseURI;

    ///@notice Node activation threshold in staked $VVV
    uint256 public activationThreshold;

    ///@notice The maximum staked $VVV which can be considered for launchpad yield points
    uint256 public maxLaunchpadStakeAmount;

    ///@notice The current tokenId
    uint256 public tokenId;

    ///@notice Maps a TokenData struct to each tokenId
    mapping(uint256 => TokenData) public tokenData;

    ///@notice Emitted when accrued yield is claimed
    event Claim(uint256 indexed tokenId, uint256 amount);

    ///@notice Emitted when launchpad yield is deposited
    event DepositLaunchpadYield(uint256 indexed tokenId, uint256 amount);

    ///@notice Emitted when node is minted
    event Mint(
        uint256 indexed tokenId,
        address indexed recipient,
        uint256 unvestedAmount,
        uint256 lockedTransactionProcessingYield,
        uint256 amountToVestPerSecond
    );

    ///@notice Emitted when the node activation threshold is set
    event SetActivationThreshold(uint256 indexed activationThreshold);

    ///@notice Emitted when $VVV is staked
    event Stake(uint256 indexed tokenId, uint256 amount);

    ///@notice Emitted when $VVV is unstaked
    event Unstake(uint256 indexed tokenId, uint256 amount);

    ///@notice Emitted when some transaction processing yield is unlocked
    event UnlockTransactionProcessingYield(uint256 indexed tokenId, uint256 unlockedAmount);

    ///@notice Emitted when a token's vestingSince is updated
    event VestingSinceUpdated(uint256 indexed tokenId, uint256 newVestingSince);

    ///@notice Thrown when input array lengths are not matched
    error ArrayLengthMismatch();

    ///@notice Thrown when the caller is not the owner of the token
    error CallerIsNotTokenOwner();

    ///@notice Thrown when an operation is attempted on an unminted token
    error UnmintedTokenId(uint256 tokenId);

    ///@notice Thrown when a mint is attempted past the total supply
    error MaxSupplyReached();

    ///@notice Thrown when msg.value doesn't match the sum of amounts to be distributed to each node
    error MsgValueDistAmountMismatch();

    ///@notice Thrown when there are no claimable tokens for a node
    error NoClaimableTokens();

    ///@notice Thrown when a node transfer is attempted while nodes are soulbound
    error NodesAreSoulbound();

    ///@notice Thrown when there is an attempt to unlock transaction processing yield when 0 yield remains to be unlocked
    error NoRemainingUnlockableYield(uint256 tokenId);

    ///@notice Thrown when a native transfer fails
    error TransferFailed();

    ///@notice Thrown when an attempt is made to stake/unstake 0 $VVV
    error ZeroTokenTransfer();

    constructor(
        address _authorizationRegistry,
        string memory _newBaseURI,
        uint256 _activationThreshold
    ) ERC721("Multi-token Nodes", "NODES") VVVAuthorizationRegistryChecker(_authorizationRegistry) {
        activationThreshold = _activationThreshold;
        authorizationRegistry = _authorizationRegistry;
        baseURI = _newBaseURI;
    }

    ///@notice Mints a node of the input tier to the recipient
    function adminMint(address _recipient, uint256 _lockedTokens) external onlyAuthorized {
        ++tokenId;
        if (tokenId > TOTAL_SUPPLY) revert MaxSupplyReached();

        uint256 unvestedAmount = (_lockedTokens * 60) / 100;
        uint256 vestingSince;
        uint256 lockedTransactionProcessingYield = _lockedTokens - unvestedAmount;
        uint256 claimableAmount;
        uint256 amountToVestPerSecond = unvestedAmount / VESTING_DURATION;
        uint256 stakedAmount;

        tokenData[tokenId] = TokenData(
            unvestedAmount,
            vestingSince,
            lockedTransactionProcessingYield,
            claimableAmount,
            amountToVestPerSecond,
            stakedAmount
        );

        _mint(_recipient, tokenId);
        emit Mint(
            tokenId,
            _recipient,
            unvestedAmount,
            lockedTransactionProcessingYield,
            amountToVestPerSecond
        );
    }

    ///@notice Stakes $VVV, handles activation if amount added causes total staked to surpass activation threshold
    function stake(uint256 _tokenId) external payable {
        if (msg.value == 0) revert ZeroTokenTransfer();
        if (msg.sender != ownerOf(_tokenId)) revert CallerIsNotTokenOwner();
        TokenData storage token = tokenData[_tokenId];

        //if node is inactive and this stake activates it, set vestingSince to the current timestamp
        if (
            !_isNodeActive(token.stakedAmount, activationThreshold) &&
            _isNodeActive(msg.value + token.stakedAmount, activationThreshold)
        ) {
            token.vestingSince = block.timestamp;
            emit VestingSinceUpdated(_tokenId, token.vestingSince);
        }

        token.stakedAmount += msg.value;

        emit Stake(_tokenId, msg.value);
    }

    ///@notice Unstakes $VVV, handles deactivation if amount removed causes total staked to fall below activation threshold
    function unstake(uint256 _tokenId, uint256 _amount) external {
        if (_amount == 0) revert ZeroTokenTransfer();
        if (msg.sender != ownerOf(_tokenId)) revert CallerIsNotTokenOwner();
        TokenData storage token = tokenData[_tokenId];

        if (
            _isNodeActive(token.stakedAmount, activationThreshold) &&
            !_isNodeActive(token.stakedAmount - _amount, activationThreshold)
        ) {
            _updateClaimableFromVesting(token);
        }

        token.stakedAmount -= _amount;

        (bool success, ) = msg.sender.call{ value: _amount }("");
        if (!success) revert TransferFailed();

        emit Unstake(_tokenId, _amount);
    }

    ///@notice Allows a node owner to claim accrued yield
    function claim(uint256 _tokenId) public {
        if (msg.sender != ownerOf(_tokenId)) revert CallerIsNotTokenOwner();
        TokenData storage token = tokenData[_tokenId];

        _updateClaimableFromVesting(token);
        uint256 amountToClaim = token.claimableAmount;

        if (amountToClaim == 0) revert NoClaimableTokens();
        token.claimableAmount = 0;

        //update vestingSince to the current timestamp to maintain correct vesting calculations
        token.vestingSince = block.timestamp;

        (bool success, ) = msg.sender.call{ value: amountToClaim }("");
        if (!success) revert TransferFailed();

        emit VestingSinceUpdated(_tokenId, token.vestingSince);
        emit Claim(_tokenId, amountToClaim);
    }

    ///@notice Claims for all input tokenIds
    function batchClaim(uint256[] calldata _tokenIds) public {
        for (uint256 i = 0; i < _tokenIds.length; i++) {
            claim(_tokenIds[i]);
        }
    }

    ///@notice Deposits launchpad yield to selectedtoken
    function depositLaunchpadYield(
        uint256[] calldata _tokenIds,
        uint256[] calldata _amounts
    ) external payable onlyAuthorized {
        if (_tokenIds.length != _amounts.length) revert ArrayLengthMismatch();
        if (msg.value == 0) revert ZeroTokenTransfer();
        uint256 amountsSum;

        for (uint256 i = 0; i < _tokenIds.length; i++) {
            uint256 thisTokenId = _tokenIds[i];
            TokenData storage thisToken = tokenData[thisTokenId];

            if (thisToken.amountToVestPerSecond == 0) revert UnmintedTokenId(thisTokenId);

            uint256 thisAmount = _amounts[i];
            thisToken.claimableAmount += thisAmount;
            amountsSum += thisAmount;

            emit DepositLaunchpadYield(thisTokenId, thisAmount);
        }

        if (amountsSum != msg.value) revert MsgValueDistAmountMismatch();
    }

    ///@notice unlocks transaction processing yield for selected tokens
    function unlockTransactionProcessingYield(
        uint256[] calldata _tokenIds,
        uint256 _amountToUnlock
    ) external onlyAuthorized {
        for (uint256 i = 0; i < _tokenIds.length; ++i) {
            uint256 thisTokenId = _tokenIds[i];
            TokenData storage thisToken = tokenData[thisTokenId];

            //revert if a selected token has not been minted
            if (thisToken.amountToVestPerSecond == 0) revert UnmintedTokenId(thisTokenId);

            uint256 tokenLockedYield = thisToken.lockedTransactionProcessingYield;

            //revert if a selected token has no unlockable yield
            if (tokenLockedYield == 0) revert NoRemainingUnlockableYield(thisTokenId);

            //unlock either _amountToUnlock or the remaining unlockable yield if _amountToUnlock is greater than the remaining unlockable yield
            uint256 yieldToUnlock = _amountToUnlock > tokenLockedYield
                ? tokenLockedYield
                : _amountToUnlock;

            thisToken.lockedTransactionProcessingYield -= yieldToUnlock;
            thisToken.claimableAmount += yieldToUnlock;
            emit UnlockTransactionProcessingYield(thisTokenId, yieldToUnlock);
        }
    }

    ///@notice Sets the node activation threshold in staked $VVV
    function setActivationThreshold(uint256 _activationThreshold) external onlyAuthorized {
        if (_activationThreshold > activationThreshold) {
            //update claimable balances of nodes which will become inactive as a result of the threshold increase
            for (uint256 i = 1; i <= tokenId; i++) {
                TokenData storage token = tokenData[i];
                if (
                    _isNodeActive(token.stakedAmount, activationThreshold) &&
                    !_isNodeActive(token.stakedAmount, _activationThreshold)
                ) {
                    _updateClaimableFromVesting(token);
                }
            }
        } else if (_activationThreshold < activationThreshold) {
            //set vestingSince for nodes which will become active as a result of the threshold decrease
            for (uint256 i = 1; i <= tokenId; i++) {
                TokenData storage token = tokenData[i];
                if (
                    _isNodeActive(token.stakedAmount, _activationThreshold) &&
                    !_isNodeActive(token.stakedAmount, activationThreshold)
                ) {
                    token.vestingSince = block.timestamp;
                    emit VestingSinceUpdated(i, token.vestingSince);
                }
            }
        }

        activationThreshold = _activationThreshold;

        emit SetActivationThreshold(_activationThreshold);
    }

    ///@notice Sets the maximum staked $VVV which can be considered for launchpad yield points
    function setMaxLaunchpadStakeAmount(uint256 _maxLaunchpadStakeAmount) external onlyAuthorized {
        maxLaunchpadStakeAmount = _maxLaunchpadStakeAmount;
    }

    ///@notice Sets whether nodes are soulbound
    function setSoulbound(bool _soulbound) external onlyAuthorized {
        soulbound = _soulbound;
    }

    ///@notice Sets token URI for token of tokenId (placeholder)
    function setBaseURI(string calldata _newBaseURI) external onlyAuthorized {
        baseURI = _newBaseURI;
    }

    ///@notice Withdraws $VVV from the contract
    function withdraw(uint256 _amount) external onlyAuthorized {
        (bool success, ) = msg.sender.call{ value: _amount }("");
        if (!success) revert TransferFailed();
    }

    ///@notice returns whether a node of input tokenId is active
    function isNodeActive(uint256 _tokenId) external view returns (bool) {
        return _isNodeActive(tokenData[_tokenId].stakedAmount, activationThreshold);
    }

    ///@notice override for ERC721:_baseURI to set baseURI
    function _baseURI() internal view override returns (string memory) {
        return baseURI;
    }

    ///@notice Override of ERC721:_update to enforce soulbound restrictions when tokens are not being minted
    function _update(
        address _to,
        uint256 _tokenId,
        address _auth
    ) internal virtual override returns (address) {
        if (soulbound && _auth != address(0)) revert NodesAreSoulbound();
        return super._update(_to, _tokenId, _auth);
    }

    ///@notice utilized in claiming and deactivation, updates the claimable tokens accumulated from vesting
    function _updateClaimableFromVesting(TokenData storage _tokenData) private {
        uint256 currentVestedAmount = _calculateVestedTokens(_tokenData);

        _tokenData.unvestedAmount -= currentVestedAmount;
        _tokenData.claimableAmount += currentVestedAmount;
    }

    ///@notice calculates vested tokens for a given tokenId since the last timestamp (vestingSince) update (does not account for claimed)
    function _calculateVestedTokens(TokenData memory _tokenData) private view returns (uint256) {
        uint256 vestingSince = _tokenData.vestingSince;

        //if node is inactive, return 0 (no vesting will occur between time of deactivation and time at which this function is called while the node is still inactive)
        if (!_isNodeActive(_tokenData.stakedAmount, activationThreshold)) return 0;

        uint256 totalUnvestedAmount = _tokenData.unvestedAmount;
        uint256 amountToVestPerSecond = _tokenData.amountToVestPerSecond;

        uint256 timeBasedVestingAmount = (block.timestamp - vestingSince) * amountToVestPerSecond;
        uint256 currentVestedAmount = totalUnvestedAmount > timeBasedVestingAmount
            ? timeBasedVestingAmount
            : totalUnvestedAmount;

        return currentVestedAmount;
    }

    ///@notice Returns whether node is active based on whether it's staked $VVV is above the activation threshold.
    function _isNodeActive(
        uint256 _stakedVVVAmount,
        uint256 _activationThreshold
    ) private pure returns (bool) {
        return _stakedVVVAmount >= _activationThreshold;
    }
}
