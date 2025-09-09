// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts/proxy/beacon/BeaconProxy.sol";
import "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

import "./IStakingAccountFactory.sol";
import "./StakingAccount.sol";
import "../admin/PerpManager.sol";

contract StakingAccountFactory is 
    IStakingAccountFactory, 
    UUPSUpgradeable, 
    OwnableUpgradeable, 
    ReentrancyGuardUpgradeable 
{
    using SafeERC20 for IERC20;
    
    UpgradeableBeacon public beacon;
    IWasabiPerps public longPool;
    IWasabiPerps public shortPool;

    mapping(address user => address stakingAccount) public userToStakingAccount;
    mapping(address token => IStakingAccount.StakingContract stakingContract) public tokenToStakingContract;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                         MODIFIERS                          */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Checks if the caller is one of the pool contracts
    modifier onlyPool() {
        if (msg.sender != address(longPool)) {
            // Nested checks save a little gas compared to using &&
            if (msg.sender != address(shortPool)) revert CallerNotPool();
        }
        _;
    }

    /// @dev Checks if the caller is an admin
    modifier onlyAdmin() {
        _getManager().isAdmin(msg.sender);
        _;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                        INITIALIZER                         */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /// @notice Initializes the factory
    /// @param _manager The PerpManager contract
    /// @param _longPool The BeraLongPool contract
    /// @param _shortPool The BeraShortPool contract
    function initialize(PerpManager _manager, IWasabiPerps _longPool, IWasabiPerps _shortPool) public initializer {
        __UUPSUpgradeable_init();
        __Ownable_init(address(_manager));
        __ReentrancyGuard_init();

        beacon = new UpgradeableBeacon(address(new StakingAccount()), address(this));
        longPool = _longPool;
        shortPool = _shortPool;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      ADMIN FUNCTIONS                       */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc IStakingAccountFactory
    function setStakingContractForToken(address _stakingToken, address _stakingContract, IStakingAccount.StakingType _stakingType) external onlyAdmin {
        tokenToStakingContract[_stakingToken] = IStakingAccount.StakingContract(_stakingContract, _stakingType);
    }

    /// @inheritdoc IStakingAccountFactory
    function upgradeBeacon(address _newImplementation) external onlyAdmin {
        beacon.upgradeTo(_newImplementation);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                       POOL FUNCTIONS                       */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc IStakingAccountFactory
    function stakePosition(IWasabiPerps.Position memory _position, IWasabiPerps.Position memory _existingPosition) public onlyPool {
        (IStakingAccount stakingAccount, IStakingAccount.StakingContract memory stakingContract)
            = _getStakingAccountAndContract(_position.trader, _position.collateralCurrency, true);

        IERC20 collateralToken = IERC20(_position.collateralCurrency);
        collateralToken.safeTransferFrom(msg.sender, address(stakingAccount), _position.collateralAmount - _existingPosition.collateralAmount);

        stakingAccount.stakePosition(_position, _existingPosition, stakingContract);

        emit StakedPosition(
            _position.trader,
            address(stakingAccount),
            stakingContract.contractAddress,
            stakingContract.stakingType,
            _position.id,
            _position.collateralAmount
        );
    }

    /// @inheritdoc IStakingAccountFactory
    function unstakePosition(IWasabiPerps.Position memory _position, uint256 _amount) public onlyPool {
        (IStakingAccount stakingAccount, IStakingAccount.StakingContract memory stakingContract)
            = _getStakingAccountAndContract(_position.trader, _position.collateralCurrency, false);

        _claimRewards(_position.collateralCurrency, _position.trader);
        stakingAccount.unstakePosition(_position, stakingContract, msg.sender, _amount);

        emit UnstakedPosition(
            _position.trader,
            address(stakingAccount),
            stakingContract.contractAddress,
            stakingContract.stakingType,
            _position.id,
            _position.collateralAmount
        );
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                           WRITES                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc IStakingAccountFactory
    function claimRewards(address _stakingToken) public nonReentrant {
        _claimRewards(_stakingToken, msg.sender);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      INTERNAL FUNCTIONS                    */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    function _claimRewards(address _stakingToken, address _user) internal {
        (IStakingAccount stakingAccount, IStakingAccount.StakingContract memory stakingContract)
            = _getStakingAccountAndContract(_user, _stakingToken, false);

        (IERC20[] memory tokens, uint256[] memory amounts) = stakingAccount.claimRewards(stakingContract);
        for (uint256 i; i < tokens.length; ) {
            emit StakingRewardsClaimed(
                _user,
                address(stakingAccount),
                stakingContract.contractAddress,
                stakingContract.stakingType,
                address(tokens[i]),
                amounts[i]
            );
            unchecked {
                i++;
            }
        }
    }

    function _getStakingAccountAndContract(
        address _user,
        address _token,
        bool _createIfNotDeployed
    ) internal returns (IStakingAccount, IStakingAccount.StakingContract memory) {
        address stakingAccount = userToStakingAccount[_user];
        if (stakingAccount == address(0)) {
            if (_createIfNotDeployed) {
                stakingAccount = address(new BeaconProxy(
                    address(beacon), 
                    abi.encodeWithSelector(StakingAccount.initialize.selector, _getManager(), _user)
                ));
                userToStakingAccount[_user] = stakingAccount;
                emit StakingAccountCreated(_user, stakingAccount);
            } else {
                revert StakingAccountNotDeployed(_user);
            }
        }
        IStakingAccount.StakingContract memory stakingContract = tokenToStakingContract[_token];
        if (stakingContract.contractAddress == address(0)) revert StakingContractNotSetForToken(_token);
        return (IStakingAccount(stakingAccount), stakingContract);
    }

    /// solhint-disable-next-line no-empty-blocks
    /// @inheritdoc UUPSUpgradeable
    function _authorizeUpgrade(address) internal view override onlyAdmin {}

    /// @dev returns the manager of the contract
    function _getManager() internal view returns (PerpManager) {
        return PerpManager(owner());
    }
}
