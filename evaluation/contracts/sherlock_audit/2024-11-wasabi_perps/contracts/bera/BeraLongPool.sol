// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "../WasabiLongPool.sol";
import "./IBeraPool.sol";
import "./IStakingAccountFactory.sol";

contract BeraLongPool is WasabiLongPool, IBeraPool {
    using SafeERC20 for IERC20;
    
    struct StakingStorage {
        mapping(uint256 => bool) isStaked;
    }

    // @notice The slot where the StakingStorage struct is stored
    // @dev This equals bytes32(uint256(keccak256("wasabi.pool.staking_storage")) - 1)
    bytes32 private constant STAKING_STORAGE_SLOT = 0xd7d4cbe20940e82007292c0d2939a485e1e8c3c257e382f7fa5a10e24698ab5d;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                        INITIALIZER                         */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev initializer for proxy
    /// @param _addressProvider address provider contract
    /// @param _manager the PerpManager contract
    function initialize(IAddressProvider _addressProvider, PerpManager _manager) public override initializer {
        __WasabiLongPool_init(_addressProvider, _manager);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                           GETTERS                          */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc IBeraPool
    function isPositionStaked(uint256 _positionId) public view returns (bool) {
        StakingStorage storage $ = _getStakingStorage();
        return $.isStaked[_positionId];
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                           WRITES                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc IWasabiPerps
    /// @dev Does not stake the position, and reverts if editing an existing staked position
    function openPositionFor(
        OpenPositionRequest calldata _request,
        Signature calldata _signature,
        address _trader
    ) public payable override(IWasabiPerps, WasabiLongPool) returns (Position memory) {
        _checkPartialStake(_request.existingPosition.id, false);
        return super.openPositionFor(_request, _signature, _trader);
    }

    /// @inheritdoc IBeraPool
    function openPositionAndStake(
        OpenPositionRequest calldata _request, 
        Signature calldata _signature
    ) external payable returns (Position memory) {
        return openPositionAndStakeFor(_request, _signature, msg.sender);
    }

    /// @inheritdoc IBeraPool
    function openPositionAndStakeFor(
        OpenPositionRequest calldata _request,
        Signature calldata _signature,
        address _trader
    ) public payable returns (Position memory) {
        _checkPartialStake(_request.existingPosition.id, true);
        Position memory position = super.openPositionFor(_request, _signature, _trader);
        _stake(position, _request.existingPosition);
        return position;
    }

    /// @inheritdoc IBeraPool
    function stakePosition(Position memory _position) external {
        if (_position.trader != msg.sender) revert SenderNotTrader();
        _stake(_position, _getEmptyPosition());
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      INTERNAL FUNCTIONS                    */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Closes a given position
    /// @param _args the close position arguments
    /// @param _position the position
    /// @param _swapFunctions the swap functions
    /// @return closeAmounts the close amounts
    function _closePositionInternal(
        ClosePositionInternalArgs memory _args,
        Position calldata _position,
        FunctionCallData[] calldata _swapFunctions
    ) internal override returns(CloseAmounts memory closeAmounts) {
        if (_args._amount == 0 || _args._amount > _position.collateralAmount) {
            _args._amount = _position.collateralAmount;
        }
        _unstakeIfStaked(_position, _args._amount);
        return super._closePositionInternal(
            _args, 
            _position, 
            _swapFunctions
        );
    }

    /// @dev Stakes the collateral of a given position via the staking account factory
    /// @param _position the position to stake
    /// @param _existingPosition the existing position, if editing an existing position
    function _stake(Position memory _position, Position memory _existingPosition) internal nonReentrant {
        if (_existingPosition.id == 0) {
            if (isPositionStaked(_position.id)) revert PositionAlreadyStaked(_position.id);
        }
        _getStakingStorage().isStaked[_position.id] = true;

        IStakingAccountFactory factory = _getStakingAccountFactory();
        IERC20(_position.collateralCurrency).forceApprove(
            address(factory),
            _position.collateralAmount - _existingPosition.collateralAmount
        );

        factory.stakePosition(_position, _existingPosition);
    }

    /// @dev Unstakes the collateral of a given position via the staking account factory if it is staked
    /// @param _position the position to unstake
    /// @param _amount the amount to unstake
    function _unstakeIfStaked(Position memory _position, uint256 _amount) internal {
        if (isPositionStaked(_position.id)) {
            _getStakingAccountFactory().unstakePosition(_position, _amount);
            if (_amount == _position.collateralAmount) {
                _getStakingStorage().isStaked[_position.id] = false;
            }
        }
    }

    /// @dev Returns the staking account factory from the address provider
    /// @return factory the staking account factory
    function _getStakingAccountFactory() internal view returns (IStakingAccountFactory) {
        return IStakingAccountFactory(addressProvider.getStakingAccountFactory());
    }

    /// @dev Returns the staking storage struct
    /// @return $ the staking storage
    function _getStakingStorage() internal pure returns (StakingStorage storage $) {
        assembly {
            $.slot := STAKING_STORAGE_SLOT
        }
    }

    function _getEmptyPosition() internal pure returns (Position memory) {
        return Position(0, address(0), address(0), address(0), 0, 0, 0, 0, 0);
    }

    function _checkPartialStake(uint256 _positionId, bool _shouldBeStaked) internal view {
        if (_positionId != 0) {
            if (isPositionStaked(_positionId) != _shouldBeStaked) revert CannotPartiallyStakePosition();
        }
    }
}
