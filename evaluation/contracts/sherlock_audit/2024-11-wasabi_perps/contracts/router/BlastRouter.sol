// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "./WasabiRouter.sol";
import "../blast/AbstractBlastContract.sol";

contract BlastRouter is WasabiRouter, AbstractBlastContract {
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
    ) public override initializer {
        __WasabiRouter_init(_longPool, _shortPool, _weth, _manager, _swapRouter, _feeReceiver, _withdrawFeeBips);
        __AbstractBlastContract_init();
    }

    /// @dev claim all gas
    function claimGas(address contractAddress, address recipientOfGas) external onlyAdmin returns (uint256) {
        return _getBlast().claimMaxGas(contractAddress, recipientOfGas);
    }
}