// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "../blast/AbstractBlastContract.sol";
import "../admin/PerpManager.sol";

contract BlastPerpManager is PerpManager, AbstractBlastContract {

    /// @dev initializer for proxy
    function initialize() public override initializer {
        __AccessManager_init(msg.sender);
        __AbstractBlastContract_init();
    }
}