// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "./IBlast.sol";

library BlastConstants {
    /// @notice Address of the Blast predeploy.
    address internal constant BLAST = 0x4300000000000000000000000000000000000002;

    /// @notice Address of the USDB predeploy.
    address internal constant USDB = 0x4300000000000000000000000000000000000003;

    /// @notice Address of the WETH predeploy.
    address internal constant WETH = 0x4300000000000000000000000000000000000004;

    /// @notice Address of the Blast points manager.
    address internal constant BLAST_POINTS = 0x2536FE9ab3F511540F2f9e2eC2A805005C3Dd800;
}