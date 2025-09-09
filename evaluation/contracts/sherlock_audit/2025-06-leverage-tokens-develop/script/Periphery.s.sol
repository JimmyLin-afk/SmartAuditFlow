// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Script, console} from "forge-std/Script.sol";

import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {IMorpho} from "@morpho-blue/interfaces/IMorpho.sol";

import {SwapAdapter} from "src/periphery/SwapAdapter.sol";
import {ISwapAdapter} from "src/interfaces/periphery/ISwapAdapter.sol";
import {LeverageRouter} from "src/periphery/LeverageRouter.sol";
import {MorphoLendingAdapter} from "src/lending/MorphoLendingAdapter.sol";
import {MorphoLendingAdapterFactory} from "src/lending/MorphoLendingAdapterFactory.sol";
import {ILeverageManager} from "src/interfaces/ILeverageManager.sol";
import {IEtherFiL2ModeSyncPool} from "src/interfaces/periphery/IEtherFiL2ModeSyncPool.sol";
import {DeployConstants} from "./DeployConstants.sol";

contract PeripheryDeploy is Script {
    function run() public {
        address deployerAddress = msg.sender;

        console.log("BlockNumber: ", block.number);
        console.log("ChainId: ", block.chainid);
        console.log("DeployerAddress: ", deployerAddress);

        console.log("Deploying...");

        vm.startBroadcast();

        SwapAdapter swapAdapter = new SwapAdapter();
        console.log("SwapAdapter deployed at: ", address(swapAdapter));

        LeverageRouter leverageRouter = new LeverageRouter(
            ILeverageManager(DeployConstants.LEVERAGE_MANAGER),
            IMorpho(DeployConstants.MORPHO),
            ISwapAdapter(swapAdapter)
        );
        console.log("LeverageRouter deployed at: ", address(leverageRouter));

        vm.stopBroadcast();
    }
}
