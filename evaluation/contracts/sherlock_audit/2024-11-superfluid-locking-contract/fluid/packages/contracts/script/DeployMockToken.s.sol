// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import { Script, console2 } from "forge-std/Script.sol";

import {
    ISuperfluid,
    ISuperToken,
    ISuperTokenFactory
} from "@superfluid-finance/ethereum-contracts/contracts/interfaces/superfluid/ISuperfluid.sol";
import { PureSuperToken } from "@superfluid-finance/ethereum-contracts/contracts/tokens/PureSuperToken.sol";

// forge script script/DeployMockToken.s.sol:DeployScript --rpc-url $RPC_URL --broadcast --verify -vvvv
contract DeployScript is Script {
    uint256 public constant INITIAL_SUPPLY = 1_000_000_000e18;
    string public constant NAME = "fake F**** Token";
    string public constant SYMBOL = "fF****";

    function run() public {
        ISuperfluid host = ISuperfluid(vm.envAddress("HOST_ADDRESS"));
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");

        vm.startBroadcast(deployerPrivateKey);

        ISuperTokenFactory factory = host.getSuperTokenFactory();

        PureSuperToken pureSuperTokenProxy = new PureSuperToken();
        factory.initializeCustomSuperToken(address(pureSuperTokenProxy));
        pureSuperTokenProxy.initialize(NAME, SYMBOL, INITIAL_SUPPLY);

        console2.log("Deployed FLUID Token Address : %s", address(pureSuperTokenProxy));
    }
}
