// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import { Script, console2 } from "forge-std/Script.sol";

import { IEPProgramManager } from "../../src/EPProgramManager.sol";

// forge script script/utils/UpdateProgramSigner.s.sol:UpdateProgramSigner --rpc-url $BASE_SEPOLIA_RPC_URL --broadcast -vvvv
contract UpdateProgramSigner is Script {
    function run() public {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        IEPProgramManager programManager = IEPProgramManager(vm.envAddress("EP_PROGRAM_MANAGER_ADDRESS"));

        address newSigner = 0xBc2cfCd4c615Ff1d06f1d07b37E3652b15bd40A2;

        vm.startBroadcast(deployerPrivateKey);
        programManager.updateProgramSigner(3756, newSigner);
        programManager.updateProgramSigner(5514, newSigner);
        programManager.updateProgramSigner(5515, newSigner);

        console2.log("Program %s updated with new signer : %s", 3756, newSigner);
        console2.log("Program %s updated with new signer : %s", 5514, newSigner);
        console2.log("Program %s updated with new signer : %s", 5515, newSigner);
    }
}
