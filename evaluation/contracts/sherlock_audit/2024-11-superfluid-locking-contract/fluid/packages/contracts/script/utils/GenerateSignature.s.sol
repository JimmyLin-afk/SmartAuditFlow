// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import { Script, console2 } from "forge-std/Script.sol";
import { ECDSA } from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

using ECDSA for bytes32;

// forge script script/utils/GenerateSignature.s.sol:GenerateSignature -vvvv
contract GenerateSignature is Script {
    uint256 public signerPrivateKey;

    function setUp() public {
        signerPrivateKey = vm.envUint("SIGNER_PRIVATE_KEY");
    }

    function run() public view {
        address user = 0x37dB1380669155d6080c04a5e6Db029E306cD964;
        uint256 unitsToGrant = 5;
        uint256 programId = 0;
        uint256 nonce = 1;

        _generateSignature(user, unitsToGrant, programId, nonce);
    }

    function _generateSignature(address _user, uint256 _unitsToGrant, uint256 _programId, uint256 _nonce)
        internal
        view
    {
        bytes32 message = keccak256(abi.encodePacked(_user, _unitsToGrant, _programId, _nonce));

        bytes32 digest = message.toEthSignedMessageHash();

        (uint8 v, bytes32 r, bytes32 s) = vm.sign(signerPrivateKey, digest);
        bytes memory validSignature = abi.encodePacked(r, s, v);

        console2.log("Message : ");
        console2.logBytes32(message);
        console2.log("");

        console2.log("Signed Message : ");
        console2.logBytes32(digest);
        console2.log("");

        console2.log("Signer Address : ");
        console2.log(vm.addr(signerPrivateKey));
        console2.log("");

        console2.log("Signature : ");
        console2.logBytes(validSignature);
        console2.log("");

        console2.log("Signature Length: ");
        console2.log(validSignature.length);
        console2.log("");
    }
}
