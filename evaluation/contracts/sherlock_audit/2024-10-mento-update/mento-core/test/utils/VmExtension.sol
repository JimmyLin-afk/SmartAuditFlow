// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase, max-line-length, no-inline-assembly

import { Vm } from "forge-std/Vm.sol";
import { ECDSA } from "openzeppelin-contracts-next/contracts/utils/cryptography/ECDSA.sol";
import { Strings } from "openzeppelin-contracts-next/contracts/utils/Strings.sol";

library VmExtension {
  /// @dev moves `block.number` and `block.timestamp` in sync
  /// @param vm The forge Vm
  /// @param blocks The number of blocks that will be moved
  function timeTravel(Vm vm, uint256 blocks) internal {
    uint256 time = blocks * 5;
    vm.roll(block.number + blocks);
    vm.warp(block.timestamp + time);
  }

  /// @dev build the KYC message hash and sign it with the provided pk
  /// @param vm The forge Vm
  /// @param signer The PK to sign the message with
  /// @param account The account to sign the message for
  /// @param credential KYC credentials
  /// @param validUntil KYC valid until this timestamp
  /// @param approvedAt KYC approved at this timestamp
  function validKycSignature(
    Vm vm,
    uint256 signer,
    address account,
    string memory credential,
    uint256 validUntil,
    uint256 approvedAt
  ) internal pure returns (bytes memory) {
    bytes32 signedMessageHash = ECDSA.toEthSignedMessageHash(
      abi.encodePacked(
        Strings.toHexString(uint256(uint160(account)), 20),
        ";",
        "fractalId",
        ";",
        Strings.toString(approvedAt),
        ";",
        Strings.toString(validUntil),
        ";",
        credential
      )
    );

    (uint8 v, bytes32 r, bytes32 s) = vm.sign(signer, signedMessageHash);
    return abi.encodePacked(r, s, v);
  }

  /// @dev Constructs a digital signature from the given components (v, r, s).
  /// @param v The recovery byte, a part of the signature (usually 1 byte).
  /// @param r The first 32 bytes of the signature, representing the R value in ECDSA.
  /// @param s The next 32 bytes of the signature, representing the S value in ECDSA.
  /// @return signature A 65-byte long digital signature composed of r, s, and v.
  function constructSignature(Vm, uint8 v, bytes32 r, bytes32 s) internal pure returns (bytes memory signature) {
    signature = new bytes(65);

    assembly {
      mstore(add(signature, 32), r)
    }

    assembly {
      mstore(add(signature, 64), s)
    }

    signature[64] = bytes1(v);

    return signature;
  }
}
