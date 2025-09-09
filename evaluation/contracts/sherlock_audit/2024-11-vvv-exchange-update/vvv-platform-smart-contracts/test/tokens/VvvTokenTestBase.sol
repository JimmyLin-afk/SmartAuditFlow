//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/**
 * @dev Base for testing VvvToken.sol
 */
import "lib/forge-std/src/Test.sol";
import { VVVToken } from "contracts/tokens/VvvToken.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";

contract VVVTokenTestBase is Test {
    VVVAuthorizationRegistry AuthRegistry;
    VVVToken vvvToken;

    uint256 public initialSupply = 1000000;
    uint256 public cap = initialSupply * 10;

    uint256 public deployerKey = 1234;
    uint256 public tokenMintManagerKey = 1235;
    address public deployer = vm.addr(deployerKey);
    address public tokenMintManager = vm.addr(tokenMintManagerKey);

    bytes32 tokenMintManagerRole = keccak256("TOKEN_MINT_MANAGER_ROLE");
    uint48 defaultAdminTransferDelay = 1 days;
}
