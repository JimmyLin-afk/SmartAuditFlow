// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;

import { Ownable } from "openzeppelin-contracts-next/contracts/access/Ownable.sol";

contract MockOwnable is Ownable {
  uint256 public protected;

  function protectedFunction(uint256 newProtected) external onlyOwner {
    protected = newProtected;
  }
}
