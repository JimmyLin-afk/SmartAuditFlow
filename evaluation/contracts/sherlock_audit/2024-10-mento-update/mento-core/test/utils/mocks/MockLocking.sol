pragma solidity ^0.8.0;

import "contracts/governance/locking/Locking.sol";

contract MockLocking {
  function initiateData(uint256 idLock, LibBrokenLine.Line memory line, address locker, address delegate) external {}
}
