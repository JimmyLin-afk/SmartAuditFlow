pragma solidity ^0.8.0;

import "openzeppelin-contracts-next/contracts/token/ERC20/IERC20.sol";

/**
 * @dev Interface of the ERC20 standard as defined in the EIP. Does not include
 * the optional functions; to access them see {ERC20Detailed}.
 */
interface IERC20MintableBurnable is IERC20 {
  function mint(address account, uint256 amount) external;
  function burn(uint256 amount) external;
}
