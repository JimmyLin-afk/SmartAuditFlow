// SPDX-License-Identifier: SEE LICENSE IN LICENSE
pragma solidity >=0.5.17 <0.8.19;
pragma experimental ABIEncoderV2;

interface IGoodDollar {
  function mint(address to, uint256 amount) external;

  function burn(uint256 amount) external;

  function safeTransferFrom(address from, address to, uint256 value) external;

  function addMinter(address _minter) external;

  function isMinter(address account) external view returns (bool);

  function balanceOf(address account) external view returns (uint256);

  // slither-disable-next-line erc721-interface
  function approve(address spender, uint256 amount) external returns (bool);
}

interface IDistributionHelper {
  function onDistribution(uint256 _amount) external;
}
