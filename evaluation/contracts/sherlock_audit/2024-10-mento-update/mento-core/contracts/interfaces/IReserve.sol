// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity >0.5.13 <0.9;

interface IReserve {
  function setTobinTaxStalenessThreshold(uint256) external;

  function addToken(address) external returns (bool);

  function removeToken(address, uint256) external returns (bool);

  function transferGold(address payable, uint256) external returns (bool);

  function transferExchangeGold(address payable, uint256) external returns (bool);

  function transferCollateralAsset(address collateralAsset, address payable to, uint256 value) external returns (bool);

  function getReserveGoldBalance() external view returns (uint256);

  function getUnfrozenReserveGoldBalance() external view returns (uint256);

  function getOrComputeTobinTax() external returns (uint256, uint256);

  function getTokens() external view returns (address[] memory);

  function getReserveRatio() external view returns (uint256);

  function addExchangeSpender(address) external;

  function removeExchangeSpender(address, uint256) external;

  function addSpender(address) external;

  function removeSpender(address) external;

  function isStableAsset(address) external view returns (bool);

  function isCollateralAsset(address) external view returns (bool);

  function getDailySpendingRatioForCollateralAsset(address collateralAsset) external view returns (uint256);

  function isExchangeSpender(address exchange) external view returns (bool);

  function addCollateralAsset(address asset) external returns (bool);

  function transferExchangeCollateralAsset(
    address collateralAsset,
    address payable to,
    uint256 value
  ) external returns (bool);

  function initialize(
    address registryAddress,
    uint256 _tobinTaxStalenessThreshold,
    uint256 _spendingRatioForCelo,
    uint256 _frozenGold,
    uint256 _frozenDays,
    bytes32[] calldata _assetAllocationSymbols,
    uint256[] calldata _assetAllocationWeights,
    uint256 _tobinTax,
    uint256 _tobinTaxReserveRatio,
    address[] calldata _collateralAssets,
    uint256[] calldata _collateralAssetDailySpendingRatios
  ) external;

  /// @notice IOwnable:
  function transferOwnership(address newOwner) external;

  function renounceOwnership() external;

  function owner() external view returns (address);

  /// @notice Getters:
  function registry() external view returns (address);

  function tobinTaxStalenessThreshold() external view returns (uint256);

  function tobinTax() external view returns (uint256);

  function tobinTaxReserveRatio() external view returns (uint256);

  function getDailySpendingRatio() external view returns (uint256);

  function checkIsCollateralAsset(address collateralAsset) external view returns (bool);

  function isToken(address) external view returns (bool);

  function getOtherReserveAddresses() external view returns (address[] memory);

  function getAssetAllocationSymbols() external view returns (bytes32[] memory);

  function getAssetAllocationWeights() external view returns (uint256[] memory);

  function collateralAssetSpendingLimit(address) external view returns (uint256);

  function getExchangeSpenders() external view returns (address[] memory);

  function getUnfrozenBalance() external view returns (uint256);

  function isOtherReserveAddress(address otherReserveAddress) external view returns (bool);

  function isSpender(address spender) external view returns (bool);

  /// @notice Setters:
  function setRegistry(address) external;

  function setTobinTax(uint256) external;

  function setTobinTaxReserveRatio(uint256) external;

  function setDailySpendingRatio(uint256 spendingRatio) external;

  function setDailySpendingRatioForCollateralAssets(
    address[] calldata _collateralAssets,
    uint256[] calldata collateralAssetDailySpendingRatios
  ) external;

  function setFrozenGold(uint256 frozenGold, uint256 frozenDays) external;

  function setAssetAllocations(bytes32[] calldata symbols, uint256[] calldata weights) external;

  function removeCollateralAsset(address collateralAsset, uint256 index) external returns (bool);

  function addOtherReserveAddress(address otherReserveAddress) external returns (bool);

  function removeOtherReserveAddress(address otherReserveAddress, uint256 index) external returns (bool);

  function collateralAssets(uint256 index) external view returns (address);
}
