// SPDX-License-Identifier: MIT
pragma solidity 0.8.18;

/**
 * @title LibIntMapping
 * @notice Library for working with int96 mappings
 */
library LibIntMapping {
  /**
   * @notice Adds value to the item in the mapping
   * @param map Mapping to add to
   * @param key Key of the item
   * @param value Value to add
   */
  function addToItem(mapping(uint256 => int96) storage map, uint256 key, int96 value) internal {
    map[key] = map[key] + (value);
  }

  /**
   * @notice Subtracts value from the item in the mapping
   * @param map Mapping to subtract from
   * @param key Key of the item
   * @param value Value to subtract
   */
  function subFromItem(mapping(uint256 => int96) storage map, uint256 key, int96 value) internal {
    map[key] = map[key] - (value);
  }
}
