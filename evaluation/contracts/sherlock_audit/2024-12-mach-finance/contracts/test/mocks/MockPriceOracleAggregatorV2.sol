// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import {PriceOracleAggregator} from "../../src/Oracles/PriceOracleAggregator.sol";
import {IOracleSource} from "../../src/Oracles/IOracleSource.sol";
import {Initializable} from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import {Ownable2StepUpgradeable} from "@openzeppelin/contracts-upgradeable/access/Ownable2StepUpgradeable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/// @custom:oz-upgrades-from PriceOracleAggregator
contract MockPriceOracleAggregatorV2 is Initializable, Ownable2StepUpgradeable, UUPSUpgradeable {
    mapping(address => IOracleSource[]) public tokenToOracleSources;

    uint256 counter;

    function setCounter(uint256 newCounter) public {
        counter = newCounter;
    }

    function getCounter() public view returns (uint256) {
        return counter;
    }

    function _authorizeUpgrade(address newImplementation) internal override onlyOwner {}
}
