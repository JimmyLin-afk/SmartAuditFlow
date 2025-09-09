// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility, const-name-snakecase, max-states-count
pragma solidity ^0.8;

import { Test } from "mento-std/Test.sol";
import { bytes32s, addresses, uints } from "mento-std/Array.sol";
import { CELO_REGISTRY_ADDRESS } from "mento-std/Constants.sol";

import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";
import { IFreezer } from "celo/contracts/common/interfaces/IFreezer.sol";

import { TestERC20 } from "test/utils/mocks/TestERC20.sol";
import { USDC } from "test/utils/mocks/USDC.sol";
import { WithRegistry } from "test/utils/WithRegistry.sol";

import { IStableTokenV2 } from "contracts/interfaces/IStableTokenV2.sol";
import { IPricingModule } from "contracts/interfaces/IPricingModule.sol";
import { IReserve } from "contracts/interfaces/IReserve.sol";
import { IBreakerBox } from "contracts/interfaces/IBreakerBox.sol";
import { ISortedOracles } from "contracts/interfaces/ISortedOracles.sol";
import { IBiPoolManager } from "contracts/interfaces/IBiPoolManager.sol";
import { IBroker } from "contracts/interfaces/IBroker.sol";
import { IPricingModule } from "contracts/interfaces/IPricingModule.sol";
import { IReserve } from "contracts/interfaces/IReserve.sol";
import { IMedianDeltaBreaker } from "contracts/interfaces/IMedianDeltaBreaker.sol";
import { IValueDeltaBreaker } from "contracts/interfaces/IValueDeltaBreaker.sol";
import { ITradingLimits } from "contracts/interfaces/ITradingLimits.sol";

contract ProtocolTest is Test, WithRegistry {
  using FixidityLib for FixidityLib.Fraction;

  uint256 constant tobinTaxStalenessThreshold = 600;
  uint256 constant dailySpendingRatio = 1000000000000000000000000;
  uint256 constant sortedOraclesDenominator = 1000000000000000000000000;
  uint256 tobinTax = FixidityLib.newFixedFraction(5, 1000).unwrap();
  uint256 tobinTaxReserveRatio = FixidityLib.newFixedFraction(2, 1).unwrap();

  event BucketsUpdated(bytes32 indexed exchangeId, uint256 bucket0, uint256 bucket1);

  mapping(address => uint256) oracleCounts;

  IBroker broker;
  IBiPoolManager biPoolManager;
  IReserve reserve;
  IPricingModule constantProduct;
  IPricingModule constantSum;

  ISortedOracles sortedOracles;
  IBreakerBox breakerBox;
  IMedianDeltaBreaker medianDeltaBreaker;
  IValueDeltaBreaker valueDeltaBreaker;

  TestERC20 celoToken;
  TestERC20 usdcToken;
  TestERC20 eurocToken;
  IStableTokenV2 cUSDToken;
  IStableTokenV2 cEURToken;
  IStableTokenV2 eXOFToken;
  IFreezer freezer;

  address cUSD_CELO_referenceRateFeedID;
  address cEUR_CELO_referenceRateFeedID;
  address cUSD_bridgedUSDC_referenceRateFeedID;
  address cEUR_bridgedUSDC_referenceRateFeedID;
  address cUSD_cEUR_referenceRateFeedID;
  address bridgedEUROC_EUR_referenceRateFeedID;
  address eXOF_bridgedEUROC_referenceRateFeedID;

  bytes32 pair_cUSD_CELO_ID;
  bytes32 pair_cEUR_CELO_ID;
  bytes32 pair_cUSD_bridgedUSDC_ID;
  bytes32 pair_cEUR_bridgedUSDC_ID;
  bytes32 pair_cUSD_cEUR_ID;
  bytes32 pair_eXOF_bridgedEUROC_ID;

  function setUp() public virtual {
    vm.warp(60 * 60 * 24 * 10); // Start at a non-zero timestamp.
    broker = IBroker(deployCode("Broker", abi.encode(true)));

    setUp_assets();
    setUp_reserve();
    setUp_sortedOracles();
    setUp_breakers();
    setUp_broker();
    setUp_freezer();
    setUp_tradingLimits();
  }

  function setUp_assets() internal {
    /* ===== Deploy collateral and stable assets ===== */

    celoToken = new TestERC20("Celo", "cGLD");
    usdcToken = new USDC("bridgedUSDC", "bridgedUSDC");
    eurocToken = new USDC("bridgedEUROC", "bridgedEUROC");

    address[] memory initialAddresses = new address[](0);
    uint256[] memory initialBalances = new uint256[](0);

    cUSDToken = IStableTokenV2(deployCode("StableTokenV2", abi.encode(false)));
    cUSDToken.initialize("cUSD", "cUSD", initialAddresses, initialBalances);
    cUSDToken.initializeV2(address(broker), address(0x0), address(0x0));

    cEURToken = IStableTokenV2(deployCode("StableTokenV2", abi.encode(false)));
    cEURToken.initialize("cEUR", "cEUR", initialAddresses, initialBalances);
    cEURToken.initializeV2(address(broker), address(0x0), address(0x0));

    eXOFToken = IStableTokenV2(deployCode("StableTokenV2", abi.encode(false)));
    eXOFToken.initialize("eXOF", "eXOF", initialAddresses, initialBalances);
    eXOFToken.initializeV2(address(broker), address(0x0), address(0x0));

    vm.label(address(cUSDToken), "cUSD");
    vm.label(address(cEURToken), "cEUR");
    vm.label(address(eXOFToken), "eXOF");
  }

  function setUp_reserve() internal {
    /* ===== Deploy reserve ===== */
    bytes32[] memory initialAssetAllocationSymbols = new bytes32[](3);
    uint256[] memory initialAssetAllocationWeights = new uint256[](3);
    initialAssetAllocationSymbols[0] = bytes32("cGLD");
    initialAssetAllocationWeights[0] = FixidityLib.newFixedFraction(1, 2).unwrap();
    initialAssetAllocationSymbols[1] = bytes32("bridgedUSDC");
    initialAssetAllocationWeights[1] = FixidityLib.newFixedFraction(1, 4).unwrap();
    initialAssetAllocationSymbols[2] = bytes32("bridgedEUROC");
    initialAssetAllocationWeights[2] = FixidityLib.newFixedFraction(1, 4).unwrap();

    address[] memory assets = new address[](3);
    uint256[] memory assetDailySpendingRatios = new uint256[](3);
    assets[0] = address(celoToken);
    assetDailySpendingRatios[0] = 100000000000000000000000;
    assets[1] = address(usdcToken);
    assetDailySpendingRatios[1] = 100000000000000000000000;
    assets[2] = address(eurocToken);
    assetDailySpendingRatios[2] = 100000000000000000000000;
    reserve = IReserve(deployCode("Reserve", abi.encode(true)));
    reserve.initialize(
      CELO_REGISTRY_ADDRESS,
      tobinTaxStalenessThreshold,
      dailySpendingRatio,
      0,
      0,
      initialAssetAllocationSymbols,
      initialAssetAllocationWeights,
      tobinTax,
      tobinTaxReserveRatio,
      assets,
      assetDailySpendingRatios
    );

    reserve.addToken(address(cUSDToken));
    reserve.addToken(address(cEURToken));
    reserve.addToken(address(eXOFToken));
  }

  function setUp_sortedOracles() internal {
    /* ===== Deploy SortedOracles ===== */

    sortedOracles = ISortedOracles(deployCode("SortedOracles", abi.encode(true)));
    sortedOracles.initialize(60 * 10);

    cUSD_CELO_referenceRateFeedID = address(cUSDToken);
    cEUR_CELO_referenceRateFeedID = address(cEURToken);
    cUSD_bridgedUSDC_referenceRateFeedID = address(bytes20(keccak256("USD/USDC")));
    cEUR_bridgedUSDC_referenceRateFeedID = address(bytes20(keccak256("EUR/USDC")));
    cUSD_cEUR_referenceRateFeedID = address(bytes20(keccak256("USD/EUR")));
    bridgedEUROC_EUR_referenceRateFeedID = address(bytes20(keccak256("EUROC/EUR")));
    eXOF_bridgedEUROC_referenceRateFeedID = address(bytes20(keccak256("XOF/EUROC")));

    initOracles(cUSD_CELO_referenceRateFeedID, 10);
    setMedianRate(cUSD_CELO_referenceRateFeedID, 5e23);

    initOracles(cEUR_CELO_referenceRateFeedID, 10);
    setMedianRate(cEUR_CELO_referenceRateFeedID, 5e23);

    initOracles(cUSD_bridgedUSDC_referenceRateFeedID, 10);
    setMedianRate(cUSD_bridgedUSDC_referenceRateFeedID, 1 * 1e24);

    initOracles(cEUR_bridgedUSDC_referenceRateFeedID, 10);
    setMedianRate(cEUR_bridgedUSDC_referenceRateFeedID, 0.9 * 1e24);

    initOracles(cUSD_cEUR_referenceRateFeedID, 10);
    setMedianRate(cUSD_cEUR_referenceRateFeedID, 1.1 * 1e24);

    initOracles(bridgedEUROC_EUR_referenceRateFeedID, 10);
    setMedianRate(bridgedEUROC_EUR_referenceRateFeedID, 1 * 1e24);

    initOracles(eXOF_bridgedEUROC_referenceRateFeedID, 10);
    setMedianRate(eXOF_bridgedEUROC_referenceRateFeedID, 656 * 1e24);
  }

  function initOracles(address rateFeedID, uint256 count) internal {
    oracleCounts[rateFeedID] = count;
    for (uint256 oracleIndex = 0; oracleIndex < count; oracleIndex++) {
      address oracleAddy = getOracleAddy(rateFeedID, oracleIndex);
      sortedOracles.addOracle(rateFeedID, oracleAddy);
    }
  }

  function setMedianRate(address rateFeedID, uint256 rate) internal {
    uint256 count = oracleCounts[rateFeedID];
    for (uint256 oracleIndex = 0; oracleIndex < count; oracleIndex++) {
      address oracleAddy = getOracleAddy(rateFeedID, oracleIndex);
      address lesserKey;
      address greaterKey;
      (address[] memory keys, uint256[] memory values, ) = sortedOracles.getRates(rateFeedID);
      for (uint256 i = 0; i < keys.length; i++) {
        if (keys[i] == oracleAddy) continue;
        if (values[i] < rate) lesserKey = keys[i];
        if (values[i] >= rate) greaterKey = keys[i];
      }

      vm.prank(oracleAddy);
      sortedOracles.report(rateFeedID, rate, lesserKey, greaterKey);
    }
  }

  function getOracleAddy(address rateFeedID, uint256 oracleIndex) internal pure returns (address) {
    return vm.addr(uint256(keccak256(abi.encodePacked(rateFeedID, oracleIndex))));
  }

  function setUp_breakers() internal {
    /* ========== Deploy Breaker Box =============== */
    address[] memory rateFeedIDs = addresses(
      cUSD_CELO_referenceRateFeedID,
      cEUR_CELO_referenceRateFeedID,
      cUSD_bridgedUSDC_referenceRateFeedID,
      cEUR_bridgedUSDC_referenceRateFeedID,
      cUSD_cEUR_referenceRateFeedID,
      bridgedEUROC_EUR_referenceRateFeedID,
      eXOF_bridgedEUROC_referenceRateFeedID
    );

    breakerBox = IBreakerBox(deployCode("BreakerBox", abi.encode(rateFeedIDs, ISortedOracles(address(sortedOracles)))));
    sortedOracles.setBreakerBox(breakerBox);

    // set rate feed dependencies

    address[] memory cEUR_bridgedUSDC_dependencies = addresses(cUSD_bridgedUSDC_referenceRateFeedID);
    breakerBox.setRateFeedDependencies(cEUR_bridgedUSDC_referenceRateFeedID, cEUR_bridgedUSDC_dependencies);

    address[] memory eXOF_bridgedEUROC_dependencies = addresses(bridgedEUROC_EUR_referenceRateFeedID);
    breakerBox.setRateFeedDependencies(eXOF_bridgedEUROC_referenceRateFeedID, eXOF_bridgedEUROC_dependencies);

    /* ========== Deploy Median Delta Breaker =============== */
    address[] memory medianDeltaBreakerRateFeedIDs = addresses(
      cUSD_CELO_referenceRateFeedID,
      cEUR_CELO_referenceRateFeedID,
      cUSD_bridgedUSDC_referenceRateFeedID,
      cEUR_bridgedUSDC_referenceRateFeedID,
      cUSD_cEUR_referenceRateFeedID
    );

    uint256[] memory medianDeltaBreakerRateChangeThresholds = uints(
      0.15 * 10 ** 24,
      0.14 * 10 ** 24,
      0.13 * 10 ** 24,
      0.12 * 10 ** 24,
      0.11 * 10 ** 24
    );
    uint256[] memory medianDeltaBreakerCooldownTimes = uints(
      5 minutes,
      0 minutes, // non recoverable median delta breaker
      5 minutes,
      5 minutes,
      5 minutes
    );

    uint256 medianDeltaBreakerDefaultThreshold = 0.15 * 10 ** 24; // 15%
    uint256 medianDeltaBreakerDefaultCooldown = 0 seconds;

    medianDeltaBreaker = IMedianDeltaBreaker(
      deployCode(
        "MedianDeltaBreaker",
        abi.encode(
          medianDeltaBreakerDefaultCooldown,
          medianDeltaBreakerDefaultThreshold,
          ISortedOracles(address(sortedOracles)),
          address(breakerBox),
          medianDeltaBreakerRateFeedIDs,
          medianDeltaBreakerRateChangeThresholds,
          medianDeltaBreakerCooldownTimes
        )
      )
    );

    breakerBox.addBreaker(address(medianDeltaBreaker), 3);

    // enable median delta breakers breakers
    breakerBox.toggleBreaker(address(medianDeltaBreaker), cUSD_CELO_referenceRateFeedID, true);
    breakerBox.toggleBreaker(address(medianDeltaBreaker), cEUR_CELO_referenceRateFeedID, true);
    breakerBox.toggleBreaker(address(medianDeltaBreaker), cUSD_bridgedUSDC_referenceRateFeedID, true);
    breakerBox.toggleBreaker(address(medianDeltaBreaker), cEUR_bridgedUSDC_referenceRateFeedID, true);
    breakerBox.toggleBreaker(address(medianDeltaBreaker), cUSD_cEUR_referenceRateFeedID, true);

    /* ============= Value Delta Breaker =============== */

    address[] memory valueDeltaBreakerRateFeedIDs = addresses(
      cUSD_bridgedUSDC_referenceRateFeedID,
      eXOF_bridgedEUROC_referenceRateFeedID,
      bridgedEUROC_EUR_referenceRateFeedID,
      cUSD_cEUR_referenceRateFeedID
    );
    uint256[] memory valueDeltaBreakerRateChangeThresholds = uints(
      0.1 * 10 ** 24,
      0.15 * 10 ** 24,
      0.05 * 10 ** 24,
      0.05 * 10 ** 24
    );
    uint256[] memory valueDeltaBreakerCooldownTimes = uints(1 seconds, 1 seconds, 1 seconds, 0 seconds);

    uint256 valueDeltaBreakerDefaultThreshold = 0.1 * 10 ** 24;
    uint256 valueDeltaBreakerDefaultCooldown = 0 seconds;

    valueDeltaBreaker = IValueDeltaBreaker(
      deployCode(
        "ValueDeltaBreaker",
        abi.encode(
          valueDeltaBreakerDefaultCooldown,
          valueDeltaBreakerDefaultThreshold,
          ISortedOracles(address(sortedOracles)),
          valueDeltaBreakerRateFeedIDs,
          valueDeltaBreakerRateChangeThresholds,
          valueDeltaBreakerCooldownTimes
        )
      )
    );

    // set reference value
    uint256[] memory valueDeltaBreakerReferenceValues = uints(1e24, 656 * 10 ** 24, 1e24, 1.1 * 10 ** 24);
    valueDeltaBreaker.setReferenceValues(valueDeltaBreakerRateFeedIDs, valueDeltaBreakerReferenceValues);

    // add value delta breaker and enable for rate feeds
    breakerBox.addBreaker(address(valueDeltaBreaker), 3);
    breakerBox.toggleBreaker(address(valueDeltaBreaker), cUSD_bridgedUSDC_referenceRateFeedID, true);
    breakerBox.toggleBreaker(address(valueDeltaBreaker), eXOF_bridgedEUROC_referenceRateFeedID, true);
    breakerBox.toggleBreaker(address(valueDeltaBreaker), bridgedEUROC_EUR_referenceRateFeedID, true);
    breakerBox.toggleBreaker(address(valueDeltaBreaker), cUSD_cEUR_referenceRateFeedID, true);
  }

  function setUp_broker() internal {
    /* ===== Deploy BiPoolManager & Broker ===== */

    constantProduct = IPricingModule(deployCode("ConstantProductPricingModule"));
    constantSum = IPricingModule(deployCode("ConstantSumPricingModule"));
    biPoolManager = IBiPoolManager(deployCode("BiPoolManager", abi.encode(true)));

    bytes32[] memory pricingModuleIdentifiers = bytes32s(
      keccak256(abi.encodePacked(constantProduct.name())),
      keccak256(abi.encodePacked(constantSum.name()))
    );

    address[] memory pricingModules = addresses(address(constantProduct), address(constantSum));

    biPoolManager.initialize(
      address(broker),
      IReserve(reserve),
      ISortedOracles(address(sortedOracles)),
      IBreakerBox(address(breakerBox))
    );
    address[] memory exchangeProviders = new address[](1);
    exchangeProviders[0] = address(biPoolManager);

    address[] memory reserves = new address[](1);
    reserves[0] = address(reserve);

    broker.initialize(exchangeProviders, reserves);
    registry.setAddressFor("Broker", address(broker));
    reserve.addExchangeSpender(address(broker));
    biPoolManager.setPricingModules(pricingModuleIdentifiers, pricingModules);

    /* ====== Create pairs for all asset combinations ======= */

    IBiPoolManager.PoolExchange memory pair_cUSD_CELO;
    pair_cUSD_CELO.asset0 = address(cUSDToken);
    pair_cUSD_CELO.asset1 = address(celoToken);
    pair_cUSD_CELO.pricingModule = constantProduct;
    pair_cUSD_CELO.lastBucketUpdate = block.timestamp;
    pair_cUSD_CELO.config.spread = FixidityLib.newFixedFraction(5, 100);
    pair_cUSD_CELO.config.referenceRateResetFrequency = 60 * 5;
    pair_cUSD_CELO.config.minimumReports = 5;
    pair_cUSD_CELO.config.referenceRateFeedID = cUSD_CELO_referenceRateFeedID;
    pair_cUSD_CELO.config.stablePoolResetSize = 1e24;

    pair_cUSD_CELO_ID = biPoolManager.createExchange(pair_cUSD_CELO);

    IBiPoolManager.PoolExchange memory pair_cEUR_CELO;
    pair_cEUR_CELO.asset0 = address(cEURToken);
    pair_cEUR_CELO.asset1 = address(celoToken);
    pair_cEUR_CELO.pricingModule = constantProduct;
    pair_cEUR_CELO.lastBucketUpdate = block.timestamp;
    pair_cEUR_CELO.config.spread = FixidityLib.newFixedFraction(5, 100);
    pair_cEUR_CELO.config.referenceRateResetFrequency = 60 * 5;
    pair_cEUR_CELO.config.minimumReports = 5;
    pair_cEUR_CELO.config.referenceRateFeedID = cEUR_CELO_referenceRateFeedID;
    pair_cEUR_CELO.config.stablePoolResetSize = 1e24;

    pair_cEUR_CELO_ID = biPoolManager.createExchange(pair_cEUR_CELO);

    IBiPoolManager.PoolExchange memory pair_cUSD_bridgedUSDC;
    pair_cUSD_bridgedUSDC.asset0 = address(cUSDToken);
    pair_cUSD_bridgedUSDC.asset1 = address(usdcToken);
    pair_cUSD_bridgedUSDC.pricingModule = constantSum;
    pair_cUSD_bridgedUSDC.lastBucketUpdate = block.timestamp;
    pair_cUSD_bridgedUSDC.config.spread = FixidityLib.newFixedFraction(5, 1000);
    pair_cUSD_bridgedUSDC.config.referenceRateResetFrequency = 60 * 5;
    pair_cUSD_bridgedUSDC.config.minimumReports = 5;
    pair_cUSD_bridgedUSDC.config.referenceRateFeedID = cUSD_bridgedUSDC_referenceRateFeedID;
    pair_cUSD_bridgedUSDC.config.stablePoolResetSize = 1e24;

    pair_cUSD_bridgedUSDC_ID = biPoolManager.createExchange(pair_cUSD_bridgedUSDC);

    IBiPoolManager.PoolExchange memory pair_cEUR_bridgedUSDC;
    pair_cEUR_bridgedUSDC.asset0 = address(cEURToken);
    pair_cEUR_bridgedUSDC.asset1 = address(usdcToken);
    pair_cEUR_bridgedUSDC.pricingModule = constantSum;
    pair_cEUR_bridgedUSDC.lastBucketUpdate = block.timestamp;
    pair_cEUR_bridgedUSDC.config.spread = FixidityLib.newFixedFraction(5, 100);
    pair_cEUR_bridgedUSDC.config.referenceRateResetFrequency = 60 * 5;
    pair_cEUR_bridgedUSDC.config.minimumReports = 5;
    pair_cEUR_bridgedUSDC.config.referenceRateFeedID = cEUR_bridgedUSDC_referenceRateFeedID;
    pair_cEUR_bridgedUSDC.config.stablePoolResetSize = 1e24;

    pair_cEUR_bridgedUSDC_ID = biPoolManager.createExchange(pair_cEUR_bridgedUSDC);

    IBiPoolManager.PoolExchange memory pair_cUSD_cEUR;
    pair_cUSD_cEUR.asset0 = address(cUSDToken);
    pair_cUSD_cEUR.asset1 = address(cEURToken);
    pair_cUSD_cEUR.pricingModule = constantProduct;
    pair_cUSD_cEUR.lastBucketUpdate = block.timestamp;
    pair_cUSD_cEUR.config.spread = FixidityLib.newFixedFraction(5, 100);
    pair_cUSD_cEUR.config.referenceRateResetFrequency = 60 * 5;
    pair_cUSD_cEUR.config.minimumReports = 5;
    pair_cUSD_cEUR.config.referenceRateFeedID = cUSD_cEUR_referenceRateFeedID;
    pair_cUSD_cEUR.config.stablePoolResetSize = 1e24;

    pair_cUSD_cEUR_ID = biPoolManager.createExchange(pair_cUSD_cEUR);

    IBiPoolManager.PoolExchange memory pair_eXOF_bridgedEUROC;
    pair_eXOF_bridgedEUROC.asset0 = address(eXOFToken);
    pair_eXOF_bridgedEUROC.asset1 = address(eurocToken);
    pair_eXOF_bridgedEUROC.pricingModule = constantSum;
    pair_eXOF_bridgedEUROC.lastBucketUpdate = block.timestamp;
    pair_eXOF_bridgedEUROC.config.spread = FixidityLib.newFixedFraction(5, 1000);
    pair_eXOF_bridgedEUROC.config.referenceRateResetFrequency = 60 * 5;
    pair_eXOF_bridgedEUROC.config.minimumReports = 5;
    pair_eXOF_bridgedEUROC.config.referenceRateFeedID = eXOF_bridgedEUROC_referenceRateFeedID;
    pair_eXOF_bridgedEUROC.config.stablePoolResetSize = 1e24;

    pair_eXOF_bridgedEUROC_ID = biPoolManager.createExchange(pair_eXOF_bridgedEUROC);
  }

  function setUp_freezer() internal {
    /* ========== Deploy Freezer =============== */

    freezer = IFreezer(deployCode("Freezer", abi.encode(true)));
    registry.setAddressFor("Freezer", address(freezer));
  }

  function setUp_tradingLimits() internal {
    /* ========== Config Trading Limits =============== */
    ITradingLimits.Config memory config = configL0L1LG(100, 10000, 1000, 100000, 1000000);
    broker.configureTradingLimit(pair_cUSD_CELO_ID, address(cUSDToken), config);
    broker.configureTradingLimit(pair_cEUR_CELO_ID, address(cEURToken), config);
    broker.configureTradingLimit(pair_cUSD_bridgedUSDC_ID, address(usdcToken), config);
    broker.configureTradingLimit(pair_cEUR_bridgedUSDC_ID, address(usdcToken), config);
    broker.configureTradingLimit(pair_cUSD_cEUR_ID, address(cUSDToken), config);
    broker.configureTradingLimit(pair_eXOF_bridgedEUROC_ID, address(eXOFToken), config);
  }

  function configL0L1LG(
    uint32 timestep0,
    int48 limit0,
    uint32 timestep1,
    int48 limit1,
    int48 limitGlobal
  ) internal pure returns (ITradingLimits.Config memory config) {
    config.timestep0 = timestep0;
    config.limit0 = limit0;
    config.timestep1 = timestep1;
    config.limit1 = limit1;
    config.limitGlobal = limitGlobal;
    config.flags = 1 | 2 | 4; //L0, L1, and LG
  }
}
