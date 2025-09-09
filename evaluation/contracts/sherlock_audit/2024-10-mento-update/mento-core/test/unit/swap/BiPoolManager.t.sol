// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase
pragma solidity ^0.8;

import { Test } from "mento-std/Test.sol";
import { bytes32s, addresses } from "mento-std/Array.sol";
import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";

import { MockReserve } from "test/utils//mocks/MockReserve.sol";
import { MockBreakerBox } from "test/utils/mocks/MockBreakerBox.sol";
import { MockERC20 } from "test/utils/mocks/MockERC20.sol";
import { MockPricingModule } from "test/utils/mocks/MockPricingModule.sol";
import { MockSortedOracles } from "test/utils/mocks/MockSortedOracles.sol";

import { IReserve } from "contracts/interfaces/IReserve.sol";
import { IBreakerBox } from "contracts/interfaces/IBreakerBox.sol";
import { IBiPoolManager } from "contracts/interfaces/IBiPoolManager.sol";
import { IExchangeProvider } from "contracts/interfaces/IExchangeProvider.sol";
import { ISortedOracles } from "contracts/interfaces/ISortedOracles.sol";
import { IPricingModule } from "contracts/interfaces/IPricingModule.sol";

// forge test --match-contract BiPoolManager -vvv
contract BiPoolManagerTest is Test {
  using FixidityLib for FixidityLib.Fraction;

  /* ------- Events from IBiPoolManager ------- */

  event ExchangeCreated(
    bytes32 indexed exchangeId,
    address indexed asset0,
    address indexed asset1,
    address pricingModule
  );
  event ExchangeDestroyed(
    bytes32 indexed exchangeId,
    address indexed asset0,
    address indexed asset1,
    address pricingModule
  );
  event BrokerUpdated(address indexed newBroker);
  event ReserveUpdated(address indexed newReserve);
  event SortedOraclesUpdated(address indexed newSortedOracles);
  event BucketsUpdated(bytes32 indexed exchangeId, uint256 bucket0, uint256 bucket1);
  event BreakerBoxUpdated(address newBreakerBox);
  event PricingModulesUpdated(bytes32[] newIdentifiers, address[] newAddresses);

  /* ------------------------------------------- */

  address deployer;
  address notDeployer;
  address broker;

  MockERC20 cUSD;
  MockERC20 cEUR;
  MockERC20 bridgedUSDC;
  MockERC20 CELO;

  IPricingModule constantProduct;
  IPricingModule constantSum;

  MockSortedOracles sortedOracles;
  MockReserve reserve;
  MockBreakerBox breaker;

  IBiPoolManager biPoolManager;

  function newMockERC20(string memory name, string memory symbol, uint256 decimals) internal returns (MockERC20 token) {
    token = new MockERC20(name, symbol, decimals);
    vm.label(address(token), symbol);
  }

  function setUp() public virtual {
    vm.warp(60 * 60 * 24 * 30); // if we start at block.timestamp == 0 we get some underflows
    deployer = makeAddr("deployer");
    notDeployer = makeAddr("notDeployer");
    broker = makeAddr("broker");

    cUSD = newMockERC20("Celo Dollar", "cUSD", 18);
    cEUR = newMockERC20("Celo Euro", "cEUR", 18);
    bridgedUSDC = newMockERC20("Bridged USDC", "bridgedUSDC", 6);
    CELO = newMockERC20("CELO", "CELO", 18);

    constantProduct = new MockPricingModule("ConstantProduct");
    constantSum = new MockPricingModule("ConstantSum");
    sortedOracles = new MockSortedOracles();

    reserve = new MockReserve();
    biPoolManager = IBiPoolManager(deployCode("BiPoolManager", abi.encode(true)));
    breaker = new MockBreakerBox();

    vm.mockCall(
      address(reserve),
      abi.encodeWithSelector(reserve.isStableAsset.selector, address(cUSD)),
      abi.encode(true)
    );

    vm.mockCall(
      address(reserve),
      abi.encodeWithSelector(reserve.isStableAsset.selector, address(cEUR)),
      abi.encode(true)
    );

    vm.mockCall(
      address(reserve),
      abi.encodeWithSelector(reserve.isCollateralAsset.selector, address(bridgedUSDC)),
      abi.encode(true)
    );

    vm.mockCall(
      address(reserve),
      abi.encodeWithSelector(reserve.isCollateralAsset.selector, address(CELO)),
      abi.encode(true)
    );

    vm.startPrank(deployer);

    biPoolManager.initialize(
      broker,
      IReserve(address(reserve)),
      ISortedOracles(address(sortedOracles)),
      IBreakerBox(address(breaker))
    );

    bytes32[] memory pricingModuleIdentifiers = bytes32s(
      keccak256(abi.encodePacked(constantProduct.name())),
      keccak256(abi.encodePacked(constantSum.name()))
    );

    address[] memory pricingModules = addresses(address(constantProduct), address(constantSum));

    biPoolManager.setPricingModules(pricingModuleIdentifiers, pricingModules);
  }

  function mockOracleRate(address target, uint256 rateNumerator) internal {
    sortedOracles.setMedianRate(target, rateNumerator);
  }

  function createExchange(MockERC20 asset0, MockERC20 asset1) internal returns (bytes32) {
    return createExchange(asset0, asset1, IPricingModule(constantProduct));
  }

  function createExchange(
    MockERC20 asset0,
    MockERC20 asset1,
    IPricingModule pricingModule
  ) internal returns (bytes32 exchangeId) {
    return createExchange(asset0, asset1, pricingModule, address(asset0));
  }

  function createExchange(
    MockERC20 asset0,
    MockERC20 asset1,
    IPricingModule pricingModule,
    address referenceRateFeedID
  ) internal returns (bytes32 exchangeId) {
    return
      createExchange(
        asset0,
        asset1,
        pricingModule,
        referenceRateFeedID,
        FixidityLib.wrap(0.1 * 1e24), // spread
        1e26 // stablePoolResetSize
      );
  }

  function createExchange(
    MockERC20 asset0,
    MockERC20 asset1,
    IPricingModule pricingModule,
    address referenceRateFeedID,
    FixidityLib.Fraction memory spread,
    uint256 stablePoolResetSize
  ) internal returns (bytes32 exchangeId) {
    return
      createExchange(
        asset0,
        asset1,
        pricingModule,
        referenceRateFeedID,
        spread,
        stablePoolResetSize,
        5 // minimumReports
      );
  }

  function createExchange(
    MockERC20 asset0,
    MockERC20 asset1,
    IPricingModule pricingModule,
    address referenceRateFeedID,
    FixidityLib.Fraction memory spread,
    uint256 stablePoolResetSize,
    uint256 minimumReports
  ) internal returns (bytes32 exchangeId) {
    IBiPoolManager.PoolExchange memory exchange;
    exchange.asset0 = address(asset0);
    exchange.asset1 = address(asset1);
    exchange.pricingModule = pricingModule;

    IBiPoolManager.PoolConfig memory config;
    config.referenceRateFeedID = referenceRateFeedID;
    config.stablePoolResetSize = stablePoolResetSize;
    config.referenceRateResetFrequency = 60 * 5; // 5 minutes
    config.minimumReports = minimumReports;
    config.spread = spread;

    exchange.config = config;

    return biPoolManager.createExchange(exchange);
  }

  function mockGetAmountIn(IPricingModule pricingModule, uint256 result) internal {
    MockPricingModule(address(pricingModule)).mockNextGetAmountIn(result);
  }

  function mockGetAmountOut(IPricingModule pricingModule, uint256 result) internal {
    MockPricingModule(address(pricingModule)).mockNextGetAmountOut(result);
  }
}

contract BiPoolManagerTest_initilizerSettersGetters is BiPoolManagerTest {
  /* ---------- Initilizer ---------- */

  function test_initilize_shouldSetOwner() public view {
    assertEq(biPoolManager.owner(), deployer);
  }

  function test_initilize_shouldSetBroker() public view {
    assertEq(biPoolManager.broker(), broker);
  }

  function test_initilize_shouldSetReserve() public view {
    assertEq(address(biPoolManager.reserve()), address(reserve));
  }

  function test_initilize_shouldSetSortedOracles() public view {
    assertEq(address(biPoolManager.sortedOracles()), address(sortedOracles));
  }

  function test_initialize_shouldSetBreakerBox() public view {
    assertEq(address(biPoolManager.breakerBox()), address(breaker));
  }

  /* ---------- Setters ---------- */

  function test_setBroker_whenSenderIsNotOwner_shouldRevert() public {
    changePrank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    biPoolManager.setBroker(address(0));
  }

  function test_setBroker_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("Broker address must be set");
    biPoolManager.setBroker(address(0));
  }

  function test_setBroker_whenSenderIsOwner_shouldUpdateAndEmit() public {
    address newBroker = makeAddr("newBroker");
    vm.expectEmit(true, true, true, true);
    emit BrokerUpdated(newBroker);

    biPoolManager.setBroker(newBroker);

    assertEq(biPoolManager.broker(), newBroker);
  }

  function test_setReserve_whenSenderIsNotOwner_shouldRevert() public {
    changePrank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    biPoolManager.setReserve(IReserve(address(0)));
  }

  function test_setReserve_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("Reserve address must be set");
    biPoolManager.setReserve(IReserve(address(0)));
  }

  function test_setReserve_whenSenderIsOwner_shouldUpdateAndEmit() public {
    address newReserve = makeAddr("newReserve");
    vm.expectEmit(true, true, true, true);
    emit ReserveUpdated(newReserve);

    biPoolManager.setReserve(IReserve(newReserve));

    assertEq(address(biPoolManager.reserve()), newReserve);
  }

  function test_setSortedOracles_whenSenderIsNotOwner_shouldRevert() public {
    changePrank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    biPoolManager.setSortedOracles(ISortedOracles(address(0)));
  }

  function test_setSortedOracles_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("SortedOracles address must be set");
    biPoolManager.setSortedOracles(ISortedOracles(address(0)));
  }

  function test_setSortedOracles_whenSenderIsOwner_shouldUpdateAndEmit() public {
    address newSortedOracles = makeAddr("newSortedOracles");
    vm.expectEmit(true, true, true, true);
    emit SortedOraclesUpdated(newSortedOracles);

    biPoolManager.setSortedOracles(ISortedOracles(newSortedOracles));

    assertEq(address(biPoolManager.sortedOracles()), newSortedOracles);
  }

  function test_setBreakerBox_whenSenderIsNotOwner_shouldRevert() public {
    changePrank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    biPoolManager.setBreakerBox(IBreakerBox(address(0)));
  }

  function test_setBreakerBox_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("BreakerBox address must be set");
    biPoolManager.setBreakerBox(IBreakerBox(address(0)));
  }

  function test_setBreakerBox_whenSenderIsOwner_shouldUpdateAndEmit() public {
    address newBreakerBox = makeAddr("newBreakerBox");
    vm.expectEmit(true, true, true, true);
    emit BreakerBoxUpdated(newBreakerBox);

    biPoolManager.setBreakerBox(IBreakerBox(newBreakerBox));

    assertEq(address(biPoolManager.breakerBox()), newBreakerBox);
  }

  function test_setPricingModules_whenCallerIsNotOwner_shouldRevert() public {
    changePrank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    biPoolManager.setPricingModules(bytes32s(""), addresses(address(0)));
  }

  function test_setPricingModules_whenArrayLengthMismatch_shouldRevert() public {
    vm.expectRevert("identifiers and modules must be the same length");
    biPoolManager.setPricingModules(bytes32s(""), addresses(address(0), address(0xf)));
  }

  function test_setPricingModules_whenCallerIsOwner_shouldUpdateAndEmit() public {
    bytes32[] memory newIdentifiers = bytes32s(
      keccak256(abi.encodePacked("TestModuleIdentifier1")),
      keccak256(abi.encodePacked("TestModuleIdentifier2"))
    );

    address[] memory newPricingModules = addresses(
      makeAddr("TestModuleIdentifier1"),
      makeAddr("TestModuleIdentifier2")
    );

    vm.expectEmit(true, true, true, true);
    emit PricingModulesUpdated(newIdentifiers, newPricingModules);

    biPoolManager.setPricingModules(newIdentifiers, newPricingModules);
  }

  /* ---------- Getters ---------- */

  function test_getPoolExchange_whenExchangeDoesNotExist_shouldRevert() public {
    bytes32 exchangeId = keccak256(abi.encodePacked(cUSD.symbol(), bridgedUSDC.symbol(), constantProduct.name()));

    vm.expectRevert("An exchange with the specified id does not exist");
    biPoolManager.getPoolExchange(exchangeId);
  }

  function test_getPoolExchange_whenPoolExists_shouldReturnPool() public {
    mockOracleRate(address(cUSD), 1e24);
    bytes32 exchangeId = createExchange(cUSD, bridgedUSDC);
    IBiPoolManager.PoolExchange memory existingExchange = biPoolManager.getPoolExchange(exchangeId);
    assertEq(existingExchange.asset0, address(cUSD));
    assertEq(existingExchange.asset1, address(bridgedUSDC));
  }
}

contract BiPoolManagerTest_createExchange is BiPoolManagerTest {
  function test_createExchange_whenNotCalledByOwner_shouldRevert() public {
    IBiPoolManager.PoolExchange memory newexchange;
    changePrank(notDeployer);

    vm.expectRevert("Ownable: caller is not the owner");
    biPoolManager.createExchange(newexchange);
  }

  function test_createExchange_whenPoolWithIdExists_shouldRevert() public {
    mockOracleRate(address(cUSD), 1e24);
    createExchange(cUSD, bridgedUSDC);

    vm.expectRevert("An exchange with the specified assets and exchange exists");
    createExchange(cUSD, bridgedUSDC);
  }

  function test_createExchange_whenAsset0IsNotRegistered_shouldRevert() public {
    MockERC20 nonReserveStable = newMockERC20("Non Reserve Stable Asset", "NRSA", 18);
    vm.expectRevert("asset0 must be a stable registered with the reserve");
    createExchange(nonReserveStable, CELO);
  }

  function test_createExchange_whenAsset0IsCollateral_shouldRevert() public {
    vm.expectRevert("asset0 must be a stable registered with the reserve");
    createExchange(bridgedUSDC, CELO);
  }

  function test_createExchange_whenAsset1IsNotRegistered_shouldRevert() public {
    MockERC20 nonReserveCollateral = newMockERC20("Non Reserve Collateral Asset", "NRCA", 18);
    vm.expectRevert("asset1 must be a stable or collateral");
    createExchange(cUSD, nonReserveCollateral);
  }

  function test_createExchange_whenMentoExchangeIsNotSet_shouldRevert() public {
    vm.expectRevert("pricingModule must be set");
    createExchange(cUSD, CELO, IPricingModule(address(0)));
  }

  function test_createExchange_whenAsset0IsNotSet_shouldRevert() public {
    vm.expectRevert("asset0 must be set");
    createExchange(MockERC20(address(0)), CELO);
  }

  function test_createExchange_whenAsset1IsNotSet_shouldRevert() public {
    vm.expectRevert("asset1 must be set");
    createExchange(cUSD, MockERC20(address(0)));
  }

  function test_createExchange_whenAssetsAreIdentical_shouldRevert() public {
    vm.expectRevert("exchange assets can't be identical");
    createExchange(cUSD, cUSD);
  }

  function test_createExchange_whenReferenceRateFeedIDIsNotSet_shouldRevert() public {
    vm.expectRevert("referenceRateFeedID must be set");
    createExchange(cUSD, CELO, constantProduct, address(0));
  }

  function test_createExchange_whenSpreadNotLTEOne_shouldRevert() public {
    vm.expectRevert("spread must be less than or equal to 1");
    createExchange(
      cUSD,
      CELO,
      constantProduct,
      address(cUSD),
      FixidityLib.wrap(2 * 1e24), // spread
      1e26 // stablePoolResetSize
    );
  }

  function test_createExchange_whenPricingModuleIsOutdated_shouldRevert() public {
    bytes32[] memory newIdentifiers = bytes32s(
      keccak256(abi.encodePacked(constantProduct.name())),
      keccak256(abi.encodePacked(constantSum.name()))
    );

    address[] memory newPricingModules = addresses(makeAddr("ConstantProduct 2.0"), address(constantSum));

    biPoolManager.setPricingModules(newIdentifiers, newPricingModules);

    vm.expectRevert("invalid pricingModule");
    createExchange(
      cUSD,
      CELO,
      constantProduct,
      address(cUSD),
      FixidityLib.wrap(2 * 1e24), // spread
      1e26 // stablePoolResetSize
    );
  }

  function test_createExchange_whenInfoIsValid_shouldUpdateMappingAndEmit() public {
    bytes32 exchangeId = keccak256(abi.encodePacked(cUSD.symbol(), CELO.symbol(), constantProduct.name()));

    mockOracleRate(address(cUSD), 2 * 1e24);
    vm.expectEmit(true, true, true, false);
    emit ExchangeCreated(exchangeId, address(cUSD), address(CELO), address(constantProduct));
    createExchange(cUSD, CELO);

    IExchangeProvider.Exchange[] memory exchanges = biPoolManager.getExchanges();
    assertEq(exchanges.length, 1);
    assertEq(exchanges[0].exchangeId, exchangeId);
  }

  function test_createExchange_whenInfoIsValid_setsBucketSizesCorrectly() public {
    mockOracleRate(address(cUSD), 2 * 1e24); // 1 CELO == 2 cUSD
    bytes32 exchangeId = createExchange(
      cUSD,
      CELO,
      constantProduct,
      address(cUSD),
      FixidityLib.wrap(0.1 * 1e24), // spread
      1e24 // stablePoolResetSize
    );

    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId);
    assertEq(exchange.bucket0, 1e24);
    assertEq(exchange.bucket1, 5e23); // exchange.bucket0 / 2
  }
}

contract BiPoolManagerTest_destroyExchange is BiPoolManagerTest {
  function test_destroyExchange_whenSenderIsNotOwner_shouldRevert() public {
    changePrank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    biPoolManager.destroyExchange(0x0, 0);
  }

  function test_destroyExchange_whenNoExchangesExist_shouldRevert() public {
    vm.expectRevert("exchangeIdIndex not in range");
    biPoolManager.destroyExchange(0x0, 0);
  }

  function test_destroyExchange_whenExchangeExistsButTheIdIsWrong_shouldRevert() public {
    mockOracleRate(address(cUSD), 2e24);
    createExchange(cUSD, bridgedUSDC);
    vm.expectRevert("exchangeId at index doesn't match");
    biPoolManager.destroyExchange(0x0, 0);
  }

  function test_destroyExchange_whenExchangeExistsButTheIndexIsTooLarge_shouldRevert() public {
    mockOracleRate(address(cUSD), 2e24);
    createExchange(cUSD, bridgedUSDC);
    vm.expectRevert("exchangeIdIndex not in range");
    biPoolManager.destroyExchange(0x0, 1);
  }

  function test_destroyExchange_whenExchangeExists_shouldUpdateAndEmit() public {
    mockOracleRate(address(cUSD), 2e24);
    bytes32 exchangeId = createExchange(cUSD, bridgedUSDC);
    vm.expectEmit(true, true, true, true);
    emit ExchangeDestroyed(exchangeId, address(cUSD), address(bridgedUSDC), address(constantProduct));
    biPoolManager.destroyExchange(exchangeId, 0);
  }

  function test_destroyExchange_whenMultipleExchangesExist_shouldUpdateTheIdList() public {
    mockOracleRate(address(cUSD), 2e24);
    bytes32 exchangeId0 = createExchange(cUSD, bridgedUSDC);
    bytes32 exchangeId1 = createExchange(cUSD, CELO);

    vm.expectEmit(true, true, true, true);
    emit ExchangeDestroyed(exchangeId0, address(cUSD), address(bridgedUSDC), address(constantProduct));
    biPoolManager.destroyExchange(exchangeId0, 0);

    IExchangeProvider.Exchange[] memory exchanges = biPoolManager.getExchanges();
    assertEq(exchanges.length, 1);
    assertEq(exchanges[0].exchangeId, exchangeId1);
  }
}

contract BiPoolManagerTest_withExchange is BiPoolManagerTest {
  bytes32 exchangeId_cUSD_CELO;
  bytes32 exchangeId_cUSD_bridgedUSDC;
  bytes32 exchangeId_cEUR_bridgedUSDC;

  function setUp() public virtual override {
    super.setUp();

    mockOracleRate(address(cUSD), 2e24);
    exchangeId_cUSD_CELO = createExchange(cUSD, CELO);

    address USDUSDC_rateFeedID = address(uint160(uint256(keccak256(abi.encodePacked("USDUSDC")))));
    mockOracleRate(USDUSDC_rateFeedID, 1e24);
    sortedOracles.setMedianTimestampToNow(USDUSDC_rateFeedID);
    exchangeId_cUSD_bridgedUSDC = createExchange(
      cUSD,
      bridgedUSDC,
      constantSum,
      USDUSDC_rateFeedID,
      FixidityLib.wrap(0.1 * 1e24), // spread
      1e24, // stablePoolResetSize
      0 // minimumReports
    );

    mockOracleRate(address(cEUR), 2e24);
    exchangeId_cEUR_bridgedUSDC = createExchange(cEUR, bridgedUSDC);
  }
}

contract BiPoolManagerTest_quote is BiPoolManagerTest_withExchange {
  /* ---------- getAmountOut ---------- */
  function test_getAmountOut_whenExchangeDoesntExist_itReverts() public {
    vm.expectRevert("An exchange with the specified id does not exist");
    biPoolManager.getAmountOut(0x0, address(0), address(0), 1e24);
  }

  function test_getAmountOut_whenTokenInNotInexchange_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.getAmountOut(exchangeId_cUSD_CELO, address(cEUR), address(cUSD), 1e24);
  }

  function test_getAmountOut_whenTokenOutNotInexchange_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.getAmountOut(exchangeId_cUSD_CELO, address(cUSD), address(cEUR), 1e24);
  }

  function test_getAmountOut_whenTokenInEqualsTokenOut_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.getAmountOut(exchangeId_cUSD_CELO, address(cUSD), address(cUSD), 1e24);
  }

  function test_getAmountOut_whenTokenInIsAsset0_itDelegatesToThePricingModule() public {
    uint256 amountIn = 1e24;
    uint256 mockAmountOut = 0.5 * 1e24;

    mockGetAmountOut(constantProduct, mockAmountOut);
    uint256 amountOut = biPoolManager.getAmountOut(exchangeId_cUSD_CELO, address(cUSD), address(CELO), amountIn);
    assertEq(amountOut, mockAmountOut);
  }

  function test_getAmountOut_whenTokenInIsAsset1_itDelegatesToThePricingModule() public {
    uint256 amountIn = 1e24;
    uint256 mockAmountOut = 0.5 * 1e24;

    mockGetAmountOut(constantProduct, mockAmountOut);
    uint256 amountOut = biPoolManager.getAmountOut(exchangeId_cUSD_CELO, address(CELO), address(cUSD), amountIn);
    assertEq(amountOut, mockAmountOut);
  }

  function test_getAmountOut_whenTokenOutHasNonstandardPrecision() public {
    uint256 amountIn = 1e18;
    uint256 mockAmountOut = 1e18;
    mockGetAmountOut(constantSum, mockAmountOut);
    uint256 amountOut = biPoolManager.getAmountOut(
      exchangeId_cUSD_bridgedUSDC,
      address(cUSD),
      address(bridgedUSDC),
      amountIn
    );
    assertEq(amountOut, 1e6);
  }

  function test_getAmountOut_whenTokenInHasNonstandardPrecision() public {
    uint256 amountIn = 1e6;
    uint256 mockAmountOut = 1e18;
    mockGetAmountOut(constantSum, mockAmountOut);
    uint256 amountOut = biPoolManager.getAmountOut(
      exchangeId_cUSD_bridgedUSDC,
      address(bridgedUSDC),
      address(cUSD),
      amountIn
    );
    assertEq(amountOut, 1e18);
  }

  /* ---------- getAmountIn ---------- */

  function test_getAmountIn_whenExchangeDoesntExist_itReverts() public {
    vm.expectRevert("An exchange with the specified id does not exist");
    biPoolManager.getAmountIn(0x0, address(0), address(0), 1e24);
  }

  function test_getAmountIn_whenTokenInNotInexchange_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.getAmountIn(exchangeId_cUSD_CELO, address(cEUR), address(cUSD), 1e24);
  }

  function test_getAmountIn_whenTokenOutNotInexchange_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.getAmountIn(exchangeId_cUSD_CELO, address(cUSD), address(cEUR), 1e24);
  }

  function test_getAmountIn_whenTokenInEqualsTokenOut_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.getAmountIn(exchangeId_cUSD_CELO, address(cUSD), address(cUSD), 1e24);
  }

  function test_getAmountIn_whenTokenInIsAsset0_itDelegatesToThePricingModule() public {
    uint256 amountOut = 1e24;
    uint256 mockAmountIn = 0.5 * 1e24;

    mockGetAmountIn(constantProduct, mockAmountIn);
    uint256 amountIn = biPoolManager.getAmountIn(exchangeId_cUSD_CELO, address(cUSD), address(CELO), amountOut);
    assertEq(amountIn, mockAmountIn);
  }

  function test_getAmountIn_whenTokenInIsAsset1_itDelegatesToThePricingModule() public {
    uint256 amountOut = 1e24;
    uint256 mockAmountIn = 0.5 * 1e24;

    mockGetAmountIn(constantProduct, mockAmountIn);
    uint256 amountIn = biPoolManager.getAmountIn(exchangeId_cUSD_CELO, address(CELO), address(cUSD), amountOut);
    assertEq(amountIn, mockAmountIn);
  }

  function test_getAmountIn_whenTokenOutHasNonstandardPrecision() public {
    uint256 amountOut = 1e6;
    uint256 mockAmountIn = 1e18;
    MockPricingModule(address(constantSum)).mockNextGetAmountIn(mockAmountIn);
    uint256 amountIn = biPoolManager.getAmountIn(
      exchangeId_cUSD_bridgedUSDC,
      address(cUSD),
      address(bridgedUSDC),
      amountOut
    );
    assertEq(amountIn, mockAmountIn);
  }

  function test_getAmountIn_whenTokenInHasNonstandardPrecision() public {
    uint256 amountOut = 1e18;
    uint256 mockAmountIn = 1e18;
    MockPricingModule(address(constantSum)).mockNextGetAmountIn(mockAmountIn);
    uint256 amountIn = biPoolManager.getAmountIn(
      exchangeId_cUSD_bridgedUSDC,
      address(bridgedUSDC),
      address(cUSD),
      amountOut
    );
    assertEq(amountIn, 1e6);
  }
}

contract BiPoolManagerTest_swap is BiPoolManagerTest_withExchange {
  function setUp() public override {
    super.setUp();
    changePrank(broker);
  }

  /* ---------- swapIn ---------- */
  function test_swapIn_whenNotBroker_itReverts() public {
    changePrank(deployer);
    vm.expectRevert("Caller is not the Broker");
    biPoolManager.swapIn(0x0, address(0), address(0), 1e24);
  }

  function test_swapIn_whenExchangeDoesntExist_itReverts() public {
    vm.expectRevert("An exchange with the specified id does not exist");
    biPoolManager.swapIn(0x0, address(0), address(0), 1e24);
  }

  function test_swapIn_whenTradingModeDoesntExist_shouldRevert() public {
    vm.mockCall(address(breaker), abi.encodeWithSelector(breaker.getRateFeedTradingMode.selector), abi.encode(1));
    vm.expectRevert("Trading is suspended for this reference rate");
    biPoolManager.swapIn(exchangeId_cUSD_CELO, address(cEUR), address(cUSD), 1e24);
  }

  function test_swapIn_whenTokenInNotInexchange_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.swapIn(exchangeId_cUSD_CELO, address(cEUR), address(cUSD), 1e24);
  }

  function test_swapIn_whenTokenOutNotInexchange_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.swapIn(exchangeId_cUSD_CELO, address(cUSD), address(cEUR), 1e24);
  }

  function test_swapIn_whenTokenInEqualsTokenOut_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.swapIn(exchangeId_cUSD_CELO, address(cUSD), address(cUSD), 1e24);
  }

  function test_swapIn_whenTokenInIsAsset0_itDelegatesToThePricingModule() public {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);

    uint256 amountIn = 1e24;
    uint256 mockAmountOut = 0.5 * 1e24;

    mockGetAmountOut(constantProduct, mockAmountOut);
    uint256 amountOut = biPoolManager.swapIn(exchangeId_cUSD_CELO, address(cUSD), address(CELO), amountIn);

    IBiPoolManager.PoolExchange memory exchangeAfter = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);
    assertEq(amountOut, mockAmountOut);
    assertEq(exchangeAfter.bucket0, exchange.bucket0 + amountIn);
    assertEq(exchangeAfter.bucket1, exchange.bucket1 - amountOut);
  }

  function test_swapIn_whenTokenInIsAsset1_itDelegatesToThePricingModule() public {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);

    uint256 amountIn = 1e24;
    uint256 mockAmountOut = 0.5 * 1e24;

    mockGetAmountOut(constantProduct, mockAmountOut);
    uint256 amountOut = biPoolManager.swapIn(exchangeId_cUSD_CELO, address(CELO), address(cUSD), amountIn);

    IBiPoolManager.PoolExchange memory exchangeAfter = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);
    assertEq(amountOut, mockAmountOut);
    assertEq(exchangeAfter.bucket0, exchange.bucket0 - amountOut);
    assertEq(exchangeAfter.bucket1, exchange.bucket1 + amountIn);
  }

  function test_swapIn_whenCSAndValidMedian_shouldNotUpdateBuckets() public {
    IBiPoolManager.PoolExchange memory exchangeBefore = biPoolManager.getPoolExchange(exchangeId_cUSD_bridgedUSDC);

    biPoolManager.swapIn(exchangeId_cUSD_bridgedUSDC, address(bridgedUSDC), address(cUSD), 1e6);

    IBiPoolManager.PoolExchange memory exchangeAfter = biPoolManager.getPoolExchange(exchangeId_cUSD_bridgedUSDC);

    assertEq(exchangeBefore.bucket0, exchangeAfter.bucket0);
    assertEq(exchangeBefore.bucket1, exchangeAfter.bucket1);
  }

  function test_swapIn_whenTokenHasNonstandardPrecision_shouldUpdateBucketsCorrectly() public {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cEUR_bridgedUSDC);
    assertEq(exchange.bucket0, 1e26); // stablePoolResetSize = 1e26
    assertEq(exchange.bucket1, 0.5 * 1e26); // mock orackle rate = 2e24

    uint256 amountIn = 1e6;
    uint256 mockAmountOut = 1e18;
    mockGetAmountOut(constantProduct, mockAmountOut);
    uint256 amountOut = biPoolManager.swapIn(
      exchangeId_cEUR_bridgedUSDC,
      address(bridgedUSDC),
      address(cEUR),
      amountIn
    );
    exchange = biPoolManager.getPoolExchange(exchangeId_cEUR_bridgedUSDC);

    assertEq(amountOut, 1e18);
    assertEq(exchange.bucket0, 1e26 - 1e18);
    assertEq(exchange.bucket1, 0.5 * 1e26 + 1e18);
  }

  /* ---------- swapOut --------- */
  function test_swapOut_whenNotBroker_itReverts() public {
    changePrank(deployer);
    vm.expectRevert("Caller is not the Broker");
    biPoolManager.swapOut(0x0, address(0), address(0), 1e24);
  }

  function test_swapOut_whenExchangeDoesntExist_itReverts() public {
    vm.expectRevert("An exchange with the specified id does not exist");
    biPoolManager.swapOut(0x0, address(0), address(0), 1e24);
  }

  function test_swapOut_whenTradingModeDoesntExist_shouldRevert() public {
    vm.mockCall(address(breaker), abi.encodeWithSelector(breaker.getRateFeedTradingMode.selector), abi.encode(2));
    vm.expectRevert("Trading is suspended for this reference rate");
    biPoolManager.swapOut(exchangeId_cUSD_CELO, address(cEUR), address(cUSD), 1e24);
  }

  function test_swapOut_whenTokenInNotInPool_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.swapOut(exchangeId_cUSD_CELO, address(cEUR), address(cUSD), 1e24);
  }

  function test_swapOut_whenTokenOutNotInexchange_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.swapOut(exchangeId_cUSD_CELO, address(cUSD), address(cEUR), 1e24);
  }

  function test_swapOut_whenTokenInEqualsTokenOut_itReverts() public {
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    biPoolManager.swapOut(exchangeId_cUSD_CELO, address(cUSD), address(cUSD), 1e24);
  }

  function test_swapOut_whenTokenInIsAsset0_itDelegatesToThePricingModule() public {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);

    uint256 amountOut = 1e24;
    uint256 mockAmountIn = 0.5 * 1e24;

    mockGetAmountIn(constantProduct, mockAmountIn);
    uint256 amountIn = biPoolManager.swapOut(exchangeId_cUSD_CELO, address(cUSD), address(CELO), amountOut);

    IBiPoolManager.PoolExchange memory exchangeAfter = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);
    assertEq(amountIn, mockAmountIn);
    assertEq(exchangeAfter.bucket0, exchange.bucket0 + amountIn);
    assertEq(exchangeAfter.bucket1, exchange.bucket1 - amountOut);
  }

  function test_swapOut_whenTokenInIsAsset1_itDelegatesToThePricingModule() public {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);

    uint256 amountOut = 1e24;
    uint256 mockAmountIn = 0.6 * 1e24;

    mockGetAmountIn(constantProduct, mockAmountIn);
    uint256 amountIn = biPoolManager.swapOut(exchangeId_cUSD_CELO, address(CELO), address(cUSD), amountOut);

    IBiPoolManager.PoolExchange memory exchangeAfter = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);
    assertEq(amountIn, mockAmountIn);
    assertEq(exchangeAfter.bucket0, exchange.bucket0 - amountOut);
    assertEq(exchangeAfter.bucket1, exchange.bucket1 + amountIn);
  }

  function test_swapOut_whenCSValidMedian_shouldNotUpdateBuckets() public {
    IBiPoolManager.PoolExchange memory exchangeBefore = biPoolManager.getPoolExchange(exchangeId_cUSD_bridgedUSDC);

    biPoolManager.swapOut(exchangeId_cUSD_bridgedUSDC, address(bridgedUSDC), address(cUSD), 1e18);

    IBiPoolManager.PoolExchange memory exchangeAfter = biPoolManager.getPoolExchange(exchangeId_cUSD_bridgedUSDC);

    assertEq(exchangeBefore.bucket0, exchangeAfter.bucket0);
    assertEq(exchangeBefore.bucket1, exchangeAfter.bucket1);
  }

  function test_swapOut_whenTokenHasNonstandardPrecision_shouldUpdateBucketsCorrectly() public {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cEUR_bridgedUSDC);
    assertEq(exchange.bucket0, 1e26); // stablePoolResetSize = 1e26
    assertEq(exchange.bucket1, 0.5 * 1e26); // mock orackle rate = 2e24

    uint256 amountOut = 1e6;
    uint256 mockAmountIn = 1e18;
    mockGetAmountIn(constantProduct, mockAmountIn);
    uint256 amountIn = biPoolManager.swapOut(
      exchangeId_cEUR_bridgedUSDC,
      address(cEUR),
      address(bridgedUSDC),
      amountOut
    );
    exchange = biPoolManager.getPoolExchange(exchangeId_cEUR_bridgedUSDC);

    assertEq(amountIn, 1e18);
    assertEq(exchange.bucket0, 1e26 + 1e18);
    assertEq(exchange.bucket1, 0.5 * 1e26 - 1e18);
  }
}

contract BiPoolManagerTest_bucketUpdates is BiPoolManagerTest_withExchange {
  function setUp() public override {
    super.setUp();
    changePrank(broker);
  }

  function swap(bytes32 exchangeId_cUSD_CELO, uint256 amountIn, uint256 amountOut) internal {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);
    mockGetAmountOut(constantProduct, amountOut);
    biPoolManager.swapIn(exchangeId_cUSD_CELO, exchange.asset0, exchange.asset1, amountIn);
  }

  function test_swapIn_whenBucketsAreStale_updatesBuckets() public {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);
    swap(exchangeId_cUSD_CELO, exchange.bucket0 / 2, exchange.bucket1 / 2); // debalance exchange

    vm.warp(exchange.config.referenceRateResetFrequency + 1);
    sortedOracles.setNumRates(address(cUSD), 10);
    sortedOracles.setMedianTimestamp(address(cUSD), block.timestamp);

    vm.expectEmit(true, true, true, true);
    uint256 stablePoolResetSize = exchange.config.stablePoolResetSize;
    emit BucketsUpdated(
      exchangeId_cUSD_CELO,
      stablePoolResetSize,
      stablePoolResetSize / 2 // due to sortedOracles exchange rate 2:1
    );

    uint256 amountIn = 1e24;
    uint256 amountOut = biPoolManager.swapIn(exchangeId_cUSD_CELO, exchange.asset0, exchange.asset1, 1e24);

    // Refresh exchange
    exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);

    assertEq(stablePoolResetSize + amountIn, exchange.bucket0);
    assertEq((stablePoolResetSize / 2) - amountOut, exchange.bucket1);
  }

  function test_swapIn_whenBucketsAreNotStale_doesNotUpdateBuckets() public {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);
    swap(exchangeId_cUSD_CELO, exchange.bucket0 / 2, exchange.bucket1 / 2); // debalance exchange
    exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO); // Refresh exchange
    uint256 bucket0BeforeSwap = exchange.bucket0;
    uint256 bucket1BeforeSwap = exchange.bucket1;

    uint256 amountIn = 1e24;
    uint256 amountOut = biPoolManager.swapIn(exchangeId_cUSD_CELO, exchange.asset0, exchange.asset1, 1e24);

    exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO); // Refresh exchange

    /*
     * XXX: Because forge doesn't support an inverse to `expectEmit` we
     * can't verify that calling swap doesn't emit BucketsUpdated
     * but we can verify that the buckets were not reset before
     * the swap, but are based on the "debalanced" exchange.
     */

    assertEq(bucket0BeforeSwap + amountIn, exchange.bucket0);
    assertEq(bucket1BeforeSwap - amountOut, exchange.bucket1);
  }

  function test_swapIn_whenBucketsAreStale_butMinReportsNotMet_doesNotUpdateBuckets() public {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);
    swap(exchangeId_cUSD_CELO, exchange.bucket0 / 2, exchange.bucket1 / 2); // debalance exchange
    exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO); // Refresh exchange
    uint256 bucket0BeforeSwap = exchange.bucket0;
    uint256 bucket1BeforeSwap = exchange.bucket1;

    vm.warp(exchange.config.referenceRateResetFrequency + 1);
    sortedOracles.setNumRates(address(cUSD), 4);
    sortedOracles.setMedianTimestampToNow(address(cUSD));

    uint256 amountIn = 1e24;
    uint256 amountOut = biPoolManager.swapIn(exchangeId_cUSD_CELO, exchange.asset0, exchange.asset1, 1e24);

    exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO); // Refresh exchange

    assertEq(bucket0BeforeSwap + amountIn, exchange.bucket0);
    assertEq(bucket1BeforeSwap - amountOut, exchange.bucket1);
  }

  function test_swapIn_whenBucketsAreStale_butReportIsExpired_doesNotUpdateBuckets() public {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);
    swap(exchangeId_cUSD_CELO, exchange.bucket0 / 2, exchange.bucket1 / 2); // debalance exchange
    exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO); // Refresh exchange
    uint256 bucket0BeforeSwap = exchange.bucket0;
    uint256 bucket1BeforeSwap = exchange.bucket1;

    vm.warp(exchange.config.referenceRateResetFrequency + 1);
    sortedOracles.setOldestReportExpired(address(cUSD));
    sortedOracles.setNumRates(address(cUSD), 10);
    sortedOracles.setMedianTimestampToNow(address(cUSD));

    uint256 amountIn = 1e24;
    uint256 amountOut = biPoolManager.swapIn(exchangeId_cUSD_CELO, exchange.asset0, exchange.asset1, 1e24);

    exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO); // Refresh exchange

    assertEq(bucket0BeforeSwap + amountIn, exchange.bucket0);
    assertEq(bucket1BeforeSwap - amountOut, exchange.bucket1);
  }

  function test_swapIn_whenBucketsAreStale_butMedianTimestampIsOld_doesNotUpdateBuckets() public {
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO);
    swap(exchangeId_cUSD_CELO, exchange.bucket0 / 2, exchange.bucket1 / 2); // debalance exchange
    exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO); // Refresh exchange
    uint256 bucket0BeforeSwap = exchange.bucket0;
    uint256 bucket1BeforeSwap = exchange.bucket1;

    vm.warp(exchange.config.referenceRateResetFrequency + 1);
    sortedOracles.setNumRates(address(cUSD), 10);
    sortedOracles.setMedianTimestamp(address(cUSD), block.timestamp - exchange.config.referenceRateResetFrequency);

    uint256 amountIn = 1e24;
    uint256 amountOut = biPoolManager.swapIn(exchangeId_cUSD_CELO, exchange.asset0, exchange.asset1, 1e24);

    exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_CELO); // Refresh exchange

    assertEq(bucket0BeforeSwap + amountIn, exchange.bucket0);
    assertEq(bucket1BeforeSwap - amountOut, exchange.bucket1);
  }
}

contract BiPoolManagerTest_ConstantSum is BiPoolManagerTest_withExchange {
  address EURUSDC_rateFeedID;

  function setUp() public override {
    super.setUp();
    EURUSDC_rateFeedID = address(uint160(uint256(keccak256(abi.encodePacked("EURUSDC")))));
    mockOracleRate(EURUSDC_rateFeedID, 1e24);
    sortedOracles.setMedianTimestampToNow(EURUSDC_rateFeedID);
    exchangeId_cEUR_bridgedUSDC = createExchange(
      cEUR,
      bridgedUSDC,
      constantSum,
      EURUSDC_rateFeedID,
      FixidityLib.wrap(0.1 * 1e24), // spread
      1e24, // stablePoolResetSize
      5 // minimumReports
    );
  }

  function test_quotesAndSwaps_whenMedianNotRecent_shouldRevert() public {
    address USDUSDC_rateFeedID = address(uint160(uint256(keccak256(abi.encodePacked("USDUSDC")))));
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(exchangeId_cUSD_bridgedUSDC);
    sortedOracles.setMedianTimestamp(USDUSDC_rateFeedID, block.timestamp - exchange.config.referenceRateResetFrequency);

    vm.expectRevert("no valid median");
    biPoolManager.getAmountOut(exchangeId_cUSD_bridgedUSDC, address(cUSD), address(bridgedUSDC), 1e18);
    vm.expectRevert("no valid median");
    biPoolManager.getAmountIn(exchangeId_cUSD_bridgedUSDC, address(cUSD), address(bridgedUSDC), 1e18);
    changePrank(broker);
    vm.expectRevert("no valid median");
    biPoolManager.swapIn(exchangeId_cUSD_bridgedUSDC, address(cUSD), address(bridgedUSDC), 1e18);
    vm.expectRevert("no valid median");
    biPoolManager.swapOut(exchangeId_cUSD_bridgedUSDC, address(cUSD), address(bridgedUSDC), 1e18);
  }

  function test_quotesAndSwaps_whenNotEnoughReports_shouldRevert() public {
    assertEq(sortedOracles.numRates(EURUSDC_rateFeedID), 0);

    vm.expectRevert("no valid median");
    biPoolManager.getAmountOut(exchangeId_cEUR_bridgedUSDC, address(cEUR), address(bridgedUSDC), 1e18);

    vm.expectRevert("no valid median");
    biPoolManager.getAmountIn(exchangeId_cEUR_bridgedUSDC, address(cEUR), address(bridgedUSDC), 1e18);

    changePrank(broker);
    vm.expectRevert("no valid median");
    biPoolManager.swapIn(exchangeId_cEUR_bridgedUSDC, address(cEUR), address(bridgedUSDC), 1e18);
    vm.expectRevert("no valid median");
    biPoolManager.swapOut(exchangeId_cEUR_bridgedUSDC, address(cEUR), address(bridgedUSDC), 1e18);
  }

  function test_quotesAndSwaps_whenOldestReportExpired_shouldRevert() public {
    address USDUSDC_rateFeedID = address(uint160(uint256(keccak256(abi.encodePacked("USDUSDC")))));
    sortedOracles.setOldestReportExpired(USDUSDC_rateFeedID);

    vm.expectRevert("no valid median");
    biPoolManager.getAmountOut(exchangeId_cUSD_bridgedUSDC, address(cUSD), address(bridgedUSDC), 1e18);
    vm.expectRevert("no valid median");
    biPoolManager.getAmountIn(exchangeId_cUSD_bridgedUSDC, address(cUSD), address(bridgedUSDC), 1e18);
    changePrank(broker);
    vm.expectRevert("no valid median");
    biPoolManager.swapIn(exchangeId_cUSD_bridgedUSDC, address(cUSD), address(bridgedUSDC), 1e18);
    vm.expectRevert("no valid median");
    biPoolManager.swapOut(exchangeId_cUSD_bridgedUSDC, address(cUSD), address(bridgedUSDC), 1e18);
  }
}
