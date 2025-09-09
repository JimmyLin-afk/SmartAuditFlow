// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase

import { Test } from "forge-std/Test.sol";
import { ERC20Mock } from "openzeppelin-contracts-next/contracts/mocks/ERC20Mock.sol";
import { GoodDollarExpansionController } from "contracts/goodDollar/GoodDollarExpansionController.sol";

import { IGoodDollarExpansionController } from "contracts/interfaces/IGoodDollarExpansionController.sol";
import { IGoodDollarExchangeProvider } from "contracts/interfaces/IGoodDollarExchangeProvider.sol";
import { IBancorExchangeProvider } from "contracts/interfaces/IBancorExchangeProvider.sol";
import { IDistributionHelper } from "contracts/goodDollar/interfaces/IGoodProtocol.sol";

import { GoodDollarExpansionControllerHarness } from "test/utils/harnesses/GoodDollarExpansionControllerHarness.sol";

contract GoodDollarExpansionControllerTest is Test {
  /* ------- Events from IGoodDollarExpansionController ------- */

  event GoodDollarExchangeProviderUpdated(address indexed exchangeProvider);

  event DistributionHelperUpdated(address indexed distributionHelper);

  event ReserveUpdated(address indexed reserve);

  event AvatarUpdated(address indexed avatar);

  event ExpansionConfigSet(bytes32 indexed exchangeId, uint64 expansionRate, uint32 expansionFrequency);

  event RewardMinted(bytes32 indexed exchangeId, address indexed to, uint256 amount);

  event InterestUBIMinted(bytes32 indexed exchangeId, uint256 amount);

  event ExpansionUBIMinted(bytes32 indexed exchangeId, uint256 amount);

  /* ------------------------------------------- */

  ERC20Mock public reserveToken;
  ERC20Mock public token;

  address public exchangeProvider;
  address public distributionHelper;
  address public reserveAddress;
  address public avatarAddress;

  bytes32 exchangeId = "ExchangeId";

  uint64 expansionRate = 1e18 * 0.01;
  uint32 expansionFrequency = uint32(1 days);

  IBancorExchangeProvider.PoolExchange pool;

  function setUp() public virtual {
    reserveToken = new ERC20Mock("cUSD", "cUSD", address(this), 1);
    token = new ERC20Mock("Good$", "G$", address(this), 1);

    exchangeProvider = makeAddr("ExchangeProvider");
    distributionHelper = makeAddr("DistributionHelper");
    reserveAddress = makeAddr("Reserve");
    avatarAddress = makeAddr("Avatar");

    pool = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: 7 * 1e9 * 1e18,
      reserveBalance: 200_000 * 1e18,
      reserveRatio: 0.2 * 1e8, // 20%
      exitContribution: 0.1 * 1e8 // 10%
    });

    vm.mockCall(
      exchangeProvider,
      abi.encodeWithSelector(IBancorExchangeProvider(exchangeProvider).getPoolExchange.selector),
      abi.encode(pool)
    );
  }

  function initializeGoodDollarExpansionController() internal returns (GoodDollarExpansionController) {
    GoodDollarExpansionController expansionController = new GoodDollarExpansionController(false);

    expansionController.initialize(exchangeProvider, distributionHelper, reserveAddress, avatarAddress);

    return expansionController;
  }
}

contract GoodDollarExpansionControllerTest_initializerSettersGetters is GoodDollarExpansionControllerTest {
  GoodDollarExpansionController expansionController;

  function setUp() public override {
    super.setUp();
    expansionController = initializeGoodDollarExpansionController();
  }

  /* ---------- Initilizer ---------- */

  function test_initializer() public view {
    assertEq(address(expansionController.distributionHelper()), distributionHelper);
    assertEq(expansionController.reserve(), reserveAddress);
    assertEq(address(expansionController.goodDollarExchangeProvider()), exchangeProvider);
    assertEq(expansionController.AVATAR(), avatarAddress);
  }

  /* ---------- Getters ---------- */

  function test_getExpansionConfig_whenConfigIsNotSet_shouldRevert() public {
    vm.expectRevert("Expansion config not set");
    expansionController.getExpansionConfig("NotSetExchangeId");
  }

  function test_getExpansionConfig_whenConfigIsSet_shouldReturnConfig() public {
    vm.prank(avatarAddress);
    expansionController.setExpansionConfig(exchangeId, expansionRate, expansionFrequency);

    IGoodDollarExpansionController.ExchangeExpansionConfig memory config = expansionController.getExpansionConfig(
      exchangeId
    );

    assertEq(config.expansionRate, expansionRate);
    assertEq(config.expansionFrequency, expansionFrequency);
    assertEq(config.lastExpansion, 0);
  }

  /* ---------- Setters ---------- */

  function test_setGoodDollarExchangeProvider_whenSenderIsNotOwner_shouldRevert() public {
    vm.prank(makeAddr("NotOwner"));
    vm.expectRevert("Ownable: caller is not the owner");
    expansionController.setGoodDollarExchangeProvider(makeAddr("NewExchangeProvider"));
  }

  function test_setGoodDollarExchangeProvider_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("GoodDollarExchangeProvider address must be set");
    expansionController.setGoodDollarExchangeProvider(address(0));
  }

  function test_setGoodDollarExchangeProvider_whenSenderIsOwner_shouldUpdateAndEmit() public {
    address newExchangeProvider = makeAddr("NewExchangeProvider");
    vm.expectEmit(true, true, true, true);
    emit GoodDollarExchangeProviderUpdated(newExchangeProvider);
    expansionController.setGoodDollarExchangeProvider(newExchangeProvider);

    assertEq(address(expansionController.goodDollarExchangeProvider()), newExchangeProvider);
  }

  function test_setDistributionHelper_whenCallerIsNotAvatar_shouldRevert() public {
    vm.prank(makeAddr("NotAvatar"));
    vm.expectRevert("Only Avatar can call this function");
    expansionController.setDistributionHelper(makeAddr("NewDistributionHelper"));
  }

  function test_setDistributionHelper_whenCallerIsOwner_shouldRevert() public {
    vm.expectRevert("Only Avatar can call this function");
    expansionController.setDistributionHelper(makeAddr("NewDistributionHelper"));
  }

  function test_setDistributionHelper_whenAddressIsZero_shouldRevert() public {
    vm.startPrank(avatarAddress);
    vm.expectRevert("Distribution helper address must be set");
    expansionController.setDistributionHelper(address(0));
    vm.stopPrank();
  }

  function test_setDistributionHelper_whenCallerIsAvatar_shouldUpdateAndEmit() public {
    vm.startPrank(avatarAddress);
    address newDistributionHelper = makeAddr("NewDistributionHelper");
    vm.expectEmit(true, true, true, true);
    emit DistributionHelperUpdated(newDistributionHelper);
    expansionController.setDistributionHelper(newDistributionHelper);

    assertEq(address(expansionController.distributionHelper()), newDistributionHelper);
    vm.stopPrank();
  }

  function test_setReserve_whenSenderIsNotOwner_shouldRevert() public {
    vm.prank(makeAddr("NotOwner"));
    vm.expectRevert("Ownable: caller is not the owner");
    expansionController.setReserve(makeAddr("NewReserve"));
  }

  function test_setReserve_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("Reserve address must be set");
    expansionController.setReserve(address(0));
  }

  function test_setReserve_whenCallerIsOwner_shouldUpdateAndEmit() public {
    address newReserve = makeAddr("NewReserve");
    vm.expectEmit(true, true, true, true);
    emit ReserveUpdated(newReserve);
    expansionController.setReserve(newReserve);

    assertEq(expansionController.reserve(), newReserve);
  }

  function test_setAvatar_whenSenderIsNotOwner_shouldRevert() public {
    vm.prank(makeAddr("NotOwner"));
    vm.expectRevert("Ownable: caller is not the owner");
    expansionController.setAvatar(makeAddr("NewAvatar"));
  }

  function test_setAvatar_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("Avatar address must be set");
    expansionController.setAvatar(address(0));
  }

  function test_setAvatar_whenCallerIsOwner_shouldUpdateAndEmit() public {
    address newAvatar = makeAddr("NewAvatar");
    vm.expectEmit(true, true, true, true);
    emit AvatarUpdated(newAvatar);
    expansionController.setAvatar(newAvatar);

    assertEq(expansionController.AVATAR(), newAvatar);
  }

  function test_setExpansionConfig_whenSenderIsNotAvatar_shouldRevert() public {
    vm.prank(makeAddr("NotAvatar"));
    vm.expectRevert("Only Avatar can call this function");
    expansionController.setExpansionConfig(exchangeId, expansionRate, expansionFrequency);
  }

  function test_setExpansionConfig_whenExpansionRateIsLargerOrEqualToOne_shouldRevert() public {
    expansionRate = 1e18;

    vm.prank(avatarAddress);
    vm.expectRevert("Expansion rate must be less than 100%");
    expansionController.setExpansionConfig(exchangeId, expansionRate, expansionFrequency);
  }

  function test_setExpansionConfig_whenExpansionRateIsZero_shouldRevert() public {
    expansionRate = 0;

    vm.prank(avatarAddress);
    vm.expectRevert("Expansion rate must be greater than 0");
    expansionController.setExpansionConfig(exchangeId, expansionRate, expansionFrequency);
  }

  function test_setExpansionConfig_whenExpansionFrequencyIsZero_shouldRevert() public {
    expansionFrequency = 0;

    vm.prank(avatarAddress);
    vm.expectRevert("Expansion frequency must be greater than 0");
    expansionController.setExpansionConfig(exchangeId, expansionRate, expansionFrequency);
  }

  function test_setExpansionConfig_whenCallerIsAvatar_shouldUpdateAndEmit() public {
    vm.prank(avatarAddress);
    vm.expectEmit(true, true, true, true);
    emit ExpansionConfigSet(exchangeId, expansionRate, expansionFrequency);
    expansionController.setExpansionConfig(exchangeId, expansionRate, expansionFrequency);

    IGoodDollarExpansionController.ExchangeExpansionConfig memory config = expansionController.getExpansionConfig(
      exchangeId
    );

    assertEq(config.expansionRate, expansionRate);
    assertEq(config.expansionFrequency, expansionFrequency);
    assertEq(config.lastExpansion, 0);
  }
}

contract GoodDollarExpansionControllerTest_mintUBIFromInterest is GoodDollarExpansionControllerTest {
  GoodDollarExpansionController expansionController;

  function setUp() public override {
    super.setUp();
    expansionController = initializeGoodDollarExpansionController();
    vm.prank(avatarAddress);
    expansionController.setExpansionConfig(exchangeId, expansionRate, expansionFrequency);

    vm.mockCall(
      exchangeProvider,
      abi.encodeWithSelector(IGoodDollarExchangeProvider(exchangeProvider).mintFromInterest.selector),
      abi.encode(1000e18)
    );
  }

  function test_mintUBIFromInterest_whenReserveInterestIs0_shouldRevert() public {
    vm.expectRevert("Reserve interest must be greater than 0");
    expansionController.mintUBIFromInterest(exchangeId, 0);
  }

  function test_mintUBIFromInterest_whenAmountToMintIsLargerThan0_shouldMintTransferAndEmit() public {
    uint256 reserveInterest = 1000e18;
    uint256 amountToMint = 1000e18;
    address interestCollector = makeAddr("InterestCollector");

    deal(address(reserveToken), interestCollector, reserveInterest);
    assertEq(reserveToken.balanceOf(interestCollector), reserveInterest);

    uint256 interestCollectorBalanceBefore = reserveToken.balanceOf(interestCollector);
    uint256 reserveBalanceBefore = reserveToken.balanceOf(reserveAddress);
    uint256 distributionHelperBalanceBefore = token.balanceOf(distributionHelper);

    vm.startPrank(interestCollector);
    reserveToken.approve(address(expansionController), reserveInterest);

    vm.expectEmit(true, true, true, true);
    emit InterestUBIMinted(exchangeId, amountToMint);
    expansionController.mintUBIFromInterest(exchangeId, reserveInterest);

    assertEq(reserveToken.balanceOf(reserveAddress), reserveBalanceBefore + reserveInterest);
    assertEq(token.balanceOf(distributionHelper), distributionHelperBalanceBefore + amountToMint);
    assertEq(reserveToken.balanceOf(interestCollector), interestCollectorBalanceBefore - reserveInterest);
  }
}

contract GoodDollarExpansionControllerTest_mintUBIFromReserveBalance is GoodDollarExpansionControllerTest {
  GoodDollarExpansionController expansionController;

  function setUp() public override {
    super.setUp();
    expansionController = initializeGoodDollarExpansionController();
    vm.prank(avatarAddress);
    expansionController.setExpansionConfig(exchangeId, expansionRate, expansionFrequency);

    vm.mockCall(
      exchangeProvider,
      abi.encodeWithSelector(IGoodDollarExchangeProvider(exchangeProvider).mintFromInterest.selector),
      abi.encode(1000e18)
    );
  }

  function test_mintUBIFromReserveBalance_whenAdditionalReserveBalanceIs0_shouldReturn0() public {
    deal(address(reserveToken), reserveAddress, pool.reserveBalance);
    uint256 amountMinted = expansionController.mintUBIFromReserveBalance(exchangeId);
    assertEq(amountMinted, 0);
  }

  function test_mintUBIFromReserveBalance_whenAdditionalReserveBalanceIsLargerThan0_shouldMintAndEmit() public {
    uint256 amountToMint = 1000e18;
    uint256 additionalReserveBalance = 1000e18;

    deal(address(reserveToken), reserveAddress, pool.reserveBalance + additionalReserveBalance);

    uint256 distributionHelperBalanceBefore = token.balanceOf(distributionHelper);

    vm.expectEmit(true, true, true, true);
    emit InterestUBIMinted(exchangeId, amountToMint);
    uint256 amountMinted = expansionController.mintUBIFromReserveBalance(exchangeId);

    assertEq(amountMinted, amountToMint);
    assertEq(token.balanceOf(distributionHelper), distributionHelperBalanceBefore + amountToMint);
  }
}

contract GoodDollarExpansionControllerTest_mintUBIFromExpansion is GoodDollarExpansionControllerTest {
  GoodDollarExpansionController expansionController;

  function setUp() public override {
    super.setUp();
    expansionController = initializeGoodDollarExpansionController();
    vm.prank(avatarAddress);
    expansionController.setExpansionConfig(exchangeId, expansionRate, expansionFrequency);

    vm.mockCall(
      exchangeProvider,
      abi.encodeWithSelector(IGoodDollarExchangeProvider(exchangeProvider).mintFromExpansion.selector),
      abi.encode(1000e18)
    );

    vm.mockCall(
      distributionHelper,
      abi.encodeWithSelector(IDistributionHelper(distributionHelper).onDistribution.selector),
      abi.encode(true)
    );
  }

  function test_mintUBIFromExpansion_whenExpansionConfigIsNotSet_shouldRevert() public {
    vm.expectRevert("Expansion config not set");
    expansionController.mintUBIFromExpansion("NotSetExchangeId");
  }

  function test_mintUBIFromExpansion_whenShouldNotExpand_shouldNotExpand() public {
    // doing one initial expansion to not be first expansion
    // since on first expansion the expansion is always applied once.
    expansionController.mintUBIFromExpansion(exchangeId);

    IGoodDollarExpansionController.ExchangeExpansionConfig memory config = expansionController.getExpansionConfig(
      exchangeId
    );
    uint256 lastExpansion = config.lastExpansion;
    skip(lastExpansion + config.expansionFrequency - 1);

    assertEq(expansionController.mintUBIFromExpansion(exchangeId), 0);
  }

  function test_mintUBIFromExpansion_whenFirstExpansionAndLessThanExpansionFrequencyPassed_shouldExpand1Time() public {
    // 1 day has passed since last expansion and expansion rate is 1% so the rate passed to the exchangeProvider
    // should be 0.99^1 = 0.99
    IGoodDollarExpansionController.ExchangeExpansionConfig memory config = expansionController.getExpansionConfig(
      exchangeId
    );
    assert(block.timestamp < config.lastExpansion + config.expansionFrequency);
    uint256 reserveRatioScalar = 1e18 * 0.99;
    uint256 amountToMint = 1000e18;
    uint256 distributionHelperBalanceBefore = token.balanceOf(distributionHelper);

    vm.expectEmit(true, true, true, true);
    emit ExpansionUBIMinted(exchangeId, amountToMint);

    vm.expectCall(
      exchangeProvider,
      abi.encodeWithSelector(
        IGoodDollarExchangeProvider(exchangeProvider).mintFromExpansion.selector,
        exchangeId,
        reserveRatioScalar
      )
    );
    vm.expectCall(
      distributionHelper,
      abi.encodeWithSelector(IDistributionHelper(distributionHelper).onDistribution.selector, amountToMint)
    );

    uint256 amountMinted = expansionController.mintUBIFromExpansion(exchangeId);
    config = expansionController.getExpansionConfig(exchangeId);

    assertEq(amountMinted, amountToMint);
    assertEq(token.balanceOf(distributionHelper), distributionHelperBalanceBefore + amountToMint);
    assertEq(config.lastExpansion, block.timestamp);
  }

  function test_mintUBIFromExpansion_whenFirstExpansionAndMultipleExpansionFrequenciesPassed_shouldExpand1Time()
    public
  {
    // 1 day has passed since last expansion and expansion rate is 1% so the rate passed to the exchangeProvider
    // should be 0.99^1 = 0.99
    IGoodDollarExpansionController.ExchangeExpansionConfig memory config = expansionController.getExpansionConfig(
      exchangeId
    );
    skip(config.expansionFrequency * 3 + 1);
    assert(block.timestamp > config.lastExpansion + config.expansionFrequency * 3);
    uint256 reserveRatioScalar = 1e18 * 0.99;
    uint256 amountToMint = 1000e18;
    uint256 distributionHelperBalanceBefore = token.balanceOf(distributionHelper);

    vm.expectEmit(true, true, true, true);
    emit ExpansionUBIMinted(exchangeId, amountToMint);

    vm.expectCall(
      exchangeProvider,
      abi.encodeWithSelector(
        IGoodDollarExchangeProvider(exchangeProvider).mintFromExpansion.selector,
        exchangeId,
        reserveRatioScalar
      )
    );
    vm.expectCall(
      distributionHelper,
      abi.encodeWithSelector(IDistributionHelper(distributionHelper).onDistribution.selector, amountToMint)
    );

    uint256 amountMinted = expansionController.mintUBIFromExpansion(exchangeId);
    config = expansionController.getExpansionConfig(exchangeId);

    assertEq(amountMinted, amountToMint);
    assertEq(token.balanceOf(distributionHelper), distributionHelperBalanceBefore + amountToMint);
    assertEq(config.lastExpansion, block.timestamp);
  }

  function test_mintUBIFromExpansion_when1DayPassed_shouldCalculateCorrectRateAndExpand() public {
    // doing one initial expansion to not be first expansion
    // since on first expansion the expansion is always applied once.
    expansionController.mintUBIFromExpansion(exchangeId);

    // 1 day has passed since last expansion and expansion rate is 1% so the rate passed to the exchangeProvider
    // should be 0.99^1 = 0.99
    uint256 reserveRatioScalar = 1e18 * 0.99;
    skip(expansionFrequency + 1);

    uint256 amountToMint = 1000e18;
    uint256 distributionHelperBalanceBefore = token.balanceOf(distributionHelper);

    vm.expectEmit(true, true, true, true);
    emit ExpansionUBIMinted(exchangeId, amountToMint);

    vm.expectCall(
      exchangeProvider,
      abi.encodeWithSelector(
        IGoodDollarExchangeProvider(exchangeProvider).mintFromExpansion.selector,
        exchangeId,
        reserveRatioScalar
      )
    );
    vm.expectCall(
      distributionHelper,
      abi.encodeWithSelector(IDistributionHelper(distributionHelper).onDistribution.selector, amountToMint)
    );

    uint256 amountMinted = expansionController.mintUBIFromExpansion(exchangeId);
    IGoodDollarExpansionController.ExchangeExpansionConfig memory config = expansionController.getExpansionConfig(
      exchangeId
    );

    assertEq(amountMinted, amountToMint);
    assertEq(token.balanceOf(distributionHelper), distributionHelperBalanceBefore + amountToMint);
    assertEq(config.lastExpansion, block.timestamp);
  }

  function test_mintUBIFromExpansion_whenMultipleDaysPassed_shouldCalculateCorrectRateAndExpand() public {
    // doing one initial expansion to not be first expansion
    // since on first expansion the expansion is always applied once.
    expansionController.mintUBIFromExpansion(exchangeId);

    // 3 days have passed since last expansion and expansion rate is 1% so the rate passed to the exchangeProvider
    // should be 0.99^3 = 0.970299
    uint256 reserveRatioScalar = 1e18 * 0.970299;

    skip(3 * expansionFrequency + 1);

    uint256 amountToMint = 1000e18;
    uint256 distributionHelperBalanceBefore = token.balanceOf(distributionHelper);

    vm.expectEmit(true, true, true, true);
    emit ExpansionUBIMinted(exchangeId, amountToMint);

    vm.expectCall(
      exchangeProvider,
      abi.encodeWithSelector(
        IGoodDollarExchangeProvider(exchangeProvider).mintFromExpansion.selector,
        exchangeId,
        reserveRatioScalar
      )
    );
    vm.expectCall(
      distributionHelper,
      abi.encodeWithSelector(IDistributionHelper(distributionHelper).onDistribution.selector, amountToMint)
    );

    uint256 amountMinted = expansionController.mintUBIFromExpansion(exchangeId);
    IGoodDollarExpansionController.ExchangeExpansionConfig memory config = expansionController.getExpansionConfig(
      exchangeId
    );

    assertEq(amountMinted, amountToMint);
    assertEq(token.balanceOf(distributionHelper), distributionHelperBalanceBefore + amountToMint);
    assertEq(config.lastExpansion, block.timestamp);
  }
}

contract GoodDollarExpansionControllerTest_getExpansionScalar is GoodDollarExpansionControllerTest {
  GoodDollarExpansionControllerHarness expansionController;

  function setUp() public override {
    super.setUp();
    expansionController = new GoodDollarExpansionControllerHarness(false);
  }

  function test_getExpansionScaler_whenExpansionRateIs0_shouldReturn1e18() public {
    IGoodDollarExpansionController.ExchangeExpansionConfig memory config = IGoodDollarExpansionController
      .ExchangeExpansionConfig(0, 1, 0);
    assertEq(expansionController.exposed_getReserveRatioScalar(config), 1e18);
  }

  function test_getExpansionScaler_whenExpansionRateIs1_shouldReturn1() public {
    IGoodDollarExpansionController.ExchangeExpansionConfig memory config = IGoodDollarExpansionController
      .ExchangeExpansionConfig(1e18 - 1, 1, 0);
    assertEq(expansionController.exposed_getReserveRatioScalar(config), 1);
  }

  function testFuzz_getExpansionScaler(
    uint256 _expansionRate,
    uint256 _expansionFrequency,
    uint256 _lastExpansion,
    uint256 _timeDelta
  ) public {
    uint64 expansionRate = uint64(bound(_expansionRate, 1, 1e18 - 1));
    uint32 expansionFrequency = uint32(bound(_expansionFrequency, 1, 1e6));
    uint32 lastExpansion = uint32(bound(_lastExpansion, 0, 1e6));
    uint32 timeDelta = uint32(bound(_timeDelta, 0, 1e6));

    skip(lastExpansion + timeDelta);

    IGoodDollarExpansionController.ExchangeExpansionConfig memory config = IGoodDollarExpansionController
      .ExchangeExpansionConfig(expansionRate, expansionFrequency, lastExpansion);
    uint256 scaler = expansionController.exposed_getReserveRatioScalar(config);
    assert(scaler >= 0 && scaler <= 1e18);
  }
}

contract GoodDollarExpansionControllerTest_mintRewardFromReserveRatio is GoodDollarExpansionControllerTest {
  GoodDollarExpansionController expansionController;

  function setUp() public override {
    super.setUp();
    expansionController = initializeGoodDollarExpansionController();
    vm.prank(avatarAddress);
    expansionController.setExpansionConfig(exchangeId, expansionRate, expansionFrequency);

    vm.mockCall(
      exchangeProvider,
      abi.encodeWithSelector(IGoodDollarExchangeProvider(exchangeProvider).updateRatioForReward.selector),
      abi.encode(true)
    );
  }

  function test_mintRewardFromReserveRatio_whenCallerIsNotAvatar_shouldRevert() public {
    vm.prank(makeAddr("NotAvatar"));
    vm.expectRevert("Only Avatar can call this function");
    expansionController.mintRewardFromReserveRatio(exchangeId, makeAddr("To"), 1000e18);
  }

  function test_mintRewardFromReserveRatio_whenToIsZero_shouldRevert() public {
    vm.prank(avatarAddress);
    vm.expectRevert("Recipient address must be set");
    expansionController.mintRewardFromReserveRatio(exchangeId, address(0), 1000e18);
  }

  function test_mintRewardFromReserveRatio_whenAmountIs0_shouldRevert() public {
    vm.prank(avatarAddress);
    vm.expectRevert("Amount must be greater than 0");
    expansionController.mintRewardFromReserveRatio(exchangeId, makeAddr("To"), 0);
  }

  function test_mintRewardFromReserveRatio_whenCallerIsAvatar_shouldMintAndEmit() public {
    uint256 amountToMint = 1000e18;
    address to = makeAddr("To");
    uint256 toBalanceBefore = token.balanceOf(to);

    vm.expectEmit(true, true, true, true);
    emit RewardMinted(exchangeId, to, amountToMint);

    vm.prank(avatarAddress);
    expansionController.mintRewardFromReserveRatio(exchangeId, to, amountToMint);

    assertEq(token.balanceOf(to), toBalanceBefore + amountToMint);
  }
}
