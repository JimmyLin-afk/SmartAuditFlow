// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase

import { Test } from "forge-std/Test.sol";
import { GoodDollarExchangeProvider } from "contracts/goodDollar/GoodDollarExchangeProvider.sol";
import { ERC20 } from "openzeppelin-contracts-next/contracts/token/ERC20/ERC20.sol";

import { IReserve } from "contracts/interfaces/IReserve.sol";
import { IExchangeProvider } from "contracts/interfaces/IExchangeProvider.sol";
import { IBancorExchangeProvider } from "contracts/interfaces/IBancorExchangeProvider.sol";

contract GoodDollarExchangeProviderTest is Test {
  /* ------- Events from IGoodDollarExchangeProvider ------- */

  event ExpansionControllerUpdated(address indexed expansionController);

  event AvatarUpdated(address indexed AVATAR);

  event ReserveRatioUpdated(bytes32 indexed exchangeId, uint32 reserveRatio);

  event ExchangeCreated(bytes32 indexed exchangeId, address indexed reserveAsset, address indexed tokenAddress);

  event ExchangeDestroyed(bytes32 indexed exchangeId, address indexed reserveAsset, address indexed tokenAddress);

  event ExitContributionSet(bytes32 indexed exchangeId, uint256 exitContribution);

  /* ------------------------------------------- */

  ERC20 public reserveToken;
  ERC20 public token;
  ERC20 public token2;

  address public reserveAddress;
  address public brokerAddress;
  address public avatarAddress;
  address public expansionControllerAddress;

  IBancorExchangeProvider.PoolExchange public poolExchange1;
  IBancorExchangeProvider.PoolExchange public poolExchange2;
  IBancorExchangeProvider.PoolExchange public poolExchange;

  function setUp() public virtual {
    reserveToken = new ERC20("cUSD", "cUSD");
    token = new ERC20("Good$", "G$");
    token2 = new ERC20("Good2$", "G2$");

    reserveAddress = makeAddr("Reserve");
    brokerAddress = makeAddr("Broker");
    avatarAddress = makeAddr("Avatar");
    expansionControllerAddress = makeAddr("ExpansionController");

    poolExchange1 = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: 300_000 * 1e18,
      reserveBalance: 60_000 * 1e18,
      reserveRatio: 0.2 * 1e8,
      exitContribution: 0.01 * 1e8
    });

    poolExchange2 = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token2),
      tokenSupply: 300_000 * 1e18,
      reserveBalance: 60_000 * 1e18,
      reserveRatio: 1e8 * 0.2,
      exitContribution: 1e8 * 0.01
    });

    poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: 7_000_000_000 * 1e18,
      reserveBalance: 200_000 * 1e18,
      reserveRatio: 1e8 * 0.28571428,
      exitContribution: 1e8 * 0.1
    });

    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isStableAsset.selector, address(token)),
      abi.encode(true)
    );
    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isStableAsset.selector, address(token2)),
      abi.encode(true)
    );
    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isCollateralAsset.selector, address(reserveToken)),
      abi.encode(true)
    );
  }

  function initializeGoodDollarExchangeProvider() internal returns (GoodDollarExchangeProvider) {
    GoodDollarExchangeProvider exchangeProvider = new GoodDollarExchangeProvider(false);

    exchangeProvider.initialize(brokerAddress, reserveAddress, expansionControllerAddress, avatarAddress);
    return exchangeProvider;
  }
}

contract GoodDollarExchangeProviderTest_initializerSettersGetters is GoodDollarExchangeProviderTest {
  GoodDollarExchangeProvider exchangeProvider;

  function setUp() public override {
    super.setUp();
    exchangeProvider = initializeGoodDollarExchangeProvider();
  }

  /* ---------- Initilizer ---------- */

  function test_initializer() public view {
    assertEq(exchangeProvider.owner(), address(this));
    assertEq(exchangeProvider.broker(), brokerAddress);
    assertEq(address(exchangeProvider.reserve()), reserveAddress);
    assertEq(address(exchangeProvider.expansionController()), expansionControllerAddress);
    assertEq(exchangeProvider.AVATAR(), avatarAddress);
  }

  /* ---------- Setters ---------- */

  function test_setAvatar_whenSenderIsNotOwner_shouldRevert() public {
    vm.prank(makeAddr("NotOwner"));
    vm.expectRevert("Ownable: caller is not the owner");
    exchangeProvider.setAvatar(makeAddr("NewAvatar"));
  }

  function test_setAvatar_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("Avatar address must be set");
    exchangeProvider.setAvatar(address(0));
  }

  function test_setAvatar_whenSenderIsOwner_shouldUpdateAndEmit() public {
    address newAvatar = makeAddr("NewAvatar");
    vm.expectEmit(true, true, true, true);
    emit AvatarUpdated(newAvatar);
    exchangeProvider.setAvatar(newAvatar);

    assertEq(exchangeProvider.AVATAR(), newAvatar);
  }

  function test_setExpansionController_whenSenderIsNotOwner_shouldRevert() public {
    vm.prank(makeAddr("NotOwner"));
    vm.expectRevert("Ownable: caller is not the owner");
    exchangeProvider.setExpansionController(makeAddr("NewExpansionController"));
  }

  function test_setExpansionController_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("ExpansionController address must be set");
    exchangeProvider.setExpansionController(address(0));
  }

  function test_setExpansionController_whenSenderIsOwner_shouldUpdateAndEmit() public {
    address newExpansionController = makeAddr("NewExpansionController");
    vm.expectEmit(true, true, true, true);
    emit ExpansionControllerUpdated(newExpansionController);
    exchangeProvider.setExpansionController(newExpansionController);

    assertEq(address(exchangeProvider.expansionController()), newExpansionController);
  }

  /* ---------- setExitContribution ---------- */
  /* Focuses only on access control, implementation details are covered in BancorExchangeProvider tests */
  function test_setExitContribution_whenSenderIsOwner_shouldRevert() public {
    vm.expectRevert("Only Avatar can call this function");
    bytes32 exchangeId = "0xexchangeId";
    exchangeProvider.setExitContribution(exchangeId, 1e5);
  }

  function test_setExitContribution_whenSenderIsNotAvatar_shouldRevert() public {
    vm.startPrank(makeAddr("NotAvatarAndNotOwner"));
    vm.expectRevert("Only Avatar can call this function");
    bytes32 exchangeId = "0xexchangeId";
    exchangeProvider.setExitContribution(exchangeId, 1e5);
    vm.stopPrank();
  }

  function test_setExitContribution_whenSenderIsAvatar_shouldUpdateAndEmit() public {
    vm.startPrank(avatarAddress);
    bytes32 exchangeId = exchangeProvider.createExchange(poolExchange1);
    uint32 newExitContribution = 1e3;
    vm.expectEmit(true, true, true, true);
    emit ExitContributionSet(exchangeId, newExitContribution);
    exchangeProvider.setExitContribution(exchangeId, newExitContribution);

    IBancorExchangeProvider.PoolExchange memory poolExchange = exchangeProvider.getPoolExchange(exchangeId);
    assertEq(poolExchange.exitContribution, newExitContribution);
    vm.stopPrank();
  }
  /* ---------- setExitContribution end ---------- */
}

/**
 * @notice createExchange tests
 * @dev These tests focus only on access control. The implementation details
 *      are covered in the BancorExchangeProvider tests.
 */
contract GoodDollarExchangeProviderTest_createExchange is GoodDollarExchangeProviderTest {
  GoodDollarExchangeProvider exchangeProvider;

  function setUp() public override {
    super.setUp();
    exchangeProvider = initializeGoodDollarExchangeProvider();
  }

  function test_createExchange_whenSenderIsNotAvatar_shouldRevert() public {
    vm.prank(makeAddr("NotAvatar"));
    vm.expectRevert("Only Avatar can call this function");
    exchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenSenderIsOwner_shouldRevert() public {
    vm.expectRevert("Only Avatar can call this function");
    exchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenSenderIsAvatar_shouldCreateExchangeAndEmit() public {
    vm.startPrank(avatarAddress);
    vm.expectEmit(true, true, true, true);
    bytes32 expectedExchangeId = keccak256(abi.encodePacked(reserveToken.symbol(), token.symbol()));
    emit ExchangeCreated(expectedExchangeId, address(reserveToken), address(token));
    bytes32 exchangeId = exchangeProvider.createExchange(poolExchange1);
    assertEq(exchangeId, expectedExchangeId);

    IBancorExchangeProvider.PoolExchange memory poolExchange = exchangeProvider.getPoolExchange(exchangeId);
    assertEq(poolExchange.reserveAsset, poolExchange1.reserveAsset);
    assertEq(poolExchange.tokenAddress, poolExchange1.tokenAddress);
    assertEq(poolExchange.tokenSupply, poolExchange1.tokenSupply);
    assertEq(poolExchange.reserveBalance, poolExchange1.reserveBalance);
    assertEq(poolExchange.reserveRatio, poolExchange1.reserveRatio);
    assertEq(poolExchange.exitContribution, poolExchange1.exitContribution);

    IExchangeProvider.Exchange[] memory exchanges = exchangeProvider.getExchanges();
    assertEq(exchanges.length, 1);
    assertEq(exchanges[0].exchangeId, exchangeId);

    assertEq(exchangeProvider.tokenPrecisionMultipliers(address(reserveToken)), 1);
    assertEq(exchangeProvider.tokenPrecisionMultipliers(address(token)), 1);
    vm.stopPrank();
  }
}

/**
 * @notice destroyExchange tests
 * @dev These tests focus only on access control. The implementation details
 *      are covered in the BancorExchangeProvider tests.
 */
contract GoodDollarExchangeProviderTest_destroyExchange is GoodDollarExchangeProviderTest {
  GoodDollarExchangeProvider exchangeProvider;

  function setUp() public override {
    super.setUp();
    exchangeProvider = initializeGoodDollarExchangeProvider();
  }

  function test_destroyExchange_whenSenderIsOwner_shouldRevert() public {
    vm.startPrank(avatarAddress);
    bytes32 exchangeId = exchangeProvider.createExchange(poolExchange1);
    vm.stopPrank();
    vm.expectRevert("Only Avatar can call this function");
    exchangeProvider.destroyExchange(exchangeId, 0);
  }

  function test_destroyExchange_whenSenderIsNotAvatar_shouldRevert() public {
    vm.startPrank(avatarAddress);
    bytes32 exchangeId = exchangeProvider.createExchange(poolExchange1);
    vm.stopPrank();

    vm.startPrank(makeAddr("NotAvatar"));
    vm.expectRevert("Only Avatar can call this function");
    exchangeProvider.destroyExchange(exchangeId, 0);
    vm.stopPrank();
  }

  function test_destroyExchange_whenExchangeExists_shouldDestroyExchangeAndEmit() public {
    vm.startPrank(avatarAddress);
    bytes32 exchangeId = exchangeProvider.createExchange(poolExchange1);
    bytes32 exchangeId2 = exchangeProvider.createExchange(poolExchange2);
    vm.stopPrank();

    vm.startPrank(avatarAddress);
    vm.expectEmit(true, true, true, true);
    emit ExchangeDestroyed(exchangeId, poolExchange1.reserveAsset, poolExchange1.tokenAddress);
    exchangeProvider.destroyExchange(exchangeId, 0);

    bytes32[] memory exchangeIds = exchangeProvider.getExchangeIds();
    assertEq(exchangeIds.length, 1);

    IExchangeProvider.Exchange[] memory exchanges = exchangeProvider.getExchanges();
    assertEq(exchanges.length, 1);
    assertEq(exchanges[0].exchangeId, exchangeId2);
    vm.stopPrank();
  }
}

contract GoodDollarExchangeProviderTest_mintFromExpansion is GoodDollarExchangeProviderTest {
  GoodDollarExchangeProvider exchangeProvider;
  bytes32 exchangeId;
  uint256 expansionRate;
  uint256 reserveRatioScalar;

  function setUp() public override {
    super.setUp();
    // based on a yearly expansion rate of 10% the daily rate is:
    // (1-x)^365 = 0.9 -> x = 1 - 0.9^(1/365) = 0.00028861728902231263...
    expansionRate = 288617289022312;
    reserveRatioScalar = 1e18 - expansionRate;
    exchangeProvider = initializeGoodDollarExchangeProvider();
    vm.prank(avatarAddress);
    exchangeId = exchangeProvider.createExchange(poolExchange);
  }

  function test_mintFromExpansion_whenCallerIsNotExpansionController_shouldRevert() public {
    vm.prank(makeAddr("NotExpansionController"));
    vm.expectRevert("Only ExpansionController can call this function");
    exchangeProvider.mintFromExpansion(exchangeId, expansionRate);
  }

  function test_mintFromExpansionRate_whenReserveRatioScalarIs0_shouldRevert() public {
    vm.prank(expansionControllerAddress);
    vm.expectRevert("Reserve ratio scalar must be greater than 0");
    exchangeProvider.mintFromExpansion(exchangeId, 0);
  }

  function test_mintFromExpansion_whenExchangeIdIsInvalid_shouldRevert() public {
    vm.prank(expansionControllerAddress);
    vm.expectRevert("Exchange does not exist");
    exchangeProvider.mintFromExpansion(bytes32(0), expansionRate);
  }

  function test_mintFromExpansion_whenNewRatioIsZero_shouldRevert() public {
    uint256 verySmallReserveRatioScalar = 1;

    vm.expectRevert("New ratio must be greater than 0");
    vm.prank(expansionControllerAddress);
    exchangeProvider.mintFromExpansion(exchangeId, verySmallReserveRatioScalar);
  }

  function test_mintFromExpansion_whenReserveRatioScalarIs100Percent_shouldReturn0() public {
    vm.prank(expansionControllerAddress);
    uint256 amountToMint = exchangeProvider.mintFromExpansion(exchangeId, 1e18);
    assertEq(amountToMint, 0, "Minted amount should be 0");
  }

  function test_mintFromExpansion_whenValidReserveRatioScalar_shouldReturnCorrectAmountAndEmit() public {
    // reserveRatioScalar is (1-0.000288617289022312) based of 10% yearly expansion rate
    // Formula: amountToMint = (tokenSupply * reserveRatio - tokenSupply * newRatio) / newRatio
    // newRatio = reserveRatio * reserveRatioScalar = 0.28571428 * (1-0.000288617289022312) = 0.285631817919071438
    // amountToMint = (7_000_000_000 * 0.28571428 - 7_000_000_000 * 0.285631817919071438) / 0.285631817919071438
    // ≈ 2_020_904,291074052815139287
    uint32 expectedReserveRatio = 28563181;
    uint256 expectedAmountToMint = 2020904291074052815139287;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.expectEmit(true, true, true, true);
    emit ReserveRatioUpdated(exchangeId, expectedReserveRatio);
    vm.prank(expansionControllerAddress);
    uint256 amountToMint = exchangeProvider.mintFromExpansion(exchangeId, reserveRatioScalar);

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertEq(amountToMint, expectedAmountToMint, "Minted amount should be correct");
    assertEq(
      poolExchangeAfter.tokenSupply,
      poolExchange.tokenSupply + amountToMint,
      "Token supply should increase by minted amount"
    );
    assertEq(poolExchangeAfter.reserveRatio, expectedReserveRatio, "Reserve ratio should be updated correctly");
    // 0.01% relative error tolerance because of precision loss when new reserve ratio is calculated
    assertApproxEqRel(priceBefore, priceAfter, 1e18 * 0.0001, "Price should remain within 0.01% of initial price");
  }

  function test_mintFromExpansion_withSmallReserveRatioScalar_shouldReturnCorrectAmount() public {
    uint256 smallReserveRatioScalar = 1e18 * 0.00001; // 0.001%
    // Formula: amountToMint = (tokenSupply * reserveRatio - tokenSupply * newRatio) / newRatio
    // newRatio = reserveRatio * reserveRatioScalar = 0.28571428 * 1e13/1e18 = 0.0000028571428
    // amountToMint = (7_000_000_000 * 0.28571428 - 7_000_000_000 * 0.0000028571428) /0.0000028571428
    // amountToMint ≈ 699993000000000
    uint32 expectedReserveRatio = 285;
    uint256 expectedAmountToMint = 699993000000000 * 1e18;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.expectEmit(true, true, true, true);
    emit ReserveRatioUpdated(exchangeId, expectedReserveRatio);
    vm.prank(expansionControllerAddress);
    uint256 amountToMint = exchangeProvider.mintFromExpansion(exchangeId, smallReserveRatioScalar);

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertEq(amountToMint, expectedAmountToMint, "Minted amount should be correct");
    assertEq(
      poolExchangeAfter.tokenSupply,
      poolExchange.tokenSupply + amountToMint,
      "Token supply should increase by minted amount"
    );
    assertEq(poolExchangeAfter.reserveRatio, expectedReserveRatio, "Reserve ratio should be updated correctly");
    // 1% relative error tolerance because of precision loss when new reserve ratio is calculated
    assertApproxEqRel(priceBefore, priceAfter, 1e18 * 0.01, "Price should remain within 1% of initial price");
  }

  function test_mintFromExpansion_withLargeReserveRatioScalar_shouldReturnCorrectAmount() public {
    uint256 largeReserveRatioScalar = 1e18 - 1; // Just below 100%
    // Formula: amountToMint = (tokenSupply * reserveRatio - tokenSupply * newRatio) / newRatio
    // newRatio = reserveRatio * reserveRatioScalar = 0.28571428 * (1e18 -1)/1e18 ≈ 0.285714279999999999
    // amountToMint = (7_000_000_000 * 0.28571428 - 7_000_000_000 * 0.285714279999999999) /0.285714279999999999
    // amountToMint ≈ 0.00000002450000049000
    uint32 expectedReserveRatio = 28571427;
    uint256 expectedAmountToMint = 24500000490;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.expectEmit(true, true, true, true);
    emit ReserveRatioUpdated(exchangeId, expectedReserveRatio);
    vm.prank(expansionControllerAddress);
    uint256 amountToMint = exchangeProvider.mintFromExpansion(exchangeId, largeReserveRatioScalar);

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertEq(amountToMint, expectedAmountToMint, "Minted amount should be correct");
    assertEq(
      poolExchangeAfter.tokenSupply,
      poolExchange.tokenSupply + amountToMint,
      "Token supply should increase by minted amount"
    );
    assertEq(poolExchangeAfter.reserveRatio, expectedReserveRatio, "Reserve ratio should be updated correctly");
    // 0.01% relative error tolerance because of precision loss when new reserve ratio is calculated
    assertApproxEqRel(priceBefore, priceAfter, 1e18 * 0.0001, "Price should remain within 0.01% of initial price");
  }

  function test_mintFromExpansion_withMultipleConsecutiveExpansions_shouldMintCorrectly() public {
    uint256 totalMinted = 0;
    uint256 initialTokenSupply = poolExchange.tokenSupply;
    uint32 initialReserveRatio = poolExchange.reserveRatio;
    uint256 initialReserveBalance = poolExchange.reserveBalance;
    uint256 initialPrice = exchangeProvider.currentPrice(exchangeId);

    vm.startPrank(expansionControllerAddress);
    for (uint256 i = 0; i < 5; i++) {
      uint256 amountToMint = exchangeProvider.mintFromExpansion(exchangeId, reserveRatioScalar);
      totalMinted += amountToMint;
      assertGt(amountToMint, 0, "Amount minted should be greater than 0");
    }
    vm.stopPrank();

    // Calculate expected reserve ratio
    // daily Scalar is applied 5 times newRatio = initialReserveRatio * (dailyScalar ** 5)
    // newRatio = 0.28571428 * (0.999711382710977688 ** 5) ≈ 0.2853022075264986
    uint256 expectedReserveRatio = 28530220;

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertEq(
      poolExchangeAfter.tokenSupply,
      initialTokenSupply + totalMinted,
      "Token supply should increase by total minted amount"
    );
    assertLt(poolExchangeAfter.reserveRatio, initialReserveRatio, "Reserve ratio should decrease");
    assertEq(poolExchangeAfter.reserveBalance, initialReserveBalance, "Reserve balance should remain unchanged");
    assertApproxEqRel(
      poolExchangeAfter.reserveRatio,
      uint32(expectedReserveRatio),
      1e18 * 0.0001, // 0.01% relative error tolerance because of precision loss when new reserve ratio is calculated
      "Reserve ratio should be updated correctly within 0.01% tolerance"
    );
    assertApproxEqRel(initialPrice, priceAfter, 1e18 * 0.0001, "Price should remain within 0.01% of initial price");
  }

  function testFuzz_mintFromExpansion(uint256 reserveRatioScalar) public {
    // 0.001% to 100%
    reserveRatioScalar = bound(reserveRatioScalar, 1e18 * 0.00001, 1e18);

    uint256 initialTokenSupply = poolExchange.tokenSupply;
    uint32 initialReserveRatio = poolExchange.reserveRatio;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    uint256 expectedReserveRatio = (uint256(initialReserveRatio) * reserveRatioScalar) / 1e18;

    vm.expectEmit(true, true, true, true);
    emit ReserveRatioUpdated(exchangeId, uint32(expectedReserveRatio));
    vm.prank(expansionControllerAddress);
    uint256 amountToMint = exchangeProvider.mintFromExpansion(exchangeId, reserveRatioScalar);

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertGe(amountToMint, 0, "Minted amount should be greater or equal than 0");
    assertGe(initialReserveRatio, poolExchangeAfter.reserveRatio, "Reserve ratio should decrease");
    assertEq(
      poolExchangeAfter.tokenSupply,
      initialTokenSupply + amountToMint,
      "Token supply should increase by minted amount"
    );
    // 1% relative error tolerance because of precision loss when new reserve ratio is calculated
    assertApproxEqRel(priceBefore, priceAfter, 1e18 * 0.01, "Price should remain within 1% of initial price");
  }
}

contract GoodDollarExchangeProviderTest_mintFromInterest is GoodDollarExchangeProviderTest {
  GoodDollarExchangeProvider exchangeProvider;
  bytes32 exchangeId;
  uint256 reserveInterest;

  function setUp() public override {
    super.setUp();
    reserveInterest = 1000 * 1e18;
    exchangeProvider = initializeGoodDollarExchangeProvider();
    vm.prank(avatarAddress);
    exchangeId = exchangeProvider.createExchange(poolExchange);
  }

  function test_mintFromInterest_whenCallerIsNotExpansionController_shouldRevert() public {
    vm.prank(makeAddr("NotExpansionController"));
    vm.expectRevert("Only ExpansionController can call this function");
    exchangeProvider.mintFromInterest(exchangeId, reserveInterest);
  }

  function test_mintFromInterest_whenExchangeIdIsInvalid_shouldRevert() public {
    vm.prank(expansionControllerAddress);
    vm.expectRevert("Exchange does not exist");
    exchangeProvider.mintFromInterest(bytes32(0), reserveInterest);
  }

  function test_mintFromInterest_whenInterestIs0_shouldReturn0() public {
    vm.prank(expansionControllerAddress);
    uint256 amountToMint = exchangeProvider.mintFromInterest(exchangeId, 0);
    assertEq(amountToMint, 0, "Minted amount should be 0");
  }

  function test_mintFromInterest_whenInterestLarger0_shouldReturnCorrectAmount() public {
    uint256 interest = 1_000 * 1e18;
    // formula: amountToMint = reserveInterest * tokenSupply / reserveBalance
    // amountToMint = 1_000 * 7_000_000_000 / 200_000 = 35_000_000
    uint256 expectedAmountToMint = 35_000_000 * 1e18;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.prank(expansionControllerAddress);
    uint256 amountToMint = exchangeProvider.mintFromInterest(exchangeId, interest);

    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);
    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);

    assertEq(amountToMint, expectedAmountToMint, "Minted amount should be correct");
    assertEq(
      poolExchangeAfter.tokenSupply,
      poolExchange.tokenSupply + amountToMint,
      "Token supply should increase by minted amount"
    );
    assertEq(
      poolExchangeAfter.reserveBalance,
      poolExchange.reserveBalance + interest,
      "Reserve balance should increase by interest amount"
    );
    assertEq(priceBefore, priceAfter, "Price should remain unchanged");
  }

  function test_mintFromInterest_whenInterestIsSmall_shouldReturnCorrectAmount() public {
    uint256 interest = 100; // 100wei
    // formula: amountToMint = reserveInterest * tokenSupply / reserveBalance
    // amountToMint = (100/1e18) * 7_000_000_000 / 200_000 = 0.000000000003500000
    uint256 expectedAmountToMint = 3_500_000;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.prank(expansionControllerAddress);
    uint256 amountToMint = exchangeProvider.mintFromInterest(exchangeId, interest);

    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);
    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);

    assertEq(amountToMint, expectedAmountToMint, "Minted amount should be correct");
    assertEq(
      poolExchangeAfter.tokenSupply,
      poolExchange.tokenSupply + amountToMint,
      "Token supply should increase by minted amount"
    );
    assertEq(
      poolExchangeAfter.reserveBalance,
      poolExchange.reserveBalance + interest,
      "Reserve balance should increase by interest amount"
    );
    assertEq(priceBefore, priceAfter, "Price should remain unchanged");
  }

  function test_mintFromInterest_whenInterestIsLarge_shouldReturnCorrectAmount() public {
    // 1_000_000 reserve tokens 5 times current reserve balance
    uint256 interest = 1_000_000 * 1e18;
    // formula: amountToMint = reserveInterest * tokenSupply / reserveBalance
    // amountToMint = 1_000_000 * 7_000_000_000 / 200_000 = 35_000_000_000
    uint256 expectedAmountToMint = 35_000_000_000 * 1e18;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.prank(expansionControllerAddress);
    uint256 amountToMint = exchangeProvider.mintFromInterest(exchangeId, interest);

    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);
    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);

    assertEq(amountToMint, expectedAmountToMint, "Minted amount should be correct");
    assertEq(
      poolExchangeAfter.tokenSupply,
      poolExchange.tokenSupply + amountToMint,
      "Token supply should increase by minted amount"
    );
    assertEq(
      poolExchangeAfter.reserveBalance,
      poolExchange.reserveBalance + interest,
      "Reserve balance should increase by interest amount"
    );
    assertEq(priceBefore, priceAfter, "Price should remain unchanged");
  }

  function test_mintFromInterest_withMultipleConsecutiveInterests_shouldMintCorrectly() public {
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.startPrank(expansionControllerAddress);
    uint256 totalMinted = 0;
    for (uint256 i = 0; i < 5; i++) {
      uint256 amountToMint = exchangeProvider.mintFromInterest(exchangeId, reserveInterest);
      totalMinted += amountToMint;
    }
    vm.stopPrank();

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertEq(
      poolExchangeAfter.tokenSupply,
      poolExchange.tokenSupply + totalMinted,
      "Token supply should increase by total minted amount"
    );
    assertEq(
      poolExchangeAfter.reserveBalance,
      poolExchange.reserveBalance + reserveInterest * 5,
      "Reserve balance should increase by total interest"
    );
    assertEq(priceBefore, priceAfter, "Price should remain unchanged");
  }

  function testFuzz_mintFromInterest(uint256 fuzzedInterest) public {
    fuzzedInterest = bound(fuzzedInterest, 1, type(uint256).max / poolExchange.tokenSupply);

    uint256 initialTokenSupply = poolExchange.tokenSupply;
    uint256 initialReserveBalance = poolExchange.reserveBalance;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.prank(expansionControllerAddress);
    uint256 amountToMint = exchangeProvider.mintFromInterest(exchangeId, fuzzedInterest);

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertGt(amountToMint, 0, "Minted amount should be greater than 0");
    assertEq(
      poolExchangeAfter.tokenSupply,
      initialTokenSupply + amountToMint,
      "Token supply should increase by minted amount"
    );
    assertEq(
      poolExchangeAfter.reserveBalance,
      initialReserveBalance + fuzzedInterest,
      "Reserve balance should increase by interest amount"
    );
    assertEq(priceBefore, priceAfter, "Price should remain unchanged");
  }
}

contract GoodDollarExchangeProviderTest_updateRatioForReward is GoodDollarExchangeProviderTest {
  GoodDollarExchangeProvider exchangeProvider;
  bytes32 exchangeId;
  uint256 reward;

  function setUp() public override {
    super.setUp();
    reward = 1000 * 1e18;
    exchangeProvider = initializeGoodDollarExchangeProvider();
    vm.prank(avatarAddress);
    exchangeId = exchangeProvider.createExchange(poolExchange);
  }

  function test_updateRatioForReward_whenCallerIsNotExpansionController_shouldRevert() public {
    vm.prank(makeAddr("NotExpansionController"));
    vm.expectRevert("Only ExpansionController can call this function");
    exchangeProvider.updateRatioForReward(exchangeId, reward);
  }

  function test_updateRatioForReward_whenExchangeIdIsInvalid_shouldRevert() public {
    vm.prank(expansionControllerAddress);
    vm.expectRevert("Exchange does not exist");
    exchangeProvider.updateRatioForReward(bytes32(0), reward);
  }

  function test_updateRatioForReward_whenRewardLarger0_shouldReturnCorrectRatioAndEmit() public {
    // formula: newRatio = reserveBalance / ((tokenSupply + reward) * currentPrice)
    // reserveRatio = 200_000 / ((7_000_000_000 + 1_000) * 0.000100000002) ≈ 0.28571423...
    uint32 expectedReserveRatio = 28571423;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.expectEmit(true, true, true, true);
    emit ReserveRatioUpdated(exchangeId, expectedReserveRatio);
    vm.prank(expansionControllerAddress);
    exchangeProvider.updateRatioForReward(exchangeId, reward);

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertEq(poolExchangeAfter.reserveRatio, expectedReserveRatio, "Reserve ratio should be updated correctly");
    assertEq(
      poolExchangeAfter.tokenSupply,
      poolExchange.tokenSupply + reward,
      "Token supply should increase by reward amount"
    );
    // 1% relative error tolerance because of precision loss when new reserve ratio is calculated
    assertApproxEqRel(priceBefore, priceAfter, 1e18 * 0.0001, "Price should remain within 0.01% of initial price");
  }

  function test_updateRatioForReward_whenRewardIsSmall_shouldReturnCorrectRatioAndEmit() public {
    uint256 reward = 1e18; // 1 token
    // formula: newRatio = reserveBalance / ((tokenSupply + reward) * currentPrice)
    // reserveRatio = 200_000 / ((7_000_000_000 + 1) * 0.000100000002) ≈ 0.2857142799
    uint32 expectedReserveRatio = 28571427;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.expectEmit(true, true, true, true);
    emit ReserveRatioUpdated(exchangeId, expectedReserveRatio);
    vm.prank(expansionControllerAddress);
    exchangeProvider.updateRatioForReward(exchangeId, reward);

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertEq(poolExchangeAfter.reserveRatio, expectedReserveRatio, "Reserve ratio should be updated correctly");
    assertEq(
      poolExchangeAfter.tokenSupply,
      poolExchange.tokenSupply + reward,
      "Token supply should increase by reward amount"
    );
    assertApproxEqRel(priceBefore, priceAfter, 1e18 * 0.0001, "Price should remain within 0.01% of initial price");
  }

  function test_updateRatioForReward_whenRewardIsLarge_shouldReturnCorrectRatioAndEmit() public {
    uint256 reward = 1_000_000_000 * 1e18; // 1 billion tokens
    // formula: newRatio = reserveBalance / ((tokenSupply + reward) * currentPrice)
    // reserveRatio = 200_000 / ((7_000_000_000 + 1_000_000_000) * 0.000100000002) ≈ 0.2499999950000...

    uint32 expectedReserveRatio = 24999999;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.expectEmit(true, true, true, true);
    emit ReserveRatioUpdated(exchangeId, expectedReserveRatio);
    vm.prank(expansionControllerAddress);
    exchangeProvider.updateRatioForReward(exchangeId, reward);

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertEq(poolExchangeAfter.reserveRatio, expectedReserveRatio, "Reserve ratio should be updated correctly");
    assertEq(
      poolExchangeAfter.tokenSupply,
      poolExchange.tokenSupply + reward,
      "Token supply should increase by reward amount"
    );
    assertApproxEqRel(priceBefore, priceAfter, 1e18 * 0.0001, "Price should remain within 0.01% of initial price");
  }

  function test_updateRatioForReward_withMultipleConsecutiveRewards() public {
    uint256 totalReward = 0;
    uint256 initialTokenSupply = poolExchange.tokenSupply;
    uint256 initialReserveBalance = poolExchange.reserveBalance;
    uint32 initialReserveRatio = poolExchange.reserveRatio;
    uint256 initialPrice = exchangeProvider.currentPrice(exchangeId);

    vm.startPrank(expansionControllerAddress);
    for (uint256 i = 0; i < 5; i++) {
      exchangeProvider.updateRatioForReward(exchangeId, reward);
      totalReward += reward;
    }
    vm.stopPrank();

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertEq(
      poolExchangeAfter.tokenSupply,
      initialTokenSupply + totalReward,
      "Token supply should increase by total reward"
    );
    assertEq(poolExchangeAfter.reserveBalance, initialReserveBalance, "Reserve balance should remain unchanged");
    assertLt(poolExchangeAfter.reserveRatio, initialReserveRatio, "Reserve ratio should decrease");
    assertApproxEqRel(initialPrice, priceAfter, 1e18 * 0.001, "Price should remain within 0.1% of initial price");
  }

  function testFuzz_updateRatioForReward(uint256 fuzzedReward) public {
    // 1 to 100 trillion tokens
    fuzzedReward = bound(fuzzedReward, 1, 100_000_000_000_000 * 1e18);

    uint256 initialTokenSupply = poolExchange.tokenSupply;
    uint256 initialReserveBalance = poolExchange.reserveBalance;
    uint32 initialReserveRatio = poolExchange.reserveRatio;
    uint256 priceBefore = exchangeProvider.currentPrice(exchangeId);

    vm.prank(expansionControllerAddress);
    exchangeProvider.updateRatioForReward(exchangeId, fuzzedReward);

    IBancorExchangeProvider.PoolExchange memory poolExchangeAfter = exchangeProvider.getPoolExchange(exchangeId);
    uint256 priceAfter = exchangeProvider.currentPrice(exchangeId);

    assertEq(
      poolExchangeAfter.tokenSupply,
      initialTokenSupply + fuzzedReward,
      "Token supply should increase by reward amount"
    );
    assertEq(poolExchangeAfter.reserveBalance, initialReserveBalance, "Reserve balance should remain unchanged");
    assertLe(poolExchangeAfter.reserveRatio, initialReserveRatio, "Reserve ratio should stay the same or decrease");
    assertApproxEqRel(
      priceBefore,
      priceAfter,
      1e18 * 0.001,
      "Price should remain unchanged, with a max relative error of 0.1%"
    );
  }
}

contract GoodDollarExchangeProviderTest_pausable is GoodDollarExchangeProviderTest {
  GoodDollarExchangeProvider exchangeProvider;
  bytes32 exchangeId;

  function setUp() public override {
    super.setUp();
    exchangeProvider = initializeGoodDollarExchangeProvider();
    vm.prank(avatarAddress);
    exchangeId = exchangeProvider.createExchange(poolExchange1);
  }

  function test_pause_whenCallerIsNotAvatar_shouldRevert() public {
    vm.prank(makeAddr("NotAvatar"));
    vm.expectRevert("Only Avatar can call this function");
    exchangeProvider.pause();
  }

  function test_unpause_whenCallerIsNotAvatar_shouldRevert() public {
    vm.prank(makeAddr("NotAvatar"));
    vm.expectRevert("Only Avatar can call this function");
    exchangeProvider.unpause();
  }

  function test_pause_whenCallerIsAvatar_shouldPauseAndDisableExchange() public {
    vm.prank(avatarAddress);
    exchangeProvider.pause();

    assert(exchangeProvider.paused());

    vm.startPrank(brokerAddress);
    vm.expectRevert("Pausable: paused");
    exchangeProvider.swapIn(exchangeId, address(reserveToken), address(token), 1e18);

    vm.expectRevert("Pausable: paused");
    exchangeProvider.swapOut(exchangeId, address(reserveToken), address(token), 1e18);

    vm.startPrank(expansionControllerAddress);
    vm.expectRevert("Pausable: paused");
    exchangeProvider.mintFromExpansion(exchangeId, 1e18);

    vm.expectRevert("Pausable: paused");
    exchangeProvider.mintFromInterest(exchangeId, 1e18);

    vm.expectRevert("Pausable: paused");
    exchangeProvider.updateRatioForReward(exchangeId, 1e18);
  }

  function test_unpause_whenCallerIsAvatar_shouldUnpauseAndEnableExchange() public {
    vm.prank(avatarAddress);
    exchangeProvider.pause();

    vm.prank(avatarAddress);
    exchangeProvider.unpause();

    assert(exchangeProvider.paused() == false);

    vm.startPrank(brokerAddress);

    exchangeProvider.swapIn(exchangeId, address(reserveToken), address(token), 1e18);
    exchangeProvider.swapOut(exchangeId, address(reserveToken), address(token), 1e18);

    vm.startPrank(expansionControllerAddress);

    exchangeProvider.mintFromExpansion(exchangeId, 1e18);
    exchangeProvider.mintFromInterest(exchangeId, 1e18);
    exchangeProvider.updateRatioForReward(exchangeId, 1e18);
  }
}
