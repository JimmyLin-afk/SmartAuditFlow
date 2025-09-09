// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

// Libraries
import { TokenHelpers } from "../helpers/TokenHelpers.sol";
import { TradingLimitHelpers } from "../helpers/TradingLimitHelpers.sol";

// Interfaces
import { IBancorExchangeProvider } from "contracts/interfaces/IBancorExchangeProvider.sol";
import { IDistributionHelper } from "contracts/goodDollar/interfaces/IGoodProtocol.sol";

// Contracts
import { GoodDollarBaseForkTest } from "./GoodDollarBaseForkTest.sol";

contract GoodDollarExpansionForkTest is GoodDollarBaseForkTest {
  using TradingLimitHelpers for *;
  using TokenHelpers for *;

  constructor(uint256 _chainId) GoodDollarBaseForkTest(_chainId) {}

  function setUp() public override {
    super.setUp();
  }

  function test_mintFromExpansion() public {
    uint256 priceBefore = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);
    uint256 distributionHelperBalanceBefore = goodDollarToken.balanceOf(distributionHelperAddress);

    vm.mockCall(
      distributionHelperAddress,
      abi.encodeWithSelector(IDistributionHelper(distributionHelperAddress).onDistribution.selector),
      abi.encode(true)
    );

    skip(2 days + 1 seconds);

    uint256 amountMinted = expansionController.mintUBIFromExpansion(exchangeId);
    uint256 priceAfter = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);
    assertApproxEqAbs(priceBefore, priceAfter, 1e11);
    assertEq(goodDollarToken.balanceOf(distributionHelperAddress), amountMinted + distributionHelperBalanceBefore);
  }

  function test_mintFromInterest() public {
    uint256 priceBefore = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);
    address reserveInterestCollector = makeAddr("reserveInterestCollector");
    uint256 reserveInterest = 1000 * 1e18;
    deal(address(reserveToken), reserveInterestCollector, reserveInterest);

    uint256 reserveBalanceBefore = reserveToken.balanceOf(address(goodDollarReserve));
    uint256 interestCollectorBalanceBefore = reserveToken.balanceOf(reserveInterestCollector);
    uint256 distributionHelperBalanceBefore = goodDollarToken.balanceOf(distributionHelperAddress);

    vm.startPrank(reserveInterestCollector);
    reserveToken.approve(address(expansionController), reserveInterest);
    expansionController.mintUBIFromInterest(exchangeId, reserveInterest);
    vm.stopPrank();

    uint256 priceAfter = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);
    uint256 reserveBalanceAfter = reserveToken.balanceOf(address(goodDollarReserve));
    uint256 interestCollectorBalanceAfter = reserveToken.balanceOf(reserveInterestCollector);
    uint256 distributionHelperBalanceAfter = goodDollarToken.balanceOf(distributionHelperAddress);

    assertEq(reserveBalanceAfter, reserveBalanceBefore + reserveInterest);
    assertEq(interestCollectorBalanceAfter, interestCollectorBalanceBefore - reserveInterest);
    assertTrue(distributionHelperBalanceBefore < distributionHelperBalanceAfter);
    assertEq(priceBefore, priceAfter);
  }
}
