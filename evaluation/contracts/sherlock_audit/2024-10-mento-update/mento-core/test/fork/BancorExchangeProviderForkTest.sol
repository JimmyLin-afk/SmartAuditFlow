// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

// Libraries
import { ERC20 } from "openzeppelin-contracts-next/contracts/token/ERC20/ERC20.sol";

// Contracts
import { BaseForkTest } from "./BaseForkTest.sol";
import { BancorExchangeProvider } from "contracts/goodDollar/BancorExchangeProvider.sol";
import { IBancorExchangeProvider } from "contracts/interfaces/IBancorExchangeProvider.sol";

contract BancorExchangeProviderForkTest is BaseForkTest {
  address ownerAddress;
  address brokerAddress;
  address reserveAddress;
  ERC20 reserveToken;
  ERC20 swapToken;

  BancorExchangeProvider bancorExchangeProvider;
  IBancorExchangeProvider.PoolExchange poolExchange;
  bytes32 exchangeId;

  constructor(uint256 _chainId) BaseForkTest(_chainId) {}

  function setUp() public override {
    super.setUp();
    ownerAddress = makeAddr("owner");
    brokerAddress = address(this.broker());
    reserveAddress = address(mentoReserve);
    reserveToken = ERC20(address(mentoReserve.collateralAssets(0))); // == CELO
    swapToken = ERC20(this.lookup("StableToken")); // == cUSD

    // Deploy and initialize BancorExchangeProvider (includes BancorFormula as part of init)
    setUpBancorExchangeProvider();
  }

  function setUpBancorExchangeProvider() public {
    vm.startPrank(ownerAddress);
    bancorExchangeProvider = new BancorExchangeProvider(false);
    bancorExchangeProvider.initialize(brokerAddress, reserveAddress);
    vm.stopPrank();

    poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(swapToken),
      tokenSupply: 300_000 * 1e18,
      reserveBalance: 60_000 * 1e18,
      reserveRatio: 0.2 * 1e8,
      exitContribution: 0.01 * 1e8
    });

    vm.prank(ownerAddress);
    exchangeId = bancorExchangeProvider.createExchange(poolExchange);
  }

  function test_init_isDeployedAndInitializedCorrectly() public view {
    assertEq(bancorExchangeProvider.owner(), ownerAddress);
    assertEq(bancorExchangeProvider.broker(), brokerAddress);
    assertEq(address(bancorExchangeProvider.reserve()), reserveAddress);

    IBancorExchangeProvider.PoolExchange memory _poolExchange = bancorExchangeProvider.getPoolExchange(exchangeId);
    assertEq(_poolExchange.reserveAsset, _poolExchange.reserveAsset);
    assertEq(_poolExchange.tokenAddress, _poolExchange.tokenAddress);
    assertEq(_poolExchange.tokenSupply, _poolExchange.tokenSupply);
    assertEq(_poolExchange.reserveBalance, _poolExchange.reserveBalance);
    assertEq(_poolExchange.reserveRatio, _poolExchange.reserveRatio);
    assertEq(_poolExchange.exitContribution, _poolExchange.exitContribution);
  }

  function test_swapIn_whenTokenInIsReserveToken_shouldSwapIn() public {
    uint256 amountIn = 1e18;
    uint256 reserveBalanceBefore = poolExchange.reserveBalance;
    uint256 swapTokenSupplyBefore = poolExchange.tokenSupply;

    uint256 expectedAmountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(swapToken),
      amountIn: amountIn
    });
    vm.prank(brokerAddress);
    uint256 amountOut = bancorExchangeProvider.swapIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(swapToken),
      amountIn: amountIn
    });
    assertEq(amountOut, expectedAmountOut);

    (, , uint256 swapTokenSupplyAfter, uint256 reserveBalanceAfter, , ) = bancorExchangeProvider.exchanges(exchangeId);

    assertEq(reserveBalanceAfter, reserveBalanceBefore + amountIn);
    assertEq(swapTokenSupplyAfter, swapTokenSupplyBefore + amountOut);
  }

  function test_swapIn_whenTokenInIsSwapToken_shouldSwapIn() public {
    uint256 amountIn = 1e18;
    uint256 reserveBalanceBefore = poolExchange.reserveBalance;
    uint256 swapTokenSupplyBefore = poolExchange.tokenSupply;
    uint256 expectedAmountOut = bancorExchangeProvider.getAmountOut(
      exchangeId,
      address(swapToken),
      address(reserveToken),
      amountIn
    );
    vm.prank(brokerAddress);
    uint256 amountOut = bancorExchangeProvider.swapIn({
      exchangeId: exchangeId,
      tokenIn: address(swapToken),
      tokenOut: address(reserveToken),
      amountIn: amountIn
    });
    assertEq(amountOut, expectedAmountOut);

    (, , uint256 swapTokenSupplyAfter, uint256 reserveBalanceAfter, , ) = bancorExchangeProvider.exchanges(exchangeId);

    assertEq(reserveBalanceAfter, reserveBalanceBefore - amountOut);
    assertEq(swapTokenSupplyAfter, swapTokenSupplyBefore - amountIn);
  }

  function test_swapOut_whenTokenInIsReserveToken_shouldSwapOut() public {
    uint256 amountOut = 1e18;
    uint256 reserveBalanceBefore = poolExchange.reserveBalance;
    uint256 swapTokenSupplyBefore = poolExchange.tokenSupply;
    uint256 expectedAmountIn = bancorExchangeProvider.getAmountIn(
      exchangeId,
      address(reserveToken),
      address(swapToken),
      amountOut
    );
    vm.prank(brokerAddress);
    uint256 amountIn = bancorExchangeProvider.swapOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(swapToken),
      amountOut: amountOut
    });
    assertEq(amountIn, expectedAmountIn);

    (, , uint256 swapTokenSupplyAfter, uint256 reserveBalanceAfter, , ) = bancorExchangeProvider.exchanges(exchangeId);

    assertEq(reserveBalanceAfter, reserveBalanceBefore + amountIn);
    assertEq(swapTokenSupplyAfter, swapTokenSupplyBefore + amountOut);
  }

  function test_swapOut_whenTokenInIsSwapToken_shouldSwapOut() public {
    uint256 amountOut = 1e18;
    uint256 reserveBalanceBefore = poolExchange.reserveBalance;
    uint256 swapTokenSupplyBefore = poolExchange.tokenSupply;
    uint256 expectedAmountIn = bancorExchangeProvider.getAmountIn(
      exchangeId,
      address(swapToken),
      address(reserveToken),
      amountOut
    );
    vm.prank(brokerAddress);

    uint256 amountIn = bancorExchangeProvider.swapOut({
      exchangeId: exchangeId,
      tokenIn: address(swapToken),
      tokenOut: address(reserveToken),
      amountOut: amountOut
    });
    assertEq(amountIn, expectedAmountIn);

    (, , uint256 swapTokenSupplyAfter, uint256 reserveBalanceAfter, , ) = bancorExchangeProvider.exchanges(exchangeId);

    assertEq(reserveBalanceAfter, reserveBalanceBefore - amountOut);
    assertEq(swapTokenSupplyAfter, swapTokenSupplyBefore - amountIn);
  }
}
