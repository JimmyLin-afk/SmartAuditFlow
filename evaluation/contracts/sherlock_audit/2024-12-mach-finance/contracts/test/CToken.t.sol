// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import "../src/CErc20Delegator.sol";
import "../src/CErc20Delegate.sol";
import "../src/ComptrollerInterface.sol";
import "../src/InterestRateModel.sol";
import "../src/Comptroller.sol";
import "../src/JumpRateModelV2.sol";
import "./BaseTest.t.sol";
import "./mocks/MockRewardDistributor.sol";
import "./mocks/SimplePriceOracle.sol";

contract CTokenTest is BaseTest {
    address public alice = makeAddr("alice");
    address public bob = makeAddr("bob");
    address public charlie = makeAddr("charlie");
    address public dave = makeAddr("dave");

    SimplePriceOracle public priceOracle;

    function setUp() public {
        _deployBaselineContracts();

        vm.startPrank(admin);
        priceOracle = new SimplePriceOracle();

        uint256 btcPrice = 100_000 * 10 ** (36 - wbtc.decimals());
        priceOracle.setUnderlyingPrice(CToken(address(cWbtcDelegator)), btcPrice);

        uint256 sonicPrice = 1 * 10 ** (36 - 18);
        priceOracle.setUnderlyingPrice(CToken(address(cSonic)), sonicPrice);

        comptroller._setPriceOracle(priceOracle);

        comptroller._setCollateralFactor(CToken(address(cWbtcDelegator)), 0.75e18);
        comptroller._setCollateralFactor(CToken(address(cSonic)), 0.5e18);
        vm.stopPrank();
    }

    function test_sweepNative() public {
        vm.deal(address(cWbtcDelegator), 100 ether);
        vm.prank(admin);
        CErc20Delegate(address(cWbtcDelegator)).sweepNative();

        assertEq(address(cWbtcDelegator).balance, 0 ether);
        assertEq(address(admin).balance, 100 ether);
    }

    function test_mint_hitSupplyCap() public {
        deal(address(wbtc), admin, 1000 ether);
        deal(admin, 1e9 ether);
        vm.startPrank(admin);

        CToken[] memory cTokens = new CToken[](2);
        cTokens[0] = CToken(address(cWbtcDelegator));
        cTokens[1] = CToken(address(cSonic));
        uint256[] memory supplyCaps = new uint256[](2);
        supplyCaps[0] = 100 ether;
        supplyCaps[1] = 1e6 ether;
        comptroller._setMarketSupplyCaps(cTokens, supplyCaps);

        // Approve allowance
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);

        cWbtcDelegator.mint(99 ether);
        cSonic.mint{value: 9.99e5 ether}();

        vm.expectRevert("market supply cap reached");
        cWbtcDelegator.mint(1.5 ether);

        vm.expectRevert("market supply cap reached");
        cSonic.mint{value: 1e5 ether}();

        // Update supply cap
        uint256[] memory newSupplyCaps = new uint256[](2);
        newSupplyCaps[0] = 150 ether;
        newSupplyCaps[1] = 1.1e6 ether;

        comptroller._setMarketSupplyCaps(cTokens, newSupplyCaps);

        cWbtcDelegator.mint(1.5 ether);
        cSonic.mint{value: 1 ether}();
    }

    function test_setProtocolSeizeShare() public {
        vm.prank(admin);
        cWbtcDelegator._setProtocolSeizeShare(0.5e18);
        vm.assertEq(cWbtcDelegator.protocolSeizeShareMantissa(), 0.5e18);

        vm.expectRevert(abi.encodeWithSelector(SetProtocolSeizeShareOwnerCheck.selector));
        vm.prank(alice);
        cWbtcDelegator._setProtocolSeizeShare(0.1e18);
    }

    function test_interestRateAccrual() public {
        // Setup initial balances (enough to cover minting)
        uint256 wBTCInitialBalance = 100 * (10 ** wbtc.decimals());
        deal(address(wbtc), alice, wBTCInitialBalance);
        deal(address(wbtc), bob, wBTCInitialBalance);
        deal(address(wbtc), charlie, wBTCInitialBalance);

        uint256 sonicInitialBalance = 1e18 ether;
        deal(alice, sonicInitialBalance);
        deal(bob, sonicInitialBalance);
        deal(charlie, sonicInitialBalance);

        uint256 wBTCMintAmount = 10 * (10 ** wbtc.decimals());

        address[] memory marketsToEnter = new address[](2);
        marketsToEnter[0] = address(cWbtcDelegator);
        marketsToEnter[1] = address(cSonic);

        // Supply wBTC
        vm.startPrank(alice);
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.mint(wBTCMintAmount);
        vm.stopPrank();

        // Supply wBTC as collateral
        vm.startPrank(bob);
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.mint(wBTCMintAmount);
        comptroller.enterMarkets(marketsToEnter);
        vm.stopPrank();

        // Supply Sonic as collateral
        vm.startPrank(charlie);
        cSonic.mint{value: 1e6 ether}();
        comptroller.enterMarkets(marketsToEnter);
        vm.stopPrank();

        // Borrow SONIC against wBTC
        vm.prank(bob);
        cSonic.borrow(1e4 ether);

        // Borrow wBTC against Sonic
        vm.prank(charlie);
        cWbtcDelegator.borrow(1 * 1e8);

        // Record initial state
        uint256 initialBTCBorrows = cWbtcDelegator.totalBorrows();
        uint256 initialBTCBorrowIndex = cWbtcDelegator.borrowIndex();

        // Simulate interest accrual daily for one year
        for (uint256 i = 0; i < 365; i++) {
            // Advance 1 day
            vm.warp(block.timestamp + 1 days);

            // Accrue interest
            cWbtcDelegator.accrueInterest();
            cSonic.accrueInterest();
        }

        // Get final state
        uint256 finalBTCBorrows = cWbtcDelegator.totalBorrows();
        uint256 finalBTCBorrowIndex = cWbtcDelegator.borrowIndex();

        // Ensure final borrow index is greater than initial borrow index
        assertGt(finalBTCBorrowIndex, initialBTCBorrowIndex, "BTC borrow index should increase");

        // Calculated via Python notebook
        uint256 expectedBTCBorrowAmount = 101265191;
        uint256 expectedBTCBorrowIndex = 1012044310264898940;

        // 0.38% delta is enough to pass, compared to Python notebook calculations
        assertApproxEqRel(finalBTCBorrows, expectedBTCBorrowAmount, 0.0038e18);
        assertApproxEqRel(finalBTCBorrowIndex, expectedBTCBorrowIndex, 0.0038e18);
    }

    function test_rewardDistributorCount() public {
        // Setup initial balances (enough to cover operations)
        deal(alice, 100 ether);
        deal(bob, 100 ether);
        deal(charlie, 100 ether);

        vm.startPrank(admin);
        MockRewardDistributor rewardDistributor = new MockRewardDistributor();
        comptroller._setRewardDistributor(rewardDistributor);
        vm.stopPrank();

        vm.startPrank(alice);
        cSonic.mint{value: 10 ether}();
        vm.stopPrank();

        vm.startPrank(bob);
        cSonic.mint{value: 10 ether}();

        address[] memory marketsToEnter = new address[](1);
        marketsToEnter[0] = address(cSonic);
        comptroller.enterMarkets(marketsToEnter);

        // Borrow Sonic
        cSonic.borrow(1 ether);
        vm.stopPrank();

        // Warp 1 day
        vm.warp(block.timestamp + 1 days);

        // Repay 0.5 sonic
        vm.prank(bob);
        cSonic.repayBorrow{value: 0.5 ether}();

        vm.prank(alice);
        cSonic.redeemUnderlying(1 ether);

        // Warp 1 day
        vm.warp(block.timestamp + 1 days);

        vm.prank(alice);
        comptroller.claimReward(alice);

        vm.prank(bob);
        comptroller.claimReward(bob);

        // Check reward distributor counts (number of times each function is called)
        assertEq(rewardDistributor.updateSupplyIndexAndDisburseSupplierRewardsCount(), 3);
        assertEq(rewardDistributor.updateBorrowIndexAndDisburseBorrowerRewardsCount(), 2);

        assertEq(rewardDistributor.disburseSupplierRewardsCount(), 4);
        assertEq(rewardDistributor.disburseBorrowerRewardsCount(), 4);

        assertEq(rewardDistributor.updateSupplyIndexCount(), 4);
        assertEq(rewardDistributor.updateBorrowIndexCount(), 4);
    }

    function test_rewardDistributorCountWithNoDistributor() public {
        // Initial setup without setting reward distributor
        deal(alice, 100 ether);
        deal(address(wbtc), alice, 100e8);

        vm.startPrank(alice);

        // Supply both tokens
        cSonic.mint{value: 10 ether}();
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.mint(10e8);

        // Enter markets
        address[] memory marketsToEnter = new address[](2);
        marketsToEnter[0] = address(cSonic);
        marketsToEnter[1] = address(cWbtcDelegator);
        comptroller.enterMarkets(marketsToEnter);

        // Borrow against collateral
        cSonic.borrow(1 ether);

        // Try to claim rewards (should not revert but no counts should increase)
        vm.warp(block.timestamp + 1 days);

        vm.expectRevert("No reward distributor configured!");
        comptroller.claimReward();
        vm.stopPrank();

        // All counts should be 0 since no distributor was set
        assertEq(address(comptroller.rewardDistributor()), address(0));
    }

    function test_rewardDistributorCountComplexMarketEntryExit() public {
        // Setup
        vm.startPrank(admin);
        MockRewardDistributor rewardDistributor = new MockRewardDistributor();
        comptroller._setRewardDistributor(rewardDistributor);
        vm.stopPrank();

        // Setup initial balances
        deal(alice, 100 ether);
        deal(bob, 100 ether);
        deal(address(wbtc), alice, 100e8);
        deal(address(wbtc), bob, 100e8);

        // Alice's actions
        vm.startPrank(alice);
        cSonic.mint{value: 20 ether}();
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.mint(10e8);

        // Enter both markets
        address[] memory marketsToEnter = new address[](2);
        marketsToEnter[0] = address(cSonic);
        marketsToEnter[1] = address(cWbtcDelegator);
        comptroller.enterMarkets(marketsToEnter);

        // Exit WBTC market
        comptroller.exitMarket(address(cWbtcDelegator));

        // Re-enter WBTC market
        address[] memory reenterMarket = new address[](1);
        reenterMarket[0] = address(cWbtcDelegator);
        comptroller.enterMarkets(reenterMarket);
        vm.stopPrank();

        // Bob's actions
        vm.startPrank(bob);
        cSonic.mint{value: 15 ether}();
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.mint(5e8);

        // Enter only Sonic market
        address[] memory sonicMarket = new address[](1);
        sonicMarket[0] = address(cSonic);
        comptroller.enterMarkets(sonicMarket);
        vm.stopPrank();

        // Advance time and claim rewards
        vm.warp(block.timestamp + 1 days);

        // Claim rewards for both users
        address[] memory holders = new address[](2);
        holders[0] = alice;
        holders[1] = bob;

        CToken[] memory allTokens = new CToken[](2);
        allTokens[0] = cSonic;
        allTokens[1] = CToken(address(cWbtcDelegator));

        comptroller.claimRewards(holders, allTokens, true, true);

        // Verify counts
        assertEq(rewardDistributor.updateSupplyIndexAndDisburseSupplierRewardsCount(), 4); // Alice's 2 mints + Bob's 2 mints
        assertEq(rewardDistributor.updateBorrowIndexAndDisburseBorrowerRewardsCount(), 0);

        // During claim: 2 holders * 2 tokens = 4 disbursements
        assertEq(rewardDistributor.disburseSupplierRewardsCount(), 4);
        assertEq(rewardDistributor.disburseBorrowerRewardsCount(), 4);

        // One update per token during claim
        assertEq(rewardDistributor.updateSupplyIndexCount(), 2);
        assertEq(rewardDistributor.updateBorrowIndexCount(), 2);
    }

    function test_rewardDistributorCountComplexBorrowingScenario() public {
        // Setup
        vm.startPrank(admin);
        MockRewardDistributor rewardDistributor = new MockRewardDistributor();
        comptroller._setRewardDistributor(rewardDistributor);
        vm.stopPrank();

        // Setup initial balances
        deal(alice, 100 ether);
        deal(bob, 100 ether);
        deal(charlie, 1e6 ether);
        deal(dave, 1e6 ether);
        deal(address(wbtc), alice, 100e8);
        deal(address(wbtc), bob, 100e8);
        deal(address(wbtc), charlie, 100e8);
        deal(address(wbtc), dave, 100e8);

        // Setup markets to enter
        address[] memory allMarkets = new address[](2);
        allMarkets[0] = address(cSonic);
        allMarkets[1] = address(cWbtcDelegator);

        // Alice supplies both assets as collateral
        vm.startPrank(alice);
        cSonic.mint{value: 50 ether}();
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.mint(20e8);
        comptroller.enterMarkets(allMarkets);
        vm.stopPrank();

        // Bob supplies WBTC and borrows Sonic
        vm.startPrank(bob);
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.mint(10e8);
        comptroller.enterMarkets(allMarkets);
        cSonic.borrow(10 ether);
        vm.stopPrank();

        // Charlie supplies Sonic and borrows WBTC
        vm.startPrank(charlie);
        cSonic.mint{value: 2.5e5 ether}();
        comptroller.enterMarkets(allMarkets);
        cWbtcDelegator.borrow(1e8);
        vm.stopPrank();

        // Dave mints Sonic
        vm.startPrank(dave);
        cSonic.mint{value: 2.5e5 ether}();
        cSonic.transfer(bob, 1 ether);
        vm.stopPrank();

        // Advance time
        vm.warp(block.timestamp + 1 days);

        // Partial repayments
        vm.startPrank(bob);
        cSonic.repayBorrow{value: 5 ether}();
        vm.stopPrank();

        vm.startPrank(charlie);
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.repayBorrow(0.5e8);
        vm.stopPrank();

        // More time passes
        vm.warp(block.timestamp + 1 days);

        // Claim rewards for all users
        address[] memory holders = new address[](3);
        holders[0] = alice;
        holders[1] = bob;
        holders[2] = charlie;

        CToken[] memory tokens = new CToken[](2);
        tokens[0] = cSonic;
        tokens[1] = CToken(address(cWbtcDelegator));

        // Claim both supplier and borrower rewards
        comptroller.claimRewards(holders, tokens, true, true);

        // Verify counts
        // Alice Mints x2, Bob Mints x1, Charlie Mints x1, Dave Mints x1 and transfers (count=2) to Bob
        assertEq(rewardDistributor.updateSupplyIndexAndDisburseSupplierRewardsCount(), 7);
        // 2 borrows + 2 repayments
        assertEq(rewardDistributor.updateBorrowIndexAndDisburseBorrowerRewardsCount(), 4);

        // During claim: 3 holders * 2 tokens = 6 disbursements each for suppliers and borrowers
        assertEq(rewardDistributor.disburseSupplierRewardsCount(), 6);
        assertEq(rewardDistributor.disburseBorrowerRewardsCount(), 6);

        // One update per token during claim
        assertEq(rewardDistributor.updateSupplyIndexCount(), 2);
        assertEq(rewardDistributor.updateBorrowIndexCount(), 2);
    }

    function test_mintAsCollateral_cSonic() public {
        vm.startPrank(alice);

        // Check membership before minting
        assertEq(comptroller.checkMembership(alice, address(cSonic)), false);
        assertEq(comptroller.checkMembership(bob, address(cSonic)), false);
        assertEq(comptroller.checkMembership(alice, address(cWbtcDelegator)), false);
        assertEq(comptroller.checkMembership(bob, address(cWbtcDelegator)), false);

        // $S price = ~$1
        deal(alice, 1e6 ether);
        cSonic.mintAsCollateral{value: 5e5 ether}();

        // Check that Alice is now a member of the Sonic market
        assertEq(comptroller.checkMembership(alice, address(cSonic)), true);
        vm.stopPrank();

        // Deposit wBTC as collateral for bob
        vm.startPrank(bob);
        deal(address(wbtc), bob, 1e8);
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.mint(1e8);
        vm.stopPrank();

        // Alice borrows wBTC
        vm.startPrank(alice);
        cWbtcDelegator.borrow(0.1e8);
        vm.stopPrank();
    }

    function test_mintAsCollateral_cWbtc() public {
        deal(address(wbtc), alice, 1e8);
        deal(address(wbtc), bob, 1e8);
        deal(bob, 1e6 ether);
        deal(alice, 1e6 ether);

        uint256 aliceWbtcMintAsCollateralAmount = 1e8;
        uint256 aliceSonicMintAmount = 1e5 ether;

        uint256 aliceWbtcBorrowAmount = 0.1e8;
        uint256 aliceSonicBorrowAmount = 1e5 ether;

        uint256 bobWbtcMintAsCollateralAmount = 1e8;
        uint256 bobSonicMintAsCollateralAmount = 1e5 ether;

        // Check membership before minting
        assertEq(comptroller.checkMembership(bob, address(cWbtcDelegator)), false);
        assertEq(comptroller.checkMembership(bob, address(cSonic)), false);
        assertEq(comptroller.checkMembership(alice, address(cWbtcDelegator)), false);
        assertEq(comptroller.checkMembership(alice, address(cSonic)), false);

        vm.startPrank(bob);
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.mintAsCollateral(bobWbtcMintAsCollateralAmount);
        cSonic.mintAsCollateral{value: bobSonicMintAsCollateralAmount}();
        vm.stopPrank();

        vm.startPrank(alice);
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.mintAsCollateral(aliceWbtcMintAsCollateralAmount);
        cSonic.mint{value: aliceSonicMintAmount}();

        // Check membership (before borrowing)
        assertEq(comptroller.checkMembership(bob, address(cWbtcDelegator)), true);
        assertEq(comptroller.checkMembership(bob, address(cSonic)), true);
        assertEq(comptroller.checkMembership(alice, address(cWbtcDelegator)), true);
        assertEq(comptroller.checkMembership(alice, address(cSonic)), false);

        // Borrow Sonic + BTC
        cSonic.borrow(aliceSonicBorrowAmount);
        cWbtcDelegator.borrow(aliceWbtcBorrowAmount);
        vm.stopPrank();

        // Borrowing should add Alice to the Sonic market
        assertEq(comptroller.checkMembership(alice, address(cSonic)), true);

        // Alice mints as collateral again
        vm.prank(alice);
        cSonic.mintAsCollateral{value: 1e5 ether}();

        // Check membership (after minting as collateral)
        assertEq(comptroller.checkMembership(alice, address(cSonic)), true);

        // Check liquidity calculations
        uint256 wBtcPrice = priceOracle.getUnderlyingPrice(CToken(address(cWbtcDelegator)));
        uint256 sonicPrice = priceOracle.getUnderlyingPrice(CToken(address(cSonic)));

        (uint256 err, uint256 liquidity, uint256 shortfall) = comptroller.getAccountLiquidity(alice);
        assertEq(err, 0);
        assertGt(liquidity, 0);
        assertEq(shortfall, 0);

        (err, liquidity, shortfall) = comptroller.getAccountLiquidity(bob);
        assertEq(err, 0);
        assertGt(liquidity, 0);
        assertEq(shortfall, 0);
    }

    function test_mintAsCollateral_previouslyEnteredMarket() public {
        // Edge cases
        // 1. Mint as collateral with no collateral before
        // 2. Mint as collateral with collateral before
        uint256 aliceWbtcMintAsCollateralAmount = 1e8;

        // Check membership before minting
        assertEq(comptroller.checkMembership(alice, address(cSonic)), false);
        assertEq(comptroller.checkMembership(alice, address(cWbtcDelegator)), false);

        deal(address(wbtc), alice, 10e8);
        deal(alice, 1e6 ether);

        vm.startPrank(alice);
        wbtc.approve(address(cWbtcDelegator), type(uint256).max);
        cWbtcDelegator.mint(aliceWbtcMintAsCollateralAmount);

        address[] memory marketsToEnter = new address[](1);
        marketsToEnter[0] = address(cWbtcDelegator);

        comptroller.enterMarkets(marketsToEnter);
        vm.stopPrank();

        // Check membership (after entering market)
        assertEq(comptroller.checkMembership(alice, address(cSonic)), false);
        assertEq(comptroller.checkMembership(alice, address(cWbtcDelegator)), true);

        // Alice mints as collateral
        vm.prank(alice);
        cWbtcDelegator.mintAsCollateral(aliceWbtcMintAsCollateralAmount);

        // Check membership (after minting as collateral)
        assertEq(comptroller.checkMembership(alice, address(cSonic)), false);
        assertEq(comptroller.checkMembership(alice, address(cWbtcDelegator)), true);
    }

    function test_mintAsCollateral_pausedMarket() public {
        vm.prank(admin);
        // Pause the market
        comptroller._setMintPaused(cSonic, true);

        vm.startPrank(alice);
        deal(alice, 1 ether);

        // Should fail when market is paused
        vm.expectRevert("mint is paused");
        cSonic.mintAsCollateral{value: 1 ether}();
        vm.stopPrank();
    }

    function test_mintAsCollateral_supplyCapReached() public {
        deal(alice, 1e6 ether);

        vm.startPrank(admin);
        // Set a low supply cap
        CToken[] memory cTokens = new CToken[](1);
        cTokens[0] = cSonic;
        uint256[] memory supplyCaps = new uint256[](1);
        supplyCaps[0] = 1 ether;
        comptroller._setMarketSupplyCaps(cTokens, supplyCaps);
        vm.stopPrank();

        vm.startPrank(alice);
        deal(alice, 2 ether);

        // Get cash for cSonic
        uint256 cash = cSonic.getCash();
        console.log("cash", cash);

        // First mint should succeed
        cSonic.mintAsCollateral{value: 0.5 ether}();

        // Second mint should fail due to cap
        vm.expectRevert("market supply cap reached");
        cSonic.mintAsCollateral{value: 1 ether}();
        vm.stopPrank();
    }
}
