// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "../contracts/TellerV2MarketForwarder_G1.sol";

import "./tokens/TestERC20Token.sol";
import "../contracts/TellerV2Context.sol";

import { Testable } from "./Testable.sol";

import { Collateral, CollateralType } from "../contracts/interfaces/escrow/ICollateralEscrowV1.sol";

import { User } from "./Test_Helpers.sol";
import { MarketLiquidityRewards_Override } from "./MarketLiquidityRewards_Override.sol";

import "../contracts/mock/MarketRegistryMock.sol";
import "../contracts/mock/CollateralManagerMock.sol";

import "../contracts/MarketLiquidityRewards.sol";

contract MarketLiquidityRewards_Test is Testable {
    MarketLiquidityUser private marketOwner;
    MarketLiquidityUser private lender;
    MarketLiquidityUser private borrower;

    uint256 immutable startTime = 1678122531;

    //  address tokenAddress;
    uint256 marketId;
    uint256 maxAmount;

    address[] emptyArray;
    address[] borrowersArray;

    uint32 maxLoanDuration;
    uint16 minInterestRate;
    uint32 expiration;

    TestERC20Token rewardToken;
    uint8 constant rewardTokenDecimals = 18;

    TestERC20Token principalToken;
    uint8 constant principalTokenDecimals = 18;

    TestERC20Token collateralToken;
    uint8 constant collateralTokenDecimals = 6;

    MarketLiquidityRewards_Override marketLiquidityRewards;

    TellerV2Mock tellerV2Mock;
    MarketRegistryMock marketRegistryMock;
    CollateralManagerMock collateralManagerMock;

    constructor()
    /*   MarketLiquidityRewards(
            address(new TellerV2Mock()),
            address(new MarketRegistryMock(address(0))),
            address(new CollateralManagerMock())
        )*/
    {

    }

    function setUp() public {
        tellerV2Mock = new TellerV2Mock();

        marketRegistryMock = new MarketRegistryMock();

        collateralManagerMock = new CollateralManagerMock();

        tellerV2Mock.setMarketRegistry(address(marketRegistryMock));

        marketLiquidityRewards = new MarketLiquidityRewards_Override(
            address(tellerV2Mock),
            address(marketRegistryMock),
            address(collateralManagerMock)
        );

        borrower = new MarketLiquidityUser(
            address(tellerV2Mock),
            address(marketLiquidityRewards)
        );
        lender = new MarketLiquidityUser(
            address(tellerV2Mock),
            address(marketLiquidityRewards)
        );

        marketOwner = new MarketLiquidityUser(
            address(tellerV2Mock),
            address(marketLiquidityRewards)
        );

        marketRegistryMock.setMarketOwner(address(marketOwner));

        //tokenAddress = address(0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174);
        marketId = 2;
        maxAmount = 100000000000000000000;
        maxLoanDuration = 2480000;
        minInterestRate = 3000;
        expiration = uint32(block.timestamp) + uint32(64000);

        //failing
        marketOwner.setTrustedMarketForwarder(
            marketId,
            address(marketLiquidityRewards)
        );
        lender.approveMarketForwarder(
            marketId,
            address(marketLiquidityRewards)
        );

        borrowersArray = new address[](1);
        borrowersArray[0] = address(borrower);

        rewardToken = new TestERC20Token(
            "Test Wrapped ETH",
            "TWETH",
            1e30,
            rewardTokenDecimals
        );

        principalToken = new TestERC20Token(
            "Test Wrapped ETH",
            "TWETH",
            1e30,
            principalTokenDecimals
        );

        collateralToken = new TestERC20Token(
            "Test USDC",
            "TUSDC",
            100000,
            collateralTokenDecimals
        );

        IERC20Upgradeable(address(rewardToken)).transfer(
            address(lender),
            10000
        );

        IERC20Upgradeable(address(rewardToken)).transfer(
            address(marketLiquidityRewards),
            1e12
        );

        vm.warp(startTime + 100);
        //delete allocationCount;

        /*  verifyLoanStartTimeWasCalled = false;
        

        verifyRewardRecipientWasCalled = false;
        verifyCollateralAmountWasCalled = false;*/
    }

    /*
FNDA:0,MarketLiquidityRewards.initialize
FNDA:0,MarketLiquidityRewards.updateAllocation
FNDA:0,MarketLiquidityRewards._verifyCollateralAmount

    */

    function _setAllocation(uint256 _allocationId, uint256 rewardTokenAmount)
        internal
    {
        MarketLiquidityRewards.RewardAllocation
            memory _allocation = IMarketLiquidityRewards.RewardAllocation({
                allocator: address(lender),
                marketId: marketId,
                rewardTokenAddress: address(rewardToken),
                rewardTokenAmount: rewardTokenAmount,
                requiredPrincipalTokenAddress: address(principalToken),
                requiredCollateralTokenAddress: address(collateralToken),
                minimumCollateralPerPrincipalAmount: 0,
                rewardPerLoanPrincipalAmount: 1e18,
                bidStartTimeMin: uint32(startTime),
                bidStartTimeMax: uint32(startTime + 10000),
                allocationStrategy: IMarketLiquidityRewards
                    .AllocationStrategy
                    .BORROWER
            });

        marketLiquidityRewards.setAllocation(_allocationId, _allocation);
    }

    function test_allocateRewards() public {
        uint256 rewardTokenAmount = 500;

        MarketLiquidityRewards.RewardAllocation
            memory _allocation = IMarketLiquidityRewards.RewardAllocation({
                allocator: address(lender),
                marketId: marketId,
                rewardTokenAddress: address(rewardToken),
                rewardTokenAmount: rewardTokenAmount,
                requiredPrincipalTokenAddress: address(principalToken),
                requiredCollateralTokenAddress: address(collateralToken),
                minimumCollateralPerPrincipalAmount: 0,
                rewardPerLoanPrincipalAmount: 0,
                bidStartTimeMin: uint32(startTime),
                bidStartTimeMax: uint32(startTime + 10000),
                allocationStrategy: IMarketLiquidityRewards
                    .AllocationStrategy
                    .BORROWER
            });

        lender._approveERC20Token(
            address(rewardToken),
            address(marketLiquidityRewards),
            rewardTokenAmount
        );

        uint256 allocationId = lender._allocateRewards(_allocation);
    }

    function test_increaseAllocationAmount() public {
        uint256 allocationId = 0;
        uint256 amountToIncrease = 100;

        _setAllocation(allocationId, 0);

        uint256 amountBefore = marketLiquidityRewards.getRewardTokenAmount(
            allocationId
        );

        lender._approveERC20Token(
            address(rewardToken),
            address(marketLiquidityRewards),
            amountToIncrease
        );

        lender._increaseAllocationAmount(allocationId, amountToIncrease);

        uint256 amountAfter = marketLiquidityRewards.getRewardTokenAmount(
            allocationId
        );

        assertEq(
            amountAfter,
            amountBefore + amountToIncrease,
            "Allocation did not increase"
        );
    }

    function test_deallocateRewards() public {
        uint256 allocationId = 0;

        _setAllocation(allocationId, 0);

        uint256 amountBefore = marketLiquidityRewards.getRewardTokenAmount(
            allocationId
        );

        lender._deallocateRewards(allocationId, amountBefore);

        uint256 amountAfter = marketLiquidityRewards.getRewardTokenAmount(
            allocationId
        );

        assertEq(amountAfter, 0, "Allocation was not deleted");
    }

    function test_claimRewards() public {
        Bid memory mockBid;

        mockBid.borrower = address(borrower);
        mockBid.lender = address(lender);
        mockBid.marketplaceId = marketId;
        mockBid.loanDetails.loanDuration = 80000;
        mockBid.loanDetails.lendingToken = (principalToken);
        mockBid.loanDetails.principal = 10000;
        mockBid.loanDetails.acceptedTimestamp = uint32(block.timestamp);
        mockBid.loanDetails.lastRepaidTimestamp = uint32(
            block.timestamp + 5000
        );
        mockBid.state = BidState.PAID;

        tellerV2Mock.setMockBid(mockBid);

        uint256 allocationId = 0;
        uint256 bidId = 0;

        _setAllocation(allocationId, 4000);

        vm.prank(address(borrower));
        marketLiquidityRewards.claimRewards(allocationId, bidId);

        assertEq(
            marketLiquidityRewards.verifyLoanStartTimeWasCalled(),
            true,
            "verifyLoanStartTime was not called"
        );

        assertEq(
            marketLiquidityRewards.verifyRewardRecipientWasCalled(),
            true,
            "verifyRewardRecipient was not called"
        );

        assertEq(
            marketLiquidityRewards.verifyCollateralAmountWasCalled(),
            true,
            "verifyCollateralAmount was not called"
        );

        //add some negative tests  (unit)
        //add comments to all of the methods
    }

    function test_claimRewards_zero_principal() public {
        Bid memory mockBid;

        mockBid.borrower = address(borrower);
        mockBid.lender = address(lender);
        mockBid.marketplaceId = marketId;
        mockBid.loanDetails.lendingToken = (principalToken);
        mockBid.loanDetails.principal = 0;
        mockBid.loanDetails.acceptedTimestamp = uint32(block.timestamp);
        mockBid.loanDetails.lastRepaidTimestamp = uint32(
            block.timestamp + 5000
        );
        mockBid.state = BidState.PAID;

        tellerV2Mock.setMockBid(mockBid);

        uint256 allocationId = 0;
        uint256 bidId = 0;

        _setAllocation(allocationId, 0);

        vm.prank(address(borrower));
        vm.expectRevert("Nothing to claim.");
        marketLiquidityRewards.claimRewards(allocationId, bidId);
    }

    function test_claimRewards_round_remainder() public {
        Bid memory mockBid;

        mockBid.borrower = address(borrower);
        mockBid.lender = address(lender);
        mockBid.marketplaceId = marketId;
        mockBid.loanDetails.lendingToken = (principalToken);
        mockBid.loanDetails.principal = 10000000;
        mockBid.loanDetails.acceptedTimestamp = uint32(block.timestamp);
        mockBid.loanDetails.lastRepaidTimestamp = uint32(
            block.timestamp + (365 days)
        );
        mockBid.state = BidState.PAID;

        tellerV2Mock.setMockBid(mockBid);

        uint256 allocationId = 0;
        uint256 bidId = 0;

        MarketLiquidityRewards.RewardAllocation
            memory _allocation = IMarketLiquidityRewards.RewardAllocation({
                allocator: address(this),
                marketId: marketId,
                rewardTokenAddress: address(rewardToken),
                rewardTokenAmount: 1000,
                requiredPrincipalTokenAddress: address(principalToken),
                requiredCollateralTokenAddress: address(collateralToken),
                minimumCollateralPerPrincipalAmount: 0,
                rewardPerLoanPrincipalAmount: 1000 * 1e18,
                bidStartTimeMin: uint32(startTime),
                bidStartTimeMax: uint32(startTime + 10000),
                allocationStrategy: IMarketLiquidityRewards
                    .AllocationStrategy
                    .BORROWER
            });

        marketLiquidityRewards.setAllocation(allocationId, _allocation);

        //send 1000 tokens to the contract
        IERC20Upgradeable(address(rewardToken)).transfer(
            address(marketLiquidityRewards),
            1000
        );

        vm.prank(address(borrower));
        marketLiquidityRewards.claimRewards(allocationId, bidId);

        assertEq(
            marketLiquidityRewards.verifyLoanStartTimeWasCalled(),
            true,
            "verifyLoanStartTime was not called"
        );

        assertEq(
            marketLiquidityRewards.verifyRewardRecipientWasCalled(),
            true,
            "verifyRewardRecipient was not called"
        );

        assertEq(
            marketLiquidityRewards.verifyCollateralAmountWasCalled(),
            true,
            "verifyCollateralAmount was not called"
        );

        uint256 remainingTokenAmount = marketLiquidityRewards
            .getRewardTokenAmount(allocationId);

        //verify that the reward status is updated to drained
        assertEq(remainingTokenAmount, 0, "Reward was not completely drained");
    }

    function test_claimRewards_zero_time_elapsed() public {
        Bid memory mockBid;

        mockBid.borrower = address(borrower);
        mockBid.lender = address(lender);
        mockBid.marketplaceId = marketId;
        mockBid.loanDetails.lendingToken = (principalToken);
        mockBid.loanDetails.principal = 10000000;
        mockBid.loanDetails.acceptedTimestamp = uint32(block.timestamp);
        mockBid.loanDetails.lastRepaidTimestamp = uint32(block.timestamp);
        mockBid.state = BidState.PAID;

        tellerV2Mock.setMockBid(mockBid);

        uint256 allocationId = 0;
        uint256 bidId = 0;

        MarketLiquidityRewards.RewardAllocation
            memory _allocation = IMarketLiquidityRewards.RewardAllocation({
                allocator: address(this),
                marketId: marketId,
                rewardTokenAddress: address(rewardToken),
                rewardTokenAmount: 1000,
                requiredPrincipalTokenAddress: address(principalToken),
                requiredCollateralTokenAddress: address(collateralToken),
                minimumCollateralPerPrincipalAmount: 0,
                rewardPerLoanPrincipalAmount: 1000 * 1e18,
                bidStartTimeMin: uint32(startTime),
                bidStartTimeMax: uint32(startTime + 10000),
                allocationStrategy: IMarketLiquidityRewards
                    .AllocationStrategy
                    .BORROWER
            });

        marketLiquidityRewards.setAllocation(allocationId, _allocation);

        //send 1000 tokens to the contract
        IERC20Upgradeable(address(rewardToken)).transfer(
            address(marketLiquidityRewards),
            1000
        );

        vm.prank(address(borrower));
        vm.expectRevert("Nothing to claim.");
        marketLiquidityRewards.claimRewards(allocationId, bidId);
    }

    function test_calculateRewardAmount_weth_principal() public {
        uint256 loanPrincipal = 1e8;
        uint256 principalTokenDecimals = 18;
        uint32 loanDuration = 60 * 60 * 24 * 365;

        uint256 rewardPerLoanPrincipalAmount = 1e16; // expanded by token decimals so really 0.01

        uint256 rewardAmount = marketLiquidityRewards.calculateRewardAmount(
            loanPrincipal,
            loanDuration,
            principalTokenDecimals,
            rewardPerLoanPrincipalAmount
        );

        assertEq(rewardAmount, 1e6, "Invalid reward amount");
    }

    function test_calculateRewardAmount_usdc_principal() public {
        uint256 loanPrincipal = 1e8;
        uint256 principalTokenDecimals = 6;
        uint32 loanDuration = 60 * 60 * 24 * 365;

        uint256 rewardPerLoanPrincipalAmount = 1e4; // expanded by token decimals so really 0.01

        uint256 rewardAmount = marketLiquidityRewards.calculateRewardAmount(
            loanPrincipal,
            loanDuration,
            principalTokenDecimals,
            rewardPerLoanPrincipalAmount
        );

        assertEq(rewardAmount, 1e6, "Invalid reward amount");
    }

    function test_requiredCollateralAmount() public {
        uint256 collateralTokenDecimals = 6;

        uint256 loanPrincipal = 1e8;
        uint256 principalTokenDecimals = 6;

        uint256 minimumCollateralPerPrincipal = 1e4 * 1e6; // expanded by token decimals so really 0.01

        uint256 minCollateral = marketLiquidityRewards.requiredCollateralAmount(
            loanPrincipal,
            principalTokenDecimals,
            collateralTokenDecimals,
            minimumCollateralPerPrincipal
        );

        assertEq(minCollateral, 1e6, "Invalid min collateral calculation");
    }

    function test_decrementAllocatedAmount() public {
        uint256 allocationId = 0;

        _setAllocation(allocationId, 0);

        uint256 amount = 100;

        marketLiquidityRewards.setAllocatedAmount(allocationId, amount);

        marketLiquidityRewards.decrementAllocatedAmount(allocationId, amount);

        assertEq(
            marketLiquidityRewards.getRewardTokenAmount(allocationId),
            0,
            "allocation amount not decremented"
        );
    }

    /* function test_verifyCollateralAmount() public {
        
        vm.expectRevert();
        super._verifyCollateralAmount(
            address(collateralToken),
            100,
            address(principalToken),
            1000,
            5000 * 1e18 * 1e18 
        );


    }*/

    function test_verifyLoanStartTime_min() public {
        vm.expectRevert(bytes("Loan was accepted before the min start time."));

        marketLiquidityRewards.verifyLoanStartTime(100, 200, 300);
    }

    function test_verifyAndReturnRewardRecipient() public {
        address recipient = marketLiquidityRewards
            .verifyAndReturnRewardRecipient(
                IMarketLiquidityRewards.AllocationStrategy.BORROWER,
                BidState.PAID,
                address(borrower),
                address(lender)
            );

        assertEq(recipient, address(borrower), "incorrect address returned");
    }

    function test_verifyAndReturnRewardRecipient_reverts() public {
        vm.expectRevert();

        address recipient = marketLiquidityRewards
            .verifyAndReturnRewardRecipient(
                IMarketLiquidityRewards.AllocationStrategy.BORROWER,
                BidState.PENDING,
                address(borrower),
                address(lender)
            );
    }

    function test_verifyLoanStartTime_max() public {
        vm.expectRevert(bytes("Loan was accepted after the max start time."));

        marketLiquidityRewards.verifyLoanStartTime(400, 200, 300);
    }
}

contract MarketLiquidityUser is User {
    MarketLiquidityRewards public immutable liquidityRewards;

    constructor(address _tellerV2, address _liquidityRewards) User(_tellerV2) {
        liquidityRewards = MarketLiquidityRewards(_liquidityRewards);
    }

    function _allocateRewards(
        MarketLiquidityRewards.RewardAllocation calldata _allocation
    ) public returns (uint256) {
        return liquidityRewards.allocateRewards(_allocation);
    }

    function _increaseAllocationAmount(
        uint256 _allocationId,
        uint256 _allocationAmount
    ) public {
        liquidityRewards.increaseAllocationAmount(
            _allocationId,
            _allocationAmount
        );
    }

    function _deallocateRewards(uint256 _allocationId, uint256 _amount) public {
        liquidityRewards.deallocateRewards(_allocationId, _amount);
    }

    function _claimRewards(uint256 _allocationId, uint256 _bidId) public {
        return liquidityRewards.claimRewards(_allocationId, _bidId);
    }

    function _approveERC20Token(address tokenAddress, address guy, uint256 wad)
        public
    {
        IERC20Upgradeable(tokenAddress).approve(guy, wad);
    }
}

contract TellerV2Mock is TellerV2Context {
    Bid mockBid;

    constructor() TellerV2Context(address(0)) {}

    function setMarketRegistry(address _marketRegistry) external {
        marketRegistry = IMarketRegistry(_marketRegistry);
    }

    function getSenderForMarket(uint256 _marketId)
        external
        view
        returns (address)
    {
        return _msgSenderForMarket(_marketId);
    }

    function getDataForMarket(uint256 _marketId)
        external
        view
        returns (bytes calldata)
    {
        return _msgDataForMarket(_marketId);
    }

    function setMockBid(Bid calldata bid) public {
        mockBid = bid;
    }

    function getLoanSummary(uint256 _bidId)
        external
        view
        returns (
            address borrower,
            address lender,
            uint256 marketId,
            address principalTokenAddress,
            uint256 principalAmount,
            uint32 acceptedTimestamp,
            uint32 lastRepaidTimestamp,
            BidState bidState
        )
    {
        Bid storage bid = mockBid;

        borrower = bid.borrower;
        lender = bid.lender;
        marketId = bid.marketplaceId;
        principalTokenAddress = address(bid.loanDetails.lendingToken);
        principalAmount = bid.loanDetails.principal;
        acceptedTimestamp = bid.loanDetails.acceptedTimestamp;
        lastRepaidTimestamp = bid.loanDetails.lastRepaidTimestamp;
        bidState = bid.state;
    }
}
