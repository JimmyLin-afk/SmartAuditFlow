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

import "../contracts/mock/MarketRegistryMock.sol";
import "../contracts/mock/CollateralManagerMock.sol";

import "../contracts/MarketLiquidityRewards.sol";

import "forge-std/console.sol";

contract MarketLiquidityRewards_Test is Testable, MarketLiquidityRewards {
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

    bool verifyLoanStartTimeWasCalled;

    bool verifyRewardRecipientWasCalled;
    bool verifyCollateralAmountWasCalled;

    constructor()
        MarketLiquidityRewards(
            address(new TellerV2Mock()),
            address(new MarketRegistryMock()),
            address(new CollateralManagerMock())
        )
    {}

    function setUp() public {
        marketOwner = new MarketLiquidityUser(address(tellerV2), (this));
        borrower = new MarketLiquidityUser(address(tellerV2), (this));
        lender = new MarketLiquidityUser(address(tellerV2), (this));
        TellerV2Mock(tellerV2).__setMarketRegistry(address(marketRegistry));

        MarketRegistryMock(marketRegistry).setMarketOwner(address(marketOwner));

        //tokenAddress = address(0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174);
        marketId = 2;
        maxAmount = 100000000000000000000;
        maxLoanDuration = 2480000;
        minInterestRate = 3000;
        expiration = uint32(block.timestamp) + uint32(64000);

        marketOwner.setTrustedMarketForwarder(marketId, address(this));
        lender.approveMarketForwarder(marketId, address(this));

        borrowersArray = new address[](1);
        borrowersArray[0] = address(borrower);

        rewardToken = new TestERC20Token(
            "Test Wrapped ETH",
            "TWETH",
            100000,
            rewardTokenDecimals
        );

        principalToken = new TestERC20Token(
            "Test Wrapped ETH",
            "TWETH",
            100000,
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

        vm.warp(startTime + 100);
        //delete allocationCount;

        verifyLoanStartTimeWasCalled = false;

        verifyRewardRecipientWasCalled = false;
        verifyCollateralAmountWasCalled = false;
    }

    function _setAllocation(uint256 allocationId) internal {
        uint256 rewardTokenAmount = 0;

        RewardAllocation memory _allocation = IMarketLiquidityRewards
            .RewardAllocation({
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
                allocationStrategy: AllocationStrategy.BORROWER
            });

        allocatedRewards[allocationId] = _allocation;
    }

    function test_allocateRewards() public {
        uint256 rewardTokenAmount = 500;

        RewardAllocation memory _allocation = IMarketLiquidityRewards
            .RewardAllocation({
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
                allocationStrategy: AllocationStrategy.BORROWER
            });

        lender._approveERC20Token(
            address(rewardToken),
            address(this),
            rewardTokenAmount
        );

        uint256 allocationId = lender._allocateRewards(_allocation);
    }

    function test_increaseAllocationAmount() public {
        uint256 allocationId = 0;
        uint256 amountToIncrease = 100;

        _setAllocation(allocationId);

        uint256 amountBefore = allocatedRewards[allocationId].rewardTokenAmount;

        lender._approveERC20Token(
            address(rewardToken),
            address(this),
            amountToIncrease
        );

        lender._increaseAllocationAmount(allocationId, amountToIncrease);

        uint256 amountAfter = allocatedRewards[allocationId].rewardTokenAmount;

        assertEq(
            amountAfter,
            amountBefore + amountToIncrease,
            "Allocation did not increase"
        );
    }

    function test_deallocateRewards() public {
        uint256 allocationId = 0;

        _setAllocation(allocationId);

        uint256 amountBefore = allocatedRewards[allocationId].rewardTokenAmount;

        lender._deallocateRewards(allocationId, amountBefore);

        uint256 amountAfter = allocatedRewards[allocationId].rewardTokenAmount;

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

        TellerV2Mock(tellerV2).setMockBid(mockBid);

        uint256 allocationId = 0;
        uint256 bidId = 0;

        _setAllocation(allocationId);
        allocatedRewards[allocationId].rewardTokenAmount = 4000;

        borrower._claimRewards(allocationId, bidId);

        assertEq(
            verifyLoanStartTimeWasCalled,
            true,
            "verifyLoanStartTime was not called"
        );

        assertEq(
            verifyRewardRecipientWasCalled,
            true,
            "verifyRewardRecipient was not called"
        );

        assertEq(
            verifyCollateralAmountWasCalled,
            true,
            "verifyCollateralAmount was not called"
        );

        //add some negative tests  (unit)
        //add comments to all of the methods
    }

    function test_calculateRewardAmount_weth_principal() public {
        uint256 loanPrincipal = 1e8;
        uint256 principalTokenDecimals = 18;
        uint32 loanDuration = 60 * 60 * 24 * 365;

        uint256 rewardPerLoanPrincipalAmount = 1e16; // expanded by token decimals so really 0.01

        uint256 rewardAmount = super._calculateRewardAmount(
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

        uint256 rewardAmount = super._calculateRewardAmount(
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

        uint256 minCollateral = super._requiredCollateralAmount(
            loanPrincipal,
            principalTokenDecimals,
            collateralTokenDecimals,
            minimumCollateralPerPrincipal
        );

        assertEq(minCollateral, 1e6, "Invalid min collateral calculation");
    }

    function test_decrementAllocatedAmount() public {
        uint256 allocationId = 0;

        _setAllocation(allocationId);

        uint256 amount = 100;

        allocatedRewards[allocationId].rewardTokenAmount += amount;

        super._decrementAllocatedAmount(allocationId, amount);

        assertEq(
            allocatedRewards[allocationId].rewardTokenAmount,
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

        super._verifyLoanStartTime(100, 200, 300);
    }

    function test_verifyAndReturnRewardRecipient() public {
        address recipient = super._verifyAndReturnRewardRecipient(
            AllocationStrategy.BORROWER,
            BidState.PAID,
            address(borrower),
            address(lender)
        );

        assertEq(recipient, address(borrower), "incorrect address returned");
    }

    function test_verifyAndReturnRewardRecipient_reverts() public {
        vm.expectRevert();

        address recipient = super._verifyAndReturnRewardRecipient(
            AllocationStrategy.BORROWER,
            BidState.PENDING,
            address(borrower),
            address(lender)
        );
    }

    function test_verifyLoanStartTime_max() public {
        vm.expectRevert(bytes("Loan was accepted after the max start time."));

        super._verifyLoanStartTime(400, 200, 300);
    }

    function allocateRewards(
        MarketLiquidityRewards.RewardAllocation calldata _allocation
    ) public override returns (uint256 allocationId_) {
        super.allocateRewards(_allocation);
    }

    function increaseAllocationAmount(
        uint256 _allocationId,
        uint256 _tokenAmount
    ) public override {
        super.increaseAllocationAmount(_allocationId, _tokenAmount);
    }

    function deallocateRewards(uint256 _allocationId, uint256 _tokenAmount)
        public
        override
    {
        super.deallocateRewards(_allocationId, _tokenAmount);
    }

    function _verifyAndReturnRewardRecipient(
        AllocationStrategy strategy,
        BidState bidState,
        address borrower,
        address lender
    ) internal override returns (address rewardRecipient) {
        verifyRewardRecipientWasCalled = true;
        return address(borrower);
    }

    function _verifyCollateralAmount(
        address _collateralTokenAddress,
        uint256 _collateralAmount,
        address _principalTokenAddress,
        uint256 _principalAmount,
        uint256 _minimumCollateralPerPrincipalAmount
    ) internal override {
        verifyCollateralAmountWasCalled = true;
    }

    function _verifyLoanStartTime(
        uint32 loanStartTime,
        uint32 minStartTime,
        uint32 maxStartTime
    ) internal override {
        verifyLoanStartTimeWasCalled = true;
    }
}

contract MarketLiquidityUser is User {
    MarketLiquidityRewards public immutable liquidityRewards;

    constructor(address _tellerV2, MarketLiquidityRewards _liquidityRewards)
        User(_tellerV2)
    {
        liquidityRewards = _liquidityRewards;
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

    function __setMarketRegistry(address _marketRegistry) external {
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
