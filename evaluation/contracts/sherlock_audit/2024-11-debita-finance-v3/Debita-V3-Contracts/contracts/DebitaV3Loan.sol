pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC721/utils/ERC721Holder.sol";

interface Aggregator {
    function s_AuctionFactory() external view returns (address);
    function emitLoanUpdated(address _loan) external;
    function feePerDay() external view returns (uint);
    function maxFEE() external view returns (uint);
    function minFEE() external view returns (uint);
}

interface IveNFTEqualizer {
    struct receiptInstance {
        uint receiptID;
        uint attachedNFT;
        uint lockedAmount;
        uint lockedDate;
        uint decimals;
        address vault;
        address underlying;
    }
    function getDataByReceipt(
        uint receiptID
    ) external returns (receiptInstance memory);
}

interface AuctionFactory {
    function createAuction(
        uint _veNFTID,
        address _veNFTAddress,
        address liquidationToken,
        uint _initAmount,
        uint _floorAmount,
        uint _duration
    ) external returns (address);

    function getLiquidationFloorPrice(
        uint initAmount
    ) external view returns (uint);
}

interface IOwnerships {
    function ownerOf(uint256 tokenId) external view returns (address);

    function transferFrom(address from, address to, uint256 tokenId) external;

    function burn(uint256 tokenId) external;

    function mint(address to) external returns (uint256);
}

interface DLOImplementation {
    struct LendInfo {
        address lendOrderAddress;
        bool perpetual;
        bool lonelyLender;
        bool[] oraclesPerPairActivated;
        uint[] maxLTVs;
        uint apr;
        uint maxDuration;
        uint minDuration;
        address owner;
        address principle;
        address[] acceptedCollaterals;
        address[] oracle_Collaterals;
        uint[] maxRatio;
        address oracle_Principle;
        uint startedLendingAmount;
        uint availableAmount;
    }

    function addFunds(uint amount) external;

    function acceptLendingOffer(uint amount) external;

    function getLendInfo() external returns (LendInfo memory);
}

contract DebitaV3Loan is Initializable, ReentrancyGuard {
    address public s_OwnershipContract;
    address public AggregatorContract;
    uint feeLender;
    address feeAddress;

    struct infoOfOffers {
        address principle; // principle of the accepted offer
        address lendOffer; // address of the lendOffer contract
        uint principleAmount; // amount of principle taken
        uint lenderID; // id of the lender ownership
        uint apr; // apr of the accepted offer
        uint ratio; // ratio of the accepted offer
        uint collateralUsed; // collateral amount used
        uint maxDeadline; // max deadline of the accepted offer
        bool paid; // if the offer has been paid
        bool collateralClaimed; // if the collateral has been claimed
        bool debtClaimed; // if the debt has been claimed
        uint interestToClaim; // available interest to claim
        uint interestPaid; // interest paid
    }

    struct LoanData {
        address collateral; // collateral of the loan
        address[] principles; // principles of the loan
        address valuableCollateralAsset; // valuable collateral of the loan (Underlying in case of fNFT)
        bool isCollateralNFT; // if the collateral is NFT
        bool auctionInitialized; // if the auction has been initialized
        bool extended; // if the loan has been extended
        uint startedAt; // timestamp of the loan
        uint initialDuration; // the initial duration that the borrower took the loan
        uint borrowerID; // id of the borrower ownership
        uint NftID; // id of the NFT that is being used as collateral (if isCollateralNFT is true)
        uint collateralAmount; // collateral amount of the loan (1 if nft)
        uint collateralValuableAmount; // valuable collateral amount of the loan (if erc20 --> same as collateralAmount, if NFT --> underlying amount)
        uint valuableCollateralUsed; // valuable collateral used in the loan
        uint totalCountPaid; // total count of offers paid
        uint[] principlesAmount; // total amount of borrowed per principles
        infoOfOffers[] _acceptedOffers;
    }

    struct AuctionData {
        address auctionAddress;
        address liquidationAddress;
        uint soldAmount;
        uint tokenPerCollateralUsed;
        bool alreadySold;
    }

    LoanData public loanData;
    AuctionData public auctionData;
    uint offersCollateralClaimed_Borrower;

    function initialize(
        address _collateral,
        address[] memory _principles,
        bool _isCollateralNFT,
        uint _NftID,
        uint _collateralAmount,
        uint _valuableCollateralAmount,
        uint valuableCollateralUsed,
        address valuableAsset,
        uint _initialDuration,
        uint[] memory _principlesAmount,
        uint _borrowerID,
        infoOfOffers[] memory _acceptedOffers,
        address m_OwnershipContract,
        uint feeInterestLender,
        address _feeAddress
    ) public initializer nonReentrant {
        // set LoanData and acceptedOffers
        require(_acceptedOffers.length < 30, "Too many offers");
        loanData = LoanData({
            collateral: _collateral,
            principles: _principles,
            valuableCollateralAsset: valuableAsset,
            isCollateralNFT: _isCollateralNFT,
            auctionInitialized: false,
            extended: false,
            startedAt: block.timestamp,
            borrowerID: _borrowerID,
            NftID: _NftID,
            collateralAmount: _collateralAmount,
            collateralValuableAmount: _valuableCollateralAmount,
            valuableCollateralUsed: valuableCollateralUsed,
            initialDuration: _initialDuration,
            totalCountPaid: 0,
            principlesAmount: _principlesAmount,
            _acceptedOffers: _acceptedOffers
        });
        s_OwnershipContract = m_OwnershipContract;
        feeLender = feeInterestLender;
        AggregatorContract = msg.sender;
        feeAddress = _feeAddress;
    }

    /** 
    @notice Function to pay the debt of the loan
    @param indexes indexes of the offers to pay (only the borrower can call this function)
     If he misses one deadline, the loan will be counted as defaulted and the collateral will be claimed by the lenders
     */
    function payDebt(uint[] memory indexes) public nonReentrant {
        IOwnerships ownershipContract = IOwnerships(s_OwnershipContract);

        require(
            ownershipContract.ownerOf(loanData.borrowerID) == msg.sender,
            "Not borrower"
        );
        // check next deadline
        require(
            nextDeadline() >= block.timestamp,
            "Deadline passed to pay Debt"
        );

        for (uint i; i < indexes.length; i++) {
            uint index = indexes[i];
            // get offer data on memory
            infoOfOffers memory offer = loanData._acceptedOffers[index];

            // change the offer to paid on storage
            loanData._acceptedOffers[index].paid = true;

            // check if it has been already paid
            require(offer.paid == false, "Already paid");

            require(offer.maxDeadline > block.timestamp, "Deadline passed");
            uint interest = calculateInterestToPay(index);
            uint feeOnInterest = (interest * feeLender) / 10000;
            uint total = offer.principleAmount + interest - feeOnInterest;
            address currentOwnerOfOffer;

            try ownershipContract.ownerOf(offer.lenderID) returns (
                address _lenderOwner
            ) {
                currentOwnerOfOffer = _lenderOwner;
            } catch {}

            DLOImplementation lendOffer = DLOImplementation(offer.lendOffer);
            DLOImplementation.LendInfo memory lendInfo = lendOffer
                .getLendInfo();

            SafeERC20.safeTransferFrom(
                IERC20(offer.principle),
                msg.sender,
                address(this),
                total
            );
            // if the lender is the owner of the offer and the offer is perpetual, then add the funds to the offer
            if (lendInfo.perpetual && lendInfo.owner == currentOwnerOfOffer) {
                loanData._acceptedOffers[index].debtClaimed = true;
                IERC20(offer.principle).approve(address(lendOffer), total);
                lendOffer.addFunds(total);
            } else {
                loanData._acceptedOffers[index].interestToClaim =
                    interest -
                    feeOnInterest;
            }

            SafeERC20.safeTransferFrom(
                IERC20(offer.principle),
                msg.sender,
                feeAddress,
                feeOnInterest
            );

            loanData._acceptedOffers[index].interestPaid += interest;
        }
        // update total count paid
        loanData.totalCountPaid += indexes.length;

        Aggregator(AggregatorContract).emitLoanUpdated(address(this));
        // check owner
    }

    function claimInterest(uint index) internal {
        IOwnerships ownershipContract = IOwnerships(s_OwnershipContract);
        infoOfOffers memory offer = loanData._acceptedOffers[index];
        uint interest = offer.interestToClaim;

        require(interest > 0, "No interest to claim");

        loanData._acceptedOffers[index].interestToClaim = 0;
        SafeERC20.safeTransfer(IERC20(offer.principle), msg.sender, interest);
        Aggregator(AggregatorContract).emitLoanUpdated(address(this));
    }

    function claimDebt(uint index) external nonReentrant {
        IOwnerships ownershipContract = IOwnerships(s_OwnershipContract);
        infoOfOffers memory offer = loanData._acceptedOffers[index];

        require(
            ownershipContract.ownerOf(offer.lenderID) == msg.sender,
            "Not lender"
        );
        // check if the offer has been paid, if not just call claimInterest function
        if (offer.paid) {
            _claimDebt(index);
        } else {
            // if not already full paid, claim interest
            claimInterest(index);
        }
    }

    function _claimDebt(uint index) internal {
        LoanData memory m_loan = loanData;
        IOwnerships ownershipContract = IOwnerships(s_OwnershipContract);

        infoOfOffers memory offer = m_loan._acceptedOffers[index];
        require(
            ownershipContract.ownerOf(offer.lenderID) == msg.sender,
            "Not lender"
        );
        require(offer.paid == true, "Not paid");
        require(offer.debtClaimed == false, "Already claimed");
        loanData._acceptedOffers[index].debtClaimed = true;
        ownershipContract.burn(offer.lenderID);
        uint interest = offer.interestToClaim;
        offer.interestToClaim = 0;

        SafeERC20.safeTransfer(
            IERC20(offer.principle),
            msg.sender,
            interest + offer.principleAmount
        );

        Aggregator(AggregatorContract).emitLoanUpdated(address(this));
    }

    // only the auction contract can call this function
    /** 
    @notice Function to handle the auction sell of the collateral. Only the auction contract can call this function
    @param amount amount of collateral sold on the auction
     */
    function handleAuctionSell(uint amount) external nonReentrant {
        require(
            msg.sender == auctionData.auctionAddress,
            "Not auction contract"
        );
        require(auctionData.alreadySold == false, "Already sold");
        LoanData memory m_loan = loanData;
        IveNFTEqualizer.receiptInstance memory nftData = IveNFTEqualizer(
            m_loan.collateral
        ).getDataByReceipt(m_loan.NftID);
        uint PRECISION = 10 ** nftData.decimals;
        auctionData.soldAmount = amount;
        auctionData.alreadySold = true;
        auctionData.tokenPerCollateralUsed = ((amount * PRECISION) /
            (loanData.valuableCollateralUsed));
        Aggregator(AggregatorContract).emitLoanUpdated(address(this));
    }

    /** 
    @notice Function to claim the collateral as lender, only in case of default. Only lenders can call this function
    @param index index of the offer to claim the collateral
     */
    function claimCollateralAsLender(uint index) external nonReentrant {
        LoanData memory m_loan = loanData;
        infoOfOffers memory offer = m_loan._acceptedOffers[index];
        IOwnerships ownershipContract = IOwnerships(s_OwnershipContract);
        require(
            ownershipContract.ownerOf(offer.lenderID) == msg.sender,
            "Not lender"
        );
        // burn ownership
        ownershipContract.burn(offer.lenderID);
        uint _nextDeadline = nextDeadline();

        require(offer.paid == false, "Already paid");
        require(
            _nextDeadline < block.timestamp && _nextDeadline != 0,
            "Deadline not passed"
        );
        require(offer.collateralClaimed == false, "Already executed");

        // claim collateral
        if (m_loan.isCollateralNFT) {
            claimCollateralAsNFTLender(index);
        } else {
            loanData._acceptedOffers[index].collateralClaimed = true;
            uint decimals = ERC20(loanData.collateral).decimals();
            SafeERC20.safeTransfer(
                IERC20(loanData.collateral),
                msg.sender,
                (offer.principleAmount * (10 ** decimals)) / offer.ratio
            );
        }
        Aggregator(AggregatorContract).emitLoanUpdated(address(this));
    }

    function claimCollateralAsNFTLender(uint index) internal returns (bool) {
        LoanData memory m_loan = loanData;
        infoOfOffers memory offer = m_loan._acceptedOffers[index];
        loanData._acceptedOffers[index].collateralClaimed = true;

        if (m_loan.auctionInitialized) {
            // if the auction has been initialized
            // check if the auction has been sold
            require(auctionData.alreadySold, "Not sold on auction");

            uint decimalsCollateral = IveNFTEqualizer(loanData.collateral)
                .getDataByReceipt(loanData.NftID)
                .decimals;

            uint payment = (auctionData.tokenPerCollateralUsed *
                offer.collateralUsed) / (10 ** decimalsCollateral);

            SafeERC20.safeTransfer(
                IERC20(auctionData.liquidationAddress),
                msg.sender,
                payment
            );

            return true;
        } else if (
            m_loan._acceptedOffers.length == 1 && !m_loan.auctionInitialized
        ) {
            // if there is only one offer and the auction has not been initialized
            // send the NFT to the lender
            IERC721(m_loan.collateral).transferFrom(
                address(this),
                msg.sender,
                m_loan.NftID
            );
            return true;
        }
        return false;
    }

    /** 
    @notice Function to create an auction for the collateral in case of a default. Only lenders can call this functions in case the borrower missed their deadline or if the borrower wants to liquidate the collateral and offers length is more than 1 (It will go anyway to the auction if the borrower has more than 1 offer)
    @param indexOfLender index of the lender in the acceptedOffers array
      */
    function createAuctionForCollateral(
        uint indexOfLender
    ) external nonReentrant {
        LoanData memory m_loan = loanData;

        address lenderAddress = safeGetOwner(
            m_loan._acceptedOffers[indexOfLender].lenderID
        );
        address borrowerAddress = safeGetOwner(m_loan.borrowerID);

        bool hasLenderRightToInitAuction = lenderAddress == msg.sender &&
            m_loan._acceptedOffers[indexOfLender].paid == false;
        bool hasBorrowerRightToInitAuction = borrowerAddress == msg.sender &&
            m_loan._acceptedOffers.length > 1;

        // check if collateral is actually NFT
        require(m_loan.isCollateralNFT, "Collateral is not NFT");

        // check that total count paid is not equal to the total offers
        require(
            m_loan.totalCountPaid != m_loan._acceptedOffers.length,
            "Already paid everything"
        );
        // check if the deadline has passed
        require(nextDeadline() < block.timestamp, "Deadline not passed");
        // check if the auction has not been already initialized
        require(m_loan.auctionInitialized == false, "Already initialized");
        // check if the lender has the right to initialize the auction
        // check if the borrower has the right to initialize the auction
        require(
            hasLenderRightToInitAuction || hasBorrowerRightToInitAuction,
            "Not involved"
        );
        // collateral has to be NFT

        AuctionFactory auctionFactory = AuctionFactory(
            Aggregator(AggregatorContract).s_AuctionFactory()
        );
        loanData.auctionInitialized = true;
        IveNFTEqualizer.receiptInstance memory receiptInfo = IveNFTEqualizer(
            m_loan.collateral
        ).getDataByReceipt(m_loan.NftID);

        // calculate floor amount for liquidations
        uint floorAmount = auctionFactory.getLiquidationFloorPrice(
            receiptInfo.lockedAmount
        );

        // create auction and save the information
        IERC721(m_loan.collateral).approve(
            address(auctionFactory),
            m_loan.NftID
        );
        address liveAuction = auctionFactory.createAuction(
            m_loan.NftID,
            m_loan.collateral,
            receiptInfo.underlying,
            receiptInfo.lockedAmount,
            floorAmount,
            864000
        );

        auctionData = AuctionData({
            auctionAddress: liveAuction,
            liquidationAddress: receiptInfo.underlying,
            soldAmount: 0,
            tokenPerCollateralUsed: 0,
            alreadySold: false
        });
        Aggregator(AggregatorContract).emitLoanUpdated(address(this));

        // emit event here
    }

    // function to claim collateral as borrower
    /** 
    @notice Function to claim the collateral as borrower. Only the borrower can call this function
    @param indexs indexes of the offers to claim the collateral
     */
    function claimCollateralAsBorrower(
        uint[] memory indexs
    ) external nonReentrant {
        IOwnerships ownershipContract = IOwnerships(s_OwnershipContract);

        require(
            ownershipContract.ownerOf(loanData.borrowerID) == msg.sender,
            "Not borrower"
        );

        // if the collateral is nft, it has another logic
        if (loanData.isCollateralNFT) {
            claimCollateralNFTAsBorrower(indexs);
        } else {
            claimCollateralERC20AsBorrower(indexs);
        }

        offersCollateralClaimed_Borrower += indexs.length;

        // In case every offer has been claimed & paid, burn the borrower ownership
        if (
            offersCollateralClaimed_Borrower == loanData._acceptedOffers.length
        ) {
            ownershipContract.burn(loanData.borrowerID);
        }
        Aggregator(AggregatorContract).emitLoanUpdated(address(this));
    }

    function claimCollateralERC20AsBorrower(uint[] memory indexs) internal {
        require(loanData.isCollateralNFT == false, "Collateral is NFT");

        uint collateralToSend;
        for (uint i; i < indexs.length; i++) {
            infoOfOffers memory offer = loanData._acceptedOffers[indexs[i]];
            require(offer.paid == true, "Not paid");
            require(offer.collateralClaimed == false, "Already executed");
            loanData._acceptedOffers[indexs[i]].collateralClaimed = true;
            uint decimalsCollateral = ERC20(loanData.collateral).decimals();
            collateralToSend +=
                (offer.principleAmount * (10 ** decimalsCollateral)) /
                offer.ratio;
        }
        SafeERC20.safeTransfer(
            IERC20(loanData.collateral),
            msg.sender,
            collateralToSend
        );
    }

    // function to extend the loan (only the borrower can call this function)
    // extend the loan to the max deadline of each offer
    function extendLoan() public {
        IOwnerships ownershipContract = IOwnerships(s_OwnershipContract);
        LoanData memory m_loan = loanData;
        require(
            ownershipContract.ownerOf(loanData.borrowerID) == msg.sender,
            "Not borrower"
        );
        require(
            nextDeadline() > block.timestamp,
            "Deadline passed to extend loan"
        );
        require(loanData.extended == false, "Already extended");
        // at least 10% of the loan duration has to be transcurred in order to extend the loan
        uint minimalDurationPayment = (m_loan.initialDuration * 1000) / 10000;
        require(
            (block.timestamp - m_loan.startedAt) > minimalDurationPayment,
            "Not enough time"
        );
        loanData.extended = true;

        // calculate fees to pay to us
        uint feePerDay = Aggregator(AggregatorContract).feePerDay();
        uint minFEE = Aggregator(AggregatorContract).minFEE();
        uint maxFee = Aggregator(AggregatorContract).maxFEE();
        uint PorcentageOfFeePaid = ((m_loan.initialDuration * feePerDay) /
            86400);
        // adjust fees

        if (PorcentageOfFeePaid > maxFee) {
            PorcentageOfFeePaid = maxFee;
        } else if (PorcentageOfFeePaid < minFEE) {
            PorcentageOfFeePaid = minFEE;
        }

        // calculate interest to pay to Debita and the subtract to the lenders

        for (uint i; i < m_loan._acceptedOffers.length; i++) {
            infoOfOffers memory offer = m_loan._acceptedOffers[i];
            // if paid, skip
            // if not paid, calculate interest to pay
            if (!offer.paid) {
                uint alreadyUsedTime = block.timestamp - m_loan.startedAt;

                uint extendedTime = offer.maxDeadline -
                    alreadyUsedTime -
                    block.timestamp;
                uint interestOfUsedTime = calculateInterestToPay(i);
                uint interestToPayToDebita = (interestOfUsedTime * feeLender) /
                    10000;

                uint misingBorrowFee;

                // if user already paid the max fee, then we dont have to charge them again
                if (PorcentageOfFeePaid != maxFee) {
                    // calculate difference from fee paid for the initialDuration vs the extra fee they should pay because of the extras days of extending the loan.  MAXFEE shouldnt be higher than extra fee + PorcentageOfFeePaid
                    uint feeOfMaxDeadline = ((offer.maxDeadline * feePerDay) /
                        86400);
                    if (feeOfMaxDeadline > maxFee) {
                        feeOfMaxDeadline = maxFee;
                    } else if (feeOfMaxDeadline < feePerDay) {
                        feeOfMaxDeadline = feePerDay;
                    }

                    misingBorrowFee = feeOfMaxDeadline - PorcentageOfFeePaid;
                }
                uint principleAmount = offer.principleAmount;
                uint feeAmount = (principleAmount * misingBorrowFee) / 10000;

                SafeERC20.safeTransferFrom(
                    IERC20(offer.principle),
                    msg.sender,
                    address(this),
                    interestOfUsedTime - interestToPayToDebita
                );

                SafeERC20.safeTransferFrom(
                    IERC20(offer.principle),
                    msg.sender,
                    feeAddress,
                    interestToPayToDebita + feeAmount
                );

                /* 
                CHECK IF CURRENT LENDER IS THE OWNER OF THE OFFER & IF IT'S PERPETUAL FOR INTEREST
                */
                DLOImplementation lendOffer = DLOImplementation(
                    offer.lendOffer
                );
                DLOImplementation.LendInfo memory lendInfo = lendOffer
                    .getLendInfo();
                address currentOwnerOfOffer;

                try ownershipContract.ownerOf(offer.lenderID) returns (
                    address _lenderOwner
                ) {
                    currentOwnerOfOffer = _lenderOwner;
                } catch {}

                if (
                    lendInfo.perpetual && lendInfo.owner == currentOwnerOfOffer
                ) {
                    IERC20(offer.principle).approve(
                        address(lendOffer),
                        interestOfUsedTime - interestToPayToDebita
                    );
                    lendOffer.addFunds(
                        interestOfUsedTime - interestToPayToDebita
                    );
                } else {
                    loanData._acceptedOffers[i].interestToClaim +=
                        interestOfUsedTime -
                        interestToPayToDebita;
                }
                loanData._acceptedOffers[i].interestPaid += interestOfUsedTime;
            }
        }
        Aggregator(AggregatorContract).emitLoanUpdated(address(this));
    }
    function claimCollateralNFTAsBorrower(uint[] memory indexes) internal {
        if (auctionData.alreadySold) {
            // in case of a partial default, borrower can claim the collateral of the offers that have been paid
            for (uint i; i < indexes.length; i++) {
                // load storage for each index
                LoanData memory m_loan = loanData;
                infoOfOffers memory offer = m_loan._acceptedOffers[indexes[i]];
                // check payment
                require(offer.paid == true, "Not paid");
                // not claimed yet
                require(offer.collateralClaimed == false, "Already executed");
                loanData._acceptedOffers[indexes[i]].collateralClaimed = true;

                uint decimalsCollateral = IveNFTEqualizer(loanData.collateral)
                    .getDataByReceipt(loanData.NftID)
                    .decimals;

                uint collateralUsed = offer.collateralUsed;

                uint payment = (auctionData.tokenPerCollateralUsed *
                    collateralUsed) / (10 ** decimalsCollateral);

                SafeERC20.safeTransfer(
                    IERC20(auctionData.liquidationAddress),
                    msg.sender,
                    payment
                );
            }
        } else {
            LoanData memory m_loan = loanData;
            // In case of NFT, borrower has to pay all the offers to claim the collateral
            require(
                m_loan.totalCountPaid == m_loan._acceptedOffers.length,
                "Not paid"
            );
            for (uint i; i < m_loan._acceptedOffers.length; i++) {
                // check payment one by one
                require(m_loan._acceptedOffers[i].paid == true, "Not paid");
                // not claimed yet
                require(
                    m_loan._acceptedOffers[i].collateralClaimed == false,
                    "Already executed"
                );
                loanData._acceptedOffers[i].collateralClaimed = true;
            }

            // send NFT to the borrower
            IERC721(m_loan.collateral).transferFrom(
                address(this),
                msg.sender,
                m_loan.NftID
            );
        }
    }

    // calculate interest to pay
    function calculateInterestToPay(uint index) public view returns (uint) {
        infoOfOffers memory offer = loanData._acceptedOffers[index];
        uint anualInterest = (offer.principleAmount * offer.apr) / 10000;
        // check already duration
        uint activeTime = block.timestamp - loanData.startedAt;
        uint minimalDurationPayment = (loanData.initialDuration * 1000) / 10000;
        uint maxDuration = offer.maxDeadline - loanData.startedAt;
        if (activeTime > maxDuration) {
            activeTime = maxDuration;
        } else if (activeTime < minimalDurationPayment) {
            activeTime = minimalDurationPayment;
        }

        uint interest = (anualInterest * activeTime) / 31536000;

        // subtract already paid interest
        return interest - offer.interestPaid;
    }

    /**
    @notice Function to get the next deadline of the loan
     */
    function nextDeadline() public view returns (uint) {
        uint _nextDeadline;
        LoanData memory m_loan = loanData;
        if (m_loan.extended) {
            for (uint i; i < m_loan._acceptedOffers.length; i++) {
                if (
                    _nextDeadline == 0 &&
                    m_loan._acceptedOffers[i].paid == false
                ) {
                    _nextDeadline = m_loan._acceptedOffers[i].maxDeadline;
                } else if (
                    m_loan._acceptedOffers[i].paid == false &&
                    _nextDeadline > m_loan._acceptedOffers[i].maxDeadline
                ) {
                    _nextDeadline = m_loan._acceptedOffers[i].maxDeadline;
                }
            }
        } else {
            _nextDeadline = m_loan.startedAt + m_loan.initialDuration;
        }
        return _nextDeadline;
    }

    function getLoanData() public view returns (LoanData memory) {
        return loanData;
    }

    function getAuctionData() public view returns (AuctionData memory) {
        return auctionData;
    }

    function safeGetOwner(uint tokenId) internal view returns (address) {
        IOwnerships ownershipContract = IOwnerships(s_OwnershipContract);
        try ownershipContract.ownerOf(tokenId) returns (address owner) {
            return owner;
        } catch {
            return address(0);
        }
    }
}
