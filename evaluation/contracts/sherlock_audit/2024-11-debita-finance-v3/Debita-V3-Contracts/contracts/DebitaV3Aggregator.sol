pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC721/utils/ERC721Holder.sol";

import "@contracts/DebitaProxyContract.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";

interface DLOFactory {
    function isLendOrderLegit(address _lendOrder) external view returns (bool);
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

    function getLendInfo() external view returns (LendInfo memory);
    function acceptLendingOffer(uint amount) external;
}

interface DebitaV3Loan {
    struct infoOfOffers {
        address principle;
        address lendOffer;
        uint principleAmount;
        uint lenderID;
        uint apr;
        uint ratio;
        uint collateralUsed;
        uint maxDeadline;
        bool paid;
        bool collateralClaimed;
        bool debtClaimed;
        uint interestToClaim;
        uint interestPaid;
    }

    struct LoanData {
        address collateral;
        address[] principles;
        address valuableCollateralAsset;
        bool isCollateralNFT;
        bool auctionInitialized;
        bool extended;
        uint startedAt;
        uint initialDuration;
        uint borrowerID;
        uint NftID;
        uint collateralAmount;
        uint collateralValuableAmount;
        uint valuableCollateralUsed;
        uint totalCountPaid;
        uint[] principlesAmount;
        infoOfOffers[] _acceptedOffers;
    }

    function getLoanData() external view returns (LoanData memory);
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
    ) external;
}

interface DebitaIncentives {
    function updateFunds(
        DebitaV3Loan.infoOfOffers[] memory informationOffers,
        address collateral,
        address[] memory lenders,
        address borrower
    ) external;
}

interface DBOFactory {
    function isBorrowOrderLegit(
        address _borrowOrder
    ) external view returns (bool);
}

interface IReceipt {
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
    ) external view returns (receiptInstance memory);
}

interface DBOImplementation {
    struct BorrowInfo {
        address borrowOrderAddress;
        bool[] oraclesPerPairActivated;
        uint[] LTVs;
        uint maxApr;
        uint duration;
        address owner;
        address[] acceptedPrinciples;
        address collateral;
        address valuableAsset;
        bool isNFT;
        uint receiptID;
        address[] oracles_Principles;
        uint[] ratio;
        address oracle_Collateral;
        uint valuableAssetAmount; // only used for auction sold NFTs
        uint availableAmount;
        uint startAmount;
    }

    function getBorrowInfo()
        external
        view
        returns (DBOImplementation.BorrowInfo memory);
    function acceptBorrowOffer(uint amount) external;
}

interface IOracle {
    function getThePrice(address _token) external view returns (uint);
}

contract DebitaV3Aggregator is ReentrancyGuard {
    event LoanCreated(
        address indexed loan,
        DebitaV3Loan.infoOfOffers[] offers,
        uint totalCountPaid,
        address collateral,
        bool auctionInit
    );
    event LoanDeleted(
        address indexed loan,
        DebitaV3Loan.infoOfOffers[] offers,
        uint totalCountPaid,
        address collateral,
        bool auctionInit
    );
    event LoanUpdated(
        address indexed loan,
        DebitaV3Loan.infoOfOffers[] offers,
        uint totalCountPaid,
        address collateral,
        bool auctionInit
    );

    address s_DLOFactory;
    address s_DBOFactory;
    address s_Incentives;
    address s_OwnershipContract;
    address s_LoanImplementation;
    address public s_AuctionFactory;

    address public feeAddress; // address where the fees are sent
    address public owner;
    uint deployedTime; // time when the contract was deployed
    uint public feePerDay = 4; // fee per day (0.04%)
    uint public maxFEE = 80; // max fee 0.8%
    uint public minFEE = 20; // min fee 0.2%
    uint public feeCONNECTOR = 1500; // 15% of the fee goes to the connector
    uint public feeInterestLender = 1500; // 15% of the paid interest
    uint public loanID;
    bool public isPaused; // aggregator is paused

    mapping(address => bool) public isSenderALoan; // if the address is a loan
    mapping(address => bool) public isCollateralAValidReceipt; // if address is a whitelisted NFT
    // id ownership => loan id
    mapping(uint => uint) public getLoanIdByOwnershipID;
    // loan id ==> loan address
    mapping(uint => address) public getAddressById;
    //
    mapping(address => bool) public oracleEnabled; // collateral address => is enabled?

    constructor(
        address _DLOFactory,
        address _DBOFactory,
        address _Incentives,
        address _OwnershipContract,
        address _auctionFactory,
        address loanImplementation
    ) {
        s_DLOFactory = _DLOFactory;
        s_DBOFactory = _DBOFactory;
        s_Incentives = _Incentives;
        s_OwnershipContract = _OwnershipContract;
        feeAddress = msg.sender;
        owner = msg.sender;
        s_AuctionFactory = _auctionFactory;
        s_LoanImplementation = loanImplementation;
        deployedTime = block.timestamp;
    }

    struct BorrowInfo {
        address borrowOrderAddress;
        bool[] oraclesPerPairActivated;
        uint[] LTVs;
        uint maxApr;
        uint duration;
        address owner;
        address[] acceptedPrinciples;
        address collateral;
        address valuableAsset;
        bool isNFT;
        uint receiptID;
        address[] oracles_Principles;
        uint[] ratio;
        address oracle_Collateral;
        uint valuableAssetAmount; // only used for auction sold NFTs
        uint availableAmount;
    }

    modifier onlyLoan() {
        require(isSenderALoan[msg.sender], "Sender is not a loan");
        _;
    }

    // lenders have multiple accepted collaterals
    // borrowers have multiple accepted principles
    /**
     * @notice Calculate ratio for each lend order and the borrower individually and then check if the ratios are within the limits
     * @dev Match offers from lenders with a borrower -- It can be called by anyone and the msg.sender will get a reward for calling this function
     * @param lendOrders array of lend orders you want to get liquidity from
     * @param lendAmountPerOrder array of amounts you want to get from each lend order
     * @param porcentageOfRatioPerLendOrder array of percentages of the ratio you want to get from each lend order (10000 = 100% of the maxRatio)
     * @param borrowOrder address of the borrow order
     * @param principles array of principles you want to borrow
     * @param indexForPrinciple_BorrowOrder array of indexes for the principles on the borrow order (in which index is the principle on acceptedPrinciples)
     * @param indexForCollateral_LendOrder array of indexes for the collateral on each lend order (in which index is the collateral on acceptedCollaterals)
     * @param indexPrinciple_LendOrder array of indexes for the principle on each lend order (in which index is the principle of the lend order on principles param)
     */
    function matchOffersV3(
        address[] memory lendOrders,
        uint[] memory lendAmountPerOrder,
        uint[] memory porcentageOfRatioPerLendOrder,
        address borrowOrder,
        address[] memory principles,
        uint[] memory indexForPrinciple_BorrowOrder,
        uint[] memory indexForCollateral_LendOrder,
        uint[] memory indexPrinciple_LendOrder
    ) external nonReentrant returns (address) {
        // Add count
        loanID++;
        DBOImplementation.BorrowInfo memory borrowInfo = DBOImplementation(
            borrowOrder
        ).getBorrowInfo();
        // check lendOrder length is less than 100
        require(lendOrders.length <= 100, "Too many lend orders");
        // check borrow order is legit
        require(
            DBOFactory(s_DBOFactory).isBorrowOrderLegit(borrowOrder),
            "Invalid borrow order"
        );
        // check if the aggregator is paused
        require(!isPaused, "New loans are paused");
        // check if valid collateral
        require(
            isCollateralAValidReceipt[borrowInfo.collateral] ||
                !borrowInfo.isNFT,
            "Invalid collateral"
        );

        // get price of collateral using borrow order oracle
        uint priceCollateral_BorrowOrder;

        if (borrowInfo.oracle_Collateral != address(0)) {
            priceCollateral_BorrowOrder = getPriceFrom(
                borrowInfo.oracle_Collateral,
                borrowInfo.valuableAsset
            );
        }
        uint[] memory ratiosForBorrower = new uint[](principles.length);

        // calculate ratio from the borrower for each principle used on this loan --  same collateral different principles
        for (uint i = 0; i < principles.length; i++) {
            // check if principle is accepted by borow order
            require(
                borrowInfo.acceptedPrinciples[
                    indexForPrinciple_BorrowOrder[i]
                ] == principles[i],
                "Invalid principle on borrow order"
            );
            // if the oracle is activated on this pair, get the price and calculate the ratio. If not use fixed ratio of the offer
            if (
                borrowInfo.oraclesPerPairActivated[
                    indexForPrinciple_BorrowOrder[i]
                ]
            ) {
                // if oracle is activated check price is not 0
                require(priceCollateral_BorrowOrder != 0, "Invalid price");
                // get principle price
                uint pricePrinciple = getPriceFrom(
                    borrowInfo.oracles_Principles[
                        indexForPrinciple_BorrowOrder[i]
                    ],
                    principles[i]
                );
                /* 
               
                pricePrinciple / priceCollateral_BorrowOrder = 100% ltv (multiply by 10^8 to get extra 8 decimals to avoid floating)

                Example:
                collateral / principle
                1.45 / 2000 = 0.000725 nominal tokens of principle per collateral for 100% LTV                                        
                */
                uint principleDecimals = ERC20(principles[i]).decimals();

                uint ValuePrincipleFullLTVPerCollateral = (priceCollateral_BorrowOrder *
                        10 ** 8) / pricePrinciple;

                // take 100% of the LTV and multiply by the LTV of the principle
                uint value = (ValuePrincipleFullLTVPerCollateral *
                    borrowInfo.LTVs[indexForPrinciple_BorrowOrder[i]]) / 10000;

                /**
                 get the ratio for the amount of principle the borrower wants to borrow
                 fix the 8 decimals and get it on the principle decimals
                 */
                uint ratio = (value * (10 ** principleDecimals)) / (10 ** 8);
                ratiosForBorrower[i] = ratio;
            } else {
                ratiosForBorrower[i] = borrowInfo.ratio[
                    indexForPrinciple_BorrowOrder[i]
                ];
            }
        }
        // calculate ratio per lenderOrder, same collateral different (for loop)
        uint amountOfCollateral;
        uint decimalsCollateral = ERC20(borrowInfo.valuableAsset).decimals();
        // weighted ratio for each principle
        uint[] memory weightedAverageRatio = new uint[](principles.length);

        // amount of collateral used per principle
        uint[] memory amountCollateralPerPrinciple = new uint[](
            principles.length
        );
        // amount of principle per principle to be lent
        uint[] memory amountPerPrinciple = new uint[](principles.length);

        // weighted APR for each principle
        uint[] memory weightedAverageAPR = new uint[](principles.length);

        // Info of each accepted offer
        address[] memory lenders = new address[](lendOrders.length);
        DebitaV3Loan.infoOfOffers[]
            memory offers = new DebitaV3Loan.infoOfOffers[](lendOrders.length);

        // percentage
        uint percentage = ((borrowInfo.duration * feePerDay) / 86400);
        uint[] memory feePerPrinciple = new uint[](principles.length);

        // init DLOFACTORY
        DLOFactory dloFactory = DLOFactory(s_DLOFactory);
        for (uint i = 0; i < lendOrders.length; i++) {
            // check lend order is legit
            require(
                dloFactory.isLendOrderLegit(lendOrders[i]),
                "Invalid lend order"
            );
            // check incentives here
            DLOImplementation.LendInfo memory lendInfo = DLOImplementation(
                lendOrders[i]
            ).getLendInfo();
            uint principleIndex = indexPrinciple_LendOrder[i];
            // check if is lonely lender, if true, only one lender is allowed
            if (lendInfo.lonelyLender) {
                require(lendOrders.length == 1, " Only one lender is allowed");
            }

            // check porcentage of ratio is between 100% and 0%
            require(
                porcentageOfRatioPerLendOrder[i] <= 10000 &&
                    porcentageOfRatioPerLendOrder[i] > 0,
                "Invalid percentage"
            );

            // check that the collateral is accepted by the lend order
            require(
                lendInfo.acceptedCollaterals[indexForCollateral_LendOrder[i]] ==
                    borrowInfo.collateral,
                "Invalid collateral Lend Offer"
            );

            // check that the principle is provided by the lend order
            require(
                lendInfo.principle == principles[principleIndex],
                "Invalid principle on lend order"
            );
            // check that the duration is between the min and max duration from the lend order
            require(
                borrowInfo.duration >= lendInfo.minDuration &&
                    borrowInfo.duration <= lendInfo.maxDuration,
                "Invalid duration"
            );
            uint collateralIndex = indexForCollateral_LendOrder[i];
            uint maxRatio;
            // check if the lend order has an oracle activated for the pair
            if (lendInfo.oraclesPerPairActivated[collateralIndex]) {
                // calculate the price for collateral and principles with each oracles provided by the lender
                uint priceCollateral_LendOrder = getPriceFrom(
                    lendInfo.oracle_Collaterals[collateralIndex],
                    borrowInfo.valuableAsset
                );
                uint pricePrinciple = getPriceFrom(
                    lendInfo.oracle_Principle,
                    principles[principleIndex]
                );

                uint fullRatioPerLending = (priceCollateral_LendOrder *
                    10 ** 8) / pricePrinciple;
                uint maxValue = (fullRatioPerLending *
                    lendInfo.maxLTVs[collateralIndex]) / 10000;
                uint principleDecimals = ERC20(principles[principleIndex])
                    .decimals();
                maxRatio = (maxValue * (10 ** principleDecimals)) / (10 ** 8);
            } else {
                maxRatio = lendInfo.maxRatio[collateralIndex];
            }
            // calculate ratio based on porcentage of the lend order
            uint ratio = (maxRatio * porcentageOfRatioPerLendOrder[i]) / 10000;
            uint m_amountCollateralPerPrinciple = amountCollateralPerPrinciple[
                principleIndex
            ];
            // calculate the amount of collateral used by the lender
            uint userUsedCollateral = (lendAmountPerOrder[i] *
                (10 ** decimalsCollateral)) / ratio;

            // get updated weight average from the last weight average
            uint updatedLastWeightAverage = (weightedAverageRatio[
                principleIndex
            ] * m_amountCollateralPerPrinciple) /
                (m_amountCollateralPerPrinciple + userUsedCollateral);

            // same with apr
            uint updatedLastApr = (weightedAverageAPR[principleIndex] *
                amountPerPrinciple[principleIndex]) /
                (amountPerPrinciple[principleIndex] + lendAmountPerOrder[i]);

            // add the amounts to the total amounts
            amountPerPrinciple[principleIndex] += lendAmountPerOrder[i];
            amountOfCollateral += userUsedCollateral;
            amountCollateralPerPrinciple[principleIndex] += userUsedCollateral;

            // calculate new weights
            uint newWeightedAverage = (ratio * userUsedCollateral) /
                (m_amountCollateralPerPrinciple + userUsedCollateral);

            uint newWeightedAPR = (lendInfo.apr * lendAmountPerOrder[i]) /
                amountPerPrinciple[principleIndex];

            // calculate the weight of the new amounts, add them to the weighted and accept offers
            weightedAverageRatio[principleIndex] =
                newWeightedAverage +
                updatedLastWeightAverage;
            weightedAverageAPR[principleIndex] =
                newWeightedAPR +
                updatedLastApr;

            // mint ownership for the lender
            uint lendID = IOwnerships(s_OwnershipContract).mint(lendInfo.owner);
            offers[i] = DebitaV3Loan.infoOfOffers({
                principle: lendInfo.principle,
                lendOffer: lendOrders[i],
                principleAmount: lendAmountPerOrder[i],
                lenderID: lendID,
                apr: lendInfo.apr,
                ratio: ratio,
                collateralUsed: userUsedCollateral,
                maxDeadline: lendInfo.maxDuration + block.timestamp,
                paid: false,
                collateralClaimed: false,
                debtClaimed: false,
                interestToClaim: 0,
                interestPaid: 0
            });
            getLoanIdByOwnershipID[lendID] = loanID;
            lenders[i] = lendInfo.owner;
            DLOImplementation(lendOrders[i]).acceptLendingOffer(
                lendAmountPerOrder[i]
            );
        }
        // fix the percentage of the fees
        if (percentage > maxFEE) {
            percentage = maxFEE;
        }

        if (percentage < minFEE) {
            percentage = minFEE;
        }

        // check ratio for each principle and check if the ratios are within the limits of the borrower
        for (uint i = 0; i < principles.length; i++) {
            require(
                weightedAverageRatio[i] >=
                    ((ratiosForBorrower[i] * 9800) / 10000) &&
                    weightedAverageRatio[i] <=
                    (ratiosForBorrower[i] * 10200) / 10000,
                "Invalid ratio"
            );

            // calculate fees --> msg.sender keeps 15% of the fee for connecting the offers
            uint feeToPay = (amountPerPrinciple[i] * percentage) / 10000;
            uint feeToConnector = (feeToPay * feeCONNECTOR) / 10000;
            feePerPrinciple[i] = feeToPay;
            // transfer fee to feeAddress
            SafeERC20.safeTransfer(
                IERC20(principles[i]),
                feeAddress,
                feeToPay - feeToConnector
            );
            // transfer fee to connector
            SafeERC20.safeTransfer(
                IERC20(principles[i]),
                msg.sender,
                feeToConnector
            );
            // check if the apr is within the limits of the borrower
            require(weightedAverageAPR[i] <= borrowInfo.maxApr, "Invalid APR");
        }
        // if collateral is an NFT, check if the amount of collateral is within the limits
        // it has a 2% margin to make easier the matching, amountOfCollateral is the amount of collateral "consumed" and the valuableAssetAmount is the underlying amount of the NFT
        if (borrowInfo.isNFT) {
            require(
                amountOfCollateral <=
                    (borrowInfo.valuableAssetAmount * 10200) / 10000 &&
                    amountOfCollateral >=
                    (borrowInfo.valuableAssetAmount * 9800) / 10000,
                "Invalid collateral amount"
            );
        }
        DBOImplementation(borrowOrder).acceptBorrowOffer(
            borrowInfo.isNFT ? 1 : amountOfCollateral
        );

        uint borrowID = IOwnerships(s_OwnershipContract).mint(borrowInfo.owner);

        // finish the loan & change the inputs

        // falta pagar incentivos y pagar fee
        DebitaProxyContract _loanProxy = new DebitaProxyContract(
            s_LoanImplementation
        );
        DebitaV3Loan deployedLoan = DebitaV3Loan(address(_loanProxy));
        // init loan
        deployedLoan.initialize(
            borrowInfo.collateral,
            principles,
            borrowInfo.isNFT,
            borrowInfo.receiptID,
            borrowInfo.isNFT ? 1 : amountOfCollateral,
            borrowInfo.valuableAssetAmount,
            amountOfCollateral,
            borrowInfo.valuableAsset,
            borrowInfo.duration,
            amountPerPrinciple,
            borrowID, //borrowInfo.id,
            offers,
            s_OwnershipContract,
            feeInterestLender,
            feeAddress
        );
        // save loan
        getAddressById[loanID] = address(deployedLoan);
        isSenderALoan[address(deployedLoan)] = true;

        // transfer the principles to the borrower
        for (uint i; i < principles.length; i++) {
            SafeERC20.safeTransfer(
                IERC20(principles[i]),
                borrowInfo.owner,
                amountPerPrinciple[i] - feePerPrinciple[i]
            );
        }
        // transfer the collateral to the loan
        if (borrowInfo.isNFT) {
            IERC721(borrowInfo.collateral).transferFrom(
                address(this),
                address(deployedLoan),
                borrowInfo.receiptID
            );
        } else {
            SafeERC20.safeTransfer(
                IERC20(borrowInfo.collateral),
                address(deployedLoan),
                amountOfCollateral
            );
        }
        // update incentives
        DebitaIncentives(s_Incentives).updateFunds(
            offers,
            borrowInfo.collateral,
            lenders,
            borrowInfo.owner
        );

        // emit
        emit LoanCreated(
            address(deployedLoan),
            offers,
            0,
            borrowInfo.collateral,
            false
        );
        return address(deployedLoan);
    }

    function statusCreateNewOffers(bool _newStatus) public {
        require(msg.sender == owner, "Invalid address");
        isPaused = _newStatus;
    }

    function setValidNFTCollateral(address _collateral, bool status) external {
        require(msg.sender == owner, "Invalid address");
        isCollateralAValidReceipt[_collateral] = status;
    }
    function setNewFee(uint _fee) external {
        require(msg.sender == owner, "Invalid address");
        require(_fee >= 1 && _fee <= 10, "Invalid fee");
        feePerDay = _fee;
    }

    function setNewMaxFee(uint _fee) external {
        require(msg.sender == owner, "Invalid address");
        require(_fee >= 50 && _fee <= 100, "Invalid fee");
        maxFEE = _fee;
    }

    function setNewMinFee(uint _fee) external {
        require(msg.sender == owner, "Invalid address");
        require(_fee >= 10 && _fee <= 50, "Invalid fee");
        minFEE = _fee;
    }

    function setNewFeeConnector(uint _fee) external {
        require(msg.sender == owner, "Invalid address");
        require(_fee >= 500 && _fee <= 2000, "Invalid fee");
        feeCONNECTOR = _fee;
    }

    function changeOwner(address owner) public {
        require(msg.sender == owner, "Only owner");
        require(deployedTime + 6 hours > block.timestamp, "6 hours passed");
        owner = owner;
    }

    function setOracleEnabled(address _oracle, bool status) external {
        require(msg.sender == owner, "Invalid address");
        oracleEnabled[_oracle] = status;
    }

    function getAllLoans(
        uint offset,
        uint limit
    ) external view returns (DebitaV3Loan.LoanData[] memory) {
        // return LoanData
        uint _limit = loanID;
        if (limit > _limit) {
            limit = _limit;
        }

        DebitaV3Loan.LoanData[] memory loans = new DebitaV3Loan.LoanData[](
            limit - offset
        );

        for (uint i = 0; i < limit - offset; i++) {
            if ((i + offset + 1) >= loanID) {
                break;
            }
            address loanAddress = getAddressById[i + offset + 1];

            DebitaV3Loan loan = DebitaV3Loan(loanAddress);
            loans[i] = loan.getLoanData();

            // loanIDs start at 1
        }
        return loans;
    }

    function getPriceFrom(
        address _oracle,
        address _token
    ) internal view returns (uint) {
        require(oracleEnabled[_oracle], "Oracle not enabled");
        return IOracle(_oracle).getThePrice(_token);
    }

    function emitLoanUpdated(address loan) public onlyLoan {
        DebitaV3Loan loanInstance = DebitaV3Loan(loan);
        DebitaV3Loan.LoanData memory loanData = loanInstance.getLoanData();
        emit LoanUpdated(
            loan,
            loanData._acceptedOffers,
            loanData.totalCountPaid,
            loanData.collateral,
            loanData.auctionInitialized
        );
    }
}
