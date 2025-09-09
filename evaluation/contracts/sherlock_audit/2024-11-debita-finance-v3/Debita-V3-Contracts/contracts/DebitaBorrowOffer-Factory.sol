pragma solidity ^0.8.0;

// if NFT --> it has to be a receipt

import "@contracts/DebitaBorrowOffer-Implementation.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@contracts/DebitaProxyContract.sol";

contract DBOFactory {
    event BorrowOrderCreated(
        address indexed borrowOrder,
        address indexed owner,
        uint maxApr,
        uint duration,
        uint[] LTVs,
        uint[] ratios,
        uint availableAmount,
        bool isActive
    );
    event BorrowOrderUpdated(
        address indexed borrowOrder,
        address indexed owner,
        uint maxApr,
        uint duration,
        uint[] LTVs,
        uint[] ratios,
        uint availableAmount,
        bool isActive
    );
    event BorrowOrderDeleted(
        address indexed borrowOrder,
        address indexed owner,
        uint maxApr,
        uint duration,
        uint[] LTVs,
        uint[] ratios,
        uint availableAmount,
        bool isActive
    );

    mapping(address => bool) public isBorrowOrderLegit;
    mapping(address => uint) public borrowOrderIndex;
    mapping(uint => address) public allActiveBorrowOrders;

    uint public activeOrdersCount;
    address aggregatorContract;
    address implementationContract;
    address public owner;

    constructor(address _implementationContract) {
        owner = msg.sender;
        implementationContract = _implementationContract;
    }

    modifier onlyBorrowOrder() {
        require(isBorrowOrderLegit[msg.sender], "Only borrow order");
        _;
    }
    /**
     * @dev Creates a new Borrow Order
     * @param _oraclesActivated Array of booleans that indicates if the oracle is activated por that pair (acceptedPrinciples[i], collateral)
     * @param _LTVs Array of LTVs for each principle (0 if not accepted)
     * @param _maxInterestRate Maximum interest rate that the borrower is willing to pay
     * @param _duration Duration of the loan
     * @param _acceptedPrinciples Array of addresses of the principles that the borrower is willing to accept
     * @param _collateral Address of the collateral token
     * @param _isNFT Boolean that indicates if the collateral is an NFT
     * @param _receiptID ID of the NFT if the collateral is an NFT
     * @param _oracleIDS_Principles Array of addresses of the oracles for each principle
     * @param _ratio Array of ratios for each principle
     * @param _oracleID_Collateral Address of the oracle for the collateral (used in case oracle is activated)
     * @param _collateralAmount Amount of collateral willing to be used
     */
    function createBorrowOrder(
        bool[] memory _oraclesActivated,
        uint[] memory _LTVs,
        uint _maxInterestRate,
        uint _duration,
        address[] memory _acceptedPrinciples,
        address _collateral,
        bool _isNFT,
        uint _receiptID,
        address[] memory _oracleIDS_Principles,
        uint[] memory _ratio,
        address _oracleID_Collateral,
        uint _collateralAmount
    ) external returns (address) {
        if (_isNFT) {
            require(_receiptID != 0, "Receipt ID cannot be 0");
            require(_collateralAmount == 1, "Started Borrow Amount must be 1");
        }

        require(_LTVs.length == _acceptedPrinciples.length, "Invalid LTVs");
        require(
            _oracleIDS_Principles.length == _acceptedPrinciples.length,
            "Invalid length"
        );
        require(
            _oraclesActivated.length == _acceptedPrinciples.length,
            "Invalid oracles"
        );
        require(_ratio.length == _acceptedPrinciples.length, "Invalid ratio");
        require(_collateralAmount > 0, "Invalid started amount");

        DBOImplementation borrowOffer = new DBOImplementation();

        borrowOffer.initialize(
            aggregatorContract,
            msg.sender,
            _acceptedPrinciples,
            _collateral,
            _oraclesActivated,
            _isNFT,
            _LTVs,
            _maxInterestRate,
            _duration,
            _receiptID,
            _oracleIDS_Principles,
            _ratio,
            _oracleID_Collateral,
            _collateralAmount
        );
        isBorrowOrderLegit[address(borrowOffer)] = true;
        if (_isNFT) {
            IERC721(_collateral).transferFrom(
                msg.sender,
                address(borrowOffer),
                _receiptID
            );
        } else {
            SafeERC20.safeTransferFrom(
                IERC20(_collateral),
                msg.sender,
                address(borrowOffer),
                _collateralAmount
            );
        }
        borrowOrderIndex[address(borrowOffer)] = activeOrdersCount;
        allActiveBorrowOrders[activeOrdersCount] = address(borrowOffer);
        activeOrdersCount++;

        uint balance = IERC20(_collateral).balanceOf(address(borrowOffer));
        require(balance >= _collateralAmount, "Invalid balance");

        emit BorrowOrderCreated(
            address(borrowOffer),
            msg.sender,
            _maxInterestRate,
            _duration,
            _LTVs,
            _ratio,
            _collateralAmount,
            true
        );
        return address(borrowOffer);
    }

    /**
     * @dev Deletes Borrow Order from index -- only callable from borrow Orders
     */
    function deleteBorrowOrder(address _borrowOrder) external onlyBorrowOrder {
        // get index of the borrow order
        uint index = borrowOrderIndex[_borrowOrder];
        borrowOrderIndex[_borrowOrder] = 0;

        // get last borrow order
        allActiveBorrowOrders[index] = allActiveBorrowOrders[
            activeOrdersCount - 1
        ];
        // take out last borrow order
        allActiveBorrowOrders[activeOrdersCount - 1] = address(0);

        // switch index of the last borrow order to the deleted borrow order
        borrowOrderIndex[allActiveBorrowOrders[index]] = index;
        activeOrdersCount--;
    }

    function getActiveBorrowOrders(
        uint offset,
        uint limit
    ) external view returns (DBOImplementation.BorrowInfo[] memory) {
        uint length = limit;
        if (limit > activeOrdersCount) {
            length = activeOrdersCount;
        }
        // chequear esto
        DBOImplementation.BorrowInfo[]
            memory result = new DBOImplementation.BorrowInfo[](length - offset);
        for (uint i = 0; (i + offset) < length; i++) {
            address order = allActiveBorrowOrders[offset + i];

            DBOImplementation.BorrowInfo memory borrowInfo = DBOImplementation(
                order
            ).getBorrowInfo();
            result[i] = borrowInfo;
        }
        return result;
    }

    function setAggregatorContract(address _aggregatorContract) external {
        require(aggregatorContract == address(0), "Already set");
        require(msg.sender == owner, "Only owner can set aggregator contract");
        aggregatorContract = _aggregatorContract;
    }

    function emitDelete(address _borrowOrder) external onlyBorrowOrder {
        DBOImplementation borrowOrder = DBOImplementation(_borrowOrder);
        DBOImplementation.BorrowInfo memory borrowInfo = borrowOrder
            .getBorrowInfo();
        emit BorrowOrderDeleted(
            _borrowOrder,
            borrowInfo.owner,
            borrowInfo.maxApr,
            borrowInfo.duration,
            borrowInfo.LTVs,
            borrowInfo.ratio,
            borrowInfo.availableAmount,
            false
        );
    }

    function emitUpdate(address _borrowOrder) external onlyBorrowOrder {
        DBOImplementation borrowOrder = DBOImplementation(_borrowOrder);
        DBOImplementation.BorrowInfo memory borrowInfo = borrowOrder
            .getBorrowInfo();
        emit BorrowOrderUpdated(
            _borrowOrder,
            borrowInfo.owner,
            borrowInfo.maxApr,
            borrowInfo.duration,
            borrowInfo.LTVs,
            borrowInfo.ratio,
            borrowInfo.availableAmount,
            borrowOrder.isActive()
        );
    }
}
