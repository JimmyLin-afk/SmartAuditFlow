pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@contracts/DebitaProxyContract.sol";

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
    function isActive() external view returns (bool);

    function initialize(
        address _aggregatorContract,
        bool _perpetual,
        bool[] memory _oraclesActivated,
        bool _lonelyLender,
        uint[] memory _LTVs,
        uint _apr,
        uint _maxDuration,
        uint _minDuration,
        address _owner,
        address _principle,
        address[] memory _acceptedCollaterals,
        address[] memory _oracles_Collateral,
        uint[] memory _ratio,
        address _oracleID_Principle,
        uint _startedLendingAmount
    ) external;
}

contract DLOFactory {
    event LendOrderCreated(
        address indexed lendOrder,
        address indexed owner,
        uint apr,
        uint maxDuration,
        uint minDuration,
        uint[] LTVs,
        uint[] Ratios,
        uint availableAmount,
        bool isActive,
        bool perpetual
    );
    event LendOrderUpdated(
        address indexed lendOrder,
        address indexed owner,
        uint apr,
        uint maxDuration,
        uint minDuration,
        uint[] LTVs,
        uint[] Ratios,
        uint availableAmount,
        bool isActive,
        bool perpetual
    );
    event LendOrderDeleted(
        address indexed lendOrder,
        address indexed owner,
        uint apr,
        uint maxDuration,
        uint minDuration,
        uint[] LTVs,
        uint[] Ratios,
        uint availableAmount,
        bool isActive,
        bool perpetual
    );

    mapping(address => bool) public isLendOrderLegit; // is lend order a legit order from the factory
    mapping(address => uint) public LendOrderIndex; // index of the lend order in the allActiveLendOrders array
    mapping(uint => address) public allActiveLendOrders; // all active lend orders

    uint public activeOrdersCount;
    address aggregatorContract;
    address public implementationContract;
    address public owner;

    constructor(address _implementationContract) {
        owner = msg.sender;

        implementationContract = _implementationContract;
    }

    modifier onlyLendOrder() {
        require(isLendOrderLegit[msg.sender], "Only lend order");
        _;
    }
    /**
     * @dev Create a new lend order
        * @param _perpetual is the lend order perpetual (compunding interest, every time the borrower pays the interest, the tokens come back to the order so the lender can lend it again)
        * @param _oraclesActivated array of booleans to activate oracles for each pair with collateral
        * @param _lonelyLender if true, the lend order is only for one lender
        * @param _LTVs array of max LTVs for each collateral
        * @param _apr annual percentage rate you want to lend the tokens for
        * @param _maxDuration max duration of the loan
        * @param _minDuration min duration of the loan
        * @param _acceptedCollaterals array of accepted collaterals
        * @param _principle address of the principle token
        * @param _oracles_Collateral array of oracles for each collateral
        * @param _ratio array of max ratios for each collateral
        * @param _oracleID_Principle address of the oracle for the principle token
        * @param _startedLendingAmount initial amount of the principle token that is available for lending
        * @return address of the new lend order

     */
    function createLendOrder(
        bool _perpetual,
        bool[] memory _oraclesActivated,
        bool _lonelyLender,
        uint[] memory _LTVs,
        uint _apr,
        uint _maxDuration,
        uint _minDuration,
        address[] memory _acceptedCollaterals,
        address _principle,
        address[] memory _oracles_Collateral,
        uint[] memory _ratio,
        address _oracleID_Principle,
        uint _startedLendingAmount
    ) external returns (address) {
        require(_minDuration <= _maxDuration, "Invalid duration");
        require(_LTVs.length == _acceptedCollaterals.length, "Invalid LTVs");
        require(
            _oracles_Collateral.length == _acceptedCollaterals.length,
            "Invalid length"
        );
        require(
            _oraclesActivated.length == _acceptedCollaterals.length,
            "Invalid oracles"
        );
        require(_ratio.length == _acceptedCollaterals.length, "Invalid ratio");

        DebitaProxyContract lendOfferProxy = new DebitaProxyContract(
            implementationContract
        );

        DLOImplementation lendOffer = DLOImplementation(
            address(lendOfferProxy)
        );

        lendOffer.initialize(
            aggregatorContract,
            _perpetual,
            _oraclesActivated,
            _lonelyLender,
            _LTVs,
            _apr,
            _maxDuration,
            _minDuration,
            msg.sender,
            _principle,
            _acceptedCollaterals,
            _oracles_Collateral,
            _ratio,
            _oracleID_Principle,
            _startedLendingAmount
        );

        SafeERC20.safeTransferFrom(
            IERC20(_principle),
            msg.sender,
            address(lendOffer),
            _startedLendingAmount
        );

        uint balance = IERC20(_principle).balanceOf(address(lendOffer));
        require(balance >= _startedLendingAmount, "Transfer failed");
        isLendOrderLegit[address(lendOffer)] = true;
        LendOrderIndex[address(lendOffer)] = activeOrdersCount;
        allActiveLendOrders[activeOrdersCount] = address(lendOffer);
        activeOrdersCount++;
        emit LendOrderCreated(
            address(lendOffer),
            msg.sender,
            _apr,
            _maxDuration,
            _minDuration,
            _LTVs,
            _ratio,
            _startedLendingAmount,
            true,
            _perpetual
        );
        return address(lendOffer);
    }

    // function to delete a lend order from index
    // only lend order can call this function
    function deleteOrder(address _lendOrder) external onlyLendOrder {
        uint index = LendOrderIndex[_lendOrder];
        LendOrderIndex[_lendOrder] = 0;

        // switch index of the last borrow order to the deleted borrow order
        allActiveLendOrders[index] = allActiveLendOrders[activeOrdersCount - 1];
        LendOrderIndex[allActiveLendOrders[activeOrdersCount - 1]] = index;

        // take out last borrow order

        allActiveLendOrders[activeOrdersCount - 1] = address(0);

        activeOrdersCount--;
    }

    function getActiveOrders(
        uint offset,
        uint limit
    ) public returns (DLOImplementation.LendInfo[] memory) {
        uint length = limit;
        if (length > activeOrdersCount) {
            length = activeOrdersCount;
        }

        DLOImplementation.LendInfo[]
            memory result = new DLOImplementation.LendInfo[](length - offset);

        for (uint i = 0; (i + offset) < limit; i++) {
            if ((i + offset) > (activeOrdersCount) - 1) {
                break;
            }
            result[i] = DLOImplementation(allActiveLendOrders[offset + i])
                .getLendInfo();
        }

        return result;
    }

    function setAggregatorContract(address _aggregatorContract) external {
        require(msg.sender == owner, "Only owner can set aggregator contract");
        require(aggregatorContract == address(0), "Already set");
        aggregatorContract = _aggregatorContract;
    }

    function emitUpdate(address _lendOrder) external onlyLendOrder {
        DLOImplementation lendOrder = DLOImplementation(_lendOrder);
        DLOImplementation.LendInfo memory lendInfo = lendOrder.getLendInfo();
        emit LendOrderUpdated(
            _lendOrder,
            lendInfo.owner,
            lendInfo.apr,
            lendInfo.maxDuration,
            lendInfo.minDuration,
            lendInfo.maxLTVs,
            lendInfo.maxRatio,
            lendInfo.availableAmount,
            lendOrder.isActive(),
            lendInfo.perpetual
        );
    }
    function emitDelete(address _lendOrder) external onlyLendOrder {
        DLOImplementation lendOrder = DLOImplementation(_lendOrder);
        DLOImplementation.LendInfo memory lendInfo = lendOrder.getLendInfo();
        emit LendOrderDeleted(
            _lendOrder,
            lendInfo.owner,
            lendInfo.apr,
            lendInfo.maxDuration,
            lendInfo.minDuration,
            lendInfo.maxLTVs,
            lendInfo.maxRatio,
            lendInfo.availableAmount,
            lendOrder.isActive(),
            lendInfo.perpetual
        );
    }
}
