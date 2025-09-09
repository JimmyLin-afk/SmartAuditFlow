pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@contracts/buyOrders/buyOrder.sol";
import "@contracts/DebitaProxyContract.sol";

contract buyOrderFactory {
    using SafeERC20 for IERC20;

    uint public sellFee = 50; // 0.5%
    address public feeAddress;

    event BuyOrderCreated(
        address indexed buyOrder,
        address indexed owner,
        address wantedToken,
        address buyToken,
        uint amount,
        uint ratio
    );

    event BuyOrderDeleted(
        address indexed buyOrder,
        address indexed owner,
        address wantedToken,
        address buyToken,
        uint amount,
        uint ratio
    );

    event BuyOrderUpdated(
        address indexed buyOrder,
        address indexed owner,
        address wantedToken,
        address buyToken,
        uint amount,
        uint ratio
    );

    // buy order address ==> is buy order
    mapping(address => bool) public isBuyOrderLegit; // is a contract a buy order created by this factory
    // buy order address ==> index
    mapping(address => uint) public BuyOrderIndex; // index of buy order address on allActiveBuyOrders

    // index ==> buy order address
    mapping(uint => address) public allActiveBuyOrders; // all active buy orders

    uint public activeOrdersCount; // count of active buy orders
    address public owner; // owner of the contract
    uint deployedTime;
    address implementationContract;
    address[] public historicalBuyOrders; // all historical buy orders

    constructor(address _implementationContract) {
        owner = msg.sender;
        feeAddress = msg.sender;
        implementationContract = _implementationContract;
        deployedTime = block.timestamp;
    }

    modifier onlyBuyOrder() {
        require(isBuyOrderLegit[msg.sender], "Only buy order");
        _;
    }

    /**
     * @dev create buy order
     * @param _token token address you want to use to buy the wanted token
     * @param wantedToken the token address you want to buy
     * @param _amount amount of token you want to use to buy the wanted token
     * @param ratio ratio you want to use to buy the wanted token (5e17:0.5)
     */

    function createBuyOrder(
        address _token,
        address wantedToken,
        uint _amount,
        uint ratio
    ) public returns (address) {
        // CHECKS
        require(_amount > 0, "Amount must be greater than 0");
        require(ratio > 0, "Ratio must be greater than 0");

        DebitaProxyContract proxy = new DebitaProxyContract(
            implementationContract
        );
        BuyOrder _createdBuyOrder = BuyOrder(address(proxy));

        // INITIALIZE THE BUY ORDER
        _createdBuyOrder.initialize(
            msg.sender,
            _token,
            wantedToken,
            address(this),
            _amount,
            ratio
        );

        // TRANSFER TOKENS TO THE BUY ORDER
        SafeERC20.safeTransferFrom(
            IERC20(_token),
            msg.sender,
            address(_createdBuyOrder),
            _amount
        );

        // INDEX
        isBuyOrderLegit[address(_createdBuyOrder)] = true;
        BuyOrderIndex[address(_createdBuyOrder)] = activeOrdersCount;
        allActiveBuyOrders[activeOrdersCount] = address(_createdBuyOrder);
        activeOrdersCount++;
        historicalBuyOrders.push(address(_createdBuyOrder));

        emit BuyOrderCreated(
            address(_createdBuyOrder),
            msg.sender,
            wantedToken,
            _token,
            _amount,
            ratio
        );
        return address(_createdBuyOrder);
    }

    // delete order from active orders
    function _deleteBuyOrder(address _buyOrder) public onlyBuyOrder {
        uint index = BuyOrderIndex[_buyOrder];
        BuyOrderIndex[_buyOrder] = 0;

        allActiveBuyOrders[index] = allActiveBuyOrders[activeOrdersCount - 1];
        allActiveBuyOrders[activeOrdersCount - 1] = address(0);

        BuyOrderIndex[allActiveBuyOrders[index]] = index;

        activeOrdersCount--;
    }

    function getActiveBuyOrders(
        uint offset,
        uint limit
    ) public view returns (BuyOrder.BuyInfo[] memory) {
        uint length = limit;

        if (limit > activeOrdersCount) {
            length = activeOrdersCount;
        }

        BuyOrder.BuyInfo[] memory _activeBuyOrders = new BuyOrder.BuyInfo[](
            limit - offset
        );
        for (uint i = offset; i < offset + limit; i++) {
            address order = allActiveBuyOrders[i];
            _activeBuyOrders[i] = BuyOrder(order).getBuyInfo();
        }
        return _activeBuyOrders;
    }

    function getHistoricalBuyOrders(
        uint offset,
        uint limit
    ) public view returns (BuyOrder.BuyInfo[] memory) {
        uint length = limit;

        if (limit > historicalBuyOrders.length) {
            length = historicalBuyOrders.length;
        }

        BuyOrder.BuyInfo[] memory _historicalBuyOrders = new BuyOrder.BuyInfo[](
            limit - offset
        );
        for (uint i = offset; i < offset + limit; i++) {
            address order = historicalBuyOrders[i];
            _historicalBuyOrders[i] = BuyOrder(order).getBuyInfo();
        }
        return _historicalBuyOrders;
    }

    function changeFee(uint _fee) public {
        require(msg.sender == owner, "Only owner");
        require(_fee >= 20 && _fee <= 100, "Invalid fee");
        sellFee = _fee;
    }

    // change owner of the contract only between 0 and 6 hours after deployment
    function changeOwner(address owner) public {
        require(msg.sender == owner, "Only owner");
        require(deployedTime + 6 hours > block.timestamp, "6 hours passed");
        owner = owner;
    }

    function emitDelete(address _buyOrder) public onlyBuyOrder {
        BuyOrder.BuyInfo memory _buyInfo = BuyOrder(_buyOrder).getBuyInfo();
        emit BuyOrderDeleted(
            _buyOrder,
            _buyInfo.owner,
            _buyInfo.wantedToken,
            _buyInfo.buyToken,
            _buyInfo.availableAmount,
            _buyInfo.buyRatio
        );
    }

    function emitUpdate(address _buyOrder) public onlyBuyOrder {
        BuyOrder.BuyInfo memory _buyInfo = BuyOrder(_buyOrder).getBuyInfo();
        emit BuyOrderUpdated(
            _buyOrder,
            _buyInfo.owner,
            _buyInfo.wantedToken,
            _buyInfo.buyToken,
            _buyInfo.availableAmount,
            _buyInfo.buyRatio
        );
    }
}
