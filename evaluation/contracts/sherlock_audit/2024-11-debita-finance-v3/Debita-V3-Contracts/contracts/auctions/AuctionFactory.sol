pragma solidity ^0.8.0;

import "@contracts/auctions/Auction.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";

interface IAggregator {
    function isSenderALoan(address) external view returns (bool);
}

contract auctionFactoryDebita {
    event createdAuction(
        address indexed auctionAddress,
        address indexed creator
    );
    event auctionEdited(
        address indexed auctionAddress,
        address indexed creator
    );
    event auctionEnded(address indexed auctionAddress, address indexed creator);

    // auction address ==> is auction
    mapping(address => bool) public isAuction; // if a contract is an auction

    // auction address ==> index
    mapping(address => uint) public AuctionOrderIndex; // index of an auction inside the active Orders

    // index ==> auction address
    mapping(uint => address) public allActiveAuctionOrders; // all active orders

    uint public activeOrdersCount; // count of active orders

    // 15%
    uint public FloorPricePercentage = 1500; // floor price for liquidations
    uint public auctionFee = 200; // fee for liquidations 2%
    uint public publicAuctionFee = 50; // fee for public auctions 0.5%
    uint deployedTime;
    address owner; // owner of the contract
    address aggregator;

    address public feeAddress; // address to send fees
    address[] public historicalAuctions; // all historical auctions

    constructor() {
        owner = msg.sender;
        feeAddress = msg.sender;
        deployedTime = block.timestamp;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner");
        _;
    }

    modifier onlyAuctions() {
        require(isAuction[msg.sender], "Only auctions");
        _;
    }
    /**
     * @dev create auction 
        * @param _veNFTID veNFT ID for the auction
        * @param _veNFTAddress veNFT address that you want to sell
        * @param liquidationToken the token address of the token you want to sell your veNFT for
        * @param _initAmount initial amount
        * @param _floorAmount floor amount of sell
        * @param _duration duration of the auction

     */
    function createAuction(
        uint _veNFTID,
        address _veNFTAddress,
        address liquidationToken,
        uint _initAmount,
        uint _floorAmount,
        uint _duration
    ) public returns (address) {
        // check if aggregator is set
        require(aggregator != address(0), "Aggregator not set");

        // initAmount should be more than floorAmount
        require(_initAmount >= _floorAmount, "Invalid amount");
        DutchAuction_veNFT _createdAuction = new DutchAuction_veNFT(
            _veNFTID,
            _veNFTAddress,
            liquidationToken,
            msg.sender,
            _initAmount,
            _floorAmount,
            _duration,
            IAggregator(aggregator).isSenderALoan(msg.sender) // if the sender is a loan --> isLiquidation = true
        );

        // Transfer veNFT
        IERC721(_veNFTAddress).safeTransferFrom(
            msg.sender,
            address(_createdAuction),
            _veNFTID,
            ""
        );

        // LOGIC INDEX
        AuctionOrderIndex[address(_createdAuction)] = activeOrdersCount;
        allActiveAuctionOrders[activeOrdersCount] = address(_createdAuction);
        activeOrdersCount++;
        historicalAuctions.push(address(_createdAuction));
        isAuction[address(_createdAuction)] = true;

        // emit event
        emit createdAuction(address(_createdAuction), msg.sender);
        return address(_createdAuction);
    }

    /**
     * @dev get active auction orders
     * @param offset offset
     * @param limit limit
     */
    function getActiveAuctionOrders(
        uint offset,
        uint limit
    ) external view returns (DutchAuction_veNFT.dutchAuction_INFO[] memory) {
        uint length = limit;
        if (limit > activeOrdersCount) {
            length = activeOrdersCount;
        }
        // chequear esto
        DutchAuction_veNFT.dutchAuction_INFO[]
            memory result = new DutchAuction_veNFT.dutchAuction_INFO[](
                length - offset
            );
        for (uint i = 0; (i + offset) < length; i++) {
            address order = allActiveAuctionOrders[offset + i];
            DutchAuction_veNFT.dutchAuction_INFO
                memory AuctionInfo = DutchAuction_veNFT(order).getAuctionData();
            result[i] = AuctionInfo;
        }
        return result;
    }

    function getLiquidationFloorPrice(
        uint initAmount
    ) public view returns (uint) {
        return (initAmount * FloorPricePercentage) / 10000;
    }

    function _deleteAuctionOrder(address _AuctionOrder) external onlyAuctions {
        // get index of the Auction order
        uint index = AuctionOrderIndex[_AuctionOrder];
        AuctionOrderIndex[_AuctionOrder] = 0;

        // get last Auction order
        allActiveAuctionOrders[index] = allActiveAuctionOrders[
            activeOrdersCount - 1
        ];
        // take out last Auction order
        allActiveAuctionOrders[activeOrdersCount - 1] = address(0);

        // switch index of the last Auction order to the deleted Auction order
        AuctionOrderIndex[allActiveAuctionOrders[index]] = index;
        activeOrdersCount--;
    }

    /**
     * @dev get historical auctions
     * @param offset offset
     * @param limit limit
     */
    function getHistoricalAuctions(
        uint offset,
        uint limit
    ) public view returns (DutchAuction_veNFT.dutchAuction_INFO[] memory) {
        uint length = limit;
        if (limit > historicalAuctions.length) {
            length = historicalAuctions.length;
        }
        DutchAuction_veNFT.dutchAuction_INFO[]
            memory result = new DutchAuction_veNFT.dutchAuction_INFO[](
                length - offset
            );
        for (uint i = 0; (i + offset) < length; i++) {
            address order = historicalAuctions[offset + i];
            DutchAuction_veNFT.dutchAuction_INFO
                memory AuctionInfo = DutchAuction_veNFT(order).getAuctionData();
            result[i] = AuctionInfo;
        }
        return result;
    }

    function getHistoricalAmount() public view returns (uint) {
        return historicalAuctions.length;
    }

    function setFloorPriceForLiquidations(uint _ratio) public onlyOwner {
        // Less than 30% and more than 5%
        require(_ratio <= 3000 && _ratio >= 500, "Invalid ratio");
        FloorPricePercentage = _ratio;
    }

    function changeAuctionFee(uint _fee) public onlyOwner {
        // between 0.5% and 4%
        require(_fee <= 400 && _fee >= 50, "Invalid fee");
        auctionFee = _fee;
    }
    function changePublicAuctionFee(uint _fee) public onlyOwner {
        // between 0% and 1%
        require(_fee <= 100 && _fee >= 0, "Invalid fee");
        publicAuctionFee = _fee;
    }

    function setAggregator(address _aggregator) public onlyOwner {
        require(aggregator == address(0), "Already set");
        aggregator = _aggregator;
    }

    function setFeeAddress(address _feeAddress) public onlyOwner {
        feeAddress = _feeAddress;
    }

    function changeOwner(address owner) public {
        require(msg.sender == owner, "Only owner");
        require(deployedTime + 6 hours > block.timestamp, "6 hours passed");
        owner = owner;
    }

    function emitAuctionDeleted(
        address _auctionAddress,
        address creator
    ) public onlyAuctions {
        emit auctionEnded(_auctionAddress, creator);
    }

    function emitAuctionEdited(
        address _auctionAddress,
        address creator
    ) public onlyAuctions {
        emit auctionEdited(_auctionAddress, creator);
    }

    // Events mints
}
