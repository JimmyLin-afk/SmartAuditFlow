pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/token/ERC721/utils/ERC721Holder.sol";

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/proxy/utils/Initializable.sol";

interface NFR {
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

interface IDBOFactory {
    function emitDelete(address _borrowOrder) external;
    function emitUpdate(address _borrowOrder) external;
    function deleteBorrowOrder(address _borrowOrder) external;
}

contract DBOImplementation is ReentrancyGuard, Initializable {
    address aggregatorContract;
    address factoryContract;
    bool public isActive;
    uint lastUpdate;

    using SafeERC20 for IERC20;

    struct BorrowInfo {
        address borrowOrderAddress; // address of the borrow order
        bool[] oraclesPerPairActivated; // oracles activated for each pair
        uint[] LTVs; // LTVs for each pair
        uint maxApr; // max APR for the borrow order
        uint duration; // duration of the borrow order
        address owner; // owner of the borrow order
        address[] acceptedPrinciples; // accepted principles for the borrow order
        address collateral; // collateral for the borrow order
        address valuableAsset; // ERC721: underlying, ERC20: Same as collateral
        bool isNFT; // is the collateral an NFT
        uint receiptID; // receipt ID of the NFT (NFT ID, since we accept only receipt type NFTs)
        address[] oracles_Principles; // oracles for each principle
        uint[] ratio; // ratio for each principle
        address oracle_Collateral; // oracle for the collateral
        uint valuableAssetAmount; // only used for auction sold NFTs
        uint availableAmount; // amount of the collateral
        uint startAmount; // amount of the collateral at the start
    }

    BorrowInfo public borrowInformation;

    modifier onlyOwner() {
        require(msg.sender == borrowInformation.owner, "Only owner");
        _;
    }

    modifier onlyAggregator() {
        require(msg.sender == aggregatorContract, "Only aggregator");
        _;
    }

    // Prevent the offer from being updated before accepted
    modifier onlyAfterTimeOut() {
        require(
            lastUpdate == 0 || (block.timestamp - lastUpdate) > 1 minutes,
            "Offer has been updated in the last minute"
        );
        _;
    }

    function initialize(
        address _aggregatorContract,
        address _owner,
        address[] memory _acceptedPrinciples,
        address _collateral,
        bool[] memory _oraclesActivated,
        bool _isNFT,
        uint[] memory _LTVs,
        uint _maxApr,
        uint _duration,
        uint _receiptID,
        address[] memory _oracleIDS_Principles,
        uint[] memory _ratio,
        address _oracleID_Collateral,
        uint _startedBorrowAmount
    ) public initializer {
        aggregatorContract = _aggregatorContract;
        isActive = true;
        // seguir aca
        address _valuableAsset;
        // if the collateral is an NFT, get the underlying asset
        if (_isNFT) {
            NFR.receiptInstance memory nftData = NFR(_collateral)
                .getDataByReceipt(_receiptID);
            _startedBorrowAmount = nftData.lockedAmount;
            _valuableAsset = nftData.underlying;
        } else {
            _valuableAsset = _collateral;
        }

        borrowInformation = BorrowInfo({
            borrowOrderAddress: address(this),
            oraclesPerPairActivated: _oraclesActivated,
            LTVs: _LTVs,
            maxApr: _maxApr,
            duration: _duration,
            owner: _owner,
            acceptedPrinciples: _acceptedPrinciples,
            collateral: _collateral,
            valuableAsset: _valuableAsset,
            isNFT: _isNFT,
            receiptID: _receiptID,
            oracles_Principles: _oracleIDS_Principles,
            ratio: _ratio,
            oracle_Collateral: _oracleID_Collateral,
            valuableAssetAmount: 0,
            availableAmount: _isNFT ? 1 : _startedBorrowAmount,
            startAmount: _startedBorrowAmount
        });
        factoryContract = msg.sender;
    }
    /**
     * @dev Accepts the borrow offer -- only callable from Aggregator
     * @param amount Amount of the collateral to be accepted
     */
    function acceptBorrowOffer(
        uint amount
    ) public onlyAggregator nonReentrant onlyAfterTimeOut {
        BorrowInfo memory m_borrowInformation = getBorrowInfo();
        require(
            amount <= m_borrowInformation.availableAmount,
            "Amount exceeds available amount"
        );
        require(amount > 0, "Amount must be greater than 0");

        borrowInformation.availableAmount -= amount;

        // transfer collateral to aggregator
        if (m_borrowInformation.isNFT) {
            IERC721(m_borrowInformation.collateral).transferFrom(
                address(this),
                aggregatorContract,
                m_borrowInformation.receiptID
            );
        } else {
            SafeERC20.safeTransfer(
                IERC20(m_borrowInformation.collateral),
                aggregatorContract,
                amount
            );
        }
        uint percentageOfAvailableCollateral = (borrowInformation
            .availableAmount * 10000) / m_borrowInformation.startAmount;

        // if available amount is less than 0.1% of the start amount, the order is no longer active and will count as completed.
        if (percentageOfAvailableCollateral <= 10) {
            isActive = false;
            // transfer remaining collateral back to owner
            if (borrowInformation.availableAmount != 0) {
                SafeERC20.safeTransfer(
                    IERC20(m_borrowInformation.collateral),
                    m_borrowInformation.owner,
                    borrowInformation.availableAmount
                );
            }
            borrowInformation.availableAmount = 0;
            IDBOFactory(factoryContract).emitDelete(address(this));
            IDBOFactory(factoryContract).deleteBorrowOrder(address(this));
        } else {
            IDBOFactory(factoryContract).emitUpdate(address(this));
        }
    }

    /**
     * @dev Cancels the borrow offer -- only callable from owner
     */
    function cancelOffer() public onlyOwner nonReentrant {
        BorrowInfo memory m_borrowInformation = getBorrowInfo();
        uint availableAmount = m_borrowInformation.availableAmount;
        require(availableAmount > 0, "No available amount");
        // set available amount to 0
        // set isActive to false
        borrowInformation.availableAmount = 0;
        isActive = false;

        // transfer collateral back to owner
        if (m_borrowInformation.isNFT) {
            if (m_borrowInformation.availableAmount > 0) {
                IERC721(m_borrowInformation.collateral).transferFrom(
                    address(this),
                    msg.sender,
                    m_borrowInformation.receiptID
                );
            }
        } else {
            SafeERC20.safeTransfer(
                IERC20(m_borrowInformation.collateral),
                msg.sender,
                availableAmount
            );
        }

        // emit canceled event on factory

        IDBOFactory(factoryContract).deleteBorrowOrder(address(this));
        IDBOFactory(factoryContract).emitDelete(address(this));
    }

    function getBorrowInfo() public view returns (BorrowInfo memory) {
        BorrowInfo memory m_borrowInformation = borrowInformation;
        // get dynamic data for NFTs
        if (m_borrowInformation.isNFT) {
            NFR.receiptInstance memory nftData = NFR(
                m_borrowInformation.collateral
            ).getDataByReceipt(m_borrowInformation.receiptID);
            m_borrowInformation.valuableAssetAmount = nftData.lockedAmount;
        }
        return m_borrowInformation;
    }

    function updateBorrowOrder(
        uint newMaxApr,
        uint newDuration,
        uint[] memory newLTVs,
        uint[] memory newRatios
    ) public onlyOwner {
        require(
            newLTVs.length == borrowInformation.acceptedPrinciples.length &&
                newRatios.length == newLTVs.length,
            "Invalid LTVs"
        );
        lastUpdate = block.timestamp;
        BorrowInfo memory m_borrowInformation = getBorrowInfo();
        m_borrowInformation.maxApr = newMaxApr;
        m_borrowInformation.duration = newDuration;
        m_borrowInformation.LTVs = newLTVs;
        m_borrowInformation.ratio = newRatios;

        borrowInformation = m_borrowInformation;
        IDBOFactory(factoryContract).emitUpdate(address(this));
    }
}
