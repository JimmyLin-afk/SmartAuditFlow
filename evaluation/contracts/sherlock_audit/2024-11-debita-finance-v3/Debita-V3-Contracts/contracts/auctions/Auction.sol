pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC721/utils/ERC721Holder.sol";
import {console} from "forge-std/Test.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

interface debitaLoan {
    function handleAuctionSell(uint amount) external;
}

interface auctionFactory {
    function auctionFee() external view returns (uint);
    function publicAuctionFee() external view returns (uint);
    function feeAddress() external view returns (address);
    function getLiquidationFloorPrice(
        uint initAmount
    ) external view returns (uint);
    function emitAuctionEdited(
        address auctionAddress,
        address creator
    ) external;
    function emitAuctionDeleted(
        address auctionAddress,
        address creator
    ) external;
    function _deleteAuctionOrder(address auctionAddress) external;
}

contract DutchAuction_veNFT is ERC721Holder {
    struct dutchAuction_INFO {
        address auctionAddress; // address of the auction
        address nftAddress; // address of the NFT being sold
        uint nftCollateralID; // ID of the NFT being sold
        address sellingToken; // token that the user is selling their veNFT for
        address owner; // owner of the auction
        uint initAmount; // initial amount of the auction
        uint floorAmount; // floor amount of the auction
        uint duration; // initial duration of the auction
        uint endBlock; // end block of the auction
        uint tickPerBlock; // amount of tokens that decrease per second
        bool isActive; // is the auction active
        uint initialBlock; // initial block of the auction (is updated after editing)
        bool isLiquidation; // is the auction a liquidation
        uint differenceDecimals; // difference in decimals between the selling token and 18 decimals
    }

    dutchAuction_INFO private s_CurrentAuction;
    address public s_ownerOfAuction; // owner of the auction
    address public factory; // auction factory
    uint decimalsDifference; // difference between 18 and the decimals of the selling token

    modifier onlyActiveAuction() {
        require(s_CurrentAuction.isActive, "Auction is not active");
        _;
    }

    modifier onlyOwner() {
        require(msg.sender == s_ownerOfAuction, "Only the owner");
        _;
    }

    constructor(
        uint _veNFTID,
        address _veNFTAddress,
        address sellingToken,
        address owner,
        uint _initAmount,
        uint _floorAmount,
        uint _duration,
        bool _isLiquidation
    ) {
        // have tickPerBlock on 18 decimals
        // check decimals of sellingToken
        // if decimals are less than 18, cure the initAmount and floorAmount
        // save the difference in decimals for later use
        uint decimalsSellingToken = ERC20(sellingToken).decimals();
        uint difference = 18 - decimalsSellingToken;
        uint curedInitAmount = _initAmount * (10 ** difference);
        uint curedFloorAmount = _floorAmount * (10 ** difference);

        s_CurrentAuction = dutchAuction_INFO({
            auctionAddress: address(this),
            nftAddress: _veNFTAddress,
            nftCollateralID: _veNFTID,
            sellingToken: sellingToken,
            owner: owner,
            initAmount: curedInitAmount,
            floorAmount: curedFloorAmount,
            duration: _duration,
            endBlock: block.timestamp + _duration,
            tickPerBlock: (curedInitAmount - curedFloorAmount) / _duration,
            isActive: true,
            initialBlock: block.timestamp,
            isLiquidation: _isLiquidation,
            differenceDecimals: difference
        });

        s_ownerOfAuction = owner;
        factory = msg.sender;
    }
    /**
     * @dev User buys the NFT 
     - sends the tokens to the owner of the auction
     - receives the NFT
     
     */
    function buyNFT() public onlyActiveAuction {
        // get memory data
        dutchAuction_INFO memory m_currentAuction = s_CurrentAuction;
        // get current price of the auction
        uint currentPrice = getCurrentPrice();
        // desactivate auction from storage
        s_CurrentAuction.isActive = false;
        uint fee;
        if (m_currentAuction.isLiquidation) {
            fee = auctionFactory(factory).auctionFee();
        } else {
            fee = auctionFactory(factory).publicAuctionFee();
        }

        // calculate fee
        uint feeAmount = (currentPrice * fee) / 10000;
        // get fee address
        address feeAddress = auctionFactory(factory).feeAddress();
        // Transfer liquidation token from the buyer to the owner of the auction
        SafeERC20.safeTransferFrom(
            IERC20(m_currentAuction.sellingToken),
            msg.sender,
            s_ownerOfAuction,
            currentPrice - feeAmount
        );

        SafeERC20.safeTransferFrom(
            IERC20(m_currentAuction.sellingToken),
            msg.sender,
            feeAddress,
            feeAmount
        );

        // If it's a liquidation, handle it properly
        if (m_currentAuction.isLiquidation) {
            debitaLoan(s_ownerOfAuction).handleAuctionSell(
                currentPrice - feeAmount
            );
        }
        IERC721 Token = IERC721(s_CurrentAuction.nftAddress);
        Token.safeTransferFrom(
            address(this),
            msg.sender,
            s_CurrentAuction.nftCollateralID
        );

        auctionFactory(factory)._deleteAuctionOrder(address(this));
        auctionFactory(factory).emitAuctionDeleted(
            address(this),
            s_ownerOfAuction
        );
        // event offerBought
    }

    /**
     * @dev User cancels the auction
     - sends the NFT back to the owner
     - desactivates the auction
     */
    function cancelAuction() public onlyActiveAuction onlyOwner {
        s_CurrentAuction.isActive = false;
        // Send NFT back to owner
        IERC721 Token = IERC721(s_CurrentAuction.nftAddress);
        Token.safeTransferFrom(
            address(this),
            s_ownerOfAuction,
            s_CurrentAuction.nftCollateralID
        );

        auctionFactory(factory)._deleteAuctionOrder(address(this));
        auctionFactory(factory).emitAuctionDeleted(
            address(this),
            s_ownerOfAuction
        );
        // event offerCanceled
    }

    /**
     * @notice the same tickPerBlock is used for the whole auction, so the duration will be longer
     * @dev User edits the auction
     - changes the floor price
     - changes the duration
     */
    function editFloorPrice(
        uint newFloorAmount
    ) public onlyActiveAuction onlyOwner {
        uint curedNewFloorAmount = newFloorAmount *
            (10 ** s_CurrentAuction.differenceDecimals);
        require(
            s_CurrentAuction.floorAmount > curedNewFloorAmount,
            "New floor lower"
        );

        dutchAuction_INFO memory m_currentAuction = s_CurrentAuction;
        uint newDuration = (m_currentAuction.initAmount - curedNewFloorAmount) /
            m_currentAuction.tickPerBlock;

        uint discountedTime = (m_currentAuction.initAmount -
            m_currentAuction.floorAmount) / m_currentAuction.tickPerBlock;

        if (
            (m_currentAuction.initialBlock + discountedTime) < block.timestamp
        ) {
            // ticket = tokens por bloque   tokens / tokens por bloque = bloques
            m_currentAuction.initialBlock = block.timestamp - (discountedTime);
        }

        m_currentAuction.duration = newDuration;
        m_currentAuction.endBlock = m_currentAuction.initialBlock + newDuration;
        m_currentAuction.floorAmount = curedNewFloorAmount;
        s_CurrentAuction = m_currentAuction;

        auctionFactory(factory).emitAuctionEdited(
            address(this),
            s_ownerOfAuction
        );
        // emit offer edited
    }

    function getCurrentPrice() public view returns (uint) {
        dutchAuction_INFO memory m_currentAuction = s_CurrentAuction;
        uint floorPrice = m_currentAuction.floorAmount;
        // Calculate the time passed since the auction started/ initial second
        uint timePassed = block.timestamp - m_currentAuction.initialBlock;

        // Calculate the amount decreased with the time passed and the tickPerBlock
        uint decreasedAmount = m_currentAuction.tickPerBlock * timePassed;
        uint currentPrice = (decreasedAmount >
            (m_currentAuction.initAmount - floorPrice))
            ? floorPrice
            : m_currentAuction.initAmount - decreasedAmount;
        // Calculate the current price in case timePassed is false
        // Check if time has passed
        currentPrice =
            currentPrice /
            (10 ** m_currentAuction.differenceDecimals);
        return currentPrice;
    }

    function getAuctionData() public view returns (dutchAuction_INFO memory) {
        return s_CurrentAuction;
    }
}
