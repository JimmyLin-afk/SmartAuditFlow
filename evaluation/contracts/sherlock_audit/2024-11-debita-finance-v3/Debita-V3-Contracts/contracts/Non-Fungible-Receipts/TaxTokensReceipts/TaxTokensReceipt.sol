pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";

interface IBorrowOrderFactory {
    function isBorrowOrderLegit(
        address _borrowOrder
    ) external view returns (bool);
}

interface ILendOrderFactory {
    function isLendOrderLegit(address _lendOrder) external view returns (bool);
}

interface IAggregator {
    function isSenderALoan(address _aggregator) external view returns (bool);
}

contract TaxTokensReceipts is ERC721Enumerable, ReentrancyGuard {
    event Deposited(address indexed user, uint amount);
    event Withdrawn(address indexed user, uint amount);
    // token ID ==> token amount
    mapping(uint => uint) public tokenAmountPerID;

    address public tokenAddress;
    address public borrowOrderFactory;
    address public lendOrderFactory;
    address public Aggregator;
    uint tokenID;

    struct receiptInstance {
        uint receiptID;
        uint attachedNFT;
        uint lockedAmount;
        uint lockedDate;
        uint decimals;
        address vault;
        address underlying;
        bool OwnerIsManager;
    }
    // change symbol for each different token
    constructor(
        address _token,
        address _borrowOrderFactory,
        address _lendOrderFactory,
        address _aggregator
    ) ERC721("TaxTokensReceipts", "TTR") {
        tokenAddress = _token;
        borrowOrderFactory = _borrowOrderFactory;
        lendOrderFactory = _lendOrderFactory;
        Aggregator = _aggregator;
    }

    // expect that owners of the token will excempt from tax this contract
    function deposit(uint amount) public nonReentrant returns (uint) {
        uint balanceBefore = ERC20(tokenAddress).balanceOf(address(this));
        SafeERC20.safeTransferFrom(
            ERC20(tokenAddress),
            msg.sender,
            address(this),
            amount
        );
        uint balanceAfter = ERC20(tokenAddress).balanceOf(address(this));
        uint difference = balanceAfter - balanceBefore;
        require(difference >= amount, "TaxTokensReceipts: deposit failed");
        tokenID++;
        tokenAmountPerID[tokenID] = amount;
        _mint(msg.sender, tokenID);
        emit Deposited(msg.sender, amount);
        return tokenID;
    }

    // withdraw the token
    function withdraw(uint _tokenID) public nonReentrant {
        require(
            ownerOf(_tokenID) == msg.sender,
            "TaxTokensReceipts: not owner"
        );
        uint amount = tokenAmountPerID[_tokenID];
        tokenAmountPerID[_tokenID] = 0;
        _burn(_tokenID);

        SafeERC20.safeTransfer(ERC20(tokenAddress), msg.sender, amount);
        emit Withdrawn(msg.sender, amount);
    }

    // Override to only interact with Debita

    function transferFrom(
        address from,
        address to,
        uint256 tokenId
    ) public virtual override(ERC721, IERC721) {
        bool isReceiverAddressDebita = IBorrowOrderFactory(borrowOrderFactory)
            .isBorrowOrderLegit(to) ||
            ILendOrderFactory(lendOrderFactory).isLendOrderLegit(to) ||
            IAggregator(Aggregator).isSenderALoan(to);
        bool isSenderAddressDebita = IBorrowOrderFactory(borrowOrderFactory)
            .isBorrowOrderLegit(from) ||
            ILendOrderFactory(lendOrderFactory).isLendOrderLegit(from) ||
            IAggregator(Aggregator).isSenderALoan(from);
        // Debita not involved --> revert
        require(
            isReceiverAddressDebita || isSenderAddressDebita,
            "TaxTokensReceipts: Debita not involved"
        );
        if (to == address(0)) {
            revert ERC721InvalidReceiver(address(0));
        }
        // Setting an "auth" arguments enables the `_isAuthorized` check which verifies that the token exists
        // (from != 0). Therefore, it is not needed to verify that the return value is not 0 here.
        address previousOwner = _update(to, tokenId, _msgSender());
        if (previousOwner != from) {
            revert ERC721IncorrectOwner(from, tokenId, previousOwner);
        }
    }

    function getDataByReceipt(
        uint receiptID
    ) public view returns (receiptInstance memory) {
        uint lockedAmount = tokenAmountPerID[receiptID];
        uint lockedDate = 0; // no locked date
        uint decimals = ERC20(tokenAddress).decimals();
        address vault = address(this);
        address underlying = tokenAddress;
        bool OwnerIsManager = true; // owner is always the manager
        return
            receiptInstance(
                receiptID,
                0, // no attached NFT
                lockedAmount,
                lockedDate,
                decimals,
                vault,
                underlying,
                OwnerIsManager
            );
    }

    function getHoldingReceiptsByAddress(
        address holder,
        uint fromIndex,
        uint stopIndex
    ) public view returns (receiptInstance[] memory) {
        uint amount = balanceOf(holder) > stopIndex
            ? stopIndex
            : balanceOf(holder);
        receiptInstance[] memory nftsDATA = new receiptInstance[](
            amount - fromIndex
        );
        for (uint i; i + fromIndex < amount; i++) {
            uint receiptID = tokenOfOwnerByIndex(holder, i + fromIndex);
            nftsDATA[i] = getDataByReceipt(receiptID);
        }
        return nftsDATA;
    }
}
