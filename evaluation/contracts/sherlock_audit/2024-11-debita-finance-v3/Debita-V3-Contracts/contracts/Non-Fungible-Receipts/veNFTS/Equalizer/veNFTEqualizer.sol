pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

interface IVotingEscrow {
    struct LockedBalance {
        int128 amount;
        uint256 end;
    }
}

interface veNFT is IVotingEscrow {
    function voter() external returns (address);
    function increaseUnlockTime(
        uint256 tokenId,
        uint256 _lock_duration
    ) external;
    function increase_unlock_time(
        uint256 tokenId,
        uint256 _lock_duration
    ) external;
    function distributor() external returns (address);
    function locked(uint id) external view returns (LockedBalance memory);
    function ownerOf(uint id) external view returns (address);
}

interface voterContract {
    function vote(
        uint256 tokenId,
        address[] memory _poolVote,
        uint256[] memory _weights
    ) external;

    function claimBribes(
        address[] memory _bribes,
        address[][] memory _tokens,
        uint256 tokenId
    ) external;

    function reset(uint256 _tokenId) external;
    function poke(uint _tokenId) external;
}

interface IReceipt {
    function burnReceipt(uint id) external;
    function transferFrom(address from, address to, uint256 tokenId) external;
    function ownerOf(uint tokenId) external returns (address);
    function decrease(address voter, uint nftID) external;
    function increase(address voter, uint nftID) external;
    function emitWithdrawn(address vault, uint amount) external;
}

contract veNFTVault is ReentrancyGuard {
    address veNFTAddress;
    address factoryAddress;

    // manager address is the wallet in charge of managing the veNFT & collecting the rewards of it
    address public managerAddress;
    uint public receiptID;
    uint public attached_NFTID;

    constructor(
        address _veAddress,
        address _factoryAddress,
        uint _receiptID,
        uint _nftID,
        address _managerAddress
    ) {
        veNFTAddress = _veAddress;
        factoryAddress = _factoryAddress;
        receiptID = _receiptID;
        attached_NFTID = _nftID;
        managerAddress = _managerAddress;
    }

    modifier onlyFactory() {
        require(msg.sender == factoryAddress, "not Factory");
        _;
    }

    function withdraw() external nonReentrant {
        IERC721 veNFTContract = IERC721(veNFTAddress);
        IReceipt receiptContract = IReceipt(factoryAddress);
        uint m_idFromNFT = attached_NFTID;
        address holder = receiptContract.ownerOf(receiptID);

        // RECEIPT HAS TO BE ON OWNER WALLET
        require(attached_NFTID != 0, "No attached nft");
        require(holder == msg.sender, "Not Holding");
        receiptContract.decrease(managerAddress, m_idFromNFT);

        delete attached_NFTID;

        // First: burn receipt
        IReceipt(factoryAddress).burnReceipt(receiptID);
        IReceipt(factoryAddress).emitWithdrawn(address(this), m_idFromNFT);
        // Second: send them their NFT
        veNFTContract.transferFrom(address(this), msg.sender, m_idFromNFT);
    }

    // Change the manager of the veNFT
    // The manager is the wallet in charge of managing the veNFT & collecting the rewards of it
    // In order to change it the caller has to be the owner of the receipt or the current manager
    function changeManager(address newManager) external {
        IReceipt receiptContract = IReceipt(factoryAddress);
        address holder = receiptContract.ownerOf(receiptID);

        require(attached_NFTID != 0, "NFT not attached");
        require(newManager != managerAddress, "same manager");
        require(
            msg.sender == holder || msg.sender == managerAddress,
            "not Allowed"
        );
        receiptContract.decrease(managerAddress, attached_NFTID);
        receiptContract.increase(newManager, attached_NFTID);
        managerAddress = newManager;
    }

    function getVoterContract_veNFT() internal returns (address) {
        return veNFT(veNFTAddress).voter();
    }

    /*
    ------------------------------------------------------------------------------------------
        CUSTOM veAERO LOGIC
    ------------------------------------------------------------------------------------------
     */
    function reset() external onlyFactory {
        voterContract _voterContract = voterContract(getVoterContract_veNFT());
        _voterContract.reset(attached_NFTID);
    }

    function vote(
        address[] calldata _poolVote,
        uint256[] calldata _weights
    ) external onlyFactory {
        require(
            _weights.length == _poolVote.length,
            "Arrays must be the same length"
        );
        voterContract voter = voterContract(getVoterContract_veNFT());
        voter.vote(attached_NFTID, _poolVote, _weights);
    }

    function claimBribes(
        address sender,
        address[] calldata _bribes,
        address[][] calldata _tokens
    ) external onlyFactory {
        voterContract voter = voterContract(getVoterContract_veNFT());
        voter.claimBribes(_bribes, _tokens, attached_NFTID);

        // Claim bribes and send it to the borrower
        for (uint256 i = 0; i < _tokens.length; i++) {
            for (uint256 j = 0; j < _tokens[i].length; j++) {
                uint256 amountToSend = ERC20(_tokens[i][j]).balanceOf(
                    address(this)
                );
                SafeERC20.safeTransfer(
                    ERC20(_tokens[i][j]),
                    sender,
                    amountToSend
                );
            }
        }
    }

    function extendLock(uint256 duration) external onlyFactory {
        veNFT(veNFTAddress).increase_unlock_time(attached_NFTID, duration);
        //        veNFT(veNFTAddress).increaseUnlockTime(attached_NFTID, duration);
    }

    function poke() external onlyFactory {
        voterContract _voterContract = voterContract(getVoterContract_veNFT());
        _voterContract.poke(attached_NFTID);
    }
}
