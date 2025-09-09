pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@contracts/Non-Fungible-Receipts/veNFTS/Equalizer/veNFTEqualizer.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract veNFTEqualizer is ReentrancyGuard, ERC721Enumerable {
    event createdVault(
        uint receiptID,
        address newVault,
        address underlying,
        uint amount
    );
    event withdrawn(
        uint receiptID,
        address newVault,
        address underlying,
        uint amount
    );

    event interactedWith(address vault, uint nftID);

    address nftAddress;

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

    mapping(uint => address) public s_ReceiptID_to_Vault;
    mapping(address => bool) internal isVaultValid;

    mapping(address => uint) public balanceOfManagement;
    mapping(address => mapping(uint => uint)) internal ownedTokens;
    mapping(address => mapping(uint => uint)) internal indexPosition; // address  => nft = length position

    uint private s_ReceiptID;
    address public _underlying;

    constructor(
        address _nftAddress,
        address underlying
    ) ERC721("NFR veEQUAL", "ReceiptveNFT") {
        nftAddress = _nftAddress;
        _underlying = underlying;
    }

    modifier onlyVault() {
        require(isVaultValid[msg.sender], "not vault");
        _;
    }

    function deposit(uint[] memory nftsID) external nonReentrant {
        // Add the receipt count & create a memory variable to save gas
        uint m_Receipt = s_ReceiptID;
        s_ReceiptID += nftsID.length;
        // For loop minting receipt tokens
        for (uint i; i < nftsID.length; i++) {
            m_Receipt++;

            // Create Vault for each deposit
            veNFTVault vault = new veNFTVault(
                nftAddress,
                address(this),
                m_Receipt,
                nftsID[i],
                msg.sender
            );
            // Transfer NFT to created Vault
            ERC721(nftAddress).transferFrom(
                msg.sender,
                address(vault),
                nftsID[i]
            );

            uint lastIndex = balanceOfManagement[msg.sender];
            // Add amount
            balanceOfManagement[msg.sender]++;
            // Update Indexs
            indexPosition[msg.sender][nftsID[i]] = lastIndex;
            ownedTokens[msg.sender][lastIndex] = nftsID[i];
            // Update vault data
            s_ReceiptID_to_Vault[m_Receipt] = address(vault);
            isVaultValid[address(vault)] = true;

            // Mint receipt to the user
            veNFT veContract = veNFT(nftAddress);
            IVotingEscrow.LockedBalance memory _locked = veContract.locked(
                nftsID[i]
            );

            uint amountOfNFT = uint(int(_locked.amount));
            _mint(msg.sender, m_Receipt);
            emit createdVault(
                m_Receipt,
                address(vault),
                _underlying,
                amountOfNFT
            );
        }
    }

    function voteMultiple(
        address[] calldata vaults,
        address[] calldata _poolVote,
        uint256[] calldata _weights
    ) external {
        for (uint i; i < vaults.length; i++) {
            require(
                msg.sender == veNFTVault(vaults[i]).managerAddress(),
                "not manager"
            );
            require(isVaultValid[vaults[i]], "not vault");
            veNFTVault(vaults[i]).vote(_poolVote, _weights);
        }
    }

    function claimBribesMultiple(
        address[] calldata vaults,
        address[] calldata _bribes,
        address[][] calldata _tokens
    ) external {
        for (uint i; i < vaults.length; i++) {
            require(
                msg.sender == veNFTVault(vaults[i]).managerAddress(),
                "not manager"
            );
            require(isVaultValid[vaults[i]], "not vault");
            veNFTVault(vaults[i]).claimBribes(msg.sender, _bribes, _tokens);
            emitInteracted(vaults[i]);
        }
    }

    function resetMultiple(address[] calldata vaults) external {
        for (uint i; i < vaults.length; i++) {
            require(
                msg.sender == veNFTVault(vaults[i]).managerAddress(),
                "not manager"
            );
            require(isVaultValid[vaults[i]], "not vault");
            veNFTVault(vaults[i]).reset();
        }
    }

    function extendMultiple(
        address[] calldata vaults,
        uint[] calldata newEnds
    ) external {
        for (uint i; i < vaults.length; i++) {
            require(
                msg.sender == veNFTVault(vaults[i]).managerAddress(),
                "not manager"
            );
            require(isVaultValid[vaults[i]], "not vault");
            veNFTVault(vaults[i]).extendLock(newEnds[i]);
        }
    }

    function pokeMultiple(address[] calldata vaults) external {
        for (uint i; i < vaults.length; i++) {
            require(
                msg.sender == veNFTVault(vaults[i]).managerAddress(),
                "not manager"
            );
            require(isVaultValid[vaults[i]], "not vault");
            veNFTVault(vaults[i]).poke();
            emitInteracted(vaults[i]);
        }
    }

    function decrease(address voter, uint nftID) external onlyVault {
        uint index_DeletingNFT = indexPosition[voter][nftID];
        uint index_LastNFT = balanceOfManagement[voter] - 1;

        if (index_DeletingNFT != index_LastNFT) {
            uint lastNFT = ownedTokens[voter][index_LastNFT];
            ownedTokens[voter][index_DeletingNFT] = lastNFT;
            indexPosition[voter][lastNFT] = index_DeletingNFT;
        }
        balanceOfManagement[voter]--;

        // Delete last NFT & put it on deleting index
        // Delete it last in case you are updating the last one
        delete ownedTokens[voter][index_LastNFT];
    }

    function increase(address voter, uint nftID) external onlyVault {
        uint nextIndex = balanceOfManagement[voter];
        balanceOfManagement[voter]++;
        indexPosition[voter][nftID] = nextIndex;
        ownedTokens[voter][nextIndex] = nftID;
    }

    function getDataFromUser(
        address manager,
        uint fromIndex,
        uint stopIndex
    ) external view returns (receiptInstance[] memory) {
        uint amount = balanceOfManagement[manager] > stopIndex
            ? stopIndex
            : balanceOfManagement[manager];

        veNFT veContract = veNFT(nftAddress);
        receiptInstance[] memory nftsDATA = new receiptInstance[](
            amount - fromIndex
        );
        for (uint i; i + fromIndex < amount; i++) {
            uint nftID = ownedTokens[manager][i + fromIndex];
            IVotingEscrow.LockedBalance memory _locked = veContract.locked(
                nftID
            );
            address vault = veContract.ownerOf(nftID);
            veNFTVault _vaultContract = veNFTVault(vault);
            uint receipt = _vaultContract.receiptID();
            uint _decimals = ERC20(_underlying).decimals();
            address manager = _vaultContract.managerAddress();
            address currentOwnerOfReceipt = ownerOf(receipt);
            nftsDATA[i] = receiptInstance({
                receiptID: receipt,
                attachedNFT: nftID,
                lockedAmount: uint(int(_locked.amount)),
                lockedDate: _locked.end,
                decimals: _decimals,
                vault: vault,
                underlying: _underlying,
                OwnerIsManager: currentOwnerOfReceipt == manager
            });
        }
        return nftsDATA;
    }

    function getDataByReceipt(
        uint receiptID
    ) public view returns (receiptInstance memory) {
        veNFT veContract = veNFT(nftAddress);
        veNFTVault vaultContract = veNFTVault(s_ReceiptID_to_Vault[receiptID]);
        uint nftID = vaultContract.attached_NFTID();
        IVotingEscrow.LockedBalance memory _locked = veContract.locked(nftID);
        uint _decimals = ERC20(_underlying).decimals();
        address manager = vaultContract.managerAddress();
        address currentOwnerOfReceipt = ownerOf(receiptID);
        receiptInstance memory receiptData = receiptInstance({
            receiptID: receiptID,
            attachedNFT: nftID,
            lockedAmount: uint(int(_locked.amount)),
            lockedDate: _locked.end,
            decimals: _decimals,
            vault: address(vaultContract),
            underlying: _underlying,
            OwnerIsManager: currentOwnerOfReceipt == manager
        });
        return receiptData;
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

    function emitInteracted(address vault) internal {
        veNFTVault vaultContract = veNFTVault(vault);
        uint nftID = vaultContract.attached_NFTID();
        emit interactedWith(vault, nftID);
    }

    function emitWithdrawn(address vault, uint nftID) external {
        require(isVaultValid[msg.sender], "not vault");
        veNFT veContract = veNFT(nftAddress);
        veNFTVault vaultContract = veNFTVault(vault);
        IVotingEscrow.LockedBalance memory _locked = veContract.locked(nftID);

        emit withdrawn(
            vaultContract.receiptID(),
            vault,
            _underlying,
            uint(int(_locked.amount))
        );
    }

    function lastReceiptID() external view returns (uint) {
        return s_ReceiptID;
    }

    function burnReceipt(uint id) external onlyVault {
        _burn(id);
    }
}
