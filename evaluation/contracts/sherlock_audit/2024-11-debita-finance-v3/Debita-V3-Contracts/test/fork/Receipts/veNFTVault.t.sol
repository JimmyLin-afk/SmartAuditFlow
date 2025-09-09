pragma solidity ^0.8.13;

import {Test, console} from "forge-std/Test.sol";
import {veNFTAerodrome} from "@contracts/Non-Fungible-Receipts/veNFTS/Aerodrome/Receipt-veNFT.sol";

import {veNFTVault} from "@contracts/Non-Fungible-Receipts/veNFTS/Equalizer/veNFTEqualizer.sol";
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";
import {VotingEscrow} from "@aerodrome/VotingEscrow.sol";

contract CounterTest is Test {
    VotingEscrow public ABIERC721Contract;
    veNFTAerodrome public receiptContract;
    veNFTVault[] public veNFTVaultContract = new veNFTVault[](3);
    ERC20Mock public TOKEN;
    address[] vaultAddress = new address[](3);
    uint[] nftID = new uint[](3);
    uint[] receiptID = new uint[](3);

    address veAERO = 0xeBf418Fe2512e7E6bd9b87a8F0f294aCDC67e6B4;
    address AERO = 0x940181a94A35A4569E4529A3CDfB74e38FD98631;

    function setUp() public {
        ABIERC721Contract = VotingEscrow(veAERO);
        TOKEN = ERC20Mock(AERO);
        receiptContract = new veNFTAerodrome(address(ABIERC721Contract), AERO);

        deal(AERO, address(this), 1000e18, true);
        TOKEN.approve(address(ABIERC721Contract), 1000e18);

        for (uint i = 0; i < 3; i++) {
            uint id = ABIERC721Contract.createLock(100e18, 365 * 4 * 86400);
            ABIERC721Contract.approve(address(receiptContract), id);
            nftID[i] = id;
        }

        receiptContract.deposit(nftID);

        for (uint i = 0; i < 3; i++) {
            receiptID[i] = receiptContract.lastReceiptID() - 2 + i;
            address vault = receiptContract.s_ReceiptID_to_Vault(receiptID[i]);
            vaultAddress[i] = vault;
            veNFTVaultContract[i] = (veNFTVault(vault));
        }
    }

    function testWithdraw() public {
        receiptContract.approve(address(veNFTVaultContract[0]), receiptID[0]);
        veNFTVaultContract[0].withdraw();
        address owner = ABIERC721Contract.ownerOf(nftID[0]);
        assertEq(owner, address(this));
    }

    function testFailWithdrawSecondTry() public {
        vm.expectRevert("No attached nft");

        receiptContract.approve(address(veNFTVaultContract[0]), receiptID[0]);
        veNFTVaultContract[0].withdraw();
        veNFTVaultContract[0].withdraw();
    }

    function testFuzz_ChangeManager(address l) public {
        vm.assume(l != address(this));

        vm.expectRevert(bytes("not Allowed"));
        vm.prank(l);
        veNFTVaultContract[0].changeManager(l);
    }

    function testChangeManagerAndInteract() public {
        address newManager = address(1);
        veNFTVaultContract[0].changeManager(newManager);

        assertEq(veNFTVaultContract[0].managerAddress(), newManager);
        receiptContract.approve(address(veNFTVaultContract[0]), receiptID[0]);
        veNFTVaultContract[0].withdraw();
        assertEq(veNFTVaultContract[0].attached_NFTID(), 0);
    }

    function testFailChangeManagerAndInteract() public {
        address newManager = address(1);
        veNFTVaultContract[0].changeManager(newManager);
        assertEq(veNFTVaultContract[0].managerAddress(), newManager);

        vm.startPrank(newManager);
        receiptContract.approve(address(veNFTVaultContract[0]), receiptID[0]);
        veNFTVaultContract[0].withdraw();
        vm.stopPrank();
    }

    function testChangeManagerReturn() public {
        address newManager = address(1);
        veNFTVaultContract[0].changeManager(newManager);
        assertEq(veNFTVaultContract[0].managerAddress(), newManager);
        vm.startPrank(newManager);
        veNFTVaultContract[0].changeManager(address(this));
        vm.stopPrank();
        assertEq(veNFTVaultContract[0].managerAddress(), address(this));
    }

    function testWithdrawAfterExpiring() public {
        receiptContract.approve(address(veNFTVaultContract[0]), receiptID[0]);
        vm.warp(block.timestamp + (365 * 4 * 86400 * 2));
        veNFTVaultContract[0].withdraw();
        address owner = ABIERC721Contract.ownerOf(nftID[0]);
        assertEq(owner, address(this));
    }

    function testNotHolding() public {
        vm.warp(block.timestamp + (365 * 4 * 86400 * 2));
        vm.startPrank(address(0x04));
        vm.expectRevert("Not Holding");
        veNFTVaultContract[0].withdraw();
    }

    function testWithdrawAndDepositAgain() public {
        uint[] memory newNFTID = getDynamicUintArray(20);
        address[] memory _vaultAddress = getDynamicAddressArray(20);
        veNFTVault[] memory _veNFTVaultContract = new veNFTVault[](20);
        uint[] memory _receiptID = new uint[](20);
        for (uint i = 0; i < 20; i++) {
            uint id = ABIERC721Contract.createLock(1e18, 365 * 4 * 86400);
            ABIERC721Contract.approve(address(receiptContract), id);

            newNFTID[i] = id;
        }

        receiptContract.deposit(newNFTID);

        for (uint i = 0; i < 20; i++) {
            _receiptID[i] = receiptContract.lastReceiptID() - 19 + i;
            address vault = receiptContract.s_ReceiptID_to_Vault(_receiptID[i]);
            _vaultAddress[i] = vault;
            _veNFTVaultContract[i] = (veNFTVault(vault));
        }
        // withdraw
        receiptContract.approve(address(_veNFTVaultContract[1]), _receiptID[1]);
        _veNFTVaultContract[1].withdraw();

        receiptContract.approve(address(_veNFTVaultContract[6]), _receiptID[6]);
        _veNFTVaultContract[6].withdraw();

        receiptContract.approve(
            address(_veNFTVaultContract[19]),
            _receiptID[19]
        );
        _veNFTVaultContract[19].withdraw();

        veNFTAerodrome.receiptInstance[]
            memory receiptCalculated = receiptContract.getDataFromUser(
                address(this),
                0,
                1000
            );

        assertEq(receiptCalculated[0].decimals, 18);
    }

    function getDynamicUintArray(
        uint256 x
    ) public pure returns (uint[] memory) {
        uint[] memory nftsID = new uint[](x);
        return nftsID;
    }

    function getDynamicAddressArray(
        uint256 x
    ) public pure returns (address[] memory) {
        address[] memory nftsID = new address[](x);
        return nftsID;
    }
}
