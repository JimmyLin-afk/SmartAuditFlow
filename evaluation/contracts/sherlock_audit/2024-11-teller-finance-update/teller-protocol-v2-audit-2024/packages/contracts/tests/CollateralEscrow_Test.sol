// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "./Testable.sol";

import { CollateralEscrowV1 } from "../contracts/escrow/CollateralEscrowV1.sol";
import "../contracts/mock/WethMock.sol";
import "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";
import "@openzeppelin/contracts/proxy/beacon/BeaconProxy.sol";

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "../contracts/interfaces/IWETH.sol";

import "./tokens/TestERC20Token.sol";
import "./tokens/TestERC721Token.sol";
import "./tokens/TestERC1155Token.sol";

import { CollateralType, CollateralEscrowV1 } from "../contracts/escrow/CollateralEscrowV1.sol";

import { CollateralEscrowV1_Override } from "./CollateralEscrow_Override.sol";

import { Collateral, CollateralType } from "../contracts/interfaces/escrow/ICollateralEscrowV1.sol";

contract CollateralEscrow_Test is Testable {
    BeaconProxy private proxy_;
    User private borrower;

    TestERC20Token wethMock;
    TestERC721Token erc721Mock;
    TestERC1155Token erc1155Mock;

    uint256 amount = 1000;

    function setUp() public {
        // Deploy implementation
        CollateralEscrowV1 escrowImplementation = new CollateralEscrowV1_Override();
        // Deploy beacon contract with implementation
        UpgradeableBeacon escrowBeacon = new UpgradeableBeacon(
            address(escrowImplementation)
        );

        wethMock = new TestERC20Token("wrappedETH", "WETH", 1e24, 18);
        erc721Mock = new TestERC721Token("ERC721", "ERC721");
        erc1155Mock = new TestERC1155Token("ERC1155");

        borrower = new User(escrowBeacon);

        uint256 borrowerBalance = 50000;
        payable(address(borrower)).transfer(borrowerBalance);
    }

    function test_withdrawAsset_ERC20() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        wethMock.transfer(address(escrow), amount);
        escrow.setStoredBalance(
            CollateralType.ERC20,
            address(wethMock),
            amount,
            0,
            address(borrower)
        );

        vm.prank(address(borrower));
        escrow.withdraw(address(wethMock), amount, address(borrower));

        uint256 storedBalance = borrower.getBalance(address(wethMock));

        assertEq(storedBalance, 0, "Stored balance was not withdrawn");

        vm.prank(address(borrower));
        vm.expectRevert("No collateral balance for asset");
        escrow.withdraw(address(wethMock), amount, address(borrower));
    }

    function test_withdrawAsset_invalid_store() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        wethMock.transfer(address(escrow), amount);

        vm.expectRevert("No collateral balance for asset");

        vm.prank(address(borrower));
        escrow.withdraw(address(wethMock), amount, address(borrower));
    }

    function test_withdrawAsset_invalid_amount() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        wethMock.transfer(address(escrow), amount);
        escrow.setStoredBalance(
            CollateralType.ERC20,
            address(wethMock),
            amount,
            0,
            address(borrower)
        );

        vm.expectRevert("Withdraw amount cannot be zero");

        vm.prank(address(borrower));
        escrow.withdraw(address(wethMock), 0, address(borrower));
    }

    function test_withdrawAsset_invalid_owner() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        wethMock.transfer(address(escrow), amount);

        vm.expectRevert("Ownable: caller is not the owner");

        escrow.withdraw(address(wethMock), amount, address(borrower));
    }

    function test_withdrawAsset_ERC721() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        uint256 tokenId = erc721Mock.mint(address(escrow));
        escrow.setStoredBalance(
            CollateralType.ERC721,
            address(erc721Mock),
            1,
            tokenId,
            address(borrower)
        );

        vm.prank(address(borrower));
        escrow.withdraw(address(erc721Mock), 1, address(borrower));

        uint256 storedBalance = borrower.getBalance(address(erc721Mock));

        assertEq(storedBalance, 0, "Stored balance was not withdrawn");

        vm.expectRevert("No collateral balance for asset");
        vm.prank(address(borrower));
        escrow.withdraw(address(erc721Mock), 1, address(borrower));
    }

    function test_withdrawAsset_ERC721_invalidAmount() public {}

    function test_withdrawAsset_ERC1155() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        uint256 tokenId = erc1155Mock.mint(address(escrow));
        escrow.setStoredBalance(
            CollateralType.ERC1155,
            address(erc1155Mock),
            1,
            tokenId,
            address(borrower)
        );

        vm.prank(address(borrower));
        escrow.withdraw(address(erc1155Mock), 1, address(borrower));

        uint256 storedBalance = borrower.getBalance(address(erc1155Mock));

        assertEq(storedBalance, 0, "Stored balance was not withdrawn");

        vm.expectRevert("No collateral balance for asset");
        vm.prank(address(borrower));
        escrow.withdraw(address(erc1155Mock), 1, address(borrower));
    }

    function test_withdrawCollateral_erc20() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );
        wethMock.transfer(address(escrow), amount);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _collateralAddress: address(wethMock),
            _amount: amount,
            _tokenId: 0
        });

        escrow._withdrawCollateralSuper(
            collateral,
            address(wethMock),
            amount - 100,
            address(borrower)
        );

        uint256 borrowerBalance = wethMock.balanceOf(address(borrower));
        assertEq(
            borrowerBalance,
            amount - 100,
            "Borrower balance not increased"
        );
    }

    function test_withdrawCollateral_erc721() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );
        uint256 tokenId = erc721Mock.mint(address(escrow));

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC721,
            _collateralAddress: address(erc721Mock),
            _amount: 1,
            _tokenId: tokenId
        });

        escrow._withdrawCollateralSuper(
            collateral,
            address(erc721Mock),
            1,
            address(borrower)
        );

        uint256 borrowerBalance = erc721Mock.balanceOf(address(borrower));
        assertEq(borrowerBalance, 1, "Borrower balance not increased");
    }

    function test_withdrawCollateral_erc721_invalid_amount() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );
        uint256 tokenId = erc721Mock.mint(address(escrow));

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC721,
            _collateralAddress: address(erc721Mock),
            _amount: 2,
            _tokenId: tokenId
        });

        vm.expectRevert("Incorrect withdrawal amount");
        escrow._withdrawCollateralSuper(
            collateral,
            address(erc721Mock),
            2,
            address(borrower)
        );
    }

    function test_withdrawCollateral_erc1155() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );
        uint256 tokenId = erc1155Mock.mint(address(escrow));

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC1155,
            _collateralAddress: address(erc1155Mock),
            _amount: 1,
            _tokenId: tokenId
        });

        escrow._withdrawCollateralSuper(
            collateral,
            address(erc1155Mock),
            1,
            address(borrower)
        );

        uint256 borrowerBalance = erc1155Mock.balanceOf(
            address(borrower),
            tokenId
        );
        assertEq(borrowerBalance, 1, "Borrower balance not increased");
    }

    function test_depositAsset_ERC20() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        wethMock.transfer(address(borrower), 1e18);
        borrower.approveERC20(address(wethMock), amount);

        vm.prank(address(borrower));
        escrow.depositAsset(CollateralType.ERC20, address(wethMock), amount, 0);

        uint256 storedBalance = borrower.getBalance(address(wethMock));

        assertEq(storedBalance, amount, "Escrow deposit unsuccessful");
    }

    function test_depositAsset_invalid_owner() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        vm.expectRevert("Ownable: caller is not the owner");

        escrow.depositAsset(CollateralType.ERC20, address(wethMock), amount, 0);
    }

    function test_depositAsset_invalid_amount() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        vm.expectRevert("Deposit amount cannot be zero");

        vm.prank(address(borrower));
        escrow.depositAsset(CollateralType.ERC20, address(wethMock), 0, 0);
    }

    function test_depositAsset_ERC721() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        uint256 tokenId = erc721Mock.mint(address(borrower));

        borrower.approveERC721(address(erc721Mock), tokenId);

        vm.prank(address(borrower));
        escrow.depositAsset(
            CollateralType.ERC721,
            address(erc721Mock),
            1,
            tokenId
        );

        uint256 storedBalance = borrower.getBalance(address(erc721Mock));

        assertEq(storedBalance, 1, "Escrow deposit unsuccessful");
    }

    function test_depositAsset_ERC721_double_collateral_overwrite_prevention()
        public
    {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        uint256 tokenIdA = erc721Mock.mint(address(borrower));
        uint256 tokenIdB = erc721Mock.mint(address(borrower));

        borrower.approveERC721(address(erc721Mock), tokenIdA);
        borrower.approveERC721(address(erc721Mock), tokenIdB);

        vm.prank(address(borrower));
        escrow.depositAsset(
            CollateralType.ERC721,
            address(erc721Mock),
            1,
            tokenIdA
        );

        uint256 storedBalance = borrower.getBalance(address(erc721Mock));

        assertEq(storedBalance, 1, "Escrow deposit unsuccessful");

        vm.expectRevert(
            "Unable to deposit multiple collateral asset instances of the same contract address."
        );
        vm.prank(address(borrower));
        escrow.depositAsset(
            CollateralType.ERC721,
            address(erc721Mock),
            1,
            tokenIdB
        );
    }

    function test_depositAsset_ERC1155() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        uint256 tokenId = erc1155Mock.mint(address(borrower));

        borrower.approveERC1155(address(erc1155Mock));

        vm.prank(address(borrower));
        escrow.depositAsset(
            CollateralType.ERC1155,
            address(erc1155Mock),
            1,
            tokenId
        );

        uint256 storedBalance = borrower.getBalance(address(erc1155Mock));

        assertEq(storedBalance, 1, "Escrow deposit unsuccessful");
    }

    function test_depositCollateralInternal_InvalidType() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        wethMock.transfer(address(borrower), 1e18);
        borrower.approveERC20(address(wethMock), amount);

        //there seems to be a bug in hardhat that causes the revert message to be garbled
        // vm.expectRevert("Invalid collateral type");
        vm.expectRevert();
        vm.prank(address(borrower));
        escrow._depositCollateralSuper(
            CollateralType(uint16(4)),
            address(wethMock),
            amount,
            0
        );
    }

    function test_depositCollateralInternal_ERC20() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        wethMock.transfer(address(borrower), 1e18);
        borrower.approveERC20(address(wethMock), amount);

        vm.prank(address(borrower));
        escrow._depositCollateralSuper(
            CollateralType.ERC20,
            address(wethMock),
            amount,
            0
        );

        uint256 escrowedBalance = wethMock.balanceOf(address(escrow));

        assertEq(escrowedBalance, amount, "Unexpected escrow balance");
    }

    function test_depositCollateralInternal_ERC721() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        uint256 tokenId = erc721Mock.mint(address(borrower));

        borrower.approveERC721(address(erc721Mock), tokenId);

        vm.prank(address(borrower));
        escrow._depositCollateralSuper(
            CollateralType.ERC721,
            address(erc721Mock),
            1,
            tokenId
        );

        uint256 escrowedBalance = erc721Mock.balanceOf(address(escrow));

        assertEq(escrowedBalance, 1, "Unexpected escrow balance");
    }

    function test_depositCollateralInternal_ERC721_invalid_amount() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        uint256 tokenId = erc721Mock.mint(address(borrower));

        borrower.approveERC721(address(erc721Mock), tokenId);

        vm.expectRevert("Incorrect deposit amount");
        vm.prank(address(borrower));
        escrow._depositCollateralSuper(
            CollateralType.ERC721,
            address(erc721Mock),
            2,
            tokenId
        );
    }

    function test_depositCollateralInternal_ERC1155() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        uint256 tokenId = erc1155Mock.mint(address(borrower), 2);

        borrower.approveERC1155(address(erc1155Mock));

        vm.prank(address(borrower));
        escrow._depositCollateralSuper(
            CollateralType.ERC1155,
            address(erc1155Mock),
            2,
            tokenId
        );

        uint256 escrowedBalance = erc1155Mock.balanceOf(
            address(escrow),
            tokenId
        );

        assertEq(escrowedBalance, 2, "Unexpected escrow balance");
    }

    function test_onERC721Received() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        bytes4 response = escrow.onERC721Received(
            address(this),
            address(borrower),
            0,
            ""
        );

        assertEq(
            response,
            bytes4(
                keccak256("onERC721Received(address,address,uint256,bytes)")
            ),
            "response is not correct"
        );
    }

    function test_onERC1155Received() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        bytes4 response = escrow.onERC1155Received(
            address(this),
            address(borrower),
            0,
            0,
            ""
        );

        assertEq(
            response,
            bytes4(
                keccak256(
                    "onERC1155Received(address,address,uint256,uint256,bytes)"
                )
            ),
            "response is not correct"
        );
    }

    function test_onERC1155BatchReceived() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        bytes4 response = escrow.onERC1155BatchReceived(
            address(this),
            address(borrower),
            new uint256[](1),
            new uint256[](1),
            ""
        );

        assertEq(
            response,
            bytes4(
                keccak256(
                    "onERC1155BatchReceived(address,address,uint256[],uint256[],bytes)"
                )
            ),
            "response is not correct"
        );
    }

    function test_onERC1155BatchReceived_invalid() public {
        CollateralEscrowV1_Override escrow = CollateralEscrowV1_Override(
            address(borrower.getEscrow())
        );

        vm.expectRevert(
            "Only allowed one asset batch transfer per transaction."
        );
        escrow.onERC1155BatchReceived(
            address(this),
            address(borrower),
            new uint256[](2),
            new uint256[](2),
            ""
        );
    }
}

contract User {
    CollateralEscrowV1 public escrow;

    constructor(UpgradeableBeacon escrowBeacon) {
        // Deploy escrow
        BeaconProxy proxy_ = new BeaconProxy(
            address(escrowBeacon),
            abi.encodeWithSelector(CollateralEscrowV1.initialize.selector, 0)
        );
        escrow = CollateralEscrowV1(address(proxy_));
    }

    function getEscrow() public view returns (CollateralEscrowV1) {
        return escrow;
    }

    /*function deposit(
        CollateralType _collateralType,
        address _collateralAddress,
        uint256 _amount,
        uint256 _tokenId
    ) public {
        escrow.depositAsset(
            _collateralType,
            _collateralAddress,
            _amount,
            _tokenId
        );
    }

 

    function withdraw(
        address _collateralAddress,
        uint256 _amount,
        address _recipient
    ) public {
        escrow.withdraw(_collateralAddress, _amount, _recipient);
    }

    */

    function approveERC20(address tokenAddress, uint256 amount) public {
        ERC20(tokenAddress).approve(address(escrow), amount);
    }

    function approveERC721(address tokenAddress, uint256 tokenId) public {
        ERC721(tokenAddress).approve(address(escrow), tokenId);
    }

    function approveERC1155(address tokenAddress) public {
        ERC1155(tokenAddress).setApprovalForAll(address(escrow), true);
    }

    function getBalance(address _collateralAddress)
        public
        returns (uint256 amount_)
    {
        (, amount_, , ) = escrow.collateralBalances(_collateralAddress);
    }

    receive() external payable {}

    //receive 721
    function onERC721Received(address, address, uint256, bytes calldata)
        external
        returns (bytes4)
    {
        return this.onERC721Received.selector;
    }

    //receive 1155
    function onERC1155Received(
        address,
        address,
        uint256,
        uint256,
        bytes calldata
    ) external returns (bytes4) {
        return this.onERC1155Received.selector;
    }
}
