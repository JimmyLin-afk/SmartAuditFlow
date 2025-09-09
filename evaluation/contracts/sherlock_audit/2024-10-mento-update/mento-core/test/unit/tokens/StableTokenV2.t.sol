// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase
pragma solidity ^0.8;

import { addresses, uints } from "mento-std/Array.sol";
import { Test } from "mento-std/Test.sol";

import { StableTokenV2 } from "contracts/tokens/StableTokenV2.sol";

contract StableTokenV2Test is Test {
  event TransferComment(string comment);
  event Transfer(address indexed from, address indexed to, uint256 value);

  bytes32 private constant _PERMIT_TYPEHASH =
    keccak256("Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)");
  bytes32 private constant _TYPE_HASH =
    keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)");
  bytes32 private _HASHED_NAME;
  bytes32 private _HASHED_VERSION;

  address validators = address(0x1001);
  address broker = address(0x1002);
  address exchange = address(0x1003);

  address holder0 = address(0x2001);
  address holder1 = address(0x2002);
  address holder2;
  uint256 holder2Pk = uint256(0x31337);

  address feeRecipient = address(0x3001);
  address gatewayFeeRecipient = address(0x3002);
  address communityFund = address(0x3003);

  StableTokenV2 private token;

  function setUp() public {
    holder2 = vm.addr(holder2Pk);

    _HASHED_NAME = keccak256(bytes("cUSD"));
    _HASHED_VERSION = keccak256(bytes("1"));

    token = new StableTokenV2(false);
    token.initialize(
      "cUSD",
      "cUSD",
      addresses(holder0, holder1, holder2, broker, exchange),
      uints(1000, 1000, 1000, 1000, 1000)
    );
    token.initializeV2(broker, validators, exchange);
  }

  function mintAndAssert(address minter, address to, uint256 value) public {
    uint256 balanceBefore = token.balanceOf(to);
    vm.prank(minter);
    token.mint(to, value);
    assertEq(token.balanceOf(to), balanceBefore + value);
  }

  function test_initializers_disabled() public {
    StableTokenV2 disabledToken = new StableTokenV2(true);

    address[] memory initialAddresses = new address[](0);
    uint256[] memory initialBalances = new uint256[](0);

    vm.expectRevert(bytes("Initializable: contract is already initialized"));
    disabledToken.initialize("cUSD", "cUSD", initialAddresses, initialBalances);

    vm.expectRevert(bytes("Initializable: contract is already initialized"));
    token.initializeV2(broker, validators, exchange);
  }

  function test_transferWithComment_shouldEmitCorrectMessage() public {
    vm.expectEmit(true, true, true, true);
    emit TransferComment("Hello World");

    vm.prank(holder0);
    token.transferWithComment(holder1, 100, "Hello World");
  }

  function test_mint_whenCalledByExchange_shouldMintTokens() public {
    mintAndAssert(exchange, holder0, 100);
  }

  function test_mint_whenCalledByValidators_shouldMintTokens() public {
    mintAndAssert(validators, holder0, 100);
  }

  function test_mint_whenCalledByBroker_shouldMintTokens() public {
    mintAndAssert(broker, holder0, 100);
  }

  function test_mint_whenSenderIsNotAuthorized_shouldRevert() public {
    vm.prank(holder0);
    vm.expectRevert(bytes("StableTokenV2: not allowed to mint"));
    token.mint(holder0, 100);
  }

  function test_burn_whenCalledByExchange_shouldBurnTokens() public {
    vm.prank(exchange);
    token.burn(100);
    assertEq(900, token.balanceOf(exchange));
  }

  function test_burn_whenCalledByBroker_shouldBurnTokens() public {
    vm.prank(broker);
    token.burn(100);
    assertEq(900, token.balanceOf(broker));
  }

  function test_burn_whenSenderIsNotAuthorized_shouldRevert() public {
    vm.prank(holder0);
    vm.expectRevert(bytes("StableTokenV2: not allowed to burn"));
    token.burn(100);
  }

  function test_permit_whenSenderPermits_shouldIncreaseAllowance() public {
    uint256 allowanceBefore = token.allowance(holder2, holder0);
    assertEq(0, allowanceBefore);

    uint256 deadline = block.timestamp + 1000;
    bytes32 structHash = buildTypedDataHash(holder2, holder0, 100, deadline);
    (uint8 v, bytes32 r, bytes32 s) = vm.sign(holder2Pk, structHash);

    vm.prank(holder0);
    token.permit(holder2, holder0, 100, deadline, v, r, s);
    assertEq(100, token.allowance(holder2, holder0));
  }

  function test_debitGasFees_whenCallerNotVM_shouldRevert() public {
    vm.expectRevert("Only VM can call");
    token.debitGasFees(holder0, 100);
  }

  function test_debitGasFees_whenCallerIsVM_shouldDebitGasFees() public {
    uint256 amount = 100;
    assertEq(token.balanceOf(holder0), 1000);

    vm.prank(address(0));
    token.debitGasFees(holder0, 100);

    assertEq(token.balanceOf(holder0), 1000 - amount);
  }

  function test_creditGasFees_whenCallerNotVM_shouldRevert() public {
    vm.expectRevert("Only VM can call");
    token.creditGasFees(holder0, feeRecipient, gatewayFeeRecipient, communityFund, 25, 25, 25, 25);
  }

  function test_creditGasFees_whenCalledByVm_shouldCreditFees() public {
    uint256 refund = 20;
    uint256 tipTxFee = 30;
    uint256 gatewayFee = 10;
    uint256 baseTxFee = 40;
    uint256 tokenSupplyBefore = token.totalSupply();

    vm.prank(address(0));
    token.creditGasFees(
      holder0,
      feeRecipient,
      gatewayFeeRecipient,
      communityFund,
      refund,
      tipTxFee,
      gatewayFee,
      baseTxFee
    );

    assertEq(token.balanceOf(holder0), 1000 + refund);
    assertEq(token.balanceOf(feeRecipient), tipTxFee);
    assertEq(token.balanceOf(gatewayFeeRecipient), gatewayFee);
    assertEq(token.balanceOf(communityFund), baseTxFee);
    assertEq(token.totalSupply(), tokenSupplyBefore + refund + tipTxFee + gatewayFee + baseTxFee);
  }

  function test_creditGasFees_whenCalledByVm_with0xFeeRecipient_shouldBurnTipTxFee() public {
    uint256 refund = 20;
    uint256 tipTxFee = 30;
    uint256 gatewayFee = 10;
    uint256 baseTxFee = 40;
    uint256 holder0InitialBalance = token.balanceOf(holder0);
    uint256 tokenSupplyBefore = token.totalSupply();
    uint256 newlyMinted = refund + tipTxFee + gatewayFee + baseTxFee;

    vm.prank(address(0));
    token.creditGasFees(
      holder0,
      address(0),
      gatewayFeeRecipient,
      communityFund,
      refund,
      tipTxFee,
      gatewayFee,
      baseTxFee
    );

    assertEq(token.balanceOf(holder0), holder0InitialBalance + refund);
    assertEq(token.balanceOf(feeRecipient), 0);
    assertEq(token.balanceOf(gatewayFeeRecipient), gatewayFee);
    assertEq(token.balanceOf(communityFund), baseTxFee);
    assertEq(token.totalSupply(), tokenSupplyBefore + newlyMinted - tipTxFee);
  }

  function test_creditGasFees_whenCalledByVm_with0xGatewayFeeRecipient_shouldBurnGateWayFee() public {
    uint256 refund = 20;
    uint256 tipTxFee = 30;
    uint256 gatewayFee = 10;
    uint256 baseTxFee = 40;
    uint256 holder0InitialBalance = token.balanceOf(holder0);
    uint256 tokenSupplyBefore = token.totalSupply();
    uint256 newlyMinted = refund + tipTxFee + gatewayFee + baseTxFee;

    vm.prank(address(0));
    token.creditGasFees(holder0, feeRecipient, address(0), communityFund, refund, tipTxFee, gatewayFee, baseTxFee);

    assertEq(token.balanceOf(holder0), holder0InitialBalance + refund);
    assertEq(token.balanceOf(feeRecipient), tipTxFee);
    assertEq(token.balanceOf(gatewayFeeRecipient), 0);
    assertEq(token.balanceOf(communityFund), baseTxFee);
    assertEq(token.totalSupply(), tokenSupplyBefore + newlyMinted - gatewayFee);
  }

  function test_creditGasFees_whenCalledByVm_with0xCommunityFund_shouldBurnBaseTxFee() public {
    uint256 refund = 20;
    uint256 tipTxFee = 30;
    uint256 gatewayFee = 10;
    uint256 baseTxFee = 40;
    uint256 holder0InitialBalance = token.balanceOf(holder0);
    uint256 tokenSupplyBefore = token.totalSupply();
    uint256 newlyMinted = refund + tipTxFee + gatewayFee + baseTxFee;

    vm.prank(address(0));
    token.creditGasFees(
      holder0,
      feeRecipient,
      gatewayFeeRecipient,
      address(0),
      refund,
      tipTxFee,
      gatewayFee,
      baseTxFee
    );

    assertEq(token.balanceOf(holder0), holder0InitialBalance + refund);
    assertEq(token.balanceOf(feeRecipient), tipTxFee);
    assertEq(token.balanceOf(gatewayFeeRecipient), gatewayFee);
    assertEq(token.balanceOf(communityFund), 0);
    assertEq(token.totalSupply(), tokenSupplyBefore + newlyMinted - baseTxFee);
  }

  function test_creditGasFees_whenCalledByVm_withMultiple0xRecipients_shouldBurnTheirRespectiveFees0() public {
    uint256 refund = 20;
    uint256 tipTxFee = 30;
    uint256 gatewayFee = 10;
    uint256 baseTxFee = 40;
    uint256 holder0InitialBalance = token.balanceOf(holder0);
    uint256 tokenSupplyBefore0 = token.totalSupply();
    uint256 newlyMinted0 = refund + tipTxFee + gatewayFee + baseTxFee;

    vm.prank(address(0));
    // gateWayFeeRecipient and communityFund both 0x
    token.creditGasFees(holder0, feeRecipient, address(0), address(0), refund, tipTxFee, gatewayFee, baseTxFee);

    assertEq(token.balanceOf(holder0), holder0InitialBalance + refund);
    assertEq(token.balanceOf(feeRecipient), tipTxFee);
    assertEq(token.balanceOf(gatewayFeeRecipient), 0);
    assertEq(token.balanceOf(communityFund), 0);
    assertEq(token.totalSupply(), tokenSupplyBefore0 + newlyMinted0 - gatewayFee - baseTxFee);
  }

  function test_creditGasFees_whenCalledByVm_withMultiple0xRecipients_shouldBurnTheirRespectiveFees1() public {
    uint256 refund = 20;
    uint256 tipTxFee = 30;
    uint256 gatewayFee = 10;
    uint256 baseTxFee = 40;
    // case with both feeRecipient and communityFund both 0x
    uint256 holder1InitialBalance = token.balanceOf(holder1);
    uint256 feeRecipientBalance = token.balanceOf(feeRecipient);
    uint256 gatewayFeeRecipientBalance = token.balanceOf(gatewayFeeRecipient);
    uint256 communityFundBalance = token.balanceOf(communityFund);
    uint256 tokenSupplyBefore1 = token.totalSupply();
    uint256 newlyMinted1 = refund + tipTxFee + gatewayFee + baseTxFee;
    vm.prank(address(0));
    token.creditGasFees(holder1, address(0), gatewayFeeRecipient, address(0), refund, tipTxFee, gatewayFee, baseTxFee);

    assertEq(token.balanceOf(holder1), holder1InitialBalance + refund);
    assertEq(token.balanceOf(feeRecipient), feeRecipientBalance);
    assertEq(token.balanceOf(gatewayFeeRecipient), gatewayFeeRecipientBalance + gatewayFee);
    assertEq(token.balanceOf(communityFund), communityFundBalance);
    assertEq(token.totalSupply(), tokenSupplyBefore1 + newlyMinted1 - tipTxFee - baseTxFee);
  }

  function buildTypedDataHash(
    address owner,
    address spender,
    uint256 amount,
    uint256 deadline
  ) internal view returns (bytes32) {
    uint256 nonce = token.nonces(owner);
    bytes32 digest = keccak256(
      abi.encodePacked(
        "\x19\x01",
        keccak256(abi.encode(_TYPE_HASH, _HASHED_NAME, _HASHED_VERSION, block.chainid, address(token))),
        keccak256(abi.encode(_PERMIT_TYPEHASH, owner, spender, amount, nonce, deadline))
      )
    );
    return digest;
  }
}
