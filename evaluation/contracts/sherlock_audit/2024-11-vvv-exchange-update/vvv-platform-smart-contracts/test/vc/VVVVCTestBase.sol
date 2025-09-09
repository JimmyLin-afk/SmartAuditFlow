//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { Test } from "lib/forge-std/src/Test.sol";
import { MockERC20 } from "contracts/mock/MockERC20.sol";
import { VVVVCInvestmentLedger } from "contracts/vc/VVVVCInvestmentLedger.sol";
import { VVVVCTokenDistributor } from "contracts/vc/VVVVCTokenDistributor.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";

/**
    @title VVVVC Test Base
 */
abstract contract VVVVCTestBase is Test {
    VVVAuthorizationRegistry AuthRegistry;
    VVVVCInvestmentLedger LedgerInstance;
    MockERC20 PaymentTokenInstance;
    MockERC20 ProjectTokenInstance;
    VVVVCTokenDistributor TokenDistributorInstance;

    //placeholders for eip712 variables before they are defined
    //by reading the token distributor contract
    bytes32 ledgerDomainSeparator;
    bytes32 distributorDomainSeparator;
    bytes32 investmentTypehash;
    bytes32 claimTypehash;
    bytes32 setInvestmentRoundStateTypehash;

    //wallet setup
    uint256 deployerKey = 1234;
    uint256 ledgerManagerKey = 1235;
    uint256 tokenDistributorManagerKey = 1236;
    uint256 testSignerKey = 12345;
    uint256 sampleUserKey = 1234567;
    uint256 sampleKycAddressKey = 12345678;
    uint256 projectTokenProxyWalletKey = 12345679;

    address deployer = vm.addr(deployerKey);
    address ledgerManager = vm.addr(ledgerManagerKey);
    address tokenDistributorManager = vm.addr(tokenDistributorManagerKey);
    address testSigner = vm.addr(testSignerKey);
    address[] testSignerArray = [testSigner];
    address sampleUser = vm.addr(sampleUserKey);
    address sampleKycAddress = vm.addr(sampleKycAddressKey);
    address[] projectTokenProxyWallets = [
        vm.addr(projectTokenProxyWalletKey),
        vm.addr(projectTokenProxyWalletKey + 1),
        vm.addr(projectTokenProxyWalletKey + 2)
    ];
    address[] users = new address[](100); // 100 users

    uint256 blockNumber;
    uint256 blockTimestamp;

    string environmentTag = "development";

    //ledger contract-specific values
    bytes32 ledgerManagerRole = keccak256("LEDGER_MANAGER_ROLE");
    bytes32 tokenDistributorManagerRole = keccak256("TOKEN_DISTRIBUTOR_MANAGER_ROLE");
    uint48 defaultAdminTransferDelay = 1 days;
    uint256 exchangeRateNumerator = 1e6;
    uint256 exchangeRateDenominator = 1e6;
    uint256 feeNumerator = 1000; //10% fee sample
    uint256 activeRoundStartTimestamp = block.timestamp;
    uint256 activeRoundEndTimestamp = block.timestamp + 1 days;

    //claim contract-specific values
    uint256 projectTokenAmountToProxyWallet = 1_000_000 * 1e18; //1 million tokens
    uint256[] sampleInvestmentRoundIds = [1, 2, 3];

    // sample invest/claim amounts, keeping relative proportions equal for both
    uint256[] sampleAmountsToInvest = [1000 * 1e6, 2000 * 1e6, 3000 * 1e6];
    uint256[] sampleTokenAmountsToClaim = [1111 * 1e18, 2222 * 1e18, 3333 * 1e18];
    uint256 sampleTokenAmountToClaim = 6666 * 1e18;

    //investment payment tokens
    uint256 paymentTokenMintAmount = 10_000 * 1e6;
    uint256 userPaymentTokenDefaultAllocation = 10_000 * 1e6;
    uint256 investmentRoundSampleLimit = 1_000_000 * 1e6;

    struct TestParams {
        uint256[] investmentRoundIds;
        uint256[] tokenAmountsToInvest;
        address[] projectTokenProxyWallets;
        uint256 claimAmount;
    }

    // generate list of random addresses and deal them payment tokens and ETH
    function generateUserAddressListAndDealEtherAndToken(MockERC20 _token) public {
        for (uint256 i = 0; i < users.length; i++) {
            users[i] = address(uint160(uint256(keccak256(abi.encodePacked(block.timestamp, i)))));
            vm.deal(users[i], 1 ether); // and YOU get an ETH
            _token.mint(users[i], paymentTokenMintAmount);
        }

        vm.deal(sampleUser, 10 ether);
        vm.deal(sampleKycAddress, 10 ether);
    }

    function advanceBlockNumberAndTimestampInBlocks(uint256 blocks) public {
        blockNumber += blocks;
        blockTimestamp += blocks * 12; //seconds per block
        vm.warp(blockTimestamp);
        vm.roll(blockNumber);
    }

    function advanceBlockNumberAndTimestampInSeconds(uint256 secondsToAdvance) public {
        blockNumber += secondsToAdvance / 12; //seconds per block
        blockTimestamp += secondsToAdvance;
        vm.warp(blockTimestamp);
        vm.roll(blockNumber);
    }

    function sum(uint256[] memory _array) public pure returns (uint256) {
        uint256 total = 0;
        for (uint256 i = 0; i < _array.length; i++) {
            total += _array[i];
        }
        return total;
    }

    // create concat'd 65 byte signature that ethers would generate instead of r,s,v
    function toBytesConcat(bytes32 r, bytes32 s, uint8 v) public pure returns (bytes memory) {
        bytes memory signature = new bytes(65);
        for (uint256 i = 0; i < 32; i++) {
            signature[i] = r[i];
            signature[i + 32] = s[i];
        }
        signature[64] = bytes1(v);
        return signature;
    }

    function getEIP712SignatureForInvest(
        bytes32 _domainSeparator,
        bytes32 _investmentTypehash,
        VVVVCInvestmentLedger.InvestParams memory _params
    ) public view returns (bytes memory) {
        bytes32 digest = keccak256(
            abi.encodePacked(
                "\x19\x01",
                _domainSeparator,
                keccak256(
                    abi.encode(
                        _investmentTypehash,
                        _params.investmentRound,
                        _params.investmentRoundLimit,
                        _params.investmentRoundStartTimestamp,
                        _params.investmentRoundEndTimestamp,
                        _params.paymentTokenAddress,
                        _params.kycAddress,
                        _params.kycAddressAllocation,
                        _params.exchangeRateNumerator,
                        _params.feeNumerator,
                        _params.deadline
                    )
                )
            )
        );

        (uint8 v, bytes32 r, bytes32 s) = vm.sign(testSignerKey, digest);
        bytes memory signature = toBytesConcat(r, s, v);

        return signature;
    }

    function generateInvestParamsWithSignature(
        uint256 _investmentRound,
        uint256 _investmentRoundLimit,
        uint256 _investmentAmount,
        uint256 _investmentAllocation,
        uint256 _exchangeRateNumerator,
        uint256 _feeNumerator,
        address _kycAddress,
        uint256 _investmentRoundStartTimestamp,
        uint256 _investmentRoundEndTimestamp
    ) public view returns (VVVVCInvestmentLedger.InvestParams memory) {
        VVVVCInvestmentLedger.InvestParams memory params = VVVVCInvestmentLedger.InvestParams({
            investmentRound: _investmentRound,
            investmentRoundLimit: _investmentRoundLimit,
            investmentRoundStartTimestamp: _investmentRoundStartTimestamp,
            investmentRoundEndTimestamp: _investmentRoundEndTimestamp,
            paymentTokenAddress: address(PaymentTokenInstance),
            kycAddress: _kycAddress,
            kycAddressAllocation: _investmentAllocation,
            amountToInvest: _investmentAmount,
            exchangeRateNumerator: _exchangeRateNumerator,
            feeNumerator: _feeNumerator,
            deadline: block.timestamp + 1 hours,
            signature: bytes("placeholder")
        });

        bytes memory sig = getEIP712SignatureForInvest(ledgerDomainSeparator, investmentTypehash, params);

        params.signature = sig;

        return params;
    }

    function investAsUser(address _investor, VVVVCInvestmentLedger.InvestParams memory _params) public {
        vm.startPrank(_investor, _investor);
        PaymentTokenInstance.approve(address(LedgerInstance), _params.amountToInvest);
        LedgerInstance.invest(_params);
        vm.stopPrank();
    }

    function batchInvestAsUser(
        address _investor,
        uint256[] memory _investmentRoundIds,
        uint256[] memory _amountsToInvest
    ) public {
        VVVVCInvestmentLedger.InvestParams memory investParams;
        for (uint256 i = 0; i < sampleInvestmentRoundIds.length; i++) {
            investParams = generateInvestParamsWithSignature(
                _investmentRoundIds[i],
                investmentRoundSampleLimit,
                _amountsToInvest[i],
                userPaymentTokenDefaultAllocation,
                exchangeRateNumerator,
                feeNumerator,
                sampleKycAddress,
                block.timestamp,
                block.timestamp + 1 days
            );
            investAsUser(_investor, investParams);
        }
    }

    function getEIP712SignatureForClaim(
        bytes32 _domainSeparator,
        bytes32 _claimTypehash,
        VVVVCTokenDistributor.ClaimParams memory _params
    ) public view returns (bytes memory) {
        bytes32 digest = keccak256(
            abi.encodePacked(
                "\x19\x01",
                _domainSeparator,
                keccak256(
                    abi.encode(
                        _claimTypehash,
                        _params.kycAddress,
                        _params.projectTokenAddress,
                        _params.projectTokenProxyWallets,
                        _params.tokenAmountsToClaim,
                        _params.nonce,
                        _params.deadline
                    )
                )
            )
        );

        (uint8 v, bytes32 r, bytes32 s) = vm.sign(testSignerKey, digest);
        bytes memory signature = toBytesConcat(r, s, v);

        return signature;
    }

    function generateClaimParamsWithSignature(
        address _kycAddress,
        address[] memory _projectTokenProxyWallets,
        uint256[] memory _tokenAmountsToClaim
    ) public view returns (VVVVCTokenDistributor.ClaimParams memory) {
        VVVVCTokenDistributor.ClaimParams memory params = VVVVCTokenDistributor.ClaimParams({
            kycAddress: _kycAddress,
            projectTokenAddress: address(ProjectTokenInstance),
            projectTokenProxyWallets: _projectTokenProxyWallets,
            tokenAmountsToClaim: _tokenAmountsToClaim,
            nonce: TokenDistributorInstance.nonces(_kycAddress) + 1,
            deadline: block.timestamp + 1 hours,
            signature: bytes("placeholder")
        });

        bytes memory sig = getEIP712SignatureForClaim(distributorDomainSeparator, claimTypehash, params);

        params.signature = sig;

        return params;
    }

    function claimAsUser(address _claimant, VVVVCTokenDistributor.ClaimParams memory _params) public {
        vm.startPrank(_claimant, _claimant);
        TokenDistributorInstance.claim(_params);
        vm.stopPrank();
    }
}
