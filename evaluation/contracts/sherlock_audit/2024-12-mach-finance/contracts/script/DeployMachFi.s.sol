// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import {Script} from "forge-std/Script.sol";
import {CErc20Delegator} from "../src/CErc20Delegator.sol";
import {CErc20Delegate} from "../src/CErc20Delegate.sol";
import {CSonic} from "../src/CSonic.sol";
import {CToken} from "../src/CToken.sol";
import {ComptrollerInterface} from "../src/ComptrollerInterface.sol";
import {InterestRateModel} from "../src/InterestRateModel.sol";
import {Comptroller} from "../src/Comptroller.sol";
import {JumpRateModelV2} from "../src/JumpRateModelV2.sol";
import {IOracleSource} from "../src/Oracles/IOracleSource.sol";
import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {PythOracle} from "../src/Oracles/Pyth/PythOracle.sol";
import {BandOracle} from "../src/Oracles/Band/BandOracle.sol";
import {API3Oracle} from "../src/Oracles/API3/API3Oracle.sol";
import {Upgrades} from "openzeppelin-foundry-upgrades/Upgrades.sol";
import {PriceOracleAggregator} from "../src/Oracles/PriceOracleAggregator.sol";
import {Maximillion} from "../src/Maximillion.sol";
import {Unitroller} from "../src/Unitroller.sol";
import {ComptrollerV1Storage} from "../src/ComptrollerStorage.sol";

import "forge-std/console.sol";

contract DeployMachFi is Script {
    bytes32 constant FTM_PRICE_FEED_ID = 0x5c6c0d2386e3352356c3ab84434fafb5ea067ac2678a38a338c4a69ddc4bdb0c;
    bytes32 constant BTC_PRICE_FEED_ID = 0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43;

    address public constant CORAL = 0xAF93888cbD250300470A1618206e036E11470149;
    // @notice - Make sure same as in .env file
    address public constant ADMIN = 0x74e57CED553740ce9238Bb0E171f77aDfAAa914C;
    address public constant NATIVE_ASSET = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE;

    address public constant PYTH_ORACLE_ADDRESS = 0x96124d1F6E44FfDf1fb5D6d74BB2DE1B7Fbe7376;
    address public constant BAND_ORACLE_ADDRESS = 0x1744a64d95059e5281Ee573BF1C26813811d9BD3;

    address constant SONIC_BLAZE_TESTNET_API3_FTM_PROXY = 0x8927DA1377C78D25E78c335F48a6f8e42Cce0C09;
    address constant SONIC_BLAZE_TESTNET_API3_WBTC_PROXY = 0x041a131Fa91Ad61dD85262A42c04975986580d50;
    address constant SONIC_BLAZE_TESTNET_API3_USDC_PROXY = 0xD3C586Eec1C6C3eC41D276a23944dea080eDCf7f;
    address constant SONIC_BLAZE_TESTNET_API3_SOLV_PROXY = 0xadf6e9419E483Cc214dfC9EF1887f3aa7e85cA09;

    CSonic public cSonic;
    CErc20Delegator public cCoral;
    CErc20Delegate public cDelegate;

    Comptroller public comptrollerImplementation;
    Comptroller public comptroller;
    Unitroller public unitroller;

    JumpRateModelV2 public interestRateModel;
    ERC20 public coral = ERC20(CORAL);
    Maximillion public maximillion;

    PythOracle public pythOracle;
    BandOracle public bandOracle;
    API3Oracle public api3Oracle;
    PriceOracleAggregator public priceOracleAggregator;

    // @notice - Admin address for the deployment
    address public admin;

    function run() public {
        uint256 privateKey = vm.envUint("ADMIN_PRIVATE_KEY");
        admin = vm.addr(privateKey);
        vm.startBroadcast(privateKey);
        vm.stopBroadcast();
    }

    function deployBaselineContracts() public {
        deployComptroller();
        deployInterestRateModel();
        address[] memory cTokens = deployCTokens();
        deployPriceOracles();
        deployMaximillion();

        comptroller._setPriceOracle(priceOracleAggregator);

        // CAREFUL of "exchange rate" manipulation attacks on Compound v2 forks
        // @dev - Before setting collateral factors -> https://x.com/hexagate_/status/1650177766187323394
        // 0. Ensure collateral factor is 0
        // 1. Mint some cTokens
        // 2. Burn them to make sure total supply doesn't go to zero
        // 3. Then set collateral factors
        // @dev - Preferably do this in a single transaction (prevent front-running)
        // https://github.com/SunWeb3Sec/DeFiHackLabs/blob/main/src/test/2023-04/HundredFinance_2_exp.sol
        supportCTokens(cTokens);
        uint256[] memory collateralFactors = new uint256[](2);
        collateralFactors[0] = 0.8e18;
        collateralFactors[1] = 0.7e18;
        setCollateralFactors(cTokens, collateralFactors);
    }

    function deployComptroller() public {
        comptrollerImplementation = new Comptroller();
        console.log("Comptroller deployed at", address(comptrollerImplementation));

        unitroller = new Unitroller();
        console.log("Unitroller deployed at", address(unitroller));

        // Set pending comptroller implementation
        unitroller._setPendingImplementation(address(comptrollerImplementation));

        // Become comptroller
        comptrollerImplementation._become(unitroller);
        comptroller = Comptroller(payable(address(unitroller)));
    }

    function deployInterestRateModel() public {
        uint256 baseRatePerYear = 0;
        uint256 multiplierPerYear = 0.25e18;
        uint256 jumpMultiplierPerYear = 5e18;
        uint256 kink_ = 0.8e18;
        interestRateModel = new JumpRateModelV2(baseRatePerYear, multiplierPerYear, jumpMultiplierPerYear, kink_, ADMIN);
        console.log("InterestRateModel deployed at", address(interestRateModel));
    }

    function deployCTokens() public returns (address[] memory) {
        cDelegate = new CErc20Delegate();
        cCoral = new CErc20Delegator(
            CORAL,
            comptroller,
            interestRateModel,
            1e18,
            "MachFi Coral",
            "cCORAL",
            8,
            payable(ADMIN),
            address(cDelegate),
            ""
        );
        cSonic = new CSonic(comptroller, interestRateModel, 1e18, "Sonic", "cSonic", 18, payable(ADMIN));

        address[] memory cTokens = new address[](2);
        cTokens[0] = address(cCoral);
        cTokens[1] = address(cSonic);

        console.log("cCoral deployed at", address(cCoral));
        console.log("cSonic deployed at", address(cSonic));

        return cTokens;
    }

    function supportCTokens(address[] memory cTokens) public {
        for (uint256 i = 0; i < cTokens.length; i++) {
            comptroller._supportMarket(CToken(cTokens[i]));
        }
    }

    function deployPriceOracles() public returns (PriceOracleAggregator) {
        _deployBandOracle();
        _deployPythOracle();
        _deployAPI3Oracle();

        address priceOracleAggregatorProxyAddress = Upgrades.deployUUPSProxy(
            "PriceOracleAggregator.sol", abi.encodeCall(PriceOracleAggregator.initialize, (ADMIN))
        );
        priceOracleAggregator = PriceOracleAggregator(payable(priceOracleAggregatorProxyAddress));

        IOracleSource[] memory oracles = new IOracleSource[](2);
        oracles[0] = pythOracle;
        oracles[1] = api3Oracle;

        priceOracleAggregator.updateTokenOracles(address(coral), oracles);
        priceOracleAggregator.updateTokenOracles(NATIVE_ASSET, oracles);
        console.log("PriceOracleAggregator deployed at", address(priceOracleAggregator));

        return priceOracleAggregator;
    }

    function _deployBandOracle() internal {
        address[] memory underlyingTokens = new address[](2);
        underlyingTokens[0] = address(coral);
        underlyingTokens[1] = NATIVE_ASSET;

        string[] memory bandSymbols = new string[](2);
        bandSymbols[0] = "BTC";
        bandSymbols[1] = "FTM";

        bandOracle = new BandOracle(admin, BAND_ORACLE_ADDRESS, underlyingTokens, bandSymbols);
        console.log("BandOracle deployed at", address(bandOracle));
    }

    function _deployPythOracle() internal {
        address[] memory underlyingTokens = new address[](2);
        underlyingTokens[0] = address(coral);
        underlyingTokens[1] = NATIVE_ASSET;

        bytes32[] memory priceFeedIds = new bytes32[](2);
        priceFeedIds[0] = BTC_PRICE_FEED_ID;
        priceFeedIds[1] = FTM_PRICE_FEED_ID;

        pythOracle = new PythOracle(admin, PYTH_ORACLE_ADDRESS, underlyingTokens, priceFeedIds);
    }

    function _deployAPI3Oracle() internal {
        address[] memory underlyingTokens = new address[](2);
        underlyingTokens[0] = address(coral);
        underlyingTokens[1] = NATIVE_ASSET;

        address[] memory api3ProxyAddresses = new address[](2);
        api3ProxyAddresses[0] = SONIC_BLAZE_TESTNET_API3_WBTC_PROXY;
        api3ProxyAddresses[1] = SONIC_BLAZE_TESTNET_API3_FTM_PROXY;

        api3Oracle = new API3Oracle(admin, underlyingTokens, api3ProxyAddresses);
        console.log("API3Oracle deployed at", address(api3Oracle));
    }

    function setCollateralFactors(address[] memory cTokens, uint256[] memory collateralFactors) public {
        for (uint256 i = 0; i < cTokens.length; i++) {
            comptroller._setCollateralFactor(CToken(cTokens[i]), collateralFactors[i]);
        }
    }

    function deployMaximillion() public {
        maximillion = new Maximillion(cSonic);
        console.log("Maximillion deployed at", address(maximillion));
    }
}
