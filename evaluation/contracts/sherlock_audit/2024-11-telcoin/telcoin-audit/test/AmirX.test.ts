import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";
import { expect } from "chai";
import { ethers } from "hardhat";
import { Stablecoin, MockAmirX, TestTokenDynamic, TestWallet, TestPlugin, TestAggregator } from "../../typechain-types";

describe("AmirX", () => {
    const MINTER_ROLE = ethers.keccak256(ethers.toUtf8Bytes('MINTER_ROLE'));
    const BURNER_ROLE = ethers.keccak256(ethers.toUtf8Bytes('BURNER_ROLE'));
    const SUPPORT_ROLE = ethers.keccak256(ethers.toUtf8Bytes('SUPPORT_ROLE'));
    const SWAPPER_ROLE = ethers.keccak256(ethers.toUtf8Bytes('SWAPPER_ROLE'));
    const MAINTAINER_ROLE = ethers.keccak256(ethers.toUtf8Bytes('MAINTAINER_ROLE'));

    const TELCOIN_ADDRESS = '0xdF7837DE1F2Fa4631D716CF2502f8b230F1dcc32';
    const POL_ADDRESS = '0x0000000000000000000000000000000000001010';
    const ZERO_ADDRESS = '0x0000000000000000000000000000000000000000';

    let deployer: SignerWithAddress;
    let holder: SignerWithAddress;
    let safe: SignerWithAddress;
    let eUSD: Stablecoin;
    let eMXN: Stablecoin;
    let USDC: TestTokenDynamic;
    let telcoin: TestTokenDynamic;
    let AmirX: MockAmirX;
    let wallet: TestWallet;
    let plugin: TestPlugin;
    let aggregator: TestAggregator;

    beforeEach("setup", async () => {
        [deployer, holder, safe] = await ethers.getSigners();

        const TestToken_Factory = await ethers.getContractFactory("TestTokenDynamic", deployer);
        telcoin = await TestToken_Factory.deploy("Telcoin", "Tel", 2, deployer.address, 10);

        const AmirX_Factory = await ethers.getContractFactory("MockAmirX", deployer);
        AmirX = await AmirX_Factory.deploy(telcoin);
    });

    describe("Static Values", () => {
        beforeEach("setup", async () => {
            const AmirX_Factory = await ethers.getContractFactory("MockAmirX", deployer);
            AmirX = await AmirX_Factory.deploy(TELCOIN_ADDRESS);
        });

        it("TELCOIN", async () => {
            expect(await AmirX.TELCOIN()).to.equal(TELCOIN_ADDRESS);
        });

        it("POL", async () => {
            expect(await AmirX.POL()).to.equal(POL_ADDRESS);
        });

        it("SUPPORT_ROLE", async () => {
            expect(await AmirX.SUPPORT_ROLE()).to.equal(SUPPORT_ROLE);
        });
    });

    describe("Swaps", () => {
        beforeEach("setup", async () => {
            const Stablecoin_Factory = await ethers.getContractFactory("Stablecoin", deployer);
            eUSD = await Stablecoin_Factory.deploy();
            eMXN = await Stablecoin_Factory.deploy();

            await eUSD.initialize("US Dollar", "eXYZ", 6);
            await eUSD.grantRole(MINTER_ROLE, AmirX);
            await eUSD.grantRole(BURNER_ROLE, AmirX);
            await eUSD.grantRole(MINTER_ROLE, deployer);

            await eMXN.initialize("Mexican Peso", "eMXN", 6);
            await eMXN.grantRole(MINTER_ROLE, AmirX);
            await eMXN.grantRole(BURNER_ROLE, AmirX);
            await eMXN.grantRole(MINTER_ROLE, deployer);

            const Token_Factory = await ethers.getContractFactory("TestTokenDynamic", deployer);
            USDC = await Token_Factory.deploy("US Dollar Coin", "USDC", 6, holder.address, 10);

            await AmirX.initialize();
            await AmirX.grantRole(MAINTAINER_ROLE, deployer);
            await AmirX.grantRole(SWAPPER_ROLE, deployer);
            await AmirX.UpdateXYZ(eUSD, true, 1000000000, 0);
            await AmirX.UpdateXYZ(eMXN, true, 1000000000, 0);

            const TestWallet_Factory = await ethers.getContractFactory("TestWallet", deployer);
            wallet = await TestWallet_Factory.deploy();

            const TestPlugin_Factory = await ethers.getContractFactory("TestPlugin", deployer);
            plugin = await TestPlugin_Factory.deploy(await telcoin.getAddress());

            const TestAggregator_Factory = await ethers.getContractFactory("TestAggregator", deployer);
            aggregator = await TestAggregator_Factory.deploy(await telcoin.getAddress());
        });

        describe("revert", () => {
            it("swap: ZeroValueInput(WALLET)", async () => {
                const stableInputs = {
                    liquiditySafe: ZERO_ADDRESS,
                    destination: ZERO_ADDRESS,
                    origin: ZERO_ADDRESS,
                    oAmount: 0,
                    target: ZERO_ADDRESS,
                    tAmount: 0,
                    stablecoinFeeCurrency: await USDC.getAddress(),
                    stablecoinFeeSafe: deployer.address,
                    feeAmount: 1
                }

                const defiInputs = {
                    defiSafe: ZERO_ADDRESS,
                    aggregator: ZERO_ADDRESS,
                    plugin: ZERO_ADDRESS,
                    feeToken: ZERO_ADDRESS,
                    referrer: ZERO_ADDRESS,
                    referralFee: 0,
                    walletData: deployer.address,
                    swapData: '0x',
                }

                await expect(AmirX.swap(ZERO_ADDRESS, true, stableInputs, defiInputs)).to.revertedWithCustomError(AmirX, "ZeroValueInput").withArgs("WALLET");
            });

            it("defiToStablecoinSwap: ZeroValueInput(WALLET)", async () => {
                const stableInputs = {
                    liquiditySafe: ZERO_ADDRESS,
                    destination: ZERO_ADDRESS,
                    origin: ZERO_ADDRESS,
                    oAmount: 0,
                    target: ZERO_ADDRESS,
                    tAmount: 0,
                    stablecoinFeeCurrency: await USDC.getAddress(),
                    stablecoinFeeSafe: deployer.address,
                    feeAmount: 1
                }

                const defiInputs = {
                    defiSafe: ZERO_ADDRESS,
                    aggregator: ZERO_ADDRESS,
                    plugin: ZERO_ADDRESS,
                    feeToken: ZERO_ADDRESS,
                    referrer: ZERO_ADDRESS,
                    referralFee: 0,
                    walletData: deployer.address,
                    swapData: '0x',
                }

                await expect(AmirX.defiToStablecoinSwap(ZERO_ADDRESS, stableInputs, defiInputs)).to.revertedWithCustomError(AmirX, "ZeroValueInput").withArgs("WALLET");
            });

            describe("_verifyDefi", () => {
                it("ZeroValueInput(BUYBACK)", async () => {
                    const defiInputs = {
                        defiSafe: ZERO_ADDRESS,
                        aggregator: ZERO_ADDRESS,
                        plugin: ZERO_ADDRESS,
                        feeToken: deployer.address,
                        referrer: ZERO_ADDRESS,
                        referralFee: 0,
                        walletData: await wallet.getTestSelector(),
                        swapData: '0x',
                    }

                    //defi.aggregator == address(0)
                    await expect(AmirX.defiSwap(wallet, defiInputs)).to.revertedWithCustomError(AmirX, "ZeroValueInput").withArgs("BUYBACK");
                    defiInputs.aggregator = deployer.address;
                    //defi.swapData.length == 0
                    await expect(AmirX.defiSwap(wallet, defiInputs)).to.revertedWithCustomError(AmirX, "ZeroValueInput").withArgs("BUYBACK");
                    defiInputs.swapData = deployer.address;
                });

                it("ZeroValueInput(PLUGIN)", async () => {
                    const defiInputs = {
                        defiSafe: ZERO_ADDRESS,
                        aggregator: deployer.address,
                        plugin: ZERO_ADDRESS,
                        feeToken: await telcoin.getAddress(),
                        referrer: deployer.address,
                        referralFee: 0,
                        walletData: await wallet.getTestSelector(),
                        swapData: await wallet.getTestSelector(),
                    }

                    //if (defi.feeToken == TELCOIN)
                    //defi.referrer == address(0)
                    await expect(AmirX.defiSwap(wallet, defiInputs)).to.revertedWithCustomError(AmirX, "ZeroValueInput").withArgs("PLUGIN");
                });
            });
        });

        // it("swap", async () => {
        //     await eUSD.mintTo(holder, 15);
        //     const stableInputs = {
        //         liquiditySafe: deployer,
        //         destination: holder,
        //         origin: eUSD,
        //         oAmount: 10,
        //         target: eMXN,
        //         tAmount: 100,
        //         stablecoinFeeCurrency: eUSD,
        //         stablecoinFeeSafe: safe.address,
        //         feeAmount: 5

        //     }

        //     const defiInputs = {
        //         defiSafe: ZERO_ADDRESS,
        //         aggregator: ZERO_ADDRESS,
        //         plugin: ZERO_ADDRESS,
        //         feeToken: ZERO_ADDRESS,
        //         referrer: ZERO_ADDRESS,
        //         referralFee: 0,
        //         walletData: '0x',
        //         swapData: '0x',
        //     }

        //     await eUSD.connect(holder).approve(AmirX, 15);
        //     await expect(AmirX.swap(holder.address, false, stableInputs, defiInputs)).to.not.be.reverted;
        //     expect(await eUSD.totalSupply()).to.equal(5);
        //     expect(await eMXN.balanceOf(holder)).to.equal(100);
        // });

        // it("convertToEXYZ", async () => {
        //     const stableInputs = {
        //         safe: deployer,
        //         destination: holder,
        //         origin: USDC,
        //         oAmount: 10,
        //         target: eMXN,
        //         tAmount: 100
        //     }

        //     const defiInputs = {
        //         defiSafe: ZERO_ADDRESS,
        //         aggregator: ZERO_ADDRESS,
        //         plugin: ZERO_ADDRESS,
        //         feeToken: ZERO_ADDRESS,
        //         referrer: ZERO_ADDRESS,
        //         referralFee: 0,
        //         walletData: '0x',
        //         swapData: '0x',
        //     }

        //     await USDC.connect(holder).approve(AmirX, 10);
        //     await expect(AmirX.defiToStablecoinSwap(holder, stableInputs, defiInputs)).to.not.be.reverted;
        //     expect(await USDC.balanceOf(deployer)).to.equal(10);
        //     expect(await eMXN.balanceOf(holder)).to.equal(100);
        // });

        // it("convertFromEXYZ", async () => {
        //     await USDC.mintTo(deployer, 10);
        //     await USDC.connect(deployer).approve(AmirX, 10);
        //     const stableInputs = {
        //         safe: deployer,
        //         destination: holder,
        //         origin: eUSD,
        //         oAmount: 10,
        //         target: USDC,
        //         tAmount: 10
        //     }

        //     const defiInputs = {
        //         safe: ZERO_ADDRESS,
        //         aggregator: ZERO_ADDRESS,
        //         plugin: ZERO_ADDRESS,
        //         feeToken: ZERO_ADDRESS,
        //         referrer: ZERO_ADDRESS,
        //         referralFee: 0,
        //         walletData: '0x',
        //         swapData: '0x',
        //     }

        //     await eUSD.mintTo(holder, 10);
        //     await eUSD.connect(holder).approve(AmirX, 10);

        //     await USDC.connect(holder).approve(AmirX, 10);
        //     await expect(AmirX.stablecoinSwap(holder, stableInputs, defiInputs)).to.not.be.reverted;
        //     expect(await eUSD.totalSupply()).to.equal(0);
        //     expect(await USDC.balanceOf(deployer)).to.equal(0);
        // });

        describe("swapping", () => {
            beforeEach("setup", async () => {
                await eUSD.grantRole(MINTER_ROLE, deployer);
            });
        });

        describe("two steps", () => {
            it("swap", async () => {
                await eUSD.mintTo(holder, 15);
                const stablecoinInputs = {
                    liquiditySafe: deployer,
                    destination: holder,
                    origin: eUSD,
                    oAmount: 10,
                    target: eMXN,
                    tAmount: 100,
                    stablecoinFeeCurrency: eUSD,
                    stablecoinFeeSafe: safe.address,
                    feeAmount: 5
                }

                const defiInputs = {
                    defiSafe: deployer,
                    aggregator: aggregator,
                    plugin: plugin,
                    feeToken: USDC,
                    referrer: ZERO_ADDRESS,
                    referralFee: 0,
                    walletData: await wallet.getTestSelector(),
                    swapData: await aggregator.getSwapSelector(),
                }

                await eUSD.connect(holder).approve(AmirX, 15);
                await telcoin.transfer(aggregator, 10);

                await expect(AmirX.swap(holder, true, stablecoinInputs, defiInputs)).to.not.be.reverted;

                expect(await eUSD.totalSupply()).to.equal(5);
                expect(await eUSD.balanceOf(safe)).to.equal(5);
                expect(await eMXN.balanceOf(holder)).to.equal(100);
                expect(await telcoin.balanceOf(deployer)).to.equal(1000);
            });

            it("swap", async () => {
                await eUSD.mintTo(holder, 15);
                const stablecoinInputs = {
                    liquiditySafe: deployer,
                    destination: holder,
                    origin: eUSD,
                    oAmount: 10,
                    target: eMXN,
                    tAmount: 100,
                    stablecoinFeeCurrency: eUSD,
                    stablecoinFeeSafe: safe.address,
                    feeAmount: 5
                }

                const defiInputs = {
                    defiSafe: deployer,
                    aggregator: aggregator,
                    plugin: plugin,
                    feeToken: USDC,
                    referrer: ZERO_ADDRESS,
                    referralFee: 0,
                    walletData: await wallet.getTestSelector(),
                    swapData: await aggregator.getSwapSelector(),
                }

                await eUSD.connect(holder).approve(AmirX, 15);
                await telcoin.transfer(aggregator, 10);

                await expect(AmirX.swap(holder, false, stablecoinInputs, defiInputs)).to.not.be.reverted;

                expect(await eUSD.totalSupply()).to.equal(5);
                expect(await eUSD.balanceOf(safe)).to.equal(5);
                expect(await eMXN.balanceOf(holder)).to.equal(100);
                expect(await telcoin.balanceOf(deployer)).to.equal(1000);
            });

            it("stable to defi", async () => {
                await eUSD.mintTo(holder, 15);
                const stablecoinInputs = {
                    liquiditySafe: deployer,
                    destination: holder,
                    origin: eUSD,
                    oAmount: 10,
                    target: eMXN,
                    tAmount: 100,
                    stablecoinFeeCurrency: eUSD,
                    stablecoinFeeSafe: safe.address,
                    feeAmount: 5
                }

                const defiInputs = {
                    defiSafe: deployer,
                    aggregator: aggregator,
                    plugin: plugin,
                    feeToken: USDC,
                    referrer: ZERO_ADDRESS,
                    referralFee: 0,
                    walletData: await wallet.getTestSelector(),
                    swapData: await aggregator.getSwapSelector(),
                }

                await eUSD.connect(holder).approve(AmirX, 15);
                await telcoin.transfer(aggregator, 10);

                await expect(AmirX.stablecoinToDefiSwap(holder, stablecoinInputs, defiInputs)).to.not.be.reverted;

                expect(await eUSD.totalSupply()).to.equal(5);
                expect(await eUSD.balanceOf(safe)).to.equal(5);
                expect(await eMXN.balanceOf(holder)).to.equal(100);
                expect(await telcoin.balanceOf(deployer)).to.equal(1000);
            });

            it("defi to stable", async () => {
                await eUSD.mintTo(holder, 15);
                const stablecoinInputs = {
                    liquiditySafe: deployer,
                    destination: holder,
                    origin: eUSD,
                    oAmount: 10,
                    target: eMXN,
                    tAmount: 100,
                    stablecoinFeeCurrency: eUSD,
                    stablecoinFeeSafe: safe.address,
                    feeAmount: 5
                }

                const defiInputs = {
                    defiSafe: deployer,
                    aggregator: aggregator,
                    plugin: plugin,
                    feeToken: USDC,
                    referrer: ZERO_ADDRESS,
                    referralFee: 0,
                    walletData: await wallet.getTestSelector(),
                    swapData: await aggregator.getSwapSelector(),
                }

                await eUSD.connect(holder).approve(AmirX, 15);
                await telcoin.transfer(aggregator, 10);

                await expect(AmirX.defiToStablecoinSwap(holder, stablecoinInputs, defiInputs)).to.not.be.reverted;

                expect(await eUSD.balanceOf(safe)).to.equal(5);
                expect(await eMXN.balanceOf(holder)).to.equal(100);
                expect(await telcoin.balanceOf(deployer)).to.equal(1000);
            });
        });

        describe("defiSwap", () => {
            it("wallet call", async () => {
                const defiInputs = {
                    defiSafe: ZERO_ADDRESS,
                    aggregator: ZERO_ADDRESS,
                    plugin: ZERO_ADDRESS,
                    feeToken: await telcoin.getAddress(),
                    referrer: ZERO_ADDRESS,
                    referralFee: 0,
                    walletData: await wallet.getTestSelector(),
                    swapData: '0x',
                }

                await telcoin.approve(AmirX, 10);
                await expect(AmirX.defiSwap(holder, defiInputs)).to.not.be.reverted;
                expect(await telcoin.balanceOf(deployer)).to.equal(1000);
            });

            it("_feeDispersal", async () => {
                const defiInputs = {
                    defiSafe: ZERO_ADDRESS,
                    aggregator: ZERO_ADDRESS,
                    plugin: plugin,
                    feeToken: await telcoin.getAddress(),
                    referrer: deployer,
                    referralFee: 10,
                    walletData: await wallet.getTestSelector(),
                    swapData: '0x',
                }

                await telcoin.transfer(AmirX, 10);
                await expect(AmirX.defiSwap(holder, defiInputs)).to.not.be.reverted;
                expect(await telcoin.balanceOf(plugin)).to.equal(10);
            });

            it("_buyBack with POL", async () => {
                const defiInputs = {
                    defiSafe: deployer,
                    aggregator: aggregator,
                    plugin: plugin,
                    feeToken: POL_ADDRESS,
                    referrer: ZERO_ADDRESS,
                    referralFee: 0,
                    walletData: await wallet.getTestSelector(),
                    swapData: await aggregator.getPOLSwapSelector(),
                }

                await telcoin.transfer(aggregator, 10);
                await expect(AmirX.defiSwap(holder, defiInputs)).to.not.be.reverted;
                expect(await telcoin.balanceOf(deployer)).to.equal(1000);
            });

            it("_buyBack wtih ERC20", async () => {
                const defiInputs = {
                    defiSafe: deployer,
                    aggregator: aggregator,
                    plugin: plugin,
                    feeToken: USDC,
                    referrer: ZERO_ADDRESS,
                    referralFee: 0,
                    walletData: await wallet.getTestSelector(),
                    swapData: await aggregator.getSwapSelector(),
                }

                await telcoin.transfer(aggregator, 10);
                await expect(AmirX.defiSwap(holder, defiInputs)).to.not.be.reverted;
                expect(await telcoin.balanceOf(deployer)).to.equal(1000);
            });
        });
    });

    describe("rescueCrypto", () => {
        beforeEach("setup", async () => {
            const AmirX_Factory = await ethers.getContractFactory("MockAmirX", deployer);
            AmirX = await AmirX_Factory.deploy(TELCOIN_ADDRESS);
            await AmirX.initialize();
            await AmirX.grantRole(SUPPORT_ROLE, deployer);
        });

        it("POL insufficient balance", async () => {
            await expect(AmirX.rescueCrypto(POL_ADDRESS, 10100000)).to.be.revertedWith("AmirX: POL send failed");
        });

        it("POL", async () => {
            deployer.sendTransaction({ value: 101, to: await AmirX.getAddress() });
            await expect(AmirX.rescueCrypto(POL_ADDRESS, 101)).to.not.be.reverted;
        });

        it("ERC20", async () => {
            await telcoin.transfer(AmirX, 101);
            await expect(AmirX.rescueCrypto(await telcoin.getAddress(), 101)).to.not.be.reverted;
            expect(await telcoin.balanceOf(deployer)).to.equal(1000);
        });
    });
});