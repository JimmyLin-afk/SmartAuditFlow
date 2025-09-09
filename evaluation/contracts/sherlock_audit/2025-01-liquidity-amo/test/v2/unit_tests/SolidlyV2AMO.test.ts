import { expect } from "chai";
import { ethers, network, upgrades } from "hardhat";
import { SignerWithAddress } from "@nomiclabs/hardhat-ethers/signers";
import {
  Minter,
  BoostStablecoin,
  MockERC20,
  V2AMO,
  IV2Voter,
  IFactory,
  MockRouter,
  IGauge,
  IERC20,
  IPoolFactory,
} from "../../../typechain-types";
import { setBalance } from "@nomicfoundation/hardhat-network-helpers";

describe("V2AMO", function () {
  const stable = false;
  const toBuy = stable ? "2000000" : "1000000";
  const fee = stable ? "0.0002" : "0.002";
  const poolFee = ethers.parseUnits(fee, 6);
  // Common variables for both pool types
  let v2AMO: V2AMO;
  let boost: BoostStablecoin;
  let testUSD: MockERC20;
  let minter: Minter;
  let router: MockRouter;
  let v2Voter: IV2Voter;
  let factory: IFactory | IPoolFactory;
  let gauge: IGauge;
  let pool: IERC20;

  // Signers
  let admin: SignerWithAddress;
  let rewardVault: SignerWithAddress;
  let setter: SignerWithAddress;
  let amoBot: SignerWithAddress;
  let withdrawer: SignerWithAddress;
  let pauser: SignerWithAddress;
  let unpauser: SignerWithAddress;
  let rewardCollector: SignerWithAddress;
  let boostMinter: SignerWithAddress;
  let user: SignerWithAddress;

  // Roles
  let SETTER_ROLE: string;
  let AMO_ROLE: string;
  let WITHDRAWER_ROLE: string;
  let PAUSER_ROLE: string;
  let UNPAUSER_ROLE: string;
  let REWARD_COLLECTOR_ROLE: string;

  // Constants
  const V2_VOTER = "0x777034fEF3CCBed74536Ea1002faec9620deAe0A";
  const V2_FACTORY = "0x777de5Fe8117cAAA7B44f396E93a401Cf5c9D4d6";
  const WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2";

  // Amounts and addresses
  const boostDesired = ethers.parseUnits("11000000", 18);
  const usdDesired = ethers.parseUnits("11000000", 6);
  const boostMin4Liquidity = ethers.parseUnits("9990000", 18);
  const usdMin4Liquidity = ethers.parseUnits("9990000", 6);
  let boostAddress: string;
  let usdAddress: string;
  let minterAddress: string;
  let routerAddress: string;
  let poolAddress: string;
  let gaugeAddress: string;
  let amoAddress: string;
  const deadline = Math.floor(Date.now() / 1000) + 60 * 100;
  const delta = ethers.parseUnits("0.01", 6);

  const boostMultiplier = ethers.parseUnits("1.1", 6);
  const validRangeWidth = ethers.parseUnits("0.01", 6);
  const validRemovingRatio = ethers.parseUnits("1.01", 6);
  const boostLowerPriceSell = ethers.parseUnits("0.99", 6);
  const boostUpperPriceBuy = ethers.parseUnits("1.01", 6);
  const boostSellRatio = ethers.parseUnits("1", 6);
  const usdBuyRatio = ethers.parseUnits("1", 6);
  const params = [
    boostMultiplier,
    validRangeWidth,
    validRemovingRatio,
    boostLowerPriceSell,
    boostUpperPriceBuy,
    boostSellRatio,
    usdBuyRatio,
  ];
  // Setup functions
  async function deployBaseContracts() {
    [
      admin,
      rewardVault,
      setter,
      amoBot,
      withdrawer,
      pauser,
      unpauser,
      boostMinter,
      user,
      rewardCollector,
    ] = await ethers.getSigners();

    const BoostFactory = await ethers.getContractFactory("BoostStablecoin");
    boost = await upgrades.deployProxy(BoostFactory, [admin.address]);
    await boost.waitForDeployment();
    boostAddress = await boost.getAddress();

    const MockErc20Factory = await ethers.getContractFactory("MockERC20");
    testUSD = await MockErc20Factory.deploy("USD", "USD", 6);
    await testUSD.waitForDeployment();
    usdAddress = await testUSD.getAddress();

    const MinterFactory = await ethers.getContractFactory("Minter");
    minter = await upgrades.deployProxy(MinterFactory, [
      boostAddress,
      usdAddress,
      admin.address,
    ]);
    await minter.waitForDeployment();
    minterAddress = await minter.getAddress();

    // Mint tokens
    await boost.grantRole(await boost.MINTER_ROLE(), minterAddress);
    await boost.grantRole(await boost.MINTER_ROLE(), boostMinter.address);
    await boost.connect(boostMinter).mint(admin.address, boostDesired);
    await testUSD.connect(boostMinter).mint(admin.address, usdDesired);
  }

  async function setupSolidlyV2Environment() {
    try {
      factory = await ethers.getContractAt("IFactory", V2_FACTORY);
      v2Voter = await ethers.getContractAt(
        [
          "function governanceWhitelist(address[] calldata tokens) external",
          "function whitelist(address token, uint256 tokenId) external",
          "function isWhitelisted(address) external view returns (bool)",
          "function createGauge(address) external returns (address)",
          "function gauges(address) external view returns (address)",
        ],
        V2_VOTER,
      );

      const createPairTx = await factory
        .connect(admin)
        .createPair(boostAddress, usdAddress, stable);
      await createPairTx.wait();

      poolAddress = await factory.getPair(boostAddress, usdAddress, stable);

      const governor = "0x9006550fAC2fe75903f9a7457E0CcF996DAd396A";
      await network.provider.send("hardhat_setBalance", [
        governor,
        "0x1000000000000000000",
      ]);
      await network.provider.request({
        method: "hardhat_impersonateAccount",
        params: [governor],
      });

      const governorSigner = await ethers.getSigner(governor);

      // Attempt whitelisting
      try {
        await v2Voter
          .connect(governorSigner)
          .governanceWhitelist([boostAddress, usdAddress, poolAddress], {
            gasLimit: 5000000,
          });
      } catch (error) {}

      // Verify whitelist status
      const isBoostWhitelisted = await v2Voter.isWhitelisted(boostAddress);
      const isUsdWhitelisted = await v2Voter.isWhitelisted(usdAddress);
      const isPoolWhitelisted = await v2Voter.isWhitelisted(poolAddress);

      // Attempt to create gauge
      if (isBoostWhitelisted && isUsdWhitelisted && isPoolWhitelisted) {
        const createGaugeTx = await v2Voter
          .connect(governorSigner)
          .createGauge(poolAddress, { gasLimit: 5000000 });
        await createGaugeTx.wait();
        gaugeAddress = await v2Voter.gauges(poolAddress);
      } else {
        throw new Error("Failed to whitelist all required components");
      }

      const RouterFactory = await ethers.getContractFactory("MockRouter");
      router = await RouterFactory.deploy(V2_FACTORY, WETH);
      await router.waitForDeployment();
      routerAddress = await router.getAddress();

      await network.provider.request({
        method: "hardhat_stopImpersonatingAccount",
        params: [governor],
      });
    } catch (error) {
      console.error("Setup failed:", error);
      throw error;
    }

    // Deploy AMO
    const SolidlyV2LiquidityAMOFactory =
      await ethers.getContractFactory("V2AMO");
    v2AMO = await upgrades.deployProxy(
      SolidlyV2LiquidityAMOFactory,
      [
        admin.address,
        boostAddress,
        usdAddress,
        stable,
        poolFee,
        0, // SOLIDLY_V2
        minterAddress,
        ethers.ZeroAddress, // No factory needed for Solidly
        routerAddress,
        gaugeAddress,
        rewardVault.address,
        0, // tokenId
        false, // useTokenId
        boostMultiplier,
        validRangeWidth,
        validRemovingRatio,
        boostLowerPriceSell,
        boostUpperPriceBuy,
        boostSellRatio,
        usdBuyRatio,
      ],
      {
        initializer:
          "initialize(address,address,address,bool,uint256,uint8,address,address,address,address,address,uint256,bool,uint256,uint24,uint24,uint256,uint256,uint256,uint256)",
        timeout: 0,
      },
    );
    await v2AMO.waitForDeployment();
    amoAddress = await v2AMO.getAddress();
  }

  async function provideLiquidity() {
    await boost.approve(routerAddress, boostDesired);
    await testUSD.approve(routerAddress, usdDesired);

    await router
      .connect(admin)
      .addLiquidity(
        usdAddress,
        boostAddress,
        stable,
        usdDesired,
        boostDesired,
        usdMin4Liquidity,
        boostMin4Liquidity,
        amoAddress,
        deadline,
      );
  }

  async function depositToGauge() {
    pool = await ethers.getContractAt(
      "@openzeppelin/contracts/token/ERC20/IERC20.sol:IERC20",
      poolAddress,
    );
    let lpBalance = await pool.balanceOf(amoAddress);
    await network.provider.request({
      method: "hardhat_impersonateAccount",
      params: [amoAddress],
    });
    await setBalance(amoAddress, ethers.parseEther("1"));
    const amoSigner = await ethers.getSigner(amoAddress);
    await pool.connect(amoSigner).approve(gaugeAddress, lpBalance);
    gauge = await ethers.getContractAt("IGauge", gaugeAddress);
    await gauge.connect(amoSigner)["deposit(uint256,uint256)"](lpBalance, 0);
    await network.provider.request({
      method: "hardhat_stopImpersonatingAccount",
      params: [amoAddress],
    });
  }

  async function setupRoles() {
    SETTER_ROLE = await v2AMO.SETTER_ROLE();
    AMO_ROLE = await v2AMO.AMO_ROLE();
    WITHDRAWER_ROLE = await v2AMO.WITHDRAWER_ROLE();
    PAUSER_ROLE = await v2AMO.PAUSER_ROLE();
    UNPAUSER_ROLE = await v2AMO.UNPAUSER_ROLE();
    REWARD_COLLECTOR_ROLE = await v2AMO.REWARD_COLLECTOR_ROLE();

    await v2AMO.grantRole(SETTER_ROLE, setter.address);
    await v2AMO.grantRole(AMO_ROLE, amoBot.address);
    await v2AMO.grantRole(WITHDRAWER_ROLE, withdrawer.address);
    await v2AMO.grantRole(PAUSER_ROLE, pauser.address);
    await v2AMO.grantRole(UNPAUSER_ROLE, unpauser.address);
    await v2AMO.grantRole(REWARD_COLLECTOR_ROLE, rewardCollector.address);
    await minter.grantRole(await minter.AMO_ROLE(), amoAddress);
  }

  before(async () => {
    await network.provider.request({
      method: "hardhat_reset",
      params: [
        {
          forking: {
            jsonRpcUrl: "https://ethereum-rpc.publicnode.com",
            blockNumber: 21328013,
          },
        },
      ],
    });
  });

  describe("SolidlyV2 Pool Tests", function () {
    beforeEach(async function () {
      await deployBaseContracts();
      await setupSolidlyV2Environment();
      await provideLiquidity();
      await depositToGauge();
      await setupRoles();
    });
    describe("Initialization", function () {
      it("should initialize with correct parameters", async function () {
        expect(await v2AMO.boost()).to.equal(boostAddress);
        expect(await v2AMO.usd()).to.equal(usdAddress);
        expect(await v2AMO.boostMinter()).to.equal(minterAddress);
      });

      it("Should set correct roles", async function () {
        expect(await v2AMO.hasRole(SETTER_ROLE, setter.address)).to.be.true;
        expect(await v2AMO.hasRole(AMO_ROLE, amoBot.address)).to.be.true;
        expect(await v2AMO.hasRole(WITHDRAWER_ROLE, withdrawer.address)).to.be
          .true;
        expect(await v2AMO.hasRole(PAUSER_ROLE, pauser.address)).to.be.true;
        expect(await v2AMO.hasRole(UNPAUSER_ROLE, unpauser.address)).to.be.true;
      });
    });

    describe("Setter Role Actions", function () {
      describe("setParams", function () {
        it("Should set params correctly", async function () {
          await expect(
            v2AMO
              .connect(setter)
              .setParams(
                boostMultiplier,
                validRangeWidth,
                validRemovingRatio,
                boostLowerPriceSell,
                boostUpperPriceBuy,
                boostSellRatio,
                usdBuyRatio,
              ),
          )
            .to.emit(v2AMO, "ParamsSet")
            .withArgs(
              boostMultiplier,
              validRangeWidth,
              validRemovingRatio,
              boostLowerPriceSell,
              boostUpperPriceBuy,
              boostSellRatio,
              usdBuyRatio,
            );

          expect(await v2AMO.boostMultiplier()).to.equal(boostMultiplier);
          expect(await v2AMO.validRangeWidth()).to.equal(validRangeWidth);
          expect(await v2AMO.validRemovingRatio()).to.equal(validRemovingRatio);
          expect(await v2AMO.boostSellRatio()).to.equal(boostSellRatio);
          expect(await v2AMO.usdBuyRatio()).to.equal(usdBuyRatio);
          expect(await v2AMO.boostLowerPriceSell()).to.equal(
            boostLowerPriceSell,
          );
          expect(await v2AMO.boostUpperPriceBuy()).to.equal(boostUpperPriceBuy);
        });

        it("Should revert when called by non-setter", async function () {
          await expect(
            v2AMO
              .connect(user)
              .setParams(
                boostMultiplier,
                validRangeWidth,
                validRemovingRatio,
                boostLowerPriceSell,
                boostUpperPriceBuy,
                boostSellRatio,
                usdBuyRatio,
              ),
          ).to.be.revertedWith(
            `AccessControl: account ${user.address.toLowerCase()} is missing role ${SETTER_ROLE}`,
          );
        });

        it("Should revert when value is out of range", async function () {
          await expect(
            v2AMO
              .connect(setter)
              .setParams(
                boostMultiplier,
                ethers.parseUnits("1.1", 6),
                validRemovingRatio,
                boostLowerPriceSell,
                boostUpperPriceBuy,
                boostSellRatio,
                usdBuyRatio,
              ),
          ).to.be.revertedWithCustomError(v2AMO, "InvalidRatioValue");

          await expect(
            v2AMO
              .connect(setter)
              .setParams(
                boostMultiplier,
                validRangeWidth,
                ethers.parseUnits("0.99", 6),
                boostLowerPriceSell,
                boostUpperPriceBuy,
                boostSellRatio,
                usdBuyRatio,
              ),
          ).to.be.revertedWithCustomError(v2AMO, "InvalidRatioValue");
        });
      });
    });

    describe("AMO Role Actions", function () {
      describe("mintAndSellBoost", function () {
        it("Should execute mintAndSellBoost successfully", async function () {
          // Setup price above peg first
          const usdToBuy = ethers.parseUnits("1000000", 6);
          await testUSD.connect(admin).mint(user.address, usdToBuy);
          await testUSD.connect(user).approve(routerAddress, usdToBuy);

          const routeBuyBoost = [
            {
              from: usdAddress,
              to: boostAddress,
              stable: stable,
            },
          ];

          await router.connect(user).swapExactTokensForTokens(
            usdToBuy,
            0, // min amount out
            routeBuyBoost,
            user.address,
            deadline,
          );

          // Test mintAndSellBoost
          // const boostAmount = ethers.parseUnits("990000", 18);
          const [boostReserve, usdReserve] = await v2AMO.getReserves();
          console.log(boostReserve);
          console.log(usdReserve);
          let boostAmount =
            BigInt(Math.sqrt(Number(usdReserve * boostReserve))) - boostReserve;
          boostAmount += (boostAmount * BigInt(poolFee)) / 1000000n;

          // Grant necessary roles
          await v2AMO.grantRole(AMO_ROLE, amoBot.address);
          await minter.grantRole(
            await minter.AMO_ROLE(),
            await v2AMO.getAddress(),
          );

          // Test unauthorized access
          await expect(
            v2AMO.connect(user).mintAndSellBoost(boostAmount),
          ).to.be.revertedWith(
            `AccessControl: account ${user.address.toLowerCase()} is missing role ${AMO_ROLE}`,
          );

          // Test authorized access
          await expect(
            v2AMO.connect(amoBot).mintAndSellBoost(boostAmount),
          ).to.emit(v2AMO, "MintSell");
        });
        it("Should revert mintAndSellBoost when called by non-amo", async function () {
          const boostAmount = ethers.parseUnits("990000", 18);
          await expect(
            v2AMO.connect(user).mintAndSellBoost(boostAmount),
          ).to.be.revertedWith(
            `AccessControl: account ${user.address.toLowerCase()} is missing role ${AMO_ROLE}`,
          );
        });
      });

      it("Should revert mintSellFarm when called by non-amo", async function () {
        const boostAmount = ethers.parseUnits("990000", 18);
        const usdAmount = ethers.parseUnits("980000", 6);
        await expect(
          v2AMO.connect(user).mintSellFarm(boostAmount, 1, 1),
        ).to.be.revertedWith(
          `AccessControl: account ${user.address.toLowerCase()} is missing role ${AMO_ROLE}`,
        );
      });

      it("Should revert unfarmBuyBurn when called by non-amo", async function () {
        // Use revertedWith for access control error
        await expect(
          v2AMO
            .connect(user)
            .unfarmBuyBurn(ethers.parseUnits("1000000", 18), 1, 1),
        ).to.be.revertedWith(
          `AccessControl: account ${user.address.toLowerCase()} is missing role ${AMO_ROLE}`,
        );
      });

      it("should only allow PAUSER_ROLE to pause and UNPAUSER_ROLE to unpause", async function () {
        // Grant roles
        await v2AMO.grantRole(AMO_ROLE, amoBot.address);
        await v2AMO.grantRole(PAUSER_ROLE, pauser.address);
        await v2AMO.grantRole(UNPAUSER_ROLE, unpauser.address);

        // Test pause
        await expect(v2AMO.connect(pauser).pause())
          .to.emit(v2AMO, "Paused")
          .withArgs(pauser.address);

        // Test operation while paused
        const boostAmount = ethers.parseUnits("1000", 18);
        await expect(
          v2AMO.connect(amoBot).mintAndSellBoost(boostAmount),
        ).to.be.revertedWith("Pausable: paused");

        // Test unpause
        await expect(v2AMO.connect(unpauser).unpause())
          .to.emit(v2AMO, "Unpaused")
          .withArgs(unpauser.address);
      });

      it("should allow WITHDRAWER_ROLE to withdraw ERC20 tokens", async function () {
        // Transfer some tokens to the contract
        await testUSD
          .connect(user)
          .mint(amoAddress, ethers.parseUnits("1000", 6));

        // Try withdrawing tokens without WITHDRAWER_ROLE
        await expect(
          v2AMO
            .connect(user)
            .withdrawERC20(
              usdAddress,
              ethers.parseUnits("1000", 6),
              user.address,
            ),
        ).to.be.revertedWith(
          `AccessControl: account ${user.address.toLowerCase()} is missing role ${WITHDRAWER_ROLE}`,
        );

        // Withdraw tokens with WITHDRAWER_ROLE
        await v2AMO
          .connect(withdrawer)
          .withdrawERC20(
            usdAddress,
            ethers.parseUnits("1000", 6),
            user.address,
          );
        const usdBalanceOfUser = await testUSD.balanceOf(
          await user.getAddress(),
        );
        expect(usdBalanceOfUser).to.be.equal(ethers.parseUnits("1000", 6));
      });

      it("should execute unfarmBuyBurn succesfully", async function () {
        const boostToBuy = ethers.parseUnits(toBuy, 18);
        const minUsdReceive = ethers.parseUnits("800000", 6);
        const routeSellBoost = [
          {
            from: boostAddress,
            to: usdAddress,
            stable: stable,
          },
        ];
        await boost.connect(boostMinter).mint(user.address, boostToBuy);
        await boost.connect(user).approve(routerAddress, boostToBuy);
        await router
          .connect(user)
          .swapExactTokensForTokens(
            boostToBuy,
            minUsdReceive,
            routeSellBoost,
            user.address,
            deadline,
          );

        expect(await v2AMO.boostPrice()).to.be.lt(ethers.parseUnits("1", 6));

        await expect(v2AMO.connect(user).unfarmBuyBurn()).to.be.emit(
          v2AMO,
          "PublicUnfarmBuyBurnExecuted",
        );
        expect(await v2AMO.boostPrice()).to.be.approximately(
          ethers.parseUnits("1", 6),
          delta,
        );
      });

      it("should correctly return boostPrice", async function () {
        expect(await v2AMO.boostPrice()).to.be.approximately(
          ethers.parseUnits("1", 6),
          delta,
        );
      });

      describe("addLiquidity", function () {
        it("should addLiquidity succesfully ", async function () {
          // Setup amounts
          const usdAmountToAdd = ethers.parseUnits("1000", 6);
          const boostMinAmount = ethers.parseUnits("900", 18);
          const usdMinAmount = ethers.parseUnits("900", 6);

          try {
            // 1. Setup roles and permissions
            await v2AMO.grantRole(AMO_ROLE, amoBot.address);
            await minter.grantRole(await minter.AMO_ROLE(), v2AMO.getAddress());

            // 2. Setup parameters with tokenId

            await v2AMO.connect(setter).setTokenId(0, true); // Set tokenId to 0 and enable it

            // 3. Mint USD to AMO contract
            await testUSD
              .connect(admin)
              .mint(await v2AMO.getAddress(), usdAmountToAdd);

            // Test non-AMO role access
            await expect(
              v2AMO
                .connect(user)
                .addLiquidity(usdAmountToAdd, boostMinAmount, usdMinAmount),
            ).to.be.revertedWith(
              `AccessControl: account ${user.address.toLowerCase()} is missing role ${AMO_ROLE}`,
            );

            // Execute addLiquidity with AMO role
            const tx = await v2AMO
              .connect(amoBot)
              .addLiquidity(usdAmountToAdd, boostMinAmount, usdMinAmount, {
                gasLimit: 5000000,
              });

            const receipt = await tx.wait();
          } catch (error) {
            console.log("\nDetailed error information:");
            console.log("Error message:", error.message);
            console.log("Error data:", error.data || "No error data");
            console.log(
              "Transaction hash:",
              error.transactionHash || "No transaction hash",
            );
            throw error;
          }
        });

        it("Should revert addLiquidity when called by non-amo", async function () {
          const usdBalance = ethers.parseUnits("980000", 6);
          await expect(
            v2AMO.connect(user).addLiquidity(usdBalance, 1, 1),
          ).to.be.revertedWith(
            `AccessControl: account ${user.address.toLowerCase()} is missing role ${AMO_ROLE}`,
          );
        });
      });

      describe("mintSellFarm", function () {
        it("Should revert mintSellFarm when called by non-amo", async function () {
          const boostAmount = ethers.parseUnits("990000", 18);
          await expect(
            v2AMO.connect(user).mintSellFarm(boostAmount, 1, 1),
          ).to.be.revertedWith(
            `AccessControl: account ${user.address.toLowerCase()} is missing role ${AMO_ROLE}`,
          );
        });
      });

      describe("unfarmBuyBurn", function () {
        it("should execute unfarmBuyBurn succesfully", async function () {
          const boostToBuy = ethers.parseUnits(toBuy, 18);
          const minUsdReceive = ethers.parseUnits("800000", 6);
          const routeSellBoost = [
            {
              from: boostAddress,
              to: usdAddress,
              stable: stable,
            },
          ];
          await boost.connect(boostMinter).mint(user.address, boostToBuy);
          await boost.connect(user).approve(routerAddress, boostToBuy);
          await router
            .connect(user)
            .swapExactTokensForTokens(
              boostToBuy,
              minUsdReceive,
              routeSellBoost,
              user.address,
              deadline,
            );

          expect(await v2AMO.boostPrice()).to.be.lt(ethers.parseUnits("1", 6));

          await expect(v2AMO.connect(user).unfarmBuyBurn()).to.be.emit(
            v2AMO,
            "PublicUnfarmBuyBurnExecuted",
          );
          expect(await v2AMO.boostPrice()).to.be.approximately(
            ethers.parseUnits("1", 6),
            delta,
          );
        });
      });
    });

    describe("Public AMO Functions", function () {
      describe("mintSellFarm", function () {
        it("should execute public mintSellFarm when price above 1", async function () {
          try {
            // 1. Setup initial parameters with BigNumber
            const usdToBuy = ethers.parseUnits(toBuy, 6);

            // 2. Setup required parameters
            await v2AMO.connect(setter).setTokenId(0, true);

            await minter.grantRole(await minter.AMO_ROLE(), v2AMO.getAddress());

            // Mint and approve USD
            await testUSD.connect(admin).mint(user.address, usdToBuy);
            await testUSD.connect(user).approve(routerAddress, usdToBuy);

            // Get initial reserves
            const [initialBoostReserve, initialUsdReserve] =
              await v2AMO.getReserves();

            // Create initial imbalance
            const routeBuyBoost = [
              {
                from: usdAddress,
                to: boostAddress,
                stable: stable,
              },
            ];

            const amountsOut = await router.getAmountsOut(
              usdToBuy,
              routeBuyBoost,
            );
            const expectedOutput = amountsOut[amountsOut.length - 1];

            // Set a lower minOutput to allow for slippage using BigInt arithmetic
            const minOutput = (expectedOutput * 99n) / 100n; // 5% slippage tolerance

            await router
              .connect(user)
              .swapExactTokensForTokens(
                usdToBuy,
                minOutput,
                routeBuyBoost,
                user.address,
                deadline,
              );

            // 4. Verify price and prepare for mintSellFarm
            const priceBeforeOperation = await v2AMO.boostPrice();
            expect(priceBeforeOperation).to.be.gt(ethers.parseUnits("1", 6));

            // Get reserves before mintSellFarm
            const [boostReserve, usdReserve] = await v2AMO.getReserves();

            // 5. Setup approvals for mintSellFarm
            const maxApproval = ethers.parseUnits("1000000000", 18);
            await boost.connect(admin).approve(routerAddress, maxApproval);
            await testUSD.connect(admin).approve(routerAddress, maxApproval);

            // 6. Execute mintSellFarm
            const tx = await v2AMO.connect(user).mintSellFarm({
              gasLimit: 5000000,
            });

            const receipt = await tx.wait();

            // 7. Verify final state
            const finalPrice = await v2AMO.boostPrice();

            expect(finalPrice).to.be.approximately(
              ethers.parseUnits("1", 6),
              ethers.parseUnits("0.01", 6),
            );
          } catch (error) {
            console.log("\nDetailed error information:");
            console.log("Error message:", error.message);
            console.log("Stack trace:", error.stack);
            throw error;
          }
        });

        it("Should revert unfarmBuyBurn when price is 1", async function () {
          // Use amoBot instead of amoAddress
          await expect(
            v2AMO.connect(amoBot).unfarmBuyBurn(),
          ).to.be.revertedWithCustomError(v2AMO, "InvalidReserveRatio");
        });
      });

      describe("unfarmBuyBurn", function () {
        it("should execute unfarmBuyBurn below 1", async function () {
          const boostToBuy = ethers.parseUnits(toBuy, 18);
          const minUsdReceive = ethers.parseUnits("800000", 6);
          const routeSellBoost = [
            {
              from: boostAddress,
              to: usdAddress,
              stable: stable,
            },
          ];
          await boost.connect(boostMinter).mint(user.address, boostToBuy);
          await boost.connect(user).approve(routerAddress, boostToBuy);
          await router
            .connect(user)
            .swapExactTokensForTokens(
              boostToBuy,
              minUsdReceive,
              routeSellBoost,
              user.address,
              deadline,
            );

          expect(await v2AMO.boostPrice()).to.be.lt(ethers.parseUnits("1", 6));

          await expect(v2AMO.connect(user).unfarmBuyBurn()).to.be.emit(
            v2AMO,
            "PublicUnfarmBuyBurnExecuted",
          );
          expect(await v2AMO.boostPrice()).to.be.approximately(
            ethers.parseUnits("1", 6),
            delta,
          );
        });

        it("Should revert mintSellFarm when price is 1", async function () {
          // Use amoBot instead of amoAddress since it's a proper signer
          await expect(
            v2AMO.connect(amoBot).mintSellFarm(),
          ).to.be.revertedWithCustomError(v2AMO, "InvalidReserveRatio");
        });
      });

      describe("MasterAMO DAO Functions", function () {
        describe("pause", function () {
          it("should allow pauser to pause the contract", async function () {
            await expect(v2AMO.connect(pauser).pause()).to.not.be.reverted;
            expect(await v2AMO.paused()).to.equal(true);
          });

          it("should not allow non-pauser to pause the contract", async function () {
            const reverteMessage = `AccessControl: account ${user.address.toLowerCase()} is missing role ${PAUSER_ROLE}`;
            await expect(v2AMO.connect(user).pause()).to.be.revertedWith(
              reverteMessage,
            );
          });

          it("should not allow operations when paused", async function () {
            // Use proper signers instead of addresses
            await v2AMO.connect(pauser).pause();

            const boostAmount = ethers.parseUnits("990000", 18);
            const usdBalance = await testUSD.balanceOf(
              await v2AMO.getAddress(),
            );

            // Use amoBot for all operations
            await expect(
              v2AMO.connect(amoBot).mintAndSellBoost(boostAmount),
            ).to.be.revertedWith("Pausable: paused");

            await expect(
              v2AMO.connect(amoBot).addLiquidity(usdBalance, 1, 1),
            ).to.be.revertedWith("Pausable: paused");

            await expect(
              v2AMO.connect(amoBot).mintSellFarm(),
            ).to.be.revertedWith("Pausable: paused");

            await expect(
              v2AMO.connect(amoBot).unfarmBuyBurn(),
            ).to.be.revertedWith("Pausable: paused");
          });
        });
      });

      describe("unpause", function () {
        it("should allow unpauser to unpause the contract", async function () {
          await v2AMO.connect(pauser).pause();
          expect(await v2AMO.paused()).to.equal(true);

          await expect(v2AMO.connect(unpauser).unpause()).to.not.be.reverted;
          expect(await v2AMO.paused()).to.equal(false);
        });

        it("should not allow non-unpauser to unpause the contract", async function () {
          await v2AMO.connect(pauser).pause();
          const reverteMessage = `AccessControl: account ${user.address.toLowerCase()} is missing role ${UNPAUSER_ROLE}`;
          await expect(v2AMO.connect(user).unpause()).to.be.revertedWith(
            reverteMessage,
          );
        });
      });

      describe("should revert when invalid parameters are set", function () {
        for (const i of [1]) {
          it(`param on index ${i}`, async function () {
            let tempParams = [...params];
            tempParams[i] = ethers.parseUnits("1.00001", 6);
            await expect(
              v2AMO.connect(setter).setParams(...tempParams),
            ).to.be.revertedWithCustomError(v2AMO, "InvalidRatioValue");
          });
        }

        for (const i of [2]) {
          it(`param on index ${i}`, async function () {
            let tempParams = [...params];
            tempParams[i] = ethers.parseUnits("0.99999", 6);
            await expect(
              v2AMO.connect(setter).setParams(...tempParams),
            ).to.be.revertedWithCustomError(v2AMO, "InvalidRatioValue");
          });
        }
      });
      describe("get reward", async function () {
        const tokens = [];
        it("should revert when token is not whitelisted", async function () {});
        it("should revert for non-setter", async function () {
          await expect(
            v2AMO.connect(user).setWhitelistedTokens(tokens, true),
          ).to.be.revertedWith(
            `AccessControl: account ${user.address.toLowerCase()} is missing role ${SETTER_ROLE}`,
          );
        });
        it("should whitelist tokens", async function () {
          await expect(
            v2AMO.connect(setter).setWhitelistedTokens(tokens, true),
          ).to.emit(v2AMO, "RewardTokensSet");
        });
        it("should revert for non-reward_collector", async function () {
          await expect(
            v2AMO.connect(user).getReward(tokens, true),
          ).to.be.revertedWith(
            `AccessControl: account ${user.address.toLowerCase()} is missing role ${REWARD_COLLECTOR_ROLE}`,
          );
        });
        it("should get reward", async function () {
          await expect(
            v2AMO.connect(rewardCollector).getReward(tokens, true),
          ).to.emit(v2AMO, "GetReward");
        });
      });
    });
  });
});
