import { loadFixture } from '@nomicfoundation/hardhat-toolbox/network-helpers.js';
import { expect } from 'chai';
import hre from 'hardhat';
import { type ReputationMarket } from '../../typechain-types/index.js';
import { createDeployer, type EthosDeployer } from '../utils/deployEthos.js';
import { type EthosUser } from '../utils/ethosUser.js';
import { DEFAULT, MarketUser } from './utils.js';

const { ethers } = hre;

describe('ReputationMarket Creation Config', () => {
  let deployer: EthosDeployer;
  let userA: MarketUser;
  let ethosUserA: EthosUser;
  let reputationMarket: ReputationMarket;

  beforeEach(async () => {
    deployer = await loadFixture(createDeployer);

    if (!deployer.reputationMarket.contract) {
      throw new Error('ReputationMarket contract not found');
    }
    ethosUserA = await deployer.createUser();
    await ethosUserA.setBalance('2000');

    userA = new MarketUser(ethosUserA.signer);
    reputationMarket = deployer.reputationMarket.contract;
    DEFAULT.reputationMarket = reputationMarket;
    DEFAULT.profileId = ethosUserA.profileId;
  });

  describe('Market Config Management', () => {
    it('should initialize with three default market configs', async () => {
      const defaultPrice = await reputationMarket.DEFAULT_PRICE();

      const config0 = await reputationMarket.marketConfigs(0);
      expect(config0.liquidity).to.equal(1000n);
      expect(config0.basePrice).to.equal(defaultPrice);
      expect(config0.creationCost).to.equal(ethers.parseEther('0.2'));

      const config1 = await reputationMarket.marketConfigs(1);
      expect(config1.liquidity).to.equal(10000n);
      expect(config1.basePrice).to.equal(defaultPrice);
      expect(config1.creationCost).to.equal(ethers.parseEther('0.5'));

      const config2 = await reputationMarket.marketConfigs(2);
      expect(config2.liquidity).to.equal(100000n);
      expect(config2.basePrice).to.equal(defaultPrice);
      expect(config2.creationCost).to.equal(ethers.parseEther('1.0'));
    });

    it('should allow admin to add new valid market config', async () => {
      const priceMaximum = await reputationMarket.DEFAULT_PRICE();
      await reputationMarket
        .connect(deployer.ADMIN)
        .addMarketConfig(priceMaximum * 200n, priceMaximum, ethers.parseEther('0.1'));

      const newConfig = await reputationMarket.marketConfigs(3);
      expect(newConfig.liquidity).to.equal(priceMaximum * 200n);
      expect(newConfig.basePrice).to.equal(priceMaximum);
      expect(newConfig.creationCost).to.equal(ethers.parseEther('0.1'));
    });

    it('should allow admin to remove market config', async () => {
      const configsBefore = await reputationMarket.marketConfigs(1);
      await reputationMarket.connect(deployer.ADMIN).removeMarketConfig(1);
      const configsAfter = await reputationMarket.marketConfigs(1);

      // Should now have the last config in this position
      expect(configsAfter.liquidity).to.not.equal(configsBefore.liquidity);
    });

    it('should revert when adding config with insufficient initial liquidity', async () => {
      const priceMaximum = await reputationMarket.DEFAULT_PRICE();
      await expect(
        reputationMarket
          .connect(deployer.ADMIN)
          .addMarketConfig(priceMaximum / 2n, 1000, priceMaximum),
      ).to.be.revertedWithCustomError(reputationMarket, 'InvalidMarketConfigOption');
    });

    it('should revert when removing last remaining config', async () => {
      // Remove all but one config
      await reputationMarket.connect(deployer.ADMIN).removeMarketConfig(2);
      await reputationMarket.connect(deployer.ADMIN).removeMarketConfig(1);

      // Try to remove the last one
      await expect(
        reputationMarket.connect(deployer.ADMIN).removeMarketConfig(0),
      ).to.be.revertedWithCustomError(reputationMarket, 'InvalidMarketConfigOption');
    });

    it('should revert when non-admin tries to add or remove config', async () => {
      const priceMaximum = await reputationMarket.DEFAULT_PRICE();
      await expect(
        reputationMarket
          .connect(userA.signer)
          .addMarketConfig(priceMaximum * 2n, 1000, priceMaximum),
      ).to.be.revertedWithCustomError(reputationMarket, 'AccessControlUnauthorizedAccount');
      await expect(
        reputationMarket.connect(userA.signer).removeMarketConfig(0),
      ).to.be.revertedWithCustomError(reputationMarket, 'AccessControlUnauthorizedAccount');
    });

    it('should emit MarketConfigAdded event when adding new config', async () => {
      const priceMaximum = await reputationMarket.DEFAULT_PRICE();
      const newLiquidity = priceMaximum * 200n;
      const newCreationCost = ethers.parseEther('0.1');

      await expect(
        reputationMarket
          .connect(deployer.ADMIN)
          .addMarketConfig(newLiquidity, priceMaximum, newCreationCost),
      )
        .to.emit(reputationMarket, 'MarketConfigAdded')
        .withArgs(3, [newLiquidity, priceMaximum, newCreationCost]);
    });

    it('should emit MarketConfigRemoved event when removing config', async () => {
      const configToRemove = await reputationMarket.marketConfigs(1);

      await expect(reputationMarket.connect(deployer.ADMIN).removeMarketConfig(1))
        .to.emit(reputationMarket, 'MarketConfigRemoved')
        .withArgs(1, [
          configToRemove.liquidity,
          configToRemove.basePrice,
          configToRemove.creationCost,
        ]);
    });
  });

  describe('Market Creation with Config', () => {
    beforeEach(async () => {
      await reputationMarket.connect(deployer.ADMIN).setAllowListEnforcement(false);
    });

    it('should create market with default config when using createMarket', async () => {
      await reputationMarket.connect(userA.signer).createMarket({ value: DEFAULT.creationCost });

      const market = await reputationMarket.getMarket(DEFAULT.profileId);
      const config0 = await reputationMarket.marketConfigs(0);
      expect(market.basePrice).to.equal(config0.basePrice);
      expect(market.liquidityParameter).to.equal(config0.liquidity);
    });

    it('should create market with specific config', async () => {
      const configIndex = 1;
      const config1 = await reputationMarket.marketConfigs(configIndex);
      await reputationMarket.connect(userA.signer).createMarketWithConfig(configIndex, {
        value: config1.creationCost,
      });

      const market = await reputationMarket.getMarket(DEFAULT.profileId);
      expect(market.basePrice).to.equal(config1.basePrice);
      expect(market.liquidityParameter).to.equal(config1.liquidity);
    });

    it('should revert when using invalid config index', async () => {
      await expect(
        reputationMarket.connect(userA.signer).createMarketWithConfig(99, {
          value: DEFAULT.creationCost,
        }),
      ).to.be.revertedWithCustomError(reputationMarket, 'InvalidMarketConfigOption');
    });

    it('should revert when providing insufficient initial liquidity', async () => {
      const config1 = await reputationMarket.marketConfigs(1);
      await expect(
        reputationMarket
          .connect(userA.signer)
          .createMarketWithConfig(1, { value: config1.creationCost / 2n }),
      ).to.be.revertedWithCustomError(reputationMarket, 'InsufficientLiquidity');
    });

    it('should emit MarketCreated event with correct config when creating market', async () => {
      const config0 = await reputationMarket.marketConfigs(0);

      await expect(
        reputationMarket.connect(userA.signer).createMarket({ value: config0.creationCost }),
      )
        .to.emit(reputationMarket, 'MarketCreated')
        .withArgs(DEFAULT.profileId, ethosUserA.signer.address, [
          config0.liquidity,
          config0.basePrice,
          config0.creationCost,
        ]);
    });

    it('should emit MarketCreated event with correct config when using createMarketWithConfig', async () => {
      const config1 = await reputationMarket.marketConfigs(1);

      await expect(
        reputationMarket
          .connect(userA.signer)
          .createMarketWithConfig(1, { value: config1.creationCost }),
      )
        .to.emit(reputationMarket, 'MarketCreated')
        .withArgs(DEFAULT.profileId, ethosUserA.signer.address, [
          config1.liquidity,
          config1.basePrice,
          config1.creationCost,
        ]);
    });
  });
});
