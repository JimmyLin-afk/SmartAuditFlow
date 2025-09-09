import { loadFixture } from '@nomicfoundation/hardhat-toolbox/network-helpers.js';
import { expect } from 'chai';
import hre from 'hardhat';
import { type ReputationMarket } from '../../typechain-types/index.js';
import { createDeployer, type EthosDeployer } from '../utils/deployEthos.js';
import { type EthosUser } from '../utils/ethosUser.js';
import { DEFAULT, MarketUser } from './utils.js';

const { ethers } = hre;

describe('ReputationMarket Base Price Tests', () => {
  let deployer: EthosDeployer;
  let userA: MarketUser;
  let reputationMarket: ReputationMarket;
  const DEFAULT_PRICE = 0.01;

  beforeEach(async () => {
    deployer = await loadFixture(createDeployer);

    if (!deployer.reputationMarket.contract) {
      throw new Error('ReputationMarket contract not found');
    }
    const ethosUserA = await deployer.createUser();
    await ethosUserA.setBalance('5000000000');

    userA = new MarketUser(ethosUserA.signer);
    reputationMarket = deployer.reputationMarket.contract;
    DEFAULT.reputationMarket = reputationMarket;
    DEFAULT.profileId = ethosUserA.profileId;
  });

  describe('Market Configuration Base Price', () => {
    it('should initialize with correct default base price', async () => {
      const config = await reputationMarket.marketConfigs(0);
      expect(config.basePrice).to.equal(ethers.parseEther(DEFAULT_PRICE.toString()));
    });

    it('should allow adding market config with higher base price', async () => {
      const higherBasePrice = ethers.parseEther((DEFAULT_PRICE * 2).toString());
      const liquidity = ethers.parseEther((DEFAULT_PRICE * 4).toString());
      await reputationMarket
        .connect(deployer.ADMIN)
        .addMarketConfig(liquidity, higherBasePrice, DEFAULT.creationCost);

      const newConfigIndex = (await reputationMarket.getMarketConfigCount()) - 1n;
      const config = await reputationMarket.marketConfigs(newConfigIndex);
      expect(config.basePrice).to.equal(higherBasePrice);
    });

    it('should revert when adding config with base price below DEFAULT_PRICE', async () => {
      const minimumBasePrice = await reputationMarket.MINIMUM_BASE_PRICE();

      await expect(
        reputationMarket
          .connect(deployer.ADMIN)
          .addMarketConfig(DEFAULT.liquidity, 1000n, minimumBasePrice - 1n),
      ).to.be.revertedWithCustomError(reputationMarket, 'InvalidMarketConfigOption');
    });

    it('should allow buying votes with a non-default base price', async () => {
      const doubleBasePrice = ethers.parseEther((DEFAULT_PRICE * 2).toString());
      await reputationMarket
        .connect(deployer.ADMIN)
        .addMarketConfig(DEFAULT.liquidity, doubleBasePrice, DEFAULT.creationCost);
      const configCount = await reputationMarket.getMarketConfigCount();
      await reputationMarket
        .connect(deployer.ADMIN)
        .createMarketWithConfigAdmin(userA.signer.address, configCount - 1n, {
          value: DEFAULT.creationCost,
        });
      const trustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      expect(trustPrice).to.equal(doubleBasePrice / 2n);
      const result = await userA.buyOneVote({
        profileId: DEFAULT.profileId,
        buyAmount: DEFAULT.buyAmount * 5n,
      });
      expect(result.fundsPaid).to.greaterThan(doubleBasePrice / 2n);
      expect(result.fundsPaid).to.lessThan(doubleBasePrice);
    });
  });

  describe('Market Creation with Different Base Prices', () => {
    let users: EthosUser[];
    const prices: bigint[] = [];
    let configCount: bigint;

    beforeEach(async () => {
      prices.push(ethers.parseEther(DEFAULT_PRICE.toString()));
      prices.push(ethers.parseEther((DEFAULT_PRICE * 2).toString()));
      prices.push(ethers.parseEther((DEFAULT_PRICE * 5).toString()));
      // remove two default configs
      await reputationMarket.connect(deployer.ADMIN).removeMarketConfig(2n);
      await reputationMarket.connect(deployer.ADMIN).removeMarketConfig(1n);
      // add new configs
      await reputationMarket
        .connect(deployer.ADMIN)
        .addMarketConfig(DEFAULT.liquidity, prices[1], DEFAULT.creationCost);
      await reputationMarket
        .connect(deployer.ADMIN)
        .addMarketConfig(DEFAULT.liquidity, prices[2], DEFAULT.creationCost);
      configCount = await reputationMarket.getMarketConfigCount();
      users = await Promise.all(
        Array.from({ length: Number(configCount) }, async () => await deployer.createUser()),
      );
      // Create markets with different base prices
      await Promise.all(
        users.map(async (user, i) => {
          await reputationMarket
            .connect(deployer.ADMIN)
            .createMarketWithConfigAdmin(user.signer.address, i, {
              value: DEFAULT.creationCost * 10n,
            });
        }),
      );
    });

    it('should create markets with different base prices and verify initial vote prices', async () => {
      for (let i = 0; i < users.length; i++) {
        const trustPrice = await reputationMarket.getVotePrice(users[i].profileId, true);
        expect(trustPrice).to.equal(prices[i] / 2n);
      }
    });

    it('should show proportional price changes across different base prices', async () => {
      // Get initial prices
      const initialPrices = await Promise.all(
        users.map(async (user) => await reputationMarket.getVotePrice(user.profileId, true)),
      );

      for (let i = 0; i < users.length; i++) {
        expect(initialPrices[i]).to.equal(prices[i] / 2n);
      }

      // Buy one vote in each market
      for (const user of users) {
        const marketUser = new MarketUser(user.signer);
        await marketUser.buyOneVote({
          profileId: user.profileId,
          buyAmount: DEFAULT.buyAmount * 5n,
        });
      }
      const afterPrices = await Promise.all(
        users.map(async (user) => await reputationMarket.getVotePrice(user.profileId, true)),
      );

      // Calculate percentage changes
      const percentageChanges = initialPrices.map((initialPrice, i) => {
        const priceChange = afterPrices[i] - initialPrice;

        // Convert to percentage (multiplied by 100_000 for precision since we're working with BigInts)
        return (priceChange * 100_000n) / initialPrice;
      });

      for (let i = 1; i < percentageChanges.length; i++) {
        const difference = percentageChanges[i] - percentageChanges[0];
        // allow 5 / 100,000 = 0.005% difference to account for rounding
        expect(difference).to.be.within(
          -5n,
          5n,
          `Price change not proportional: market 0: ${percentageChanges[0]}%, market ${i}: ${percentageChanges[i]}%`,
        );
      }
    });
  });

  describe('Very high price limits', () => {
    let basePrice100x: bigint;
    beforeEach(async () => {
      basePrice100x = ethers.parseEther((DEFAULT_PRICE * 100).toString());
      await reputationMarket
        .connect(deployer.ADMIN)
        .addMarketConfig(DEFAULT.liquidity, basePrice100x, DEFAULT.creationCost);
      const configCount = await reputationMarket.getMarketConfigCount();

      await reputationMarket
        .connect(deployer.ADMIN)
        .createMarketWithConfigAdmin(userA.signer.address, configCount - 1n, {
          value: DEFAULT.creationCost,
        });
    });

    it('should respect a high base price as maximum for trust votes', async () => {
      const billionEth = ethers.parseEther('1000000000');
      const maxVotes = 133000n - 1n;
      // buy many votes to push price up
      await userA.buyVotes({ votesToBuy: maxVotes, buyAmount: billionEth });
      const currentPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);

      expect(Number(currentPrice)).to.be.lte(Number(basePrice100x), 'Price is too high'); // less than maximum
      expect(Number(currentPrice)).to.be.gt(
        Number((basePrice100x * 99n) / 100n),
        'Price is too low',
      ); // more than 99% of maximum
    });

    it('should maintain zero as minimum price for distrust votes', async () => {
      const billionEth = ethers.parseEther('1000000000');
      const maxVotes = 133000n - 1n;
      // Buy many distrust votes to push trust price down
      await userA.buyVotes({
        votesToBuy: maxVotes,
        buyAmount: billionEth,
        isPositive: false,
      });
      const trustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      expect(Number(trustPrice)).to.be.gte(0, 'Price is too low');
      expect(Number(trustPrice)).to.be.lte(
        Number((basePrice100x * 1n) / 100n),
        'Price is too high',
      ); // less than 1% of base price
    });
  });
});
