import { loadFixture } from '@nomicfoundation/hardhat-toolbox/network-helpers.js';
import { expect } from 'chai';
import hre from 'hardhat';
import { type ReputationMarket } from '../../typechain-types/index.js';
import { createDeployer, type EthosDeployer } from '../utils/deployEthos.js';
import { type EthosUser } from '../utils/ethosUser.js';
import { DEFAULT, MarketUser } from './utils.js';

const { ethers } = hre;

describe('Reputation Market Bonding Curve', () => {
  let deployer: EthosDeployer;
  let ethosUserA: EthosUser;
  let ethosUserB: EthosUser;
  let userA: MarketUser;
  let userB: MarketUser;
  let reputationMarket: ReputationMarket;

  beforeEach(async () => {
    deployer = await loadFixture(createDeployer);

    if (!deployer.reputationMarket.contract) {
      throw new Error('ReputationMarket contract not found');
    }
    ethosUserA = await deployer.createUser();
    ethosUserB = await deployer.createUser();
    await ethosUserA.setBalance('200000');
    await ethosUserB.setBalance('200000');

    userA = new MarketUser(ethosUserA.signer);
    userB = new MarketUser(ethosUserB.signer);
    reputationMarket = deployer.reputationMarket.contract;
    DEFAULT.reputationMarket = reputationMarket;
    DEFAULT.profileId = ethosUserA.profileId;
    await reputationMarket
      .connect(deployer.ADMIN)
      .setUserAllowedToCreateMarket(DEFAULT.profileId, true);
    await reputationMarket.connect(userA.signer).createMarket({ value: DEFAULT.creationCost });
  });

  describe('Reflexivity & Price Feedback Loops', () => {
    it('should make subsequent trust purchases more expensive after initial trust purchases', async () => {
      const initialPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);

      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n });

      const priceAfterFirstPurchase = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      expect(priceAfterFirstPurchase).to.be.greaterThan(initialPrice);

      await userA.buyVotes();

      const priceAfterSecondPurchase = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      expect(priceAfterSecondPurchase).to.be.greaterThan(priceAfterFirstPurchase);
    });

    it('should make subsequent distrust purchases cheaper after initial trust purchases', async () => {
      const initialDistrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, false);

      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n });

      const distrustPriceAfterTrust = await reputationMarket.getVotePrice(DEFAULT.profileId, false);
      expect(distrustPriceAfterTrust).to.be.lessThan(initialDistrustPrice);

      await userA.buyVotes();

      const distrustPriceAfterMoreTrust = await reputationMarket.getVotePrice(
        DEFAULT.profileId,
        false,
      );
      expect(distrustPriceAfterMoreTrust).to.be.lessThan(distrustPriceAfterTrust);
    });

    it('should show diminishing price changes on repeated purchases of same size', async () => {
      const initialPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);

      await userA.buyVotes();
      const priceAfterFirst = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const firstPriceChange = priceAfterFirst - initialPrice;

      await userA.buyVotes();
      const priceAfterSecond = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const secondPriceChange = priceAfterSecond - priceAfterFirst;

      await userA.buyVotes();
      const priceAfterThird = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const thirdPriceChange = priceAfterThird - priceAfterSecond;

      expect(secondPriceChange).to.be.lessThan(firstPriceChange);
      expect(thirdPriceChange).to.be.lessThan(secondPriceChange);
    });

    it('should show accelerating price movement as position size increases', async () => {
      const initialPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);

      await userA.buyVotes();
      const priceAfterSmall = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const smallPriceChange = priceAfterSmall - initialPrice;

      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n });
      const priceAfterMedium = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const mediumPriceChange = priceAfterMedium - priceAfterSmall;

      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 10n });
      const priceAfterLarge = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const largePriceChange = priceAfterLarge - priceAfterMedium;

      expect(mediumPriceChange).to.be.greaterThan(
        (smallPriceChange * 15n) / 10n,
        'Medium price change is not greater than small price change * 1.5',
      );
      expect(largePriceChange).to.be.greaterThan(
        (smallPriceChange * 15n) / 10n,
        'Large price change is not greater than cumulative price change * 1.5',
      );
    });

    it('should maintain price stability after equal and opposite trades', async () => {
      const initialTrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const initialDistrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, false);

      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n });
      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n, isPositive: false });
      const intermediateTrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const intermediateDistrustPrice = await reputationMarket.getVotePrice(
        DEFAULT.profileId,
        false,
      );
      expect(intermediateTrustPrice).to.equal(initialTrustPrice);
      expect(intermediateDistrustPrice).to.equal(initialDistrustPrice);

      await userA.sellVotes({ sellVotes: DEFAULT.votesToBuy * 2n });
      await userA.sellVotes({ sellVotes: DEFAULT.votesToBuy * 2n, isPositive: false });

      const finalTrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const finalDistrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, false);

      expect(finalTrustPrice).to.equal(initialTrustPrice);
      expect(finalDistrustPrice).to.equal(initialDistrustPrice);
    });
  });

  describe('Multi-User Interactions', () => {
    it('should increase price when multiple users buy votes', async () => {
      const initialPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);

      await userA.buyVotes();
      const priceAfterFirstUser = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      expect(priceAfterFirstUser).to.be.greaterThan(initialPrice);

      await userB.buyVotes();
      const priceAfterSecondUser = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      expect(priceAfterSecondUser).to.be.greaterThan(priceAfterFirstUser);
    });

    it('should decrease price when multiple users sell votes', async () => {
      // First give both users some votes to sell
      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n });
      await userB.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n });
      const initialPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);

      await userA.sellVotes({ sellVotes: DEFAULT.votesToBuy * 2n });
      const priceAfterFirstUser = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      expect(priceAfterFirstUser).to.be.lessThan(initialPrice);

      await userB.sellVotes({ sellVotes: DEFAULT.votesToBuy * 2n });
      const priceAfterSecondUser = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      expect(priceAfterSecondUser).to.be.lessThan(priceAfterFirstUser);
    });

    it('should maintain equal price when buying and selling between users', async () => {
      // Give userB some votes to sell
      await userB.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n });
      const initialPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);

      // UserA buys while UserB sells same amount
      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 2n });
      await userB.sellVotes({ sellVotes: DEFAULT.votesToBuy * 2n });

      const finalPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      expect(finalPrice).to.equal(initialPrice);
    });

    it('should maintain equal price when buying trust and distrust between users', async () => {
      const initialTrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const initialDistrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, false);

      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n, isPositive: false });
      await userB.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n });

      const intermediateTrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const intermediateDistrustPrice = await reputationMarket.getVotePrice(
        DEFAULT.profileId,
        false,
      );
      expect(intermediateTrustPrice).to.equal(initialTrustPrice);
      expect(intermediateDistrustPrice).to.equal(initialDistrustPrice);

      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 2n });
      await userB.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 2n, isPositive: false });

      const finalTrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const finalDistrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, false);

      expect(finalTrustPrice).to.equal(initialTrustPrice);
      expect(finalDistrustPrice).to.equal(initialDistrustPrice);
    });

    it('should maintain equal price when selling trust and distrust between users', async () => {
      const initialTrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const initialDistrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, false);

      // Give users positions to sell
      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n });
      await userB.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n, isPositive: false });

      const intermediateTrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const intermediateDistrustPrice = await reputationMarket.getVotePrice(
        DEFAULT.profileId,
        false,
      );
      expect(intermediateTrustPrice).to.equal(initialTrustPrice);
      expect(intermediateDistrustPrice).to.equal(initialDistrustPrice);

      // Both users sell part of their positions
      await userA.sellVotes({ sellVotes: DEFAULT.votesToBuy * 2n });
      await userB.sellVotes({ sellVotes: DEFAULT.votesToBuy * 2n, isPositive: false });

      const finalTrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const finalDistrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, false);
      expect(finalTrustPrice).to.equal(initialTrustPrice);
      expect(finalDistrustPrice).to.equal(initialDistrustPrice);
    });
  });

  describe('Complex Trading Patterns', () => {
    it('should maintain market funds with arbitrary trading patterns', async () => {
      const initialMarketFunds = await reputationMarket.marketFunds(DEFAULT.profileId);
      const firstContractBalance = await ethers.provider.getBalance(reputationMarket.getAddress());

      const trades = [
        { isBuy: true, amount: DEFAULT.votesToBuy * 5n },
        { isBuy: true, amount: DEFAULT.votesToBuy * 2n, isPositive: false },
        { isBuy: false, amount: DEFAULT.votesToBuy * 2n },
        { isBuy: true, amount: DEFAULT.votesToBuy * 3n },
        { isBuy: true, amount: DEFAULT.votesToBuy * 4n, isPositive: false },
        { isBuy: false, amount: DEFAULT.votesToBuy },
      ];

      for (const trade of trades) {
        if (trade.isBuy) {
          await userA.buyVotes({ votesToBuy: trade.amount, isPositive: trade.isPositive });
        } else {
          await userA.sellVotes({ sellVotes: trade.amount, isPositive: trade.isPositive });
        }
      }

      // Calculate final positions
      let trust = 0n;
      let distrust = 0n;

      for (const trade of trades) {
        const amount = trade.isBuy ? trade.amount : -trade.amount;
        trade.isPositive === false ? (distrust += amount) : (trust += amount);
      }

      // Sell all remaining votes
      if (trust > 0) await userA.sellVotes({ sellVotes: trust });
      if (distrust > 0) await userA.sellVotes({ sellVotes: distrust, isPositive: false });

      const finalMarketFunds = await reputationMarket.marketFunds(DEFAULT.profileId);
      const finalContractBalance = await ethers.provider.getBalance(reputationMarket.getAddress());
      expect(finalMarketFunds).to.be.equal(initialMarketFunds, 'Market funds should be equal');
      expect(finalContractBalance).to.be.equal(
        firstContractBalance,
        'Contract balance should be equal',
      );
    });

    it('should maintain market funds when alternating small purchases and large sales', async () => {
      const initialMarketFunds = await reputationMarket.marketFunds(DEFAULT.profileId);
      const firstContractBalance = await ethers.provider.getBalance(reputationMarket.getAddress());

      for (let i = 1; i <= 50; i++) {
        await userA.buyVotes({ votesToBuy: 1n });
        await userA.buyVotes({ votesToBuy: 1n, isPositive: false });
      }

      await userA.sellVotes({ sellVotes: 50n });
      await userA.sellVotes({ sellVotes: 50n, isPositive: false });

      const finalMarketFunds = await reputationMarket.marketFunds(DEFAULT.profileId);
      const finalContractBalance = await ethers.provider.getBalance(reputationMarket.getAddress());
      expect(finalMarketFunds).to.be.equal(initialMarketFunds, 'Market funds should be equal');
      expect(finalContractBalance).to.be.equal(
        firstContractBalance,
        'Contract balance should be equal',
      );
    });

    it('should produce same final price whether done as many small trades or one large trade', async () => {
      const initialPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const initialFunds = await reputationMarket.marketFunds(DEFAULT.profileId);
      const firstContractBalance = await ethers.provider.getBalance(reputationMarket.getAddress());

      // First approach: many small trades
      for (let i = 1; i <= 100; i++) {
        await userA.buyVotes({ votesToBuy: 1n });
      }
      const priceAfterSmallTrades = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const fundsAfterSmallTrades = await reputationMarket.marketFunds(DEFAULT.profileId);
      await userA.sellVotes({ sellVotes: 100n });

      // Second approach: one large trade
      await userA.buyVotes({ votesToBuy: 100n });
      const priceAfterLargeTrade = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const fundsAfterLargeTrade = await reputationMarket.marketFunds(DEFAULT.profileId);
      await userA.sellVotes({ sellVotes: 100n });

      // Final state should match initial state
      const finalPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
      const finalFunds = await reputationMarket.marketFunds(DEFAULT.profileId);
      const finalContractBalance = await ethers.provider.getBalance(reputationMarket.getAddress());

      expect(priceAfterSmallTrades).to.equal(
        priceAfterLargeTrade,
        'Price after small trades should be equal to price after large trade',
      );
      expect(fundsAfterSmallTrades).to.equal(
        fundsAfterLargeTrade,
        'Funds after small trades should be equal to funds after large trade',
      );
      expect(finalPrice).to.equal(initialPrice, 'Final price should be equal to initial price');
      expect(finalFunds).to.equal(initialFunds, 'Final funds should be equal to initial funds');
      expect(finalContractBalance).to.be.equal(
        firstContractBalance,
        'Contract balance should be equal',
      );
    });
  });

  describe('Market Invariants', () => {
    it('should maintain trust + distrust prices equal to basePrice', async () => {
      const basePrice = await reputationMarket.marketConfigs(0).then((config) => config.basePrice);

      async function checkPriceInvariant(message: string): Promise<void> {
        const trustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);
        const distrustPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, false);
        const sum = trustPrice + distrustPrice;
        // We allow a 1 wei difference when rounding down
        expect(sum).to.be.oneOf([basePrice - 1n, basePrice], message);
      }

      await checkPriceInvariant('Initial state');

      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n });
      await checkPriceInvariant('After trust purchase');

      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 3n, isPositive: false });
      await checkPriceInvariant('After distrust purchase');

      await userA.sellVotes({ sellVotes: DEFAULT.votesToBuy * 2n });
      await checkPriceInvariant('After trust sale');
    });

    it('should maintain contract balance matching sum of all market funds', async () => {
      // Create a second market
      const secondProfileId = ethosUserB.profileId;
      await reputationMarket
        .connect(deployer.ADMIN)
        .setUserAllowedToCreateMarket(secondProfileId, true);
      await reputationMarket
        .connect(ethosUserB.signer)
        .createMarket({ value: DEFAULT.creationCost });

      // Do some trades on both markets
      await userA.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 5n });
      await userB.buyVotes({ votesToBuy: DEFAULT.votesToBuy * 3n, profileId: secondProfileId });

      const firstMarketFunds = await reputationMarket.marketFunds(DEFAULT.profileId);
      const secondMarketFunds = await reputationMarket.marketFunds(secondProfileId);
      const totalMarketFunds = firstMarketFunds + secondMarketFunds;
      const contractBalance = await ethers.provider.getBalance(reputationMarket.getAddress());

      expect(contractBalance).to.equal(totalMarketFunds);
    });
  });
});
