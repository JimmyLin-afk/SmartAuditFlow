import { loadFixture } from '@nomicfoundation/hardhat-toolbox/network-helpers.js';
import { expect } from 'chai';
import hre from 'hardhat';
import { type ReputationMarket } from '../../typechain-types/index.js';
import { createDeployer, type EthosDeployer } from '../utils/deployEthos.js';
import { type EthosUser } from '../utils/ethosUser.js';
import { DEFAULT, MarketUser } from './utils.js';

const { ethers } = hre;

describe('ReputationMarket', () => {
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
    await ethosUserA.setBalance('200000');
    ethosUserB = await deployer.createUser();
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

  describe('createMarket', () => {
    it('should create a new market for self', async () => {
      // check market is created
      const market = await reputationMarket.getMarket(DEFAULT.profileId);
      expect(market.profileId).to.equal(DEFAULT.profileId);
      expect(market.trustVotes).to.equal(1);
      expect(market.distrustVotes).to.equal(1);
      // Check number of votes for userA
      let { trustVotes, distrustVotes } = await userA.getVotes();
      expect(trustVotes).to.equal(0);
      expect(distrustVotes).to.equal(0);
      // Check number of votes for userB
      ({ trustVotes, distrustVotes } = await userB.getVotes());
      expect(trustVotes).to.equal(0);
      expect(distrustVotes).to.equal(0);
    });

    it('should revert with MarketAlreadyExists when creating a market that already exists', async () => {
      await expect(
        reputationMarket.connect(userA.signer).createMarket({ value: DEFAULT.creationCost }),
      )
        .to.be.revertedWithCustomError(reputationMarket, 'MarketAlreadyExists')
        .withArgs(DEFAULT.profileId);
    });

    it('should revert with MarketDoesNotExist when buying votes for a non-existent market', async () => {
      const nonExistentProfileId = 999n;
      await expect(userA.buyOneVote({ profileId: nonExistentProfileId }))
        .to.be.revertedWithCustomError(reputationMarket, 'MarketDoesNotExist')
        .withArgs(nonExistentProfileId);
    });

    it('should allow ADMIN to create a market for any profileId', async () => {
      await reputationMarket
        .connect(deployer.ADMIN)
        .createMarketWithConfigAdmin(ethosUserB.signer.address, 0, {
          value: DEFAULT.creationCost,
        });
      const market = await reputationMarket.getMarket(ethosUserB.profileId);

      expect(market.profileId).to.equal(ethosUserB.profileId);
      expect(market.trustVotes).to.equal(1);
      expect(market.distrustVotes).to.equal(1);
    });

    it('should revert when ADMIN attempts to create a market for an address that does not have a profile', async () => {
      const newWallet = await deployer.newWallet();
      await expect(
        reputationMarket.connect(deployer.ADMIN).createMarketWithConfigAdmin(newWallet.address, 0, {
          value: DEFAULT.creationCost,
        }),
      ).to.be.revertedWithCustomError(deployer.ethosProfile.contract, 'ProfileNotFoundForAddress');
    });

    it('should not allow ADMIN to create a market for an invalid profileId', async () => {
      await reputationMarket
        .connect(deployer.ADMIN)
        .createMarketWithConfigAdmin(ethosUserB.signer.address, 0, {
          value: DEFAULT.creationCost,
        });
      const market = await reputationMarket.getMarket(ethosUserB.profileId);

      expect(market.profileId).to.equal(ethosUserB.profileId);
      expect(market.trustVotes).to.equal(1);
      expect(market.distrustVotes).to.equal(1);
    });
  });

  it('should allow a user to buy positive votes', async () => {
    const { trustVotes: positive, distrustVotes: negative } = await userA.buyVotes();
    expect(positive).to.equal(DEFAULT.votesToBuy);
    expect(negative).to.equal(0n);
  });

  it('should allow a user to buy one positive vote', async () => {
    // buy positive votes
    await userA.buyOneVote();

    const { trustVotes, distrustVotes } = await userA.getVotes();
    expect(trustVotes).to.equal(1);
    expect(distrustVotes).to.equal(0);
  });

  it('should allow a user to buy one negative vote', async () => {
    // buy negative votes
    await userA.buyOneVote({
      isPositive: false,
    });

    const { trustVotes, distrustVotes } = await userA.getVotes();
    expect(trustVotes).to.equal(0);
    expect(distrustVotes).to.equal(1);
  });

  it('should allow a user to sell positive vote', async () => {
    // buy positive votes
    await userA.buyVotes({
      votesToBuy: 10n,
      minVotesToBuy: 10n,
    });

    const { trustVotes: positiveBefore } = await userA.getVotes();

    await userA.sellOneVote();

    const { trustVotes: positiveAfter } = await userA.getVotes();
    expect(positiveAfter).to.equal(positiveBefore - 1n);
  });

  it('should allow a user to sell negative stake', async () => {
    // buy negative votes
    await userA.buyVotes({
      isPositive: false,
    });

    const { distrustVotes: negativeBefore } = await userA.getVotes();

    await userA.sellOneVote({ isPositive: false });

    const { distrustVotes: negativeAfter } = await userA.getVotes();
    expect(negativeAfter).to.equal(negativeBefore - 1n);
  });

  it('should update the price of votes when buying', async () => {
    const priceBefore = await DEFAULT.reputationMarket.getVotePrice(
      DEFAULT.profileId,
      DEFAULT.isPositive,
    );

    await userA.buyOneVote();

    const priceAfter = await DEFAULT.reputationMarket.getVotePrice(
      DEFAULT.profileId,
      DEFAULT.isPositive,
    );
    expect(priceAfter).to.be.greaterThan(priceBefore);
  });

  it('should update the price of votes when selling', async () => {
    await userA.buyOneVote();
    const priceBefore = await DEFAULT.reputationMarket.getVotePrice(
      DEFAULT.profileId,
      DEFAULT.isPositive,
    );
    // sell positive votes
    await userA.sellOneVote();
    const priceAfter = await DEFAULT.reputationMarket.getVotePrice(
      DEFAULT.profileId,
      DEFAULT.isPositive,
    );
    expect(priceAfter).to.be.lessThan(priceBefore);
  });

  it('should pay the user exactly what they paid when selling votes', async () => {
    const { fundsPaid } = await userA.buyOneVote();
    const { fundsReceived } = await userA.sellOneVote();
    expect(fundsPaid).to.equal(fundsReceived);
  });

  it('should allow a user to sell multiple votes', async () => {
    const amountToBuy = DEFAULT.buyAmount * 100n;

    await userA.buyVotes({ buyAmount: amountToBuy });
    const { trustVotes: initialPositiveVotes, balance: initialBalance } = await userA.getVotes();
    const { trustVotes: finalPositiveVotes, balance: finalBalance, gas } = await userA.sellVotes();
    expect(initialPositiveVotes - finalPositiveVotes).to.equal(DEFAULT.sellVotes);
    const balanceDifference = finalBalance - initialBalance - gas;
    expect(balanceDifference).to.be.gt(0);
  });

  it('should correctly return user votes', async () => {
    // Buy some trust votes
    await userA.buyOneVote({
      isPositive: true,
    });

    // Buy some distrust votes
    await userA.buyOneVote({
      isPositive: false,
    });
    await userA.buyOneVote({
      isPositive: false,
    });

    // Get user votes directly from the contract
    const userVotes = await reputationMarket.getUserVotes(
      await userA.signer.getAddress(),
      DEFAULT.profileId,
    );

    // Check if the returned values match the expected votes
    expect(userVotes.profileId).to.equal(DEFAULT.profileId);
    expect(userVotes.trustVotes).to.equal(1n);
    expect(userVotes.distrustVotes).to.equal(2n);
  });

  it('should emit VotesBought event with correct parameters when buying votes', async () => {
    const buyAmount = ethers.parseEther('0.1');
    const { simulatedVotesBought, simulatedFundsPaid } = await userA.simulateBuy({ buyAmount });

    const address = await userA.signer.getAddress();
    const transaction = await reputationMarket
      .connect(userA.signer)
      .buyVotes(DEFAULT.profileId, DEFAULT.isPositive, simulatedVotesBought, 1n, {
        value: buyAmount,
      });
    await expect(transaction)
      .to.emit(reputationMarket, 'VotesBought')
      .withArgs(
        DEFAULT.profileId, // profileId
        address, // buyer
        DEFAULT.isPositive, // isPositive
        simulatedVotesBought, // amount
        simulatedFundsPaid, // funds
        (await transaction.getBlock())?.timestamp, // boughtAt
      );
  });

  it('should emit VotesSold event with correct parameters when selling votes', async () => {
    const buyAmount = ethers.parseEther('0.1');
    await userA.buyVotes({ buyAmount });

    const { simulatedVotesSold, simulatedFundsReceived } = await userA.simulateSell({
      sellVotes: DEFAULT.sellVotes,
    });

    const address = await userA.signer.getAddress();
    const transaction = await reputationMarket
      .connect(userA.signer)
      .sellVotes(DEFAULT.profileId, DEFAULT.isPositive, DEFAULT.sellVotes, 0n);

    await expect(transaction)
      .to.emit(reputationMarket, 'VotesSold')
      .withArgs(
        DEFAULT.profileId, // profileId
        address, // seller
        DEFAULT.isPositive, // isPositive
        simulatedVotesSold, // amount
        simulatedFundsReceived, // funds
        (await transaction.getBlock())?.timestamp, // soldAt
      );
  });

  describe('Slippage', () => {
    it('should succeed when price marginally changes from another user buying', async () => {
      // User A prepares to buy some votes
      const { simulatedVotesBought, simulatedFundsPaid } = await userA.simulateBuy({
        buyAmount: DEFAULT.buyAmount * 100n,
        minVotesToBuy: 100n,
        votesToBuy: 100n,
      });

      // UserB makes a tiny purchase to create minimal price impact
      await userB.buyVotes();

      // Should succeed with 1% slippage tolerance
      await expect(
        userA.buyVotes({
          buyAmount: simulatedFundsPaid,
          minVotesToBuy: (simulatedVotesBought * 99n) / 100n,
          votesToBuy: simulatedVotesBought,
        }),
      ).to.not.be.reverted;
    });

    it('should succeed with moderate price changes when given sufficient slippage', async () => {
      // User A prepares to buy some votes
      const { simulatedVotesBought, simulatedFundsPaid } = await userA.simulateBuy({
        buyAmount: DEFAULT.buyAmount * 100n,
        minVotesToBuy: 100n,
        votesToBuy: 100n,
      });

      // UserB makes a moderate purchase that impacts price
      await userB.buyVotes({
        buyAmount: DEFAULT.buyAmount * 5n,
        votesToBuy: 5n,
        minVotesToBuy: 5n,
      });

      // Should succeed with 5% slippage tolerance
      await expect(
        userA.buyVotes({
          buyAmount: simulatedFundsPaid,
          minVotesToBuy: (simulatedVotesBought * 95n) / 100n,
          votesToBuy: simulatedVotesBought,
        }),
      ).to.not.be.reverted;
    });

    it('should revert when another user buying significantly impacts price', async () => {
      const { simulatedVotesBought, simulatedFundsPaid } = await userA.simulateBuy({
        buyAmount: DEFAULT.buyAmount * 100n,
        minVotesToBuy: 100n,
        votesToBuy: 100n,
      });

      await userB.buyVotes({
        buyAmount: DEFAULT.buyAmount * 100n,
        votesToBuy: 100n,
        minVotesToBuy: 100n,
      });

      await expect(
        userA.buyVotes({
          buyAmount: simulatedFundsPaid,
          minVotesToBuy: simulatedVotesBought,
          votesToBuy: simulatedVotesBought,
        }),
      ).to.be.revertedWithCustomError(reputationMarket, 'InsufficientFunds');
    });

    it('should revert when another buys a single vote with 0% slippage', async () => {
      const { simulatedVotesBought, simulatedFundsPaid } = await userA.simulateBuy({
        buyAmount: DEFAULT.buyAmount * 100n,
        minVotesToBuy: 100n,
        votesToBuy: 100n,
      });

      await userB.buyVotes({
        buyAmount: DEFAULT.buyAmount * 100n,
        votesToBuy: 100n,
        minVotesToBuy: 100n,
      });

      await expect(
        userA.buyVotes({
          buyAmount: simulatedFundsPaid,
          minVotesToBuy: simulatedVotesBought,
          votesToBuy: simulatedVotesBought,
        }),
      ).to.be.revertedWithCustomError(reputationMarket, 'InsufficientFunds');
    });

    describe('Sell', () => {
      let initialVotes: bigint;
      const sellVotes = 5n;

      beforeEach(async () => {
        // Buy some votes first
        await userA.buyVotes({
          buyAmount: DEFAULT.buyAmount * 100n,
          votesToBuy: 100n,
          minVotesToBuy: 100n,
        });
        const { trustVotes } = await userA.getVotes();
        initialVotes = trustVotes;
        expect(initialVotes).to.be.greaterThan(sellVotes);
      });

      it('should fail when actual proceeds are less than expected', async () => {
        const { simulatedSellPrice } = await userA.simulateSell({
          sellVotes,
        });

        await expect(
          userA.sellVotes({
            sellVotes,
            minSellPrice: simulatedSellPrice + 1n,
          }),
        ).to.be.revertedWithCustomError(reputationMarket, 'SellSlippageLimitExceeded');
      });

      it('should succeed when no minimum price is set', async () => {
        await expect(
          userA.sellVotes({
            sellVotes,
            minSellPrice: 0n,
          }),
        ).to.not.be.reverted;
      });

      it('should succeed when actual proceeds are more than expected', async () => {
        const { simulatedSellPrice } = await userA.simulateSell({
          sellVotes,
        });

        await expect(
          userA.sellVotes({
            sellVotes,
            minSellPrice: simulatedSellPrice - 1n,
          }),
        ).to.not.be.reverted;
      });
    });
  });

  describe('Simulations', () => {
    it('should correctly simulate buying votes', async () => {
      // Simulate buying votes
      const { simulatedVotesBought, simulatedFundsPaid } = await userA.simulateBuy();

      // Actually buy votes
      const { trustVotes: actualVotesBought, fundsPaid: actualFundsPaid } = await userA.buyVotes();

      // Compare simulated results with actual results
      expect(simulatedVotesBought).to.equal(actualVotesBought);
      expect(simulatedFundsPaid).to.equal(actualFundsPaid);
    });

    it('should correctly simulate selling votes', async () => {
      const votesToSell = 5n;

      // Buy votes first
      const { trustVotes: initialTrustVotesOwned } = await userA.buyVotes({
        buyAmount: DEFAULT.buyAmount * 100n,
        votesToBuy: 5n,
      });

      // Simulate selling votes
      const { simulatedVotesSold, simulatedFundsReceived } = await userA.simulateSell({
        sellVotes: votesToSell,
      });

      // Actually sell votes
      const { trustVotes: trustVotesRemaining, fundsReceived: actualFundsReceived } =
        await userA.sellVotes({
          sellVotes: votesToSell,
        });

      // Compare simulated results with actual results
      expect(trustVotesRemaining).to.equal(initialTrustVotesOwned - simulatedVotesSold);
      expect(simulatedFundsReceived).to.equal(actualFundsReceived);
    });

    it('should correctly simulate selling zero votes', async () => {
      const votesToBuyAndSell = 0n;

      // Buy votes first
      const { trustVotes: initialTrustVotesOwned } = await userA.buyVotes({
        buyAmount: DEFAULT.buyAmount * 100n,
        votesToBuy: votesToBuyAndSell,
      });

      // Simulate selling votes
      const { simulatedVotesSold, simulatedFundsReceived } = await userA.simulateSell({
        sellVotes: votesToBuyAndSell,
      });

      // // Actually sell votes
      const { trustVotes: trustVotesRemaining, fundsReceived: actualFundsReceived } =
        await userA.sellVotes({
          sellVotes: votesToBuyAndSell,
        });
      // Compare simulated results with actual results
      expect(trustVotesRemaining).to.equal(initialTrustVotesOwned - simulatedVotesSold);
      expect(simulatedFundsReceived).to.equal(actualFundsReceived);
    });

    it('should not change contract state when simulating buy', async () => {
      const initialMarketState = await reputationMarket.getMarket(DEFAULT.profileId);
      const initialUserVotes = await userA.getVotes();
      const votesToBuy = 5n;

      // Simulate buying votes
      await reputationMarket.simulateBuy(DEFAULT.profileId, DEFAULT.isPositive, votesToBuy);

      const finalMarketState = await reputationMarket.getMarket(DEFAULT.profileId);
      const finalUserVotes = await userA.getVotes();

      // Verify that the market state and user votes haven't changed
      expect(initialMarketState.trustVotes).to.equal(finalMarketState.trustVotes);
      expect(initialMarketState.distrustVotes).to.equal(finalMarketState.distrustVotes);
      expect(initialUserVotes.trustVotes).to.equal(finalUserVotes.trustVotes);
      expect(initialUserVotes.distrustVotes).to.equal(finalUserVotes.distrustVotes);
    });

    it('should not change contract state when simulating sell', async () => {
      const votesToBuyAndSell = 5n;

      // Buy votes first
      await userA.buyVotes({
        buyAmount: DEFAULT.buyAmount * 100n,
        votesToBuy: votesToBuyAndSell,
        minVotesToBuy: votesToBuyAndSell,
      });

      const initialMarketState = await reputationMarket.getMarket(DEFAULT.profileId);
      const initialUserVotes = await userA.getVotes();

      // Simulate selling votes
      await userA.simulateSell({
        sellVotes: votesToBuyAndSell,
      });

      const finalMarketState = await reputationMarket.getMarket(DEFAULT.profileId);
      const finalUserVotes = await userA.getVotes();

      // Verify that the market state and user votes haven't changed
      expect(initialMarketState.trustVotes).to.equal(finalMarketState.trustVotes);
      expect(initialMarketState.distrustVotes).to.equal(finalMarketState.distrustVotes);
      expect(initialUserVotes.trustVotes).to.equal(finalUserVotes.trustVotes);
      expect(initialUserVotes.distrustVotes).to.equal(finalUserVotes.distrustVotes);
    });

    it('should return correct cost when simulating buying votes', async () => {
      const { simulatedFundsPaid, simulatedVotesBought } = await userA.simulateBuy({
        buyAmount: DEFAULT.buyAmount * 100n,
        votesToBuy: 50n,
      });
      const { fundsPaid: actualFundsPaid, trustVotes: actualVotesBought } = await userA.buyVotes({
        buyAmount: DEFAULT.buyAmount * 100n,
        votesToBuy: 50n,
      });

      expect(simulatedFundsPaid).to.equal(actualFundsPaid);
      expect(simulatedVotesBought).to.equal(actualVotesBought);
    });

    it('should return correct cost when simulating selling votes', async () => {
      // First buy some votes to sell
      const votesToBuy = 50n;
      await userA.buyVotes({ buyAmount: DEFAULT.buyAmount * 100n, votesToBuy });

      const { simulatedFundsReceived, simulatedVotesSold } = await userA.simulateSell({
        sellVotes: 5n,
      });
      const { fundsReceived: actualFundsReceived, trustVotes: trustVotesRemaining } =
        await userA.sellVotes({
          sellVotes: 5n,
        });

      expect(simulatedFundsReceived).to.equal(actualFundsReceived);
      expect(simulatedVotesSold).to.equal(votesToBuy - trustVotesRemaining);
    });

    it('should correctly return new vote price after simulated buy', async () => {
      const votesToBuy = 5n;

      // Get current price before simulation
      const currentPrice = await reputationMarket.getVotePrice(
        DEFAULT.profileId,
        DEFAULT.isPositive,
      );

      // Simulate buying votes
      const { newVotePrice } = await userA.simulateBuy({
        votesToBuy,
      });

      // Actually buy votes to verify simulation
      await userA.buyVotes({ votesToBuy });

      // Get actual new price after purchase
      const actualNewPrice = await reputationMarket.getVotePrice(
        DEFAULT.profileId,
        DEFAULT.isPositive,
      );

      // Simulated new price should match actual new price
      expect(newVotePrice).to.equal(actualNewPrice);
      // New price should be higher than original price for buys
      expect(newVotePrice).to.be.greaterThan(currentPrice);
    });

    it('should correctly return new vote price after simulated sell', async () => {
      // First buy some votes to sell
      const votesToBuy = 10n;
      const votesToSell = 5n;
      await userA.buyVotes({ votesToBuy });

      // Get current price before simulation
      const currentPrice = await reputationMarket.getVotePrice(
        DEFAULT.profileId,
        DEFAULT.isPositive,
      );

      // Simulate selling votes
      const { newVotePrice } = await userA.simulateSell({
        sellVotes: votesToSell,
      });

      // Actually sell votes to verify simulation
      await userA.sellVotes({ sellVotes: votesToSell });

      // Get actual new price after sale
      const actualNewPrice = await reputationMarket.getVotePrice(
        DEFAULT.profileId,
        DEFAULT.isPositive,
      );

      // Simulated new price should match actual new price
      expect(newVotePrice).to.equal(actualNewPrice);
      // New price should be lower than original price for sells
      expect(newVotePrice).to.be.lessThan(currentPrice);
    });
  });

  describe('Participants', () => {
    it('should add a user to participants when buying votes', async () => {
      // Check that the user is not a participant initially
      expect(
        await reputationMarket.isParticipant(DEFAULT.profileId, await userA.signer.getAddress()),
      ).to.equal(false);

      // Buy votes
      await userA.buyVotes();

      // Check that the user is now a participant
      expect(
        await reputationMarket.isParticipant(DEFAULT.profileId, await userA.signer.getAddress()),
      ).to.equal(true);

      // Check that the user is in the participants array
      const participantCount = await reputationMarket.getParticipantCount(DEFAULT.profileId);
      let userFound = false;

      for (let i = 0; i < participantCount; i++) {
        const participant = await reputationMarket.participants(DEFAULT.profileId, i);

        if (participant === (await userA.signer.getAddress())) {
          userFound = true;
          break;
        }
      }
      expect(userFound).to.equal(true);
    });

    it('should not add a user to participants multiple times', async () => {
      // Buy votes twice
      await userA.buyVotes();
      await userA.buyVotes();

      // Check that the user is a participant
      expect(
        await reputationMarket.isParticipant(DEFAULT.profileId, await userA.signer.getAddress()),
      ).to.equal(true);

      // Check that the user appears only once in the participants array
      const participantCount = await reputationMarket.getParticipantCount(DEFAULT.profileId);
      let userCount = 0;

      for (let i = 0; i < participantCount; i++) {
        const participant = await reputationMarket.participants(DEFAULT.profileId, i);

        if (participant === (await userA.signer.getAddress())) {
          userCount++;
        }
      }
      expect(userCount).to.equal(1);
    });

    it('should show a user as a participant even after selling all votes', async () => {
      // Buy votes
      await userA.buyOneVote();

      // Check that the user is a participant
      expect(
        await reputationMarket.isParticipant(DEFAULT.profileId, await userA.signer.getAddress()),
      ).to.equal(true);

      // Sell all votes
      await userA.sellOneVote();

      // Check that the user is still a participant
      expect(
        await reputationMarket.isParticipant(DEFAULT.profileId, await userA.signer.getAddress()),
      ).to.equal(true);
    });

    it('should keep a user as a participant when selling only some votes', async () => {
      const amountToBuy = DEFAULT.buyAmount * 20n;

      // Buy votes
      await userA.buyVotes({ buyAmount: amountToBuy });

      // Sell half of the votes
      await userA.sellVotes({ sellVotes: DEFAULT.sellVotes / 2n });

      // Check that the user is still a participant
      expect(
        await reputationMarket.isParticipant(DEFAULT.profileId, await userA.signer.getAddress()),
      ).to.equal(true);
    });
  });

  it('should match logistic sigmoid value (73.11%) when buying 1000 votes', async () => {
    const marketConfig = await reputationMarket.marketConfigs(0);
    const basePrice = marketConfig.basePrice;
    // Buy 1000 votes
    await userA.buyVotes({
      votesToBuy: 1000n,
      buyAmount: DEFAULT.buyAmount * 1000n,
    });

    // Get final price
    const finalPrice = await reputationMarket.getVotePrice(DEFAULT.profileId, true);

    // e/(e+1) ≈ 0.7310585786300048822130804829419
    const expectedPrice = (basePrice * 731058578630004879n) / 1000000000000000000n;

    expect(finalPrice).to.be.closeTo(expectedPrice, 1);
  });
});
