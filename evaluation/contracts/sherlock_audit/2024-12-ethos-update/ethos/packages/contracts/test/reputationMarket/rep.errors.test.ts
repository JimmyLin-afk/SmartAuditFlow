import { loadFixture } from '@nomicfoundation/hardhat-toolbox/network-helpers.js';
import { expect, use } from 'chai';
import chaiAsPromised from 'chai-as-promised';
import hre from 'hardhat';
import { type ReputationMarket } from '../../typechain-types/index.js';
import { createDeployer, type EthosDeployer } from '../utils/deployEthos.js';
import { DEFAULT, MarketUser } from './utils.js';

const { ethers } = hre;

/* eslint-disable react-hooks/rules-of-hooks */
use(chaiAsPromised as Chai.ChaiPlugin);

describe('ReputationMarket Errors', () => {
  let deployer: EthosDeployer;
  let userA: MarketUser;
  let reputationMarket: ReputationMarket;

  beforeEach(async () => {
    deployer = await loadFixture(createDeployer);

    if (!deployer.reputationMarket.contract) {
      throw new Error('ReputationMarket contract not found');
    }
    const ethosUserA = await deployer.createUser();
    await ethosUserA.setBalance('2000');

    userA = new MarketUser(ethosUserA.signer);

    reputationMarket = deployer.reputationMarket.contract;
    DEFAULT.reputationMarket = reputationMarket;
    DEFAULT.profileId = ethosUserA.profileId;

    await reputationMarket
      .connect(deployer.ADMIN)
      .createMarketWithConfigAdmin(ethosUserA.signer.address, 0, {
        value: DEFAULT.creationCost,
      });
  });

  it('should revert with InsufficientFunds when buying a vote with insufficient funds', async () => {
    await expect(userA.buyVotes({ buyAmount: 1n })).to.be.revertedWithCustomError(
      reputationMarket,
      'InsufficientFunds',
    );
  });

  it('should revert with InsufficientLiquidity when a user creates a market with insufficient initial liquidity', async () => {
    const ethosUserB = await deployer.createUser();
    await reputationMarket
      .connect(deployer.ADMIN)
      .setUserAllowedToCreateMarket(ethosUserB.profileId, true);
    const marketConfig = await reputationMarket.marketConfigs(0n);
    // should fail if they don't pay enough
    await expect(
      reputationMarket
        .connect(ethosUserB.signer)
        .createMarket({ value: marketConfig.creationCost - 1n }),
    ).to.be.revertedWithCustomError(reputationMarket, 'InsufficientLiquidity');
    // should work if they pay up
    await expect(
      reputationMarket
        .connect(ethosUserB.signer)
        .createMarket({ value: marketConfig.creationCost }),
    ).to.be.not.reverted;
  });

  it('should allow admin creates a market with arbitrary creation cost', async () => {
    const ethosUserB = await deployer.createUser();
    const arbitraryCreationCost = ethers.parseEther('.123456');
    await expect(
      reputationMarket
        .connect(deployer.ADMIN)
        .createMarketWithConfigAdmin(ethosUserB.signer.address, 0, {
          value: arbitraryCreationCost,
        }),
    ).to.be.not.reverted;
    // check the market funds
    const marketFunds = await reputationMarket.marketFunds(ethosUserB.profileId);
    expect(marketFunds).to.equal(arbitraryCreationCost);
  });

  it('should revert with InsufficientVotesOwned when selling positive votes without owning any', async () => {
    await expect(reputationMarket.connect(userA.signer).sellVotes(DEFAULT.profileId, true, 1, 0n))
      .to.be.revertedWithCustomError(reputationMarket, 'InsufficientVotesOwned')
      .withArgs(DEFAULT.profileId, userA.signer.getAddress());
  });

  it('should revert with InsufficientVotesToSell when selling negative votes without owning any', async () => {
    await userA.buyOneVote();
    await expect(reputationMarket.connect(userA.signer).sellVotes(DEFAULT.profileId, false, 1, 0n))
      .to.be.revertedWithCustomError(reputationMarket, 'InsufficientVotesOwned')
      .withArgs(DEFAULT.profileId, userA.signer.getAddress());
  });
});
