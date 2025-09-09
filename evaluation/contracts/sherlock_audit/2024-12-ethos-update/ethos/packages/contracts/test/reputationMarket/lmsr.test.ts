import { loadFixture } from '@nomicfoundation/hardhat-toolbox/network-helpers.js';
import { expect } from 'chai';
import hre from 'hardhat';
import { type TestLMSR } from '../../typechain-types/index.js';
import { createDeployer } from '../utils/deployEthos.js';

const { ethers } = hre;

describe('LMSR (Logarithmic Market Scoring Rule)', () => {
  let lmsr: TestLMSR;
  const LIQUIDITY_PARAMETER = 1000;
  const QUOTIENT = 1000000000000000000n; // 1e18 - this is the denominator of the ratio returned by the price function

  beforeEach(async () => {
    await loadFixture(createDeployer);

    // Deploy LMSR library first
    const LMSRFactory = await ethers.getContractFactory('LMSR');
    const lmsrLib = await LMSRFactory.deploy();
    await lmsrLib.waitForDeployment();

    // Get TestLMSR factory and link the library
    const TestLMSRFactory = await ethers.getContractFactory('TestLMSR', {
      libraries: {
        'contracts/utils/LMSR.sol:LMSR': await lmsrLib.getAddress(),
      },
    });

    // Deploy TestLMSR
    lmsr = await TestLMSRFactory.deploy();
  });

  describe('Vote Price Calculations', () => {
    it('should calculate correct initial prices with equal votes', async () => {
      const votes = 1000n;

      const trustPrice = await lmsr.getOdds(votes, votes, LIQUIDITY_PARAMETER, true);
      const distrustPrice = await lmsr.getOdds(votes, votes, LIQUIDITY_PARAMETER, false);

      expect(trustPrice).to.equal(distrustPrice, 'Trust and distrust prices should be equal');
      expect(trustPrice).to.equal(QUOTIENT / 2n, 'Trust price should be half of max price');
    });

    it('should calculate higher price for side with more votes', async () => {
      const votes = 1000n;

      const trustPrice = await lmsr.getOdds(votes + 1n, votes, LIQUIDITY_PARAMETER, true);
      const distrustPrice = await lmsr.getOdds(votes, votes, LIQUIDITY_PARAMETER, false);

      expect(trustPrice).to.be.gt(distrustPrice);
    });

    it('should maintain price bounds between 0 and basePrice', async () => {
      const largeVotes = 10000n;
      const smallVotes = 1n;

      const highPrice = await lmsr.getOdds(largeVotes, smallVotes, LIQUIDITY_PARAMETER, true);
      const lowPrice = await lmsr.getOdds(largeVotes, smallVotes, LIQUIDITY_PARAMETER, false);

      expect(highPrice).to.be.lte(QUOTIENT);
      expect(lowPrice).to.be.gt(0);
    });

    it('should continue to increase price as votes increase', async () => {
      const liquidity = 10000;
      const range = 277601; // 277,601 is the point where the price stops increasing with this liquidity parameter

      for (let i = 1; i < range; i += 500) {
        const previousPrice = await lmsr.getOdds(i - 1, 0n, liquidity, true);
        const newPrice = await lmsr.getOdds(i, 0, liquidity, true);
        expect(newPrice).to.be.gt(previousPrice, `Price should increase as votes increase (${i})`);
      }
    });

    it('should match logistic sigmoid value (73.11%) when buying 1000 votes at 1000 liquidity', async () => {
      const startingVotes = 0n;
      const votesToBuy = 1000n;
      // e/(e+1) ≈ 0.7310585786300048822130804829419
      const expectedOdds = (QUOTIENT * 731058578630004879n) / 1000000000000000000n;

      const odds = await lmsr.getOdds(votesToBuy, startingVotes, LIQUIDITY_PARAMETER, true);

      expect(odds).to.be.closeTo(expectedOdds, 1);
    });
  });

  describe('Cost Calculations', () => {
    it('should calculate cost of zero votes changing', async () => {
      const votes = 0n;

      const cost = await lmsr.getCost(votes, votes, votes, votes, LIQUIDITY_PARAMETER);

      expect(cost).to.be.eq(0);
    });

    it('should cost more than zero to buy votes', async () => {
      const currentVotes = 1000n;
      const votesToBuy = 10n;

      const cost = await lmsr.getCost(
        currentVotes, // trust votes
        currentVotes, // distrust votes
        currentVotes + votesToBuy, // intended trust votes
        currentVotes, // intended distrust votes
        LIQUIDITY_PARAMETER,
      );

      expect(cost).to.be.gt(0);
    });

    it('should cost less than 1x basePrice to buy 1 votes when the market is new', async () => {
      const currentVotes = 1n;
      const votesToBuy = 1n;
      const liquidity = ethers.parseEther('0.02');

      const cost = await lmsr.getCost(
        currentVotes,
        currentVotes,
        currentVotes + votesToBuy,
        currentVotes,
        liquidity,
      );

      expect(cost).to.be.lt(QUOTIENT);
    });

    it('should calculate higher cost for adding votes to majority side', async () => {
      const majorityVotes = 2000n;
      const minorityVotes = 1500n;
      const addAmount = 10n;

      const costToMajority = await lmsr.getCost(
        majorityVotes,
        minorityVotes,
        majorityVotes + addAmount,
        minorityVotes,
        LIQUIDITY_PARAMETER,
      );

      const costToMinority = await lmsr.getCost(
        majorityVotes,
        minorityVotes,
        majorityVotes,
        minorityVotes + addAmount,
        LIQUIDITY_PARAMETER,
      );

      expect(costToMajority).to.be.gt(costToMinority);
    });

    it('should calculate lower cost for removing votes from majority side', async () => {
      const majorityVotes = 100n;
      const minorityVotes = 99n;
      const removeAmount = 1n;

      const costFromMajority = await lmsr.getCost(
        majorityVotes,
        minorityVotes,
        majorityVotes - removeAmount,
        minorityVotes,
        LIQUIDITY_PARAMETER,
      );

      const costFromMinority = await lmsr.getCost(
        majorityVotes,
        minorityVotes,
        majorityVotes,
        minorityVotes - removeAmount,
        LIQUIDITY_PARAMETER,
      );

      expect(costFromMajority).to.be.lt(costFromMinority);
    });

    it('should be the same cost when adding and removing votes', async () => {
      const initialVotes = 1000n;
      const voteDelta = 100n;

      const costToAdd = await lmsr.getCost(
        initialVotes,
        initialVotes,
        initialVotes + voteDelta,
        initialVotes,
        LIQUIDITY_PARAMETER,
      );

      const costToRemove = await lmsr.getCost(
        initialVotes + voteDelta,
        initialVotes,
        initialVotes,
        initialVotes,
        LIQUIDITY_PARAMETER,
      );

      expect(costToAdd).to.be.gt(0);
      const netCost = costToAdd + costToRemove;
      expect(netCost).to.be.eq(0);
    });
  });

  describe('Market Behavior', () => {
    it('should handle zero votes gracefully', async () => {
      await expect(lmsr.getOdds(0n, 1000n, LIQUIDITY_PARAMETER, true)).to.not.be.rejected;
      await expect(lmsr.getCost(0n, 1000n, 0n, 1000n, LIQUIDITY_PARAMETER)).to.not.be.rejected;
    });

    it('should handle very large vote numbers', async () => {
      const MAX_SAFE_VOTES = 133_000n - 1n;
      // TODO there is a "trick" to increasing support for large exponents without losing precision
      // ask cursor about Factoring Out the Maximum Exponent
      // (but I didn't want to apply it yet, as that adds more complexity)
      await expect(lmsr.getOdds(MAX_SAFE_VOTES, 1, LIQUIDITY_PARAMETER, true)).not.to.be.reverted;
      await expect(lmsr.getOdds(MAX_SAFE_VOTES, 2, LIQUIDITY_PARAMETER, true)).not.to.be.reverted;
    });

    it('should charge cost between current and next price when buying one vote', async () => {
      for (let yes = 1n; yes < 1000n; yes += 100n) {
        for (let no = 0n; no < 1000n; no += 100n) {
          const yesOdds = await lmsr.getOdds(yes, no, LIQUIDITY_PARAMETER, true);
          const nextYesOdds = await lmsr.getOdds(yes + 1n, no, LIQUIDITY_PARAMETER, true);
          const yesCost = await lmsr.getCost(yes, no, yes + 1n, no, LIQUIDITY_PARAMETER);
          expect(yesCost).to.be.lt(nextYesOdds, `yes: ${yes}, no: ${no}`);
          expect(yesCost).to.be.gt(yesOdds, `yes: ${yes}, no: ${no}`);
          const noOdds = await lmsr.getOdds(yes, no, LIQUIDITY_PARAMETER, false);
          const nextNoOdds = await lmsr.getOdds(yes, no + 1n, LIQUIDITY_PARAMETER, false);
          const noCost = await lmsr.getCost(yes, no, yes, no + 1n, LIQUIDITY_PARAMETER);
          expect(noCost).to.be.lt(nextNoOdds, `yes: ${yes}, no: ${no}`);
          expect(noCost).to.be.gt(noOdds, `yes: ${yes}, no: ${no}`);
        }
      }
    });

    it('should pay between current and next price when selling one vote', async () => {
      for (let yes = 1n; yes < 1000n; yes += 100n) {
        for (let no = 1n; no < 1000n; no += 100n) {
          const yesOdds = await lmsr.getOdds(yes, no, LIQUIDITY_PARAMETER, true);
          const nextYesOdds = await lmsr.getOdds(yes - 1n, no, LIQUIDITY_PARAMETER, true);
          const yesCost = await lmsr.getCost(yes, no, yes - 1n, no, LIQUIDITY_PARAMETER);
          expect(yesCost).to.be.lt(-nextYesOdds, `yes: ${yes}, no: ${no}`);
          expect(yesCost).to.be.gt(-yesOdds, `yes: ${yes}, no: ${no}`);
          const noOdds = await lmsr.getOdds(yes, no, LIQUIDITY_PARAMETER, false);
          const nextNoOdds = await lmsr.getOdds(yes, no - 1n, LIQUIDITY_PARAMETER, false);
          const noCost = await lmsr.getCost(yes, no, yes, no - 1n, LIQUIDITY_PARAMETER);
          expect(noCost).to.be.lt(-nextNoOdds, `yes: ${yes}, no: ${no}`);
          expect(noCost).to.be.gt(-noOdds, `yes: ${yes}, no: ${no}`);
        }
      }
    });
  });

  describe('Liquidity Parameter', () => {
    it('should change the price less when liquidity is higher', async () => {
      const yesVotes = 2000n;
      const noVotes = 1000n;
      const lessLiquidOdds = await lmsr.getOdds(yesVotes, noVotes, LIQUIDITY_PARAMETER, true);
      const moreLiquidOdds = await lmsr.getOdds(yesVotes, noVotes, LIQUIDITY_PARAMETER * 2, true);
      expect(moreLiquidOdds).to.be.lt(lessLiquidOdds);
    });
  });

  describe('Contract Funds Conservation', () => {
    it('should conserve contract funds when buying and selling votes in series or in parallel', async () => {
      const balance = { user: 0n };
      const mrkt = { yes: 0n, no: 0n };
      const l = LIQUIDITY_PARAMETER; // just to keep shit on one line

      // buy 100 yes and no votes, one at a time alternating
      for (let i = 0; i < 100; i++) {
        balance.user += await lmsr.getCost(mrkt.yes, mrkt.no, mrkt.yes + 1n, mrkt.no, l);
        mrkt.yes += 1n;
        balance.user += await lmsr.getCost(mrkt.yes, mrkt.no, mrkt.yes, mrkt.no + 1n, l);
        mrkt.no += 1n;
      }
      // sell 100 yes votes in bulk
      balance.user += await lmsr.getCost(mrkt.yes, mrkt.no, mrkt.yes - 100n, mrkt.no, l);
      mrkt.yes -= 100n;
      // sell 100 no votes in bulk
      balance.user += await lmsr.getCost(mrkt.yes, mrkt.no, mrkt.yes, mrkt.no - 100n, l);
      mrkt.no -= 100n;
      expect(balance.user).to.be.eq(0n);
    });

    it('should conserve contract funds when buying votes in arbitrary order', async () => {
      const balance = { user: 0n };
      const mrkt = { yes: 0n, no: 0n };
      const l = LIQUIDITY_PARAMETER; // just to keep shit on one line
      const Y = true;
      const N = false;

      const arbitraryOrder = [Y, N, Y, Y, Y, N, Y, N, N, N, Y, N, Y, Y, N, Y, N, Y, N, Y, N, Y, N];

      for (const buyYesOrNo of arbitraryOrder) {
        if (buyYesOrNo) {
          balance.user += await lmsr.getCost(mrkt.yes, mrkt.no, mrkt.yes + 1n, mrkt.no, l);
          mrkt.yes += 1n;
        } else {
          balance.user += await lmsr.getCost(mrkt.yes, mrkt.no, mrkt.yes, mrkt.no + 1n, l);
          mrkt.no += 1n;
        }
      }

      // sell everything in bulk
      balance.user += await lmsr.getCost(mrkt.yes, mrkt.no, 0n, 0n, l);
      expect(balance.user).to.be.eq(0n);
    });

    it('should conserve contract funds when selling votes in an arbitrary order', async () => {
      const balance = { user: 0n };
      const mrkt = { yes: 0n, no: 0n };
      const l = LIQUIDITY_PARAMETER; // just to keep shit on one line
      const Y = true;
      const N = false;

      const arbitraryOrder = [Y, N, Y, Y, Y, N, Y, N, N, N, Y, N, Y, N, N, Y, N, Y, N, N, N, Y, N];
      const numYes = arbitraryOrder.filter((x) => x).length;
      const numNo = arbitraryOrder.filter((x) => !x).length;

      // buy yes and no votes in bulk
      balance.user += await lmsr.getCost(mrkt.yes, mrkt.no, numYes, numNo, l);
      mrkt.yes += BigInt(numYes);
      mrkt.no += BigInt(numNo);

      // sell in arbitrary order
      for (const sellYesOrNo of arbitraryOrder) {
        if (sellYesOrNo) {
          balance.user += await lmsr.getCost(mrkt.yes, mrkt.no, mrkt.yes - 1n, mrkt.no, l);
          mrkt.yes -= 1n;
        } else {
          balance.user += await lmsr.getCost(mrkt.yes, mrkt.no, mrkt.yes, mrkt.no - 1n, l);
          mrkt.no -= 1n;
        }
      }
      expect(balance.user).to.be.eq(0n);
    });
  });
});
