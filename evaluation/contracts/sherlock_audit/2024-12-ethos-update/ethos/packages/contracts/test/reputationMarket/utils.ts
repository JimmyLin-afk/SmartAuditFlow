import { type HardhatEthersSigner } from '@nomicfoundation/hardhat-ethers/signers.js';
import { type ContractTransactionResponse, type Log, type EventLog } from 'ethers';
import hre from 'hardhat';

import { type ReputationMarket } from '../../typechain-types/index.js';

const { ethers } = hre;

export const DEFAULT = {
  reputationMarket: undefined as unknown as ReputationMarket,
  profileId: 1n,
  liquidity: 1000n,
  buyAmount: ethers.parseEther('.01'),
  value: { value: ethers.parseEther('.1') },
  isPositive: true,
  sellVotes: 10n,
  votesToBuy: 10n,
  creationCost: ethers.parseEther('1'),
};

type Params = {
  reputationMarket: ReputationMarket;
  profileId: bigint;
  isPositive: boolean;
  buyAmount: bigint;
  sellVotes: bigint;
  votesToBuy: bigint;
  minVotesToBuy: bigint;
  minSellPrice: bigint;
};

type Result = {
  balance: bigint;
  gas: bigint;
  trustVotes: bigint;
  distrustVotes: bigint;
  fundsPaid?: bigint;
  fundsReceived?: bigint;
};

function getParams(params?: Partial<Params>): Params {
  const votesToBuy = params?.votesToBuy ?? DEFAULT.votesToBuy;
  const minVotesToBuy = params?.minVotesToBuy ?? votesToBuy;

  return {
    reputationMarket: params?.reputationMarket ?? DEFAULT.reputationMarket,
    profileId: params?.profileId ?? DEFAULT.profileId,
    isPositive: params?.isPositive ?? DEFAULT.isPositive,
    buyAmount: params?.buyAmount ?? DEFAULT.buyAmount * votesToBuy,
    sellVotes: params?.sellVotes ?? DEFAULT.sellVotes,
    votesToBuy,
    minVotesToBuy,
    minSellPrice: params?.minSellPrice ?? 0n,
  };
}

function isEventLog(log: Log): log is EventLog {
  return 'args' in log && typeof log.args === 'object';
}

function isVotesBoughtEvent(reputationMarket: ReputationMarket) {
  return function (log: Log): log is EventLog {
    return (
      isEventLog(log) &&
      log.topics[0] === reputationMarket.interface.getEvent('VotesBought')?.topicHash
    );
  };
}

function isVotesSoldEvent(reputationMarket: ReputationMarket) {
  return function (log: Log): log is EventLog {
    return (
      isEventLog(log) &&
      log.topics[0] === reputationMarket.interface.getEvent('VotesSold')?.topicHash
    );
  };
}

function isWithdrawDonationsEvent(reputationMarket: ReputationMarket) {
  return function (log: Log): log is EventLog {
    return (
      isEventLog(log) &&
      log.topics[0] === reputationMarket.interface.getEvent('DonationWithdrawn')?.topicHash
    );
  };
}

export class MarketUser {
  public readonly signer: HardhatEthersSigner;
  constructor(signer: HardhatEthersSigner) {
    this.signer = signer;
  }

  async getVotes(
    params?: Partial<Params>,
  ): Promise<{ trustVotes: bigint; distrustVotes: bigint; balance: bigint }> {
    const { reputationMarket, profileId } = getParams(params);
    const { trustVotes, distrustVotes } = await reputationMarket
      .connect(this.signer)
      .getUserVotes(this.signer.getAddress(), profileId);
    const balance = await ethers.provider.getBalance(this.signer.address);

    return { trustVotes, distrustVotes, balance };
  }

  async getGas(tx: ContractTransactionResponse): Promise<{ gas: bigint }> {
    const receipt = await tx.wait();

    if (!receipt?.status) {
      throw new Error('Transaction failed');
    }

    return { gas: receipt.gasUsed };
  }

  async simulateBuy(params?: Partial<Params>): Promise<{
    simulatedVotesBought: bigint;
    simulatedFundsPaid: bigint;
    simulatedProtocolFee: bigint;
    simulatedDonation: bigint;
    newVotePrice: bigint;
  }> {
    const { reputationMarket, profileId, isPositive, votesToBuy } = getParams(params);
    const { purchaseCostBeforeFees, protocolFee, donation, newVotePrice } = await reputationMarket
      .connect(this.signer)
      .simulateBuy(profileId, isPositive, votesToBuy);

    return {
      simulatedVotesBought: votesToBuy,
      simulatedFundsPaid: purchaseCostBeforeFees + protocolFee + donation,
      simulatedProtocolFee: protocolFee,
      simulatedDonation: donation,
      newVotePrice,
    };
  }

  async buyVotes(params?: Partial<Params>): Promise<Result> {
    const { reputationMarket, profileId, isPositive, buyAmount, minVotesToBuy, votesToBuy } =
      getParams(params);

    const tx: ContractTransactionResponse = await reputationMarket
      .connect(this.signer)
      .buyVotes(profileId, isPositive, votesToBuy, minVotesToBuy, {
        value: buyAmount,
      });
    const { gas } = await this.getGas(tx);
    const { trustVotes, distrustVotes, balance } = await this.getVotes(params);
    const receipt = await tx.wait();
    const event = receipt?.logs.find(isVotesBoughtEvent(reputationMarket));
    const fundsPaid = event ? event.args.funds : 0n;

    return { gas, trustVotes, distrustVotes, balance, fundsPaid };
  }

  async buyOneVote(params?: Partial<Params>): Promise<Result> {
    return await this.buyVotes(getParams({ ...params, votesToBuy: 1n }));
  }

  async simulateSell(params?: Partial<Params>): Promise<{
    simulatedVotesSold: bigint;
    simulatedFundsReceived: bigint;
    simulatedProtocolFee: bigint;
    simulatedSellPrice: bigint;
    newVotePrice: bigint;
  }> {
    const { reputationMarket, profileId, isPositive, sellVotes } = getParams(params);
    const { proceedsBeforeFees, protocolFee, newVotePrice } = await reputationMarket
      .connect(this.signer)
      .simulateSell(profileId, isPositive, sellVotes);

    return {
      simulatedVotesSold: sellVotes,
      simulatedFundsReceived: proceedsBeforeFees - protocolFee,
      simulatedProtocolFee: protocolFee,
      simulatedSellPrice: sellVotes > 0n ? proceedsBeforeFees / sellVotes : 0n,
      newVotePrice,
    };
  }

  async sellVotes(params?: Partial<Params>): Promise<Result> {
    const { reputationMarket, profileId, isPositive, sellVotes, minSellPrice } = getParams(params);
    const tx: ContractTransactionResponse = await reputationMarket
      .connect(this.signer)
      .sellVotes(profileId, isPositive, sellVotes, minSellPrice);
    const { gas } = await this.getGas(tx);
    const { trustVotes, distrustVotes, balance } = await this.getVotes(params);
    const receipt = await tx.wait();
    const event = receipt?.logs.find(isVotesSoldEvent(reputationMarket));
    const fundsReceived = event ? event.args.funds : 0n;

    return { gas, trustVotes, distrustVotes, balance, fundsReceived };
  }

  async sellOneVote(params?: Partial<Params>): Promise<Result> {
    return await this.sellVotes(getParams({ ...params, sellVotes: 1n }));
  }

  async withdrawDonations(): Promise<{ donationsWithdrawn: bigint }> {
    const { reputationMarket } = getParams();
    const tx = await reputationMarket.connect(this.signer).withdrawDonations();
    const receipt = await tx.wait();
    const event = receipt?.logs.find(isWithdrawDonationsEvent(reputationMarket));
    const donationsWithdrawn = event ? event.args.amount : 0n;

    return { donationsWithdrawn };
  }
}
