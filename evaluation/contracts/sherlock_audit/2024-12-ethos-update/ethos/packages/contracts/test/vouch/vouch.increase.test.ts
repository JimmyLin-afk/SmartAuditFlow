import { loadFixture } from '@nomicfoundation/hardhat-toolbox/network-helpers.js';
import { expect } from 'chai';
import hre from 'hardhat';
import { calcFeeDistribution, calculateFee } from '../utils/common.js';
import { createDeployer, type EthosDeployer } from '../utils/deployEthos.js';
import { type EthosUser } from '../utils/ethosUser.js';

const { ethers } = hre;

describe('EthosVouch Increasing', () => {
  let deployer: EthosDeployer;
  let userA: EthosUser;
  let userB: EthosUser;
  const entryFee = 50n;
  const donationFee = 150n;
  const vouchIncentives = 200n;
  const exitFeeBasisPoints = 100n;
  const initialAmount = ethers.parseEther('0.1');
  const increaseAmount = ethers.parseEther('0.05');

  beforeEach(async () => {
    deployer = await loadFixture(createDeployer);
    [userA, userB] = await Promise.all([deployer.createUser(), deployer.createUser()]);

    // Set up all fees
    await Promise.all([
      deployer.ethosVouch.contract.connect(deployer.ADMIN).setEntryProtocolFeeBasisPoints(entryFee),
      deployer.ethosVouch.contract
        .connect(deployer.ADMIN)
        .setEntryDonationFeeBasisPoints(donationFee),
      deployer.ethosVouch.contract
        .connect(deployer.ADMIN)
        .setEntryVouchersPoolFeeBasisPoints(vouchIncentives),
      deployer.ethosVouch.contract
        .connect(deployer.ADMIN)
        .setExitFeeBasisPoints(exitFeeBasisPoints),
    ]);
  });

  it('should successfully increase vouch amount', async () => {
    const { vouchId, balance } = await userA.vouch(userB, { paymentAmount: initialAmount });

    // Calculate fees and expected deposit
    const { deposit } = calcFeeDistribution(increaseAmount, {
      entry: entryFee,
      donation: donationFee,
      vouchIncentives,
    });

    await deployer.ethosVouch.contract
      .connect(userA.signer)
      .increaseVouch(vouchId, { value: increaseAmount });

    const finalBalance = await userA.getVouchBalance(vouchId);
    expect(finalBalance).to.be.closeTo(balance + deposit, 1n);
  });

  it('should emit VouchIncreased event', async () => {
    const { vouchId } = await userA.vouch(userB, { paymentAmount: initialAmount });

    await expect(
      deployer.ethosVouch.contract
        .connect(userA.signer)
        .increaseVouch(vouchId, { value: increaseAmount }),
    ).to.emit(deployer.ethosVouch.contract, 'VouchIncreased');
  });

  it('should apply protocol entry fee correctly on increased amount', async () => {
    const { vouchId } = await userA.vouch(userB, { paymentAmount: initialAmount });

    // Get vault address and initial protocol fee recipient balance
    const protocolFeeAddress = await deployer.ethosVouch.contract.protocolFeeAddress();
    const initialFeeBalance = await ethers.provider.getBalance(protocolFeeAddress);

    // Increase vouch
    await deployer.ethosVouch.contract
      .connect(userA.signer)
      .increaseVouch(vouchId, { value: increaseAmount });

    // Check protocol fee recipient's balance increased by expected amount
    const finalFeeBalance = await ethers.provider.getBalance(protocolFeeAddress);

    const { shares } = calcFeeDistribution(increaseAmount, {
      entry: entryFee,
      donation: donationFee,
      vouchIncentives,
    });

    expect(finalFeeBalance - initialFeeBalance).to.equal(shares.protocol);
  });

  it('should apply donation fee correctly on increased amount', async () => {
    const { vouchId } = await userA.vouch(userB, { paymentAmount: initialAmount });

    // Get initial rewards balance for userB
    const initialRewardsBalance = await userB.getRewardsBalance();

    // Increase vouch
    await deployer.ethosVouch.contract
      .connect(userA.signer)
      .increaseVouch(vouchId, { value: increaseAmount });

    // Check userB's rewards balance increased by expected amount
    const finalRewardsBalance = await userB.getRewardsBalance();

    const { shares } = calcFeeDistribution(increaseAmount, {
      entry: entryFee,
      donation: donationFee,
      vouchIncentives,
    });

    expect(finalRewardsBalance - initialRewardsBalance).to.equal(shares.donation);
  });

  it('should apply vouchers pool fee correctly on increased amount', async () => {
    const { vouchId } = await userA.vouch(userB, { paymentAmount: initialAmount });

    // another user needs to vouch for the same person so they earn vouch pool incentives
    const userC = await deployer.createUser();
    const { vouchId: vouchIdC, balance: balanceC } = await userC.vouch(userB, {
      paymentAmount: initialAmount,
    });

    // Calculate expected pool incentives
    const { shares } = calcFeeDistribution(increaseAmount, {
      entry: entryFee,
      donation: donationFee,
      vouchIncentives,
    });

    // Increase vouch
    await deployer.ethosVouch.contract
      .connect(userA.signer)
      .increaseVouch(vouchId, { value: increaseAmount });

    const finalBalanceC = await userC.getVouchBalance(vouchIdC);
    const actualIncrease = finalBalanceC - balanceC;

    // Allow for small rounding differences
    expect(actualIncrease).to.be.closeTo(shares.vouchersPool, 1n);
  });

  it('should revert when non-author tries to increase vouch', async () => {
    const { vouchId } = await userA.vouch(userB, { paymentAmount: initialAmount });

    await expect(
      deployer.ethosVouch.contract
        .connect(userB.signer)
        .increaseVouch(vouchId, { value: increaseAmount }),
    )
      .to.be.revertedWithCustomError(deployer.ethosVouch.contract, 'NotAuthorForVouch')
      .withArgs(vouchId, userB.profileId);
  });

  it('should revert when trying to increase an unvouched vouch', async () => {
    const { vouchId } = await userA.vouch(userB, { paymentAmount: initialAmount });
    await userA.unvouch(vouchId);

    await expect(
      deployer.ethosVouch.contract
        .connect(userA.signer)
        .increaseVouch(vouchId, { value: increaseAmount }),
    )
      .to.be.revertedWithCustomError(deployer.ethosVouch.contract, 'AlreadyUnvouched')
      .withArgs(vouchId);
  });

  it('should revert when trying to increase a non-existent vouch', async () => {
    const nonExistentVouchId = 999;

    await expect(
      deployer.ethosVouch.contract
        .connect(userA.signer)
        .increaseVouch(nonExistentVouchId, { value: increaseAmount }),
    )
      .to.be.revertedWithCustomError(deployer.ethosVouch.contract, 'NotAuthorForVouch')
      .withArgs(nonExistentVouchId, userA.profileId);
  });

  it('should handle multiple consecutive increases', async () => {
    const { vouchId } = await userA.vouch(userB, { paymentAmount: initialAmount });
    const initialBalance = await userA.getVouchBalance(vouchId);

    // Calculate fees for first increase
    const { deposit: firstDeposit } = calcFeeDistribution(increaseAmount, {
      entry: entryFee,
      donation: donationFee,
      vouchIncentives,
    });

    // First increase
    await deployer.ethosVouch.contract
      .connect(userA.signer)
      .increaseVouch(vouchId, { value: increaseAmount });

    // Calculate fees for second increase
    const { deposit: secondDeposit } = calcFeeDistribution(increaseAmount, {
      entry: entryFee,
      donation: donationFee,
      vouchIncentives,
    });

    // Second increase
    await deployer.ethosVouch.contract
      .connect(userA.signer)
      .increaseVouch(vouchId, { value: increaseAmount });

    // The balance should have increased by both deposits
    const finalBalance = await userA.getVouchBalance(vouchId);
    const actualIncrease = finalBalance - initialBalance;
    const expectedIncrease = firstDeposit + secondDeposit;

    // Allow for small rounding differences
    expect(actualIncrease).to.be.closeTo(expectedIncrease, 1n);
  });

  it('should revert increase with zero value', async () => {
    const { vouchId } = await userA.vouch(userB, { paymentAmount: initialAmount });

    await expect(
      deployer.ethosVouch.contract.connect(userA.signer).increaseVouch(vouchId, { value: 0 }),
    )
      .to.be.revertedWithCustomError(deployer.ethosVouch.contract, 'MinimumVouchAmount')
      .withArgs(await deployer.ethosVouch.contract.configuredMinimumVouchAmount());
  });

  it('should revert when fee transfer fails during increase', async () => {
    const { vouchId } = await userA.vouch(userB, { paymentAmount: initialAmount });
    // Set protocol fee address to rejecting contract
    await deployer.ethosVouch.contract
      .connect(deployer.OWNER)
      .setProtocolFeeAddress(deployer.rejectETHReceiver.address);
    await expect(
      deployer.ethosVouch.contract
        .connect(userA.signer)
        .increaseVouch(vouchId, { value: increaseAmount }),
    )
      .to.be.revertedWithCustomError(deployer.ethosVouch.contract, 'FeeTransferFailed')
      .withArgs('Protocol fee deposit failed');
  });

  it('should return full amount (initial + increase) when unvouching', async () => {
    const { vouchId } = await userA.vouch(userB, { paymentAmount: initialAmount });

    // Calculate initial deposit after entry fees
    const { deposit: initialDeposit } = calcFeeDistribution(initialAmount, {
      entry: entryFee,
      donation: donationFee,
      vouchIncentives,
    });

    // Increase vouch
    await deployer.ethosVouch.contract
      .connect(userA.signer)
      .increaseVouch(vouchId, { value: increaseAmount });

    // Calculate increase deposit after entry fees
    const { deposit: increaseDeposit } = calcFeeDistribution(increaseAmount, {
      entry: entryFee,
      donation: donationFee,
      vouchIncentives,
    });

    // Get userA's balance before unvouching
    const balanceBefore = await ethers.provider.getBalance(userA.signer.address);

    // Calculate total deposit before exit fee
    const totalDeposit = initialDeposit + increaseDeposit;

    // Calculate exit fee on total deposit
    const { deposit: expectedReturn } = calculateFee(totalDeposit, exitFeeBasisPoints);

    // Unvouch and get transaction details
    const unvouchTx = await userA.unvouch(vouchId);
    const receipt = await unvouchTx.wait();
    const gasUsed = receipt ? receipt.gasUsed * receipt.gasPrice : 0n;

    // Get balance after unvouching
    const balanceAfter = await ethers.provider.getBalance(userA.signer.address);

    // Actual change in balance (accounting for gas)
    const actualReturn = balanceAfter - balanceBefore + gasUsed;

    // Allow for small rounding differences
    expect(actualReturn).to.be.closeTo(expectedReturn, 1n);
  });
});
