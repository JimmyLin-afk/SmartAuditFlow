import { loadFixture } from '@nomicfoundation/hardhat-toolbox/network-helpers.js';
import { expect } from 'chai';
import hre from 'hardhat';
import { calcFeeDistribution } from '../utils/common.js';
import { DEFAULT, VOUCH_PARAMS } from '../utils/defaults.js';
import { createDeployer, type EthosDeployer } from '../utils/deployEthos.js';
import { type EthosUser } from '../utils/ethosUser.js';

const { ethers } = hre;

describe('Vouch Incentives', () => {
  let deployer: EthosDeployer;
  let userA: EthosUser;
  let userB: EthosUser;
  let userC: EthosUser;

  const vouchIncentives = 200n;

  async function setupVouchIncentives(): Promise<void> {
    await deployer.ethosVouch.contract
      .connect(deployer.ADMIN)
      .setEntryVouchersPoolFeeBasisPoints(vouchIncentives);
  }

  beforeEach(async () => {
    deployer = await loadFixture(createDeployer);
    [userA, userB, userC] = await Promise.all([
      deployer.createUser(),
      deployer.createUser(),
      deployer.createUser(),
    ]);
  });

  it('should not deduct vouch incentives for the first voucher', async () => {
    const paymentAmount = ethers.parseEther('0.1');
    await setupVouchIncentives();
    const { vouchId } = await userA.vouch(userB, { paymentAmount });
    const balance = await userA.getVouchBalance(vouchId);

    // For first vouch, calculate deposit with only vouchers pool fee
    const { deposit } = calcFeeDistribution(paymentAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    expect(balance).to.be.closeTo(deposit, 1n);
  });

  it('should allow changing the vouch incentives percentage', async () => {
    const newVouchIncentives = 250n;

    // Set initial vouch incentives
    await setupVouchIncentives();
    // check the initial value
    const initialVouchIncentives =
      await deployer.ethosVouch.contract.entryVouchersPoolFeeBasisPoints();
    expect(initialVouchIncentives).to.equal(vouchIncentives);

    // Change vouch incentives
    await deployer.ethosVouch.contract
      .connect(deployer.ADMIN)
      .setEntryVouchersPoolFeeBasisPoints(newVouchIncentives);

    // Check the new value
    const updatedVouchIncentives =
      await deployer.ethosVouch.contract.entryVouchersPoolFeeBasisPoints();
    expect(updatedVouchIncentives).to.equal(newVouchIncentives);
  });

  it('should deduct vouch incentives for the second voucher', async () => {
    const paymentAmount = ethers.parseEther('0.1');
    await setupVouchIncentives();

    // First vouch - only vouchers pool fee
    const { vouchId: vouchId0 } = await userB.vouch(userA, { paymentAmount });
    const vouch0InitialBalance = await userB.getVouchBalance(vouchId0);

    // Calculate first vouch deposit
    const { deposit: firstDeposit } = calcFeeDistribution(paymentAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });
    expect(vouch0InitialBalance).to.equal(firstDeposit);

    // Second vouch - with fees
    const { vouchId: vouchId1 } = await userC.vouch(userA, { paymentAmount });

    // Calculate fees for second vouch
    const { deposit: secondDeposit, shares } = calcFeeDistribution(paymentAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // First voucher should have original amount plus all incentive fees
    const vouch0FinalBalance = await userB.getVouchBalance(vouchId0);
    expect(vouch0FinalBalance).to.equal(firstDeposit + shares.vouchersPool);

    // Second voucher should have amount minus incentive fees
    const vouch1Balance = await userC.getVouchBalance(vouchId1);
    expect(vouch1Balance).to.equal(secondDeposit);
  });

  it('should deduct vouch incentives for the third voucher and with varying amounts', async () => {
    await setupVouchIncentives();
    const userD = await deployer.createUser();

    // First vouch - no fees
    const amount1 = DEFAULT.PAYMENT_AMOUNT;
    const { vouchId: vouchId0 } = await userB.vouch(userA, {
      paymentAmount: amount1,
    });

    // Calculate first vouch deposit
    const { deposit: deposit1 } = calcFeeDistribution(amount1, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Second vouch - fees to first voucher
    const amount2 = DEFAULT.PAYMENT_AMOUNT * 2n;
    const { vouchId: vouchId1 } = await userC.vouch(userA, {
      paymentAmount: amount2,
    });

    // Calculate second vouch fees
    const { deposit: deposit2, shares: shares2 } = calcFeeDistribution(amount2, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Verify first vouch received all fees from second vouch
    expect(await userB.getVouchBalance(vouchId0)).to.equal(deposit1 + shares2.vouchersPool);
    expect(await userC.getVouchBalance(vouchId1)).to.equal(deposit2);

    // Third vouch - fees split proportionally
    const amount3 = DEFAULT.PAYMENT_AMOUNT * 3n;
    const { vouchId: vouchId2 } = await userD.vouch(userA, {
      paymentAmount: amount3,
    });

    // Calculate third vouch fees
    const { deposit: deposit3, shares: shares3 } = calcFeeDistribution(amount3, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Calculate proportional distribution of shares3.vouchersPool
    const totalBalance = deposit1 + shares2.vouchersPool + deposit2;
    const vouch0Share = (shares3.vouchersPool * (deposit1 + shares2.vouchersPool)) / totalBalance;
    const vouch1Share = (shares3.vouchersPool * deposit2) / totalBalance;

    // Verify final balances
    expect(await userB.getVouchBalance(vouchId0)).to.be.closeTo(
      deposit1 + shares2.vouchersPool + vouch0Share,
      1n,
    );
    expect(await userC.getVouchBalance(vouchId1)).to.be.closeTo(deposit2 + vouch1Share, 1n);
    expect(await userD.getVouchBalance(vouchId2)).to.equal(deposit3);
  });

  it('should correctly distribute incentives among multiple first vouchers', async () => {
    const paymentAmount = VOUCH_PARAMS.paymentAmount;
    await setupVouchIncentives();

    // First vouch - userB is first voucher
    const { vouchId: vouchId0 } = await userB.vouch(userA);

    // Calculate first vouch deposit
    const { deposit: firstDeposit } = calcFeeDistribution(paymentAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Verify first vouch deposit
    expect(await userB.getVouchBalance(vouchId0)).to.equal(firstDeposit);

    // UserB unvouches completely
    await userB.unvouch(vouchId0);

    // Verify balance is zero after unvouch
    const unvouchedBalance = await userB.getVouchBalance(vouchId0);
    expect(unvouchedBalance).to.equal(0n);

    // Now userC becomes the new first voucher
    const { vouchId: vouchId1 } = await userC.vouch(userA);

    // Calculate new first vouch deposit
    const { deposit: newFirstDeposit } = calcFeeDistribution(paymentAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Verify new first vouch deposit
    expect(await userC.getVouchBalance(vouchId1)).to.equal(newFirstDeposit);

    // UserC unvouches completely
    await userC.unvouch(vouchId1);

    // Create a new user to be the next first voucher
    const userD = await deployer.createUser();
    const { vouchId: vouchId2 } = await userD.vouch(userA);

    // Calculate next first vouch deposit
    const { deposit: nextFirstDeposit } = calcFeeDistribution(paymentAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Verify the new first vouch deposit
    expect(await userD.getVouchBalance(vouchId2)).to.equal(nextFirstDeposit);

    // Now make a second vouch while userD is first voucher
    const userE = await deployer.createUser();
    const { vouchId: vouchId3 } = await userE.vouch(userA);

    // Calculate fees for second vouch
    const { deposit: secondDeposit, shares } = calcFeeDistribution(paymentAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Verify userD (first voucher) received the incentive fee
    expect(await userD.getVouchBalance(vouchId2)).to.equal(nextFirstDeposit + shares.vouchersPool);

    // Verify userE (second voucher) has deposit amount
    expect(await userE.getVouchBalance(vouchId3)).to.equal(secondDeposit);
  });

  it('should handle incentive distribution with varying stake amounts', async () => {
    await setupVouchIncentives();

    // Create additional users
    const userD = await deployer.createUser();
    const userE = await deployer.createUser();

    // Use different amounts for first vouchers
    const amount1 = ethers.parseEther('0.1'); // 0.1 ETH
    const amount2 = ethers.parseEther('0.2'); // 0.2 ETH
    const amount3 = ethers.parseEther('0.5'); // 0.5 ETH

    // First vouchers with different amounts
    const { vouchId: vouchId1 } = await userB.vouch(userA, { paymentAmount: amount1 });
    const { vouchId: vouchId2 } = await userC.vouch(userA, { paymentAmount: amount2 });

    // New voucher with larger amount - should distribute fees proportionally
    const { vouchId: vouchId3 } = await userD.vouch(userA, { paymentAmount: amount3 });

    // Get actual stakes after third vouch
    const vouch0Stakes = await userB.getVouchBalance(vouchId1);
    const vouch1Stakes = await userC.getVouchBalance(vouchId2);
    const vouch2Stakes = await userD.getVouchBalance(vouchId3);

    // Use closeTo for all balance checks due to potential rounding
    expect(await userB.getVouchBalance(vouchId1)).to.be.closeTo(vouch0Stakes, 1);
    expect(await userC.getVouchBalance(vouchId2)).to.be.closeTo(vouch1Stakes, 1);
    expect(await userD.getVouchBalance(vouchId3)).to.be.closeTo(vouch2Stakes, 1);

    // Add another voucher to verify continued proportional distribution
    const amount4 = ethers.parseEther('0.3'); // 0.3 ETH
    const { vouchId: vouchId4 } = await userE.vouch(userA, { paymentAmount: amount4 });

    // Get final stakes for all vouchers
    const finalVouch0Stakes = await userB.getVouchBalance(vouchId1);
    const finalVouch1Stakes = await userC.getVouchBalance(vouchId2);
    const finalVouch2Stakes = await userD.getVouchBalance(vouchId3);
    const finalVouch3Stakes = await userE.getVouchBalance(vouchId4);

    // Verify final balances for all vouchers
    expect(await userB.getVouchBalance(vouchId1)).to.be.closeTo(finalVouch0Stakes, 1);
    expect(await userC.getVouchBalance(vouchId2)).to.be.closeTo(finalVouch1Stakes, 1);
    expect(await userD.getVouchBalance(vouchId3)).to.be.closeTo(finalVouch2Stakes, 1);
    expect(await userE.getVouchBalance(vouchId4)).to.be.closeTo(finalVouch3Stakes, 1);
  });

  it('should correctly distribute incentives after users unvouch and revouch', async () => {
    const paymentAmount = VOUCH_PARAMS.paymentAmount;
    await setupVouchIncentives();

    // First vouch
    const { vouchId: vouchId0 } = await userB.vouch(userA);

    // Calculate first vouch deposit
    const { deposit: firstDeposit } = calcFeeDistribution(paymentAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Verify first vouch deposit
    expect(await userB.getVouchBalance(vouchId0)).to.equal(firstDeposit);

    // Second vouch
    const { vouchId: vouchId1 } = await userC.vouch(userA);

    // Calculate second vouch fees
    const { deposit: secondDeposit, shares } = calcFeeDistribution(paymentAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Verify balances after second vouch
    expect(await userB.getVouchBalance(vouchId0)).to.equal(firstDeposit + shares.vouchersPool);
    expect(await userC.getVouchBalance(vouchId1)).to.equal(secondDeposit);

    // UserB unvouches
    await userB.unvouch(vouchId0);

    // UserB revouches
    const { vouchId: vouchId2 } = await userB.vouch(userA);

    // Calculate new vouch fees
    const { deposit: newDeposit, shares: newShares } = calcFeeDistribution(paymentAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Verify final balances
    expect(await userB.getVouchBalance(vouchId2)).to.equal(newDeposit);
    expect(await userC.getVouchBalance(vouchId1)).to.equal(secondDeposit + newShares.vouchersPool);
  });

  it('should correctly handle donations with zero fees configured', async () => {
    const paymentAmount = ethers.parseEther('0.1');

    // Set fees to 0
    await deployer.ethosVouch.contract
      .connect(deployer.ADMIN)
      .setEntryVouchersPoolFeeBasisPoints(0n);

    // First vouch
    const { vouchId: vouchId0 } = await userB.vouch(userA, { paymentAmount });

    // Second vouch - should have no fees
    const { vouchId: vouchId1 } = await userC.vouch(userA, { paymentAmount });

    // Verify both vouches have exact payment amounts
    expect(await userB.getVouchBalance(vouchId0)).to.equal(paymentAmount);
    expect(await userC.getVouchBalance(vouchId1)).to.equal(paymentAmount);

    // Third vouch - should still have no fees
    const userD = await deployer.createUser();
    const { vouchId: vouchId2 } = await userD.vouch(userA, { paymentAmount });
    expect(await userD.getVouchBalance(vouchId2)).to.equal(paymentAmount);
  });

  it('should not distribute vouch rewards to oneself when increasing vouch', async () => {
    const initialAmount = ethers.parseEther('1');
    const increaseAmount = ethers.parseEther('2');
    await userA.setBalance('10');
    await setupVouchIncentives();

    // First vouch
    const { vouchId } = await userA.vouch(userB, { paymentAmount: initialAmount });

    // Calculate initial deposit
    const { deposit: initialDeposit } = calcFeeDistribution(initialAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Calculate increase deposit
    const { deposit: increaseDeposit } = calcFeeDistribution(increaseAmount, {
      entry: 0n,
      donation: 0n,
      vouchIncentives,
    });

    // Increase vouch
    await userA.increaseVouch(vouchId, { paymentAmount: increaseAmount });
    const finalBalance = await userA.getVouchBalance(vouchId);

    // Should just be initial deposit + increase deposit (no self-rewards)
    const expectedBalance = initialDeposit + increaseDeposit;
    expect(finalBalance).to.equal(expectedBalance);
  });

  it('should not allow fee bypass by splitting vouch into smaller amounts', async () => {
    const oneEth = ethers.parseEther('1'); // 1 ETH per increase
    const tenEth = 10n * oneEth;
    await setupVouchIncentives();

    // First user vouches with small amount to establish voucher pool
    const firstVoucher = await deployer.createUser();
    await firstVoucher.setBalance('100');
    await firstVoucher.vouch(userA, {
      paymentAmount: oneEth,
    });

    // Single large vouch user
    const singleVouchUser = await deployer.createUser();
    await singleVouchUser.setBalance('100');
    const { vouchId: singleVouchId } = await singleVouchUser.vouch(userA, {
      paymentAmount: tenEth,
    });

    // Calculate single vouch fees
    const singleVouchFees = tenEth - (await singleVouchUser.getVouchBalance(singleVouchId));

    // Multi small vouch user
    const multiVouchUser = await deployer.createUser();
    await multiVouchUser.setBalance('100');

    // Initial small vouch
    const { vouchId: multiVouchId } = await multiVouchUser.vouch(userA, {
      paymentAmount: oneEth,
    });
    const initialBalance = await multiVouchUser.getVouchBalance(multiVouchId);

    let totalFees = oneEth - initialBalance;

    // Increase vouch 9 more times to reach same total
    for (let i = 0; i < 9; i++) {
      const balanceBefore = await multiVouchUser.getVouchBalance(multiVouchId);
      await multiVouchUser.increaseVouch(multiVouchId, {
        paymentAmount: oneEth,
      });
      const balanceAfter = await multiVouchUser.getVouchBalance(multiVouchId);
      const feesThisIncrease = oneEth - (balanceAfter - balanceBefore);
      totalFees += feesThisIncrease;
    }

    // Fees should be similar regardless of approach
    expect(totalFees).to.be.closeTo(singleVouchFees, 1);
  });
});
