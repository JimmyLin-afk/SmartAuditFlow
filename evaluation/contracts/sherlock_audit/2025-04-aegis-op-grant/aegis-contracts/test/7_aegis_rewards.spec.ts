import { ethers } from 'hardhat'
import { loadFixture, time } from '@nomicfoundation/hardhat-network-helpers'
import { expect } from 'chai'

import {
  REWARDS_MANAGER_ROLE,
  OrderType,
  deployFixture,
  encodeString,
  signClaimRequest,
  signClaimRequestByWallet,
  signOrder,
  FUNDS_MANAGER_ROLE,
  SETTINGS_MANAGER_ROLE,
} from './helpers'

describe('AegisRewards', () => {
  describe('#depositRewards', () => {
    describe('success', () => {
      it('should add rewards to a total amount', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisRewardsContract, yusdContract } = await loadFixture(deployFixture)

        await yusdContract.setMinter(owner)
        await aegisRewardsContract.setAegisMintingAddress(owner)

        const snapshotId = 'test'
        const amount = ethers.parseEther('1')
        await yusdContract.mint(owner, amount)
        await yusdContract.approve(aegisRewardsContract, amount)

        const blockTime = await time.latest()

        await expect(aegisRewardsContract.depositRewards(encodeString(snapshotId), amount)).to.
          emit(aegisRewardsContract, 'DepositRewards').
          withArgs(ethers.encodeBytes32String(snapshotId), amount, blockTime+1)
      })
    })

    describe('error', () => {
      it('should revert when caller is not AegisMinting contract', async () => {
        const { aegisRewardsContract } = await loadFixture(deployFixture)

        await expect(aegisRewardsContract.depositRewards(encodeString('test'), ethers.parseEther('1'))).to.be.reverted
      })
    })
  })

  describe('#claimRewards', () => {
    describe('success', () => {
      it('should claim rewards to account', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, aegisRewardsContract, aegisRewardsAddress, assetContract, assetAddress, yusdContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisRewardsContract.grantRole(REWARDS_MANAGER_ROLE, owner)
        await aegisMintingContract.setInsuranceFundAddress(ethers.ZeroAddress)

        const snapshotId = 'test'
        const amount = ethers.parseEther('2')
        {
          await assetContract.mint(aegisMintingAddress, amount)
          const order = {
            orderType: OrderType.DEPOSIT_INCOME,
            userWallet: owner.address,
            beneficiary: ethers.ZeroAddress,
            collateralAsset: assetAddress,
            collateralAmount: amount,
            yusdAmount: amount,
            slippageAdjustedAmount: 0,
            expiry: (await time.latest()) + 10000,
            nonce: Date.now(),
            additionalData: encodeString(snapshotId),
          }
          const signature = await signOrder(order, aegisMintingAddress)

          await aegisMintingContract.depositIncome(order, signature)
          await aegisRewardsContract.finalizeRewards(ethers.encodeBytes32String(snapshotId), 0)
        }

        const snapshot2Id = 'test2'
        const amount2 = ethers.parseEther('2')
        {
          await assetContract.mint(aegisMintingAddress, amount2)
          const order = {
            orderType: OrderType.DEPOSIT_INCOME,
            userWallet: owner.address,
            beneficiary: ethers.ZeroAddress,
            collateralAsset: assetAddress,
            collateralAmount: amount2,
            yusdAmount: amount2,
            slippageAdjustedAmount: 0,
            expiry: (await time.latest()) + 10000,
            nonce: Date.now(),
            additionalData: encodeString(snapshot2Id),
          }
          const signature = await signOrder(order, aegisMintingAddress)

          await aegisMintingContract.depositIncome(order, signature)
          await aegisRewardsContract.finalizeRewards(ethers.encodeBytes32String(snapshot2Id), 0)
        }

        const contractYUSDBalanceBefore = await yusdContract.balanceOf(aegisRewardsAddress)
        const userYUSDBalanceBefore = await yusdContract.balanceOf(owner.address)

        const claimRequest = {
          claimer: owner.address,
          ids: [ethers.encodeBytes32String(snapshotId), ethers.encodeBytes32String(snapshot2Id)],
          amounts: [amount, amount2],
        }
        const signature = await signClaimRequest(claimRequest, aegisRewardsAddress)

        await expect(aegisRewardsContract.claimRewards(claimRequest, signature)).to.
          emit(aegisRewardsContract, 'ClaimRewards').
          withArgs(owner.address, [ethers.encodeBytes32String(snapshotId), ethers.encodeBytes32String(snapshot2Id)], amount + amount2)

        await expect(yusdContract.balanceOf(aegisRewardsAddress)).eventually.to.be.equal(contractYUSDBalanceBefore - amount - amount2)
        await expect(yusdContract.balanceOf(owner.address)).eventually.to.be.equal(userYUSDBalanceBefore + amount + amount2)

        const reward = await aegisRewardsContract.rewardById(snapshotId)
        expect(reward.amount).to.be.equal(0)
      })
    })

    describe('error', () => {
      it('should revert when deposit for snapshot id does not exist', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisRewardsContract, aegisRewardsAddress } = await loadFixture(deployFixture)

        const amount = ethers.parseEther('1')
        const snapshotId = 'test'
        const claimRequest = {
          claimer: owner.address,
          ids: [ethers.encodeBytes32String(snapshotId)],
          amounts: [amount],
        }
        const signature = await signClaimRequest(claimRequest, aegisRewardsAddress)

        await expect(aegisRewardsContract.claimRewards(claimRequest, signature)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'ZeroRewards')
      })

      it('should revert when caller is not a claimer', async () => {
        const [owner, sender] = await ethers.getSigners()
        const { aegisRewardsContract, aegisRewardsAddress, yusdContract } = await loadFixture(deployFixture)

        await yusdContract.setMinter(owner.address)
        await aegisRewardsContract.setAegisMintingAddress(owner)

        const snapshotId = 'test'
        const amount = ethers.parseEther('1')
        await yusdContract.mint(owner, amount)
        await yusdContract.approve(aegisRewardsContract, amount)
        await aegisRewardsContract.depositRewards(encodeString(snapshotId), amount)

        const claimRequest = {
          claimer: owner.address,
          ids: [ethers.encodeBytes32String(snapshotId)],
          amounts: [amount],
        }
        const signature = await signClaimRequest(claimRequest, aegisRewardsAddress)

        await expect(aegisRewardsContract.connect(sender).claimRewards(claimRequest, signature)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'InvalidClaimer')
      })

      it('should revert when signed by unknown account', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisRewardsContract, aegisRewardsAddress, yusdContract } = await loadFixture(deployFixture)

        await yusdContract.setMinter(owner.address)
        await aegisRewardsContract.setAegisMintingAddress(owner)

        const snapshotId = 'test'
        const amount = ethers.parseEther('1')
        await yusdContract.mint(owner, amount)
        await yusdContract.approve(aegisRewardsContract, amount)
        await aegisRewardsContract.depositRewards(encodeString(snapshotId), amount)

        const unknownSigner = await ethers.Wallet.createRandom()
        const claimRequest = {
          claimer: owner.address,
          ids: [ethers.encodeBytes32String(snapshotId)],
          amounts: [amount],
        }
        const signature = await signClaimRequestByWallet(claimRequest, aegisRewardsAddress, unknownSigner)

        await expect(aegisRewardsContract.claimRewards(claimRequest, signature)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'InvalidSignature')
      })

      it('should revert when length of ids and amounts does not match', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisRewardsContract, aegisRewardsAddress, yusdContract } = await loadFixture(deployFixture)

        await yusdContract.setMinter(owner.address)
        await aegisRewardsContract.setAegisMintingAddress(owner)

        const snapshotId = 'test'
        const amount = ethers.parseEther('1')
        await yusdContract.mint(owner, amount)
        await yusdContract.approve(aegisRewardsContract, amount)
        await aegisRewardsContract.depositRewards(encodeString(snapshotId), amount)

        const claimRequest = {
          claimer: owner.address,
          ids: [ethers.encodeBytes32String(snapshotId), ethers.encodeBytes32String('test2')],
          amounts: [amount],
        }
        const signature = await signClaimRequest(claimRequest, aegisRewardsAddress)

        await expect(aegisRewardsContract.claimRewards(claimRequest, signature)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'InvalidParams')
      })

      it('should revert when snapshot rewards are zero', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, aegisRewardsContract, aegisRewardsAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisRewardsContract.grantRole(REWARDS_MANAGER_ROLE, owner.address)
        await aegisMintingContract.setInsuranceFundAddress(ethers.ZeroAddress)

        const snapshotId = 'test'
        const amount = ethers.parseEther('1')
        {
          await assetContract.mint(aegisMintingAddress, amount)
          const order = {
            orderType: OrderType.DEPOSIT_INCOME,
            userWallet: owner.address,
            beneficiary: ethers.ZeroAddress,
            collateralAsset: assetAddress,
            collateralAmount: amount,
            yusdAmount: amount,
            slippageAdjustedAmount: 0,
            expiry: (await time.latest()) + 10000,
            nonce: Date.now(),
            additionalData: encodeString(snapshotId),
          }
          const signature = await signOrder(order, aegisMintingAddress)

          await aegisMintingContract.depositIncome(order, signature)
          await aegisRewardsContract.finalizeRewards(ethers.encodeBytes32String(snapshotId), 0)
        }

        const claimRequest = {
          claimer: owner.address,
          ids: [ethers.encodeBytes32String(snapshotId)],
          amounts: [amount],
        }
        const signature = await signClaimRequest(claimRequest, aegisRewardsAddress)

        await expect(aegisRewardsContract.claimRewards(claimRequest, signature)).to.be.not.reverted

        await expect(aegisRewardsContract.claimRewards(claimRequest, signature)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'ZeroRewards')
      })

      it('should revert when rewards were already claimed by an address', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, aegisRewardsContract, aegisRewardsAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisRewardsContract.grantRole(REWARDS_MANAGER_ROLE, owner.address)
        await aegisMintingContract.setInsuranceFundAddress(ethers.ZeroAddress)

        const snapshotId = 'test'
        const amount = ethers.parseEther('2')
        {
          await assetContract.mint(aegisMintingAddress, amount)
          const order = {
            orderType: OrderType.DEPOSIT_INCOME,
            userWallet: owner.address,
            beneficiary: ethers.ZeroAddress,
            collateralAsset: assetAddress,
            collateralAmount: amount,
            yusdAmount: amount,
            slippageAdjustedAmount: 0,
            expiry: (await time.latest()) + 10000,
            nonce: Date.now(),
            additionalData: encodeString(snapshotId),
          }
          const signature = await signOrder(order, aegisMintingAddress)

          await aegisMintingContract.depositIncome(order, signature)
          await aegisRewardsContract.finalizeRewards(ethers.encodeBytes32String(snapshotId), 0)
        }

        const claimRequest = {
          claimer: owner.address,
          ids: [ethers.encodeBytes32String(snapshotId)],
          amounts: [amount / 2n],
        }
        const signature = await signClaimRequest(claimRequest, aegisRewardsAddress)

        await expect(aegisRewardsContract.claimRewards(claimRequest, signature)).to.be.not.reverted

        await expect(aegisRewardsContract.claimRewards(claimRequest, signature)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'ZeroRewards')
      })
    })
  })

  describe('#finalizeRewards', () => {
    describe('success', () => {
      it('should finalize rewards with id', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisRewardsContract } = await loadFixture(deployFixture)

        await aegisRewardsContract.grantRole(REWARDS_MANAGER_ROLE, owner)

        const snapshotId = ethers.encodeBytes32String('test')
        const claimDuration = 10
        const timestamp = await time.latest()
        await expect(aegisRewardsContract.finalizeRewards(snapshotId, claimDuration)).to.
          emit(aegisRewardsContract, 'FinalizeRewards').
          withArgs(snapshotId, timestamp+1+claimDuration)

        const reward = await aegisRewardsContract.rewardById('test')
        expect(reward.finalized).to.be.equal(true)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have REWARDS_MANAGER_ROLE role', async () => {
        const [,unknownUser] = await ethers.getSigners()
        const { aegisRewardsContract } = await loadFixture(deployFixture)

        await expect(aegisRewardsContract.connect(unknownUser).finalizeRewards(ethers.encodeBytes32String('test'), 0)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when already finalized', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisRewardsContract } = await loadFixture(deployFixture)

        await aegisRewardsContract.grantRole(REWARDS_MANAGER_ROLE, owner)

        await expect(aegisRewardsContract.finalizeRewards(ethers.encodeBytes32String('test'), 0)).not.to.be.reverted

        await expect(aegisRewardsContract.finalizeRewards(ethers.encodeBytes32String('test'), 0)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'UnknownRewards')
      })
    })
  })

  describe('#withdrawExpiredRewards', () => {
    describe('success', () => {
      it('should withdraw expired rewards', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, aegisRewardsContract, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisRewardsContract.grantRole(REWARDS_MANAGER_ROLE, owner.address)
        await aegisMintingContract.setInsuranceFundAddress(ethers.ZeroAddress)

        const snapshotId = 'test'
        const amount = ethers.parseEther('2')
        {
          await assetContract.mint(aegisMintingAddress, amount)
          const order = {
            orderType: OrderType.DEPOSIT_INCOME,
            userWallet: owner.address,
            beneficiary: ethers.ZeroAddress,
            collateralAsset: assetAddress,
            collateralAmount: amount,
            yusdAmount: amount,
            slippageAdjustedAmount: 0,
            expiry: (await time.latest()) + 10000,
            nonce: Date.now(),
            additionalData: encodeString(snapshotId),
          }
          const signature = await signOrder(order, aegisMintingAddress)

          await aegisMintingContract.depositIncome(order, signature)
          await aegisRewardsContract.finalizeRewards(ethers.encodeBytes32String(snapshotId), 1)
        }

        await time.increase(2)

        await expect(aegisRewardsContract.withdrawExpiredRewards(ethers.encodeBytes32String(snapshotId), owner.address)).to.
          emit(aegisRewardsContract, 'WithdrawExpiredRewards').
          withArgs(ethers.encodeBytes32String(snapshotId), owner.address, amount)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have REWARDS_MANAGER_ROLE role', async () => {
        const [,unknownUser] = await ethers.getSigners()
        const { aegisRewardsContract } = await loadFixture(deployFixture)

        await expect(aegisRewardsContract.connect(unknownUser).withdrawExpiredRewards(ethers.encodeBytes32String('test'), unknownUser.address)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when reward is not finalized', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisRewardsContract } = await loadFixture(deployFixture)

        await aegisRewardsContract.grantRole(REWARDS_MANAGER_ROLE, owner)

        await expect(aegisRewardsContract.withdrawExpiredRewards(ethers.encodeBytes32String('test'), owner.address)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'UnknownRewards')
      })

      it('should revert when amount equals to zero', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, aegisRewardsContract, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisRewardsContract.grantRole(REWARDS_MANAGER_ROLE, owner.address)
        await aegisMintingContract.setInsuranceFundAddress(ethers.ZeroAddress)

        const snapshotId = 'test'
        const amount = ethers.parseEther('2')
        {
          await assetContract.mint(aegisMintingAddress, amount)
          const order = {
            orderType: OrderType.DEPOSIT_INCOME,
            userWallet: owner.address,
            beneficiary: ethers.ZeroAddress,
            collateralAsset: assetAddress,
            collateralAmount: amount,
            yusdAmount: amount,
            slippageAdjustedAmount: 0,
            expiry: (await time.latest()) + 10000,
            nonce: Date.now(),
            additionalData: encodeString(snapshotId),
          }
          const signature = await signOrder(order, aegisMintingAddress)

          await aegisMintingContract.depositIncome(order, signature)
          await aegisRewardsContract.finalizeRewards(ethers.encodeBytes32String(snapshotId), 1)
        }

        await time.increase(2)

        await expect(aegisRewardsContract.withdrawExpiredRewards(ethers.encodeBytes32String(snapshotId), owner.address)).not.to.be.reverted

        await expect(aegisRewardsContract.withdrawExpiredRewards(ethers.encodeBytes32String(snapshotId), owner.address)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'UnknownRewards')
      })

      it('should revert when expiry equals to zero', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisRewardsContract, yusdContract } = await loadFixture(deployFixture)

        await yusdContract.setMinter(owner.address)
        await aegisRewardsContract.setAegisMintingAddress(owner)

        await aegisRewardsContract.grantRole(REWARDS_MANAGER_ROLE, owner)

        const snapshotId = 'test'
        const amount = ethers.parseEther('2')
        await yusdContract.mint(owner, amount)
        await yusdContract.approve(aegisRewardsContract, amount)
        await aegisRewardsContract.depositRewards(encodeString(snapshotId), amount)

        const bytes32SnapshotId = ethers.encodeBytes32String(snapshotId)
        await expect(aegisRewardsContract.finalizeRewards(bytes32SnapshotId, 0)).not.to.be.reverted

        await expect(aegisRewardsContract.withdrawExpiredRewards(bytes32SnapshotId, owner.address)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'UnknownRewards')
      })

      it('should revert when not expired', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisRewardsContract, yusdContract } = await loadFixture(deployFixture)

        await yusdContract.setMinter(owner.address)
        await aegisRewardsContract.setAegisMintingAddress(owner)

        await aegisRewardsContract.grantRole(REWARDS_MANAGER_ROLE, owner)

        const snapshotId = 'test'
        const amount = ethers.parseEther('2')
        await yusdContract.mint(owner, amount)
        await yusdContract.approve(aegisRewardsContract, amount)
        await aegisRewardsContract.depositRewards(encodeString(snapshotId), amount)

        const bytes32SnapshotId = ethers.encodeBytes32String(snapshotId)
        await expect(aegisRewardsContract.finalizeRewards(bytes32SnapshotId, 100)).not.to.be.reverted

        await expect(aegisRewardsContract.withdrawExpiredRewards(bytes32SnapshotId, owner.address)).to.be.
          revertedWithCustomError(aegisRewardsContract, 'UnknownRewards')
      })
    })
  })
})