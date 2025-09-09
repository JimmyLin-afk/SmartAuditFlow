import { ethers } from 'hardhat'
import { loadFixture, time } from '@nomicfoundation/hardhat-network-helpers'
import { expect } from 'chai'

import {
  MAX_BPS,
  FUNDS_MANAGER_ROLE,
  COLLATERAL_MANAGER_ROLE,
  INCOME_FEE_BP,
  OrderType,
  deployFixture,
  custodianAccount,
  insuranceFundAccount,
  signOrder,
  signOrderByWallet,
  encodeString,
  SETTINGS_MANAGER_ROLE,
} from './helpers'

describe('AegisMinting', () => {
  describe('#transferToCustody', () => {
    describe('success', () => {
      it('should transfer correct amount of selected asset to a custody', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)
        await aegisConfig.setOperator(owner, true)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('1000'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const collateralAmount = ethers.parseEther('100')
        const yusdAmount = ethers.parseEther('99.999')

        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.MINT,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        // Mint tokens first
        await expect(aegisMintingContract.connect(sender).mint(order, signature)).not.to.be.reverted

        const mintingContractAssetBalanceBefore = await assetContract.balanceOf(aegisMintingAddress)
        const custodyAvailableAssetBalanceBefore = await aegisMintingContract.custodyAvailableAssetBalance(assetAddress)
        const custodyBalanceBefore = await assetContract.balanceOf(custodianAccount.address)

        const transferAmount = ethers.parseEther('60')
        await expect(aegisMintingContract.transferToCustody(custodianAccount.address, assetAddress, transferAmount)).to.
          emit(aegisMintingContract, 'CustodyTransfer').
          withArgs(custodianAccount.address, assetAddress, transferAmount)

        await expect(assetContract.balanceOf(aegisMintingAddress)).eventually.to.be.equal(mintingContractAssetBalanceBefore - transferAmount)
        await expect(aegisMintingContract.custodyAvailableAssetBalance(assetAddress)).eventually.to.be.equal(custodyAvailableAssetBalanceBefore - transferAmount)
        await expect(assetContract.balanceOf(custodianAccount.address)).eventually.to.be.equal(custodyBalanceBefore + transferAmount)
      })

      it('should transfer correct amount of selected asset except frozen funds to a custody', async () => {
        const [owner, sender] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('1000'))
        await assetContract.connect(sender).approve(aegisMintingContract, ethers.parseEther('1000'))

        const collateralAmount = ethers.parseEther('100')
        const yusdAmount = ethers.parseEther('99.999')

        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.MINT,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        // Mint tokens first
        await expect(aegisMintingContract.connect(sender).mint(order, signature)).not.to.be.reverted

        const freezeFunds = ethers.parseEther('10')
        await aegisMintingContract.freezeFunds(assetContract, freezeFunds)

        const transferAmount = collateralAmount - freezeFunds
        await expect(aegisMintingContract.transferToCustody(custodianAccount.address, assetContract, transferAmount)).to.
          emit(aegisMintingContract, 'CustodyTransfer').
          withArgs(custodianAccount.address, assetAddress, transferAmount)

        await expect(aegisMintingContract.custodyAvailableAssetBalance(assetAddress)).to.be.eventually.equal(0)
        await expect(aegisMintingContract.assetFrozenFunds(assetAddress)).to.be.eventually.equal(freezeFunds)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have COLLATERAL_MANAGER_ROLE role', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).transferToCustody(custodianAccount.address, assetAddress, ethers.parseEther('1'))).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when transferring unsupported asset', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)

        const unknownAsset = await ethers.Wallet.createRandom()

        await expect(aegisMintingContract.transferToCustody(custodianAccount.address, unknownAsset.address, ethers.parseEther('1'))).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })

      it('should revert when transferring to unknown custody wallet', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)

        const unknownCustody = await ethers.Wallet.createRandom()

        await expect(aegisMintingContract.transferToCustody(unknownCustody.address, assetAddress, ethers.parseEther('1'))).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidCustodianAddress')
      })

      it('should revert when transferring amount is greater than available balance', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.transferToCustody(custodianAccount.address, assetAddress, ethers.parseEther('1'))).to.be.
          revertedWithCustomError(aegisMintingContract, 'NotEnoughFunds')
      })

      it('should revert when transferring amount is greater than available balance with frozen funds', async () => {
        const [owner, sender] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('1000'))
        await assetContract.connect(sender).approve(aegisMintingContract, ethers.parseEther('1000'))

        const collateralAmount = ethers.parseEther('100')
        const yusdAmount = ethers.parseEther('99.999')

        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.MINT,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        // Mint tokens first
        await expect(aegisMintingContract.connect(sender).mint(order, signature)).not.to.be.reverted

        await aegisMintingContract.freezeFunds(assetContract, ethers.parseEther('10'))

        await expect(aegisMintingContract.transferToCustody(custodianAccount.address, assetContract, collateralAmount)).to.be.
          revertedWithCustomError(aegisMintingContract, 'NotEnoughFunds')
      })
    })
  })

  describe('#forceTransferToCustody', () => {
    describe('success', () => {
      it('should transfer all available funds of selected asset to a custody', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('1000'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const collateralAmount = ethers.parseEther('100')
        const yusdAmount = ethers.parseEther('99.999')
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.MINT,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).not.to.be.reverted

        const custodianBalanceBefore = await assetContract.balanceOf(custodianAccount)

        await expect(aegisMintingContract.forceTransferToCustody(custodianAccount, assetAddress)).to.
          emit(aegisMintingContract, 'ForceCustodyTransfer').
          withArgs(custodianAccount, assetAddress, collateralAmount)

        await expect(assetContract.balanceOf(aegisMintingContract)).to.be.eventually.equal(0)
        await expect(assetContract.balanceOf(custodianAccount)).to.be.eventually.equal(custodianBalanceBefore + collateralAmount)
      })

      it('should transfer all available funds of selected asset except frozen funds to a custody', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('1000'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const collateralAmount = ethers.parseEther('100')
        const yusdAmount = ethers.parseEther('99.999')
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.MINT,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).not.to.be.reverted

        const freezeAmount = ethers.parseEther('50')
        await aegisMintingContract.freezeFunds(assetAddress, freezeAmount)

        const transferAmount = collateralAmount - freezeAmount
        const custodianBalanceBefore = await assetContract.balanceOf(custodianAccount)

        await expect(aegisMintingContract.forceTransferToCustody(custodianAccount, assetAddress)).to.
          emit(aegisMintingContract, 'ForceCustodyTransfer').
          withArgs(custodianAccount, assetAddress, transferAmount)

        await expect(assetContract.balanceOf(aegisMintingContract)).to.be.eventually.equal(freezeAmount)
        await expect(assetContract.balanceOf(custodianAccount)).to.be.eventually.equal(custodianBalanceBefore + transferAmount)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have COLLATERAL_MANAGER_ROLE role', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).forceTransferToCustody(custodianAccount.address, assetAddress)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when transferring unsupported asset', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)

        const unknownAsset = await ethers.Wallet.createRandom()

        await expect(aegisMintingContract.forceTransferToCustody(custodianAccount.address, unknownAsset.address)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })

      it('should revert when transferring to unknown custody wallet', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)

        const unknownCustody = await ethers.Wallet.createRandom()

        await expect(aegisMintingContract.forceTransferToCustody(unknownCustody.address, assetAddress)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidCustodianAddress')
      })

      it('should revert when transferring amount greater than available balance', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(COLLATERAL_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.forceTransferToCustody(custodianAccount.address, assetAddress)).to.be.
          revertedWithCustomError(aegisMintingContract, 'NotEnoughFunds')
      })
    })
  })

  describe('#depositIncome', () => {
    describe('success', () => {
      it('should split YUSD rewards between InsuranceFund and AegisRewards addresses', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisRewardsAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await assetContract.mint(aegisMintingAddress, ethers.parseEther('100'))

        const snapshotId = 'test'
        const collateralAmount = ethers.parseEther('100')
        const yusdAmount = ethers.parseEther('99.9999')
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: owner.address,
          beneficiary: ethers.ZeroAddress,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        const mintingContractAssetBalanceBefore = await assetContract.balanceOf(aegisMintingAddress)
        const custodyAvailableAssetBalanceBefore = await aegisMintingContract.custodyAvailableAssetBalance(assetAddress)
        const insuranceFundYUSDBalanceBefore = await yusdContract.balanceOf(insuranceFundAccount.address)
        const aegisRewardsYUSDBalanceBefore = await yusdContract.balanceOf(aegisRewardsAddress)

        const incomeFee = (yusdAmount * INCOME_FEE_BP) / MAX_BPS
        const aegisRewardsYUSDRewardsAmount = yusdAmount - incomeFee

        await expect(aegisMintingContract.depositIncome(order, signature)).to.
          emit(yusdContract, 'Transfer').
          withArgs(ethers.ZeroAddress, aegisRewardsAddress, aegisRewardsYUSDRewardsAmount).
          emit(aegisMintingContract, 'DepositIncome').
          withArgs(snapshotId, owner.address, assetAddress, collateralAmount, aegisRewardsYUSDRewardsAmount, incomeFee, blockTime+1)

        await expect(assetContract.balanceOf(aegisMintingAddress)).eventually.to.be.equal(mintingContractAssetBalanceBefore)
        await expect(aegisMintingContract.custodyAvailableAssetBalance(assetAddress)).eventually.to.be.equal(custodyAvailableAssetBalanceBefore + collateralAmount)
        await expect(yusdContract.balanceOf(insuranceFundAccount.address)).eventually.to.be.equal(insuranceFundYUSDBalanceBefore + incomeFee)
        await expect(yusdContract.balanceOf(aegisRewardsAddress)).eventually.to.be.equal(aegisRewardsYUSDBalanceBefore + aegisRewardsYUSDRewardsAmount)
      })

      it('should mint all YUSD rewards to AegisRewards when insuranceFundPercentBP is zero', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisRewardsAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await assetContract.mint(aegisMintingAddress, ethers.parseEther('100'))
        await assetContract.approve(aegisMintingAddress, ethers.parseEther('100'))

        // Set insurance fund percent to 0
        await aegisMintingContract.setIncomeFeeBP(0)

        const snapshotId = 'test'
        const collateralAmount = ethers.parseEther('100')
        const yusdAmount = ethers.parseEther('99.9999')
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: owner.address,
          beneficiary: ethers.ZeroAddress,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        const mintingContractAssetBalanceBefore = await assetContract.balanceOf(aegisMintingAddress)
        const custodyAvailableAssetBalanceBefore = await aegisMintingContract.custodyAvailableAssetBalance(assetAddress)
        const aegisRewardsYUSDBalanceBefore = await yusdContract.balanceOf(aegisRewardsAddress)

        await expect(aegisMintingContract.depositIncome(order, signature)).to.
          emit(aegisMintingContract, 'DepositIncome').
          withArgs(snapshotId, owner.address, assetAddress, collateralAmount, yusdAmount, 0, blockTime+1)


        await expect(assetContract.balanceOf(aegisMintingAddress)).eventually.to.be.equal(mintingContractAssetBalanceBefore)
        await expect(aegisMintingContract.custodyAvailableAssetBalance(assetAddress)).eventually.to.be.equal(custodyAvailableAssetBalanceBefore + collateralAmount)
        await expect(yusdContract.balanceOf(aegisRewardsAddress)).eventually.to.be.equal(aegisRewardsYUSDBalanceBefore + yusdAmount)
      })

      it('should mint all YUSD rewards to AegisRewards when InsuranceFund address is zero', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisRewardsAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await assetContract.mint(aegisMintingAddress, ethers.parseEther('100'))

        // Set insurance fund to zero address
        await aegisMintingContract.setInsuranceFundAddress(ethers.ZeroAddress)

        const snapshotId = 'test'
        const collateralAmount = ethers.parseEther('100')
        const yusdAmount = ethers.parseEther('99.9999')
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: owner.address,
          beneficiary: ethers.ZeroAddress,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        const mintingContractAssetBalanceBefore = await assetContract.balanceOf(aegisMintingAddress)
        const custodyAvailableAssetBalanceBefore = await aegisMintingContract.custodyAvailableAssetBalance(assetAddress)
        const aegisRewardsYUSDBalanceBefore = await yusdContract.balanceOf(aegisRewardsAddress)

        await expect(aegisMintingContract.depositIncome(order, signature)).to.
          emit(aegisMintingContract, 'DepositIncome').
          withArgs(snapshotId, owner.address, assetAddress, collateralAmount, yusdAmount, 0, blockTime+1)


        await expect(assetContract.balanceOf(aegisMintingAddress)).eventually.to.be.equal(mintingContractAssetBalanceBefore)
        await expect(aegisMintingContract.custodyAvailableAssetBalance(assetAddress)).eventually.to.be.equal(custodyAvailableAssetBalanceBefore + collateralAmount)
        await expect(yusdContract.balanceOf(aegisRewardsAddress)).eventually.to.be.equal(aegisRewardsYUSDBalanceBefore + yusdAmount)
      })

      it('should not revert when beneficiary address is zero', async () => {
        const [sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, sender)

        await assetContract.mint(aegisMintingAddress, ethers.parseEther('100'))

        const snapshotId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: sender.address,
          beneficiary: ethers.ZeroAddress,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).depositIncome(order, signature)).not.to.be.reverted
      })
    })

    describe('error', () => {
      it('should be reverted when caller does not have FUNDS_MANAGER_ROLE role', async () => {
        const signers = await ethers.getSigners()
        const sender = signers[1]

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const snapshotId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: sender.address,
          beneficiary: ethers.ZeroAddress,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).depositIncome(order, signature)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when OrderType is not DEPOSIT_INCOME', async () => {
        const signers = await ethers.getSigners()
        const sender = signers[1]

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, sender)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const snapshotId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).depositIncome(order, signature)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidOrder')
      })

      it('should revert when collateral asset is not supported', async () => {
        const signers = await ethers.getSigners()
        const sender = signers[1]

        const { aegisMintingContract, aegisMintingAddress, assetContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, sender)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const fakeAsset = await ethers.Wallet.createRandom()

        const snapshotId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: sender.address,
          collateralAsset: fakeAsset.address,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).depositIncome(order, signature)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })

      it('should revert when signed by unknown signer', async () => {
        const signers = await ethers.getSigners()
        const sender = signers[1]

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, sender)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const unknownSigner = await ethers.Wallet.createRandom()

        const snapshotId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrderByWallet(order, aegisMintingAddress, unknownSigner)

        await expect(aegisMintingContract.connect(sender).depositIncome(order, signature)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidSignature')
      })

      it('should revert when collateral amount is zero', async () => {
        const signers = await ethers.getSigners()
        const sender = signers[1]

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, sender)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const snapshotId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: 0,
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).depositIncome(order, signature)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAmount')
      })

      it('should revert when yusd amount is zero', async () => {
        const signers = await ethers.getSigners()
        const sender = signers[1]

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, sender)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const snapshotId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: 0,
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).depositIncome(order, signature)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAmount')
      })

      it('should revert when order expired', async () => {
        const signers = await ethers.getSigners()
        const sender = signers[1]

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, sender)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const blockTime = await time.latest()

        const snapshotId = 'test'
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: 0,
          expiry: blockTime - 1000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).depositIncome(order, signature)).to.be.
          revertedWithCustomError(aegisMintingContract, 'SignatureExpired')
      })

      it('should revert when order was already processed', async () => {
        const signers = await ethers.getSigners()
        const sender = signers[1]

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, sender)

        await assetContract.mint(aegisMintingContract, ethers.parseEther('100'))

        const blockTime = await time.latest()

        const snapshotId = 'test'
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: 0,
          expiry: blockTime + 1000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).depositIncome(order, signature)).to.be.not.reverted

        await expect(aegisMintingContract.connect(sender).depositIncome(order, signature)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidNonce')
      })

      it('should revert when caller is not benefactor', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, sender)

        await assetContract.mint(aegisMintingContract, ethers.parseEther('100'))

        const blockTime = await time.latest()

        const snapshotId = 'test'
        const order = {
          orderType: OrderType.DEPOSIT_INCOME,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: 0,
          expiry: blockTime + 1000,
          nonce: Date.now(),
          additionalData: encodeString(snapshotId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.depositIncome(order, signature)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidSender')
      })
    })
  })
})