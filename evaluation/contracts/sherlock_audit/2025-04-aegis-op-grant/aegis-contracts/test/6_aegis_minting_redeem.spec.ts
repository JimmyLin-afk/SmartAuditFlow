import { ethers } from 'hardhat'
import { loadFixture, time } from '@nomicfoundation/hardhat-network-helpers'
import { expect } from 'chai'

import {
  MAX_BPS,
  FUNDS_MANAGER_ROLE,
  OrderType,
  deployFixture,
  insuranceFundAccount,
  signOrder,
  signOrderByWallet,
  encodeString,
  SETTINGS_MANAGER_ROLE,
  RedeemRequestStatus,
  USD_FEED_ADDRESS,
} from './helpers'

describe('AegisMinting', () => {
  describe('#requestRedeem', () => {
    describe('success', () => {
      it('should create RedeemRequest', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.99')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        const userYUSDBalanceBefore = await yusdContract.balanceOf(sender)
        const contractYUSDBalanceBefore = await yusdContract.balanceOf(aegisMintingContract)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.
          emit(aegisMintingContract, 'CreateRedeemRequest').
          withArgs(requestId, sender.address, assetAddress, collateralAmount, yusdAmount).
          emit(yusdContract, 'Transfer').
          withArgs(sender, aegisMintingContract, yusdAmount)

        const redeemRequest = await aegisMintingContract.getRedeemRequest(requestId)
        const blockTime = await time.latest()

        await expect(yusdContract.balanceOf(sender)).to.be.eventually.equal(userYUSDBalanceBefore - yusdAmount)
        await expect(yusdContract.balanceOf(aegisMintingContract)).to.be.eventually.equal(contractYUSDBalanceBefore + yusdAmount)
        expect(redeemRequest.status).to.be.equal(RedeemRequestStatus.PENDING)
        expect(redeemRequest.timestamp).to.be.equal(blockTime)
        expect(redeemRequest.order.orderType).to.be.equal(redeemOrder.orderType)
        expect(redeemRequest.order.userWallet).to.be.equal(redeemOrder.userWallet)
        expect(redeemRequest.order.collateralAsset).to.be.equal(redeemOrder.collateralAsset)
        expect(redeemRequest.order.collateralAmount).to.be.equal(redeemOrder.collateralAmount)
        expect(redeemRequest.order.yusdAmount).to.be.equal(redeemOrder.yusdAmount)
        expect(redeemRequest.order.slippageAdjustedAmount).to.be.equal(redeemOrder.slippageAdjustedAmount)
        expect(redeemRequest.order.expiry).to.be.equal(redeemOrder.expiry)
        expect(redeemRequest.order.additionalData).to.be.equal(redeemOrder.additionalData)
      })

      it('should create RedeemRequest within limits', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        // Set limits
        const redeemMaxAmount = ethers.parseEther('10')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setRedeemLimits(60, redeemMaxAmount)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.99')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        const redeemLimitBefore = await aegisMintingContract.redeemLimit()

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        const redeemLimitAfter = await aegisMintingContract.redeemLimit()
        expect(redeemLimitAfter.currentPeriodTotalAmount).to.be.equal(redeemLimitBefore.currentPeriodTotalAmount + yusdAmount)
      })

      it('should reset redeem limit counters at the beginning of new period', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        // Set limits
        const redeemMaxAmount = ethers.parseEther('10')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setRedeemLimits(60, redeemMaxAmount)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        {
          const requestId = 'test'
          const collateralAmount = ethers.parseEther('10')
          const yusdAmount = ethers.parseEther('9.99')
          const redeemOrder = {
            orderType: OrderType.REDEEM,
            userWallet: sender.address,
            collateralAsset: assetAddress,
            collateralAmount: collateralAmount,
            yusdAmount: yusdAmount,
            slippageAdjustedAmount: yusdAmount,
            expiry: (await time.latest()) + 10000,
            nonce: Date.now(),
            additionalData: encodeString(requestId),
          }
          const signature = await signOrder(redeemOrder, aegisMintingAddress)

          await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted
        }

        await time.increase(60)

        const requestId = 'test2'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.99')
        let blockTime = await time.latest()
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        blockTime = await time.latest()
        const redeemLimit = await aegisMintingContract.redeemLimit()

        expect(redeemLimit.currentPeriodTotalAmount).to.be.equal(yusdAmount)
        expect(redeemLimit.currentPeriodStartTime).to.be.equal(blockTime)
      })
    })

    describe('error', () => {
      it('should revert when OrderType is not REDEEM', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const requestId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.MINT,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: ethers.parseEther('1'),
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidOrder')
      })

      it('should revert when collateral asset is not supported', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const fakeAsset = await ethers.Wallet.createRandom()

        const requestId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: fakeAsset.address,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: ethers.parseEther('1'),
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })

      it('should revert when signed by unknown signer', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const unknownSigner = await ethers.Wallet.createRandom()

        const requestId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: ethers.parseEther('1'),
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrderByWallet(order, aegisMintingAddress, unknownSigner)

        await expect(aegisMintingContract.connect(sender).requestRedeem(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidSignature')
      })

      it('should revert when collateral amount is zero', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const requestId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: 0,
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: ethers.parseEther('1'),
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidAmount')
      })

      it('should revert when yusd amount is zero', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const requestId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: 0,
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidAmount')
      })

      it('should revert when order expired', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const requestId = 'test'
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: ethers.parseEther('1'),
          expiry: blockTime - 1000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'SignatureExpired')
      })

      it('should revert when redeem is paused', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](owner, true)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner.address)
        await aegisMintingContract.setRedeemPaused(true)

        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.REDEEM,
          userWallet: owner.address,
          beneficiary: owner.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: ethers.parseEther('1'),
          expiry: blockTime - 1000,
          nonce: Date.now(),
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.requestRedeem(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'RedeemPaused')
      })

      it('should revert when benefactor is not in whitelist', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        // Mint asset to sender
        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        // Mint asset to owner
        await assetContract.mint(owner.address, ethers.parseEther('100'))
        await assetContract.connect(owner).approve(aegisMintingAddress, ethers.parseEther('100'))

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.99')
        const requestId = 'test'
        const blockTime = await time.latest()
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'NotWhitelisted')
      })

      it('should revert when RedeemRequest with id already exist', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('10'),
          yusdAmount: ethers.parseEther('9.99'),
          slippageAdjustedAmount: ethers.parseEther('9.99'),
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.revertedWithCustomError(aegisMintingContract, 'InvalidRedeemRequest')
      })

      it('should revert when redeeming amount exceeds max amount within period', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        // Set limits
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setRedeemLimits(60, ethers.parseEther('15'))

        // Mint YUSD to sender
        {
          const blockTime = await time.latest()

          const mintOrder = {
            orderType: OrderType.MINT,
            userWallet: sender.address,
            collateralAsset: assetAddress,
            collateralAmount: ethers.parseEther('50'),
            yusdAmount: ethers.parseEther('50'),
            slippageAdjustedAmount: ethers.parseEther('50'),
            expiry: blockTime + 10000,
            nonce: Date.now(),
            additionalData: encodeString(''),
          }
          const signature = await signOrder(mintOrder, aegisMintingAddress)

          await expect(aegisMintingContract.connect(sender).mint(mintOrder, signature)).not.to.be.reverted
        }

        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        // First redeem
        {
          const requestId = 'test'
          await assetContract.mint(aegisMintingAddress, ethers.parseEther('10'))

          const redeemOrder = {
            orderType: OrderType.REDEEM,
            userWallet: sender.address,
            collateralAsset: assetAddress,
            collateralAmount: ethers.parseEther('10'),
            yusdAmount: ethers.parseEther('9.99'),
            slippageAdjustedAmount: ethers.parseEther('9.99'),
            expiry: (await time.latest()) + 10000,
            nonce: Date.now(),
            additionalData: encodeString(requestId),
          }
          const signature = await signOrder(redeemOrder, aegisMintingAddress)

          await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted
        }

        const requestId = 'test2'
        await assetContract.mint(aegisMintingAddress, ethers.parseEther('10'))

        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('10'),
          yusdAmount: ethers.parseEther('9.99'),
          slippageAdjustedAmount: ethers.parseEther('9.99'),
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.revertedWithCustomError(aegisMintingContract, 'LimitReached')
      })

      it('should revert when redeeming amount exceeds max amount at the beginning of period', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        // Mint asset to owner
        await assetContract.mint(owner, ethers.parseEther('100'))
        await assetContract.connect(owner).approve(aegisMintingAddress, ethers.parseEther('100'))

        // Set limits
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setRedeemLimits(60, ethers.parseEther('15'))

        // Mint YUSD to sender
        {
          const blockTime = await time.latest()

          const mintOrder = {
            orderType: OrderType.MINT,
            userWallet: sender.address,
            collateralAsset: assetAddress,
            collateralAmount: ethers.parseEther('10'),
            yusdAmount: ethers.parseEther('50'),
            slippageAdjustedAmount: ethers.parseEther('50'),
            expiry: blockTime + 10000,
            nonce: Date.now(),
            additionalData: encodeString(''),
          }
          const signature = await signOrder(mintOrder, aegisMintingAddress)

          await expect(aegisMintingContract.connect(sender).mint(mintOrder, signature)).not.to.be.reverted
        }

        // Approve YUSD to be locked by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('10'))

        const requestId = 'test2'
        await assetContract.mint(aegisMintingAddress, ethers.parseEther('16'))

        const blockTime = await time.latest()
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('10'),
          yusdAmount: ethers.parseEther('16'),
          slippageAdjustedAmount: ethers.parseEther('16'),
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'LimitReached')
      })

      it('should revert when calculated collateral amount by Chainlink price is less than min receive amount', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, assetContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        const feedRegistry = await ethers.deployContract('FeedRegistry')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setFeedRegistryAddress(feedRegistry)

        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, '100100000')

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('10')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: collateralAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'PriceSlippage')
      })

      it('should revert when calculated collateral amount by AegisOracle price is less than min receive amount', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, assetContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        const feedRegistry = await ethers.deployContract('FeedRegistry')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setFeedRegistryAddress(feedRegistry)

        // Set feed price to 1 asset/USD to pass check
        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, '100000000')

        const aegisOracle = await ethers.deployContract('AegisOracle', [[owner], owner])
        await aegisMintingContract.setAegisOracleAddress(aegisOracle)

        await aegisOracle.updateYUSDPrice('99963000')

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('10')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: collateralAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'PriceSlippage')
      })

      it('should revert when caller is not benefator', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.99')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.requestRedeem(redeemOrder, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidSender')
      })
    })
  })

  describe('#approveRedeemRequest', () => {
    describe('success', () => {
      it('should approve RedeemRequest', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, assetContract, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const yusdAmount = ethers.parseEther('9.99')
        const collateralAmount = ethers.parseEther('10')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: collateralAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        const custodianAvailableAssetBalanceBefore = await aegisMintingContract.custodyAvailableAssetBalance(assetAddress)

        // Mint asset funds
        await assetContract.mint(aegisMintingContract, collateralAmount)

        const untrackedAvailableAssetBalanceBefore = await aegisMintingContract.untrackedAvailableAssetBalance(assetAddress)
        const contractYUSDBalanceBefore = await yusdContract.balanceOf(aegisMintingContract)
        const userAssetBalanceBefore = await assetContract.balanceOf(sender)

        await expect(aegisMintingContract.approveRedeemRequest(requestId, collateralAmount)).to.
          emit(aegisMintingContract, 'ApproveRedeemRequest').
          withArgs(requestId, owner.address, sender.address, assetAddress, collateralAmount, yusdAmount, 0).
          emit(yusdContract, 'Transfer').
          withArgs(aegisMintingContract, ethers.ZeroAddress, yusdAmount).
          emit(assetContract, 'Transfer').
          withArgs(aegisMintingContract, sender, collateralAmount)

        const redeemRequest = await aegisMintingContract.getRedeemRequest(requestId)

        await expect(aegisMintingContract.custodyAvailableAssetBalance(assetAddress)).to.be.eventually.equal(custodianAvailableAssetBalanceBefore)
        await expect(aegisMintingContract.untrackedAvailableAssetBalance(assetAddress)).to.be.eventually.equal(untrackedAvailableAssetBalanceBefore - collateralAmount)
        await expect(yusdContract.balanceOf(aegisMintingContract)).to.be.eventually.equal(contractYUSDBalanceBefore - yusdAmount)
        await expect(assetContract.balanceOf(sender)).to.be.eventually.equal(userAssetBalanceBefore + collateralAmount)
        expect(redeemRequest.status).to.be.equal(RedeemRequestStatus.APPROVED)
      })

      it('should approve RedeemRequest and take fee', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, assetContract, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        const feeBP = 500n

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setRedeemFeeBP(feeBP)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const yusdAmount = ethers.parseEther('9.99')
        const collateralAmount = ethers.parseEther('10')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: collateralAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        const custodianAvailableAssetBalanceBefore = await aegisMintingContract.custodyAvailableAssetBalance(assetAddress)

        // Mint asset funds
        await assetContract.mint(aegisMintingContract, collateralAmount)

        const untrackedAvailableAssetBalanceBefore = await aegisMintingContract.untrackedAvailableAssetBalance(assetAddress)
        const contractYUSDBalanceBefore = await yusdContract.balanceOf(aegisMintingContract)
        const userAssetBalanceBefore = await assetContract.balanceOf(sender)

        const fee = (yusdAmount * feeBP) / MAX_BPS
        const receiveYUSDAmount = yusdAmount - fee

        await expect(aegisMintingContract.approveRedeemRequest(requestId, collateralAmount)).to.
          emit(aegisMintingContract, 'ApproveRedeemRequest').
          withArgs(requestId, owner.address, sender.address, assetAddress, collateralAmount, receiveYUSDAmount, fee).
          emit(yusdContract, 'Transfer').
          withArgs(aegisMintingContract, ethers.ZeroAddress, receiveYUSDAmount).
          emit(yusdContract, 'Transfer').
          withArgs(aegisMintingContract, insuranceFundAccount, fee).
          emit(assetContract, 'Transfer').
          withArgs(aegisMintingContract, sender, collateralAmount)

        const redeemRequest = await aegisMintingContract.getRedeemRequest(requestId)

        await expect(aegisMintingContract.custodyAvailableAssetBalance(assetAddress)).to.be.eventually.equal(custodianAvailableAssetBalanceBefore)
        await expect(aegisMintingContract.untrackedAvailableAssetBalance(assetAddress)).to.be.eventually.equal(untrackedAvailableAssetBalanceBefore - collateralAmount)
        await expect(yusdContract.balanceOf(aegisMintingContract)).to.be.eventually.equal(contractYUSDBalanceBefore - yusdAmount)
        await expect(assetContract.balanceOf(sender)).to.be.eventually.equal(userAssetBalanceBefore + collateralAmount)
        expect(redeemRequest.status).to.be.equal(RedeemRequestStatus.APPROVED)
      })

      it('should approve RedeemRequest and transfer smallest collateral amount by Chainlink price', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        const feedRegistry = await ethers.deployContract('FeedRegistry')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setFeedRegistryAddress(feedRegistry)

        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, '100000000')

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('10')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: collateralAmount - ethers.parseEther('1'),
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        // Mint asset funds
        await assetContract.mint(aegisMintingContract, collateralAmount)

        const chainlinkPrice = 100100000n
        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, chainlinkPrice)

        const chainlinkCollateralAmount = yusdAmount * 10n ** 8n / chainlinkPrice
        const untrackedAvailableAssetBalanceBefore = await aegisMintingContract.untrackedAvailableAssetBalance(assetAddress)
        const userAssetBalanceBefore = await assetContract.balanceOf(sender)

        await expect(aegisMintingContract.approveRedeemRequest(requestId, collateralAmount)).to.
          emit(aegisMintingContract, 'ApproveRedeemRequest').
          withArgs(requestId, owner.address, sender.address, assetAddress, chainlinkCollateralAmount, yusdAmount, 0).
          emit(yusdContract, 'Transfer').
          withArgs(aegisMintingContract, ethers.ZeroAddress, yusdAmount).
          emit(assetContract, 'Transfer').
          withArgs(aegisMintingContract, sender, chainlinkCollateralAmount)

        await expect(aegisMintingContract.untrackedAvailableAssetBalance(assetAddress)).to.be.eventually.equal(untrackedAvailableAssetBalanceBefore - chainlinkCollateralAmount)
        await expect(assetContract.balanceOf(sender)).to.be.eventually.equal(userAssetBalanceBefore + chainlinkCollateralAmount)
      })

      it('should approve RedeemRequest and transfer smallest collateral amount of initial, Chainlink price based and AegisOracle price based', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        const feedRegistry = await ethers.deployContract('FeedRegistry')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setFeedRegistryAddress(feedRegistry)

        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, '100000000')

        const aegisOracle = await ethers.deployContract('AegisOracle', [[owner], owner])
        await aegisOracle.setOperator(owner, true)
        await aegisMintingContract.setAegisOracleAddress(aegisOracle)

        await aegisOracle.updateYUSDPrice('100000000')

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('10')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: collateralAmount - ethers.parseEther('1'),
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        // Mint asset funds
        await assetContract.mint(aegisMintingContract, collateralAmount)

        const chainlinkPrice = 100100000n
        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, chainlinkPrice)

        const oraclePrice = 99963000n
        await aegisOracle.updateYUSDPrice(oraclePrice)

        const chainlinkCollateralAmount = yusdAmount * 10n ** 8n / chainlinkPrice
        const oracleCollateralAmount = yusdAmount * 10n ** 8n / (chainlinkPrice * 10n ** 8n / oraclePrice)
        let smalletCollateralAmount = collateralAmount
        if (chainlinkCollateralAmount < smalletCollateralAmount) {
          smalletCollateralAmount = chainlinkCollateralAmount
        }
        if (oracleCollateralAmount < smalletCollateralAmount) {
          smalletCollateralAmount = oracleCollateralAmount
        }

        const untrackedAvailableAssetBalanceBefore = await aegisMintingContract.untrackedAvailableAssetBalance(assetAddress)
        const userAssetBalanceBefore = await assetContract.balanceOf(sender)

        await expect(aegisMintingContract.approveRedeemRequest(requestId, collateralAmount)).to.
          emit(aegisMintingContract, 'ApproveRedeemRequest').
          withArgs(requestId, owner.address, sender.address, assetAddress, smalletCollateralAmount, yusdAmount, 0).
          emit(yusdContract, 'Transfer').
          withArgs(aegisMintingContract, ethers.ZeroAddress, yusdAmount).
          emit(assetContract, 'Transfer').
          withArgs(aegisMintingContract, sender, smalletCollateralAmount)

        await expect(aegisMintingContract.untrackedAvailableAssetBalance(assetAddress)).to.be.eventually.equal(untrackedAvailableAssetBalanceBefore - smalletCollateralAmount)
        await expect(assetContract.balanceOf(sender)).to.be.eventually.equal(userAssetBalanceBefore + smalletCollateralAmount)
      })

      it('should reject RedeemRequest when underlying order is expired', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.99')
        const orderExpiry = (await time.latest()) + 10000
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: collateralAmount,
          expiry: orderExpiry,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        // Mint asset funds
        await assetContract.mint(aegisMintingContract, collateralAmount)

        await time.increase(orderExpiry)

        await expect(aegisMintingContract.approveRedeemRequest('test', collateralAmount)).to.
          emit(aegisMintingContract, 'RejectRedeemRequest').
          withArgs(requestId, owner.address, sender.address, yusdAmount)
      })

      it('should reject RedeemRequest when collateral amount is less than min receive amount', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.99')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: collateralAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        // Mint asset funds
        await assetContract.mint(aegisMintingContract, collateralAmount)

        await expect(aegisMintingContract.approveRedeemRequest('test', 1)).to.
          emit(aegisMintingContract, 'RejectRedeemRequest').
          withArgs(requestId, owner.address, sender.address, yusdAmount)
      })

      it('should reject RedeemRequest when calculated collateral amount by Chainlink price is less than min receive amount', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        const feedRegistry = await ethers.deployContract('FeedRegistry')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setFeedRegistryAddress(feedRegistry)

        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, '99963000')

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('10')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: collateralAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        // Mint asset funds
        await assetContract.mint(aegisMintingContract, collateralAmount)

        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, '100100000')

        await expect(aegisMintingContract.approveRedeemRequest('test', collateralAmount)).to.
          emit(aegisMintingContract, 'RejectRedeemRequest').
          withArgs(requestId, owner.address, sender.address, yusdAmount)
      })

      it('should reject RedeemRequest when calculated collateral amount by AegisOracle price is less than min receive amount', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('10')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: collateralAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        // Mint asset funds
        await assetContract.mint(aegisMintingContract, collateralAmount)

        const feedRegistry = await ethers.deployContract('FeedRegistry')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setFeedRegistryAddress(feedRegistry)

        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, '100000000')

        const aegisOracle = await ethers.deployContract('AegisOracle', [[owner], owner])
        await aegisOracle.setOperator(owner, true)
        await aegisMintingContract.setAegisOracleAddress(aegisOracle)

        await aegisOracle.updateYUSDPrice('99963000')

        await expect(aegisMintingContract.approveRedeemRequest('test', collateralAmount)).to.
          emit(aegisMintingContract, 'RejectRedeemRequest').
          withArgs(requestId, owner.address, sender.address, yusdAmount)
      })
    })

    describe('error', () => {
      it('should be reverted when caller does not have FUNDS_MANAGER_ROLE role', async () => {
        const signers = await ethers.getSigners()
        const sender = signers[1]

        const { aegisMintingContract, aegisMintingAddress, assetContract } = await loadFixture(deployFixture)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        await expect(aegisMintingContract.connect(sender).approveRedeemRequest('test', 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when RedeemRequest does not exist', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.approveRedeemRequest('test', 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidRedeemRequest')
      })

      it('should revert when contract has zero asset balance', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: ethers.parseEther('9.99'),
          slippageAdjustedAmount: collateralAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.approveRedeemRequest(requestId, collateralAmount)).to.be.
          revertedWithCustomError(aegisMintingContract, 'NotEnoughFunds')
      })

      it('should revert when contract has only asset funds for custodian', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, assetContract, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        const collateralAmount = ethers.parseEther('10')

        await assetContract.mint(sender, collateralAmount)
        await assetContract.connect(sender).approve(aegisMintingContract, collateralAmount)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        const yusdAmount = ethers.parseEther('9.99')
        // Mint YUSD
        {
          const order = {
            orderType: OrderType.MINT,
            userWallet: sender.address,
            collateralAsset: assetAddress,
            collateralAmount: collateralAmount,
            yusdAmount: yusdAmount,
            slippageAdjustedAmount: yusdAmount,
            expiry: (await time.latest()) + 10000,
            nonce: Date.now(),
            additionalData: encodeString(''),
          }
          const signature = await signOrder(order, aegisMintingAddress)

          await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.not.reverted
        }

        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: collateralAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.approveRedeemRequest(requestId, collateralAmount)).to.be.
          revertedWithCustomError(aegisMintingContract, 'NotEnoughFunds')
      })

      it('should revert when RedeemRequest already processed', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('10'),
          yusdAmount: ethers.parseEther('9.99'),
          slippageAdjustedAmount: ethers.parseEther('9.99'),
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.rejectRedeemRequest(requestId)).to.not.reverted

        await expect(aegisMintingContract.approveRedeemRequest(requestId, 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidRedeemRequest')
      })

      it('should revert when passed zero amount', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('10'),
          yusdAmount: ethers.parseEther('9.99'),
          slippageAdjustedAmount: ethers.parseEther('9.99'),
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.approveRedeemRequest(requestId, 0)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAmount')
      })

      it('should revert when passed amount is greter than order amount', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: ethers.parseEther('9.99'),
          slippageAdjustedAmount: ethers.parseEther('9.99'),
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.approveRedeemRequest(requestId, collateralAmount + 1n)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAmount')
      })

      it('should revert when paused', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner.address)
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner.address)
        await aegisMintingContract.setRedeemPaused(true)

        await expect(aegisMintingContract.approveRedeemRequest('test', 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'RedeemPaused')
      })
    })
  })

  describe('#rejectRedeemRequest', () => {
    describe('success', () => {
      it('should reject pending RedeemRequest', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.99')
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: collateralAmount,
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        const userYUSDBalanceBefore = await yusdContract.balanceOf(sender)
        const contractYUSDBalanceBefore = await yusdContract.balanceOf(aegisMintingContract)

        await expect(aegisMintingContract.rejectRedeemRequest(requestId)).to.
          emit(aegisMintingContract, 'RejectRedeemRequest').
          withArgs(requestId, owner.address, sender.address, yusdAmount).
          emit(yusdContract, 'Transfer').
          withArgs(aegisMintingContract, sender, yusdAmount)

        const redeemRequest = await aegisMintingContract.getRedeemRequest(requestId)

        await expect(yusdContract.balanceOf(sender)).to.eventually.equal(userYUSDBalanceBefore + yusdAmount)
        await expect(yusdContract.balanceOf(aegisMintingContract)).to.eventually.equal(contractYUSDBalanceBefore - yusdAmount)
        expect(redeemRequest.status).to.equal(RedeemRequestStatus.REJECTED)
      })
    })

    describe('error', () => {
      it('should be reverted when caller does not have FUNDS_MANAGER_ROLE role', async () => {
        const signers = await ethers.getSigners()
        const sender = signers[1]

        const { aegisMintingContract, aegisMintingAddress, assetContract } = await loadFixture(deployFixture)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        await expect(aegisMintingContract.connect(sender).rejectRedeemRequest('test')).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when RedeemRequest does not exist', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.rejectRedeemRequest('test')).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidRedeemRequest')
      })

      it('should revert when RedeemRequest already processed', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('10'),
          yusdAmount: ethers.parseEther('9.99'),
          slippageAdjustedAmount: ethers.parseEther('9.99'),
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.rejectRedeemRequest('test')).to.not.reverted

        await expect(aegisMintingContract.rejectRedeemRequest('test')).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidRedeemRequest')
      })

      it('should revert when paused', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner.address)
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner.address)
        await aegisMintingContract.setRedeemPaused(true)

        await expect(aegisMintingContract.rejectRedeemRequest('test')).to.be.
          revertedWithCustomError(aegisMintingContract, 'RedeemPaused')
      })
    })
  })

  describe('#withdrawRedeemRequest', () => {
    describe('success', () => {
      it('should withdraw expired deposit redeem', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const yusdAmount = ethers.parseEther('9.99')
        const orderExpiry = (await time.latest()) + 10000
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('10'),
          yusdAmount: yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: orderExpiry,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        await time.increase(orderExpiry)

        const benefactorYUSDBalanceBefore = await yusdContract.balanceOf(sender)

        await expect(aegisMintingContract.withdrawRedeemRequest(requestId)).to.
          emit(aegisMintingContract, 'WithdrawRedeemRequest').
          withArgs(requestId, sender.address, yusdAmount)

        await expect(assetContract.balanceOf(aegisMintingAddress)).to.be.eventually.equal(0)
        await expect(yusdContract.balanceOf(sender)).to.be.eventually.equal(benefactorYUSDBalanceBefore + yusdAmount)
      })
    })

    describe('error', () => {
      it('should revert when RedeemRequest does not exist', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.withdrawRedeemRequest('test')).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidRedeemRequest')
      })

      it('should revert when RedeemRequest already processed', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('10'),
          yusdAmount: ethers.parseEther('9.99'),
          slippageAdjustedAmount: ethers.parseEther('9.99'),
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner)
        await aegisMintingContract.rejectRedeemRequest(requestId)

        await expect(aegisMintingContract.connect(sender).withdrawRedeemRequest('test')).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidRedeemRequest')
      })

      it('should revert when RedeemRequest\'s underlying order is not expired', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('10'),
          yusdAmount: ethers.parseEther('9.99'),
          slippageAdjustedAmount: ethers.parseEther('9.99'),
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        await expect(aegisMintingContract.withdrawRedeemRequest('test')).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidRedeemRequest')
      })

      it('should revert when redeem is paused', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await yusdContract.setMinter(owner)
        await yusdContract.mint(sender, ethers.parseEther('100'))
        await yusdContract.setMinter(aegisMintingAddress)
        // Approve YUSD to be sent by AegisMinting contract from sender
        await yusdContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('1000'))

        const requestId = 'test'
        const redeemOrder = {
          orderType: OrderType.REDEEM,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('10'),
          yusdAmount: ethers.parseEther('9.99'),
          slippageAdjustedAmount: ethers.parseEther('9.99'),
          expiry: (await time.latest()) + 10000,
          nonce: Date.now(),
          additionalData: encodeString(requestId),
        }
        const signature = await signOrder(redeemOrder, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).requestRedeem(redeemOrder, signature)).to.be.not.reverted

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setRedeemPaused(true)

        await expect(aegisMintingContract.connect(sender).withdrawRedeemRequest('test')).to.be.
          revertedWithCustomError(aegisMintingContract, 'RedeemPaused')
      })
    })
  })
})
