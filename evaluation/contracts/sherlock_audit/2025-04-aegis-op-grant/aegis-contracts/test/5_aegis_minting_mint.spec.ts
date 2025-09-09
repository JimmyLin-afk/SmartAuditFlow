import { ethers } from 'hardhat'
import { loadFixture, time } from '@nomicfoundation/hardhat-network-helpers'
import { expect } from 'chai'

import {
  MAX_BPS,
  OrderType,
  deployFixture,
  insuranceFundAccount,
  signOrder,
  signOrderByWallet,
  encodeString,
  SETTINGS_MANAGER_ROLE,
  USD_FEED_ADDRESS,
} from './helpers'

describe('AegisMinting', () => {
  describe('#mint', () => {
    describe('success', () => {
      it('should mint correct amount of YUSD in exchange for collateral asset', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.999')

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

        const mintingContractAssetBalanceBefore = await assetContract.balanceOf(aegisMintingAddress)
        const custodyAvailableAssetBalanceBefore = await aegisMintingContract.custodyAvailableAssetBalance(assetAddress)

        const senderYUSDBalanceBefore = await yusdContract.balanceOf(sender.address)
        const senderAssetBalanceBefore = await assetContract.balanceOf(sender.address)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.
          emit(aegisMintingContract, 'Mint').
          withArgs(sender.address, order.collateralAsset, order.collateralAmount, order.yusdAmount, 0)

        await expect(assetContract.balanceOf(aegisMintingAddress)).eventually.to.be.equal(mintingContractAssetBalanceBefore + collateralAmount)
        await expect(aegisMintingContract.custodyAvailableAssetBalance(assetAddress)).eventually.to.be.equal(custodyAvailableAssetBalanceBefore + collateralAmount)
        await expect(yusdContract.balanceOf(sender.address)).eventually.to.be.equal(senderYUSDBalanceBefore + yusdAmount)
        await expect(assetContract.balanceOf(sender.address)).eventually.to.be.equal(senderAssetBalanceBefore - collateralAmount)
      })

      it('should mint when benefactor is in whitelist', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.999')

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

        const mintingContractAssetBalanceBefore = await assetContract.balanceOf(aegisMintingAddress)
        const custodyAvailableAssetBalanceBefore = await aegisMintingContract.custodyAvailableAssetBalance(assetAddress)

        const senderYUSDBalanceBefore = await yusdContract.balanceOf(sender.address)
        const senderAssetBalanceBefore = await assetContract.balanceOf(sender.address)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.
          emit(aegisMintingContract, 'Mint').
          withArgs(sender.address, order.collateralAsset, order.collateralAmount, order.yusdAmount, 0)

        await expect(assetContract.balanceOf(aegisMintingAddress)).eventually.to.be.equal(mintingContractAssetBalanceBefore + collateralAmount)
        await expect(aegisMintingContract.custodyAvailableAssetBalance(assetAddress)).eventually.to.be.equal(custodyAvailableAssetBalanceBefore + collateralAmount)
        await expect(yusdContract.balanceOf(sender.address)).eventually.to.be.equal(senderYUSDBalanceBefore + yusdAmount)
        await expect(assetContract.balanceOf(sender.address)).eventually.to.be.equal(senderAssetBalanceBefore - collateralAmount)
      })

      it('should mint correct amount of YUSD in exchange for collateral asset and take fee', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, yusdContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        const feeBP = 500n
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner.address)
        await aegisMintingContract.setMintFeeBP(feeBP)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.999')
        const feeAmount = yusdAmount * feeBP / MAX_BPS
        const mintAmount = yusdAmount - feeAmount

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

        const mintingContractAssetBalanceBefore = await assetContract.balanceOf(aegisMintingAddress)
        const custodyAvailableAssetBalanceBefore = await aegisMintingContract.custodyAvailableAssetBalance(assetAddress)

        const senderYUSDBalanceBefore = await yusdContract.balanceOf(sender.address)
        const senderAssetBalanceBefore = await assetContract.balanceOf(sender.address)
        const insuranceFundYUSDBalanceBefore = await yusdContract.balanceOf(insuranceFundAccount.address)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.
          emit(aegisMintingContract, 'Mint').
          withArgs(sender, order.collateralAsset, order.collateralAmount, mintAmount, feeAmount).
          emit(yusdContract, 'Transfer').
          withArgs(ethers.ZeroAddress, insuranceFundAccount.address, feeAmount)

        await expect(assetContract.balanceOf(aegisMintingAddress)).eventually.to.be.equal(mintingContractAssetBalanceBefore + collateralAmount)
        await expect(aegisMintingContract.custodyAvailableAssetBalance(assetAddress)).eventually.to.be.equal(custodyAvailableAssetBalanceBefore + collateralAmount)
        await expect(yusdContract.balanceOf(sender.address)).eventually.to.be.equal(senderYUSDBalanceBefore + mintAmount)
        await expect(assetContract.balanceOf(sender.address)).eventually.to.be.equal(senderAssetBalanceBefore - collateralAmount)
        await expect(yusdContract.balanceOf(insuranceFundAccount.address)).eventually.to.be.equal(insuranceFundYUSDBalanceBefore + feeAmount)
      })

      it('should mint within limits', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        // Set limits
        const mintMaxAmount = ethers.parseEther('10')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setMintLimits(60, mintMaxAmount)

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.999')

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

        const mintLimitBefore = await aegisMintingContract.mintLimit()

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.not.reverted

        const mintLimitAfter = await aegisMintingContract.mintLimit()
        expect(mintLimitAfter.currentPeriodTotalAmount).to.be.equal(mintLimitBefore.currentPeriodTotalAmount + yusdAmount)
      })

      it('should reset mint limit counters at the beginning of new period', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        // Set limits
        const mintMaxAmount = ethers.parseEther('10')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setMintLimits(60, mintMaxAmount)

        const collateralAmount = ethers.parseEther('10')
        let blockTime = await time.latest()

        {
          const yusdAmount = ethers.parseEther('9.999')

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

          await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.not.reverted
        }

        await time.increase(60)

        const yusdAmount = ethers.parseEther('8')

        blockTime = await time.latest()
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

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.not.reverted

        blockTime = await time.latest()

        const mintLimit = await aegisMintingContract.mintLimit()

        expect(mintLimit.currentPeriodTotalAmount).to.be.equal(yusdAmount)
        expect(mintLimit.currentPeriodStartTime).to.be.equal(blockTime)
      })
    })

    describe('error', () => {
      it('should revert when OrderType is not MINT', async () => {
        const [,sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

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
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidOrder')
      })

      it('should revert when collateral asset is not supported', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const fakeAsset = await ethers.Wallet.createRandom()

        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.MINT,
          userWallet: sender.address,
          collateralAsset: fakeAsset.address,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: ethers.parseEther('1'),
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })

      it('should revert when signer by unknown signer', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const unknownSigner = await ethers.Wallet.createRandom()

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
          additionalData: encodeString(''),
        }
        const signature = await signOrderByWallet(order, aegisMintingAddress, unknownSigner)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidSignature')
      })

      it('should revert when collateral amount is zero', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.MINT,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: 0,
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: ethers.parseEther('1'),
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidAmount')
      })

      it('should revert when yusd amount is zero', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.MINT,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: 0,
          slippageAdjustedAmount: 0,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidAmount')
      })

      it('should revert when order expired', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const blockTime = await time.latest()

        const order = {
          orderType: OrderType.MINT,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('1'),
          yusdAmount: ethers.parseEther('1'),
          slippageAdjustedAmount: ethers.parseEther('1'),
          expiry: blockTime - 1000,
          nonce: Date.now(),
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'SignatureExpired')
      })

      it('should revert when paused', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](owner, true)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner.address)
        await aegisMintingContract.setMintPaused(true)

        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.MINT,
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

        await expect(aegisMintingContract.mint(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'MintPaused')
      })

      it('should revert when minting amount exceeds max amount within period', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        // Set limits
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setMintLimits(60, ethers.parseEther('15'))

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('9.999')

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

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.not.reverted

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.revertedWithCustomError(aegisMintingContract, 'LimitReached')
      })

      it('should revert when minting amount exceeds max amount at the beginning of period', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        // Set limits
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setMintLimits(60, ethers.parseEther('15'))

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('16')

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

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.revertedWithCustomError(aegisMintingContract, 'LimitReached')
      })

      it('should revert when benefactor is not in whitelist', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetAddress } = await loadFixture(deployFixture)

        const yusdAmount = ethers.parseEther('9.999')
        const blockTime = await time.latest()
        const order = {
          orderType: OrderType.MINT,
          userWallet: sender.address,
          collateralAsset: assetAddress,
          collateralAmount: ethers.parseEther('10'),
          yusdAmount,
          slippageAdjustedAmount: yusdAmount,
          expiry: blockTime + 10000,
          nonce: Date.now(),
          additionalData: encodeString(''),
        }
        const signature = await signOrder(order, aegisMintingAddress)

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'NotWhitelisted')
      })

      it('should revert when Chainlink asset price is less than 0', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const feedRegistry = await ethers.deployContract('FeedRegistry')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setFeedRegistryAddress(feedRegistry)

        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, '0')

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('10')

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

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.revertedWith('Invalid price')
      })

      it('should revert when Chainlink asset price is stale', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const feedRegistry = await ethers.deployContract('FeedRegistry')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setFeedRegistryAddress(feedRegistry)

        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, '99963000')
        await feedRegistry.setUpdatedAt((await time.latest()) - 86400)

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('10')

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

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.revertedWith('Stale price')
      })

      it('should revert when calculated YUSD amount by Chainlink price is less than min receive amount', async () => {
        const [owner, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const feedRegistry = await ethers.deployContract('FeedRegistry')
        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)
        await aegisMintingContract.setFeedRegistryAddress(feedRegistry)

        await feedRegistry.setPrice(assetContract, USD_FEED_ADDRESS, '99963000')

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('10')

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

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'PriceSlippage')
      })

      it('should revert when order was already processed', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('10')

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

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.not.reverted

        await expect(aegisMintingContract.connect(sender).mint(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidNonce')
      })

      it('should revert when caller is not benefactor', async () => {
        const [, sender] = await ethers.getSigners()

        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress, aegisConfig } = await loadFixture(deployFixture)

        await aegisConfig['whitelistAddress(address,bool)'](sender, true)

        await assetContract.mint(sender.address, ethers.parseEther('100'))
        await assetContract.connect(sender).approve(aegisMintingAddress, ethers.parseEther('100'))

        const collateralAmount = ethers.parseEther('10')
        const yusdAmount = ethers.parseEther('10')

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

        await expect(aegisMintingContract.mint(order, signature)).to.be.revertedWithCustomError(aegisMintingContract, 'InvalidSender')
      })
    })
  })
})
