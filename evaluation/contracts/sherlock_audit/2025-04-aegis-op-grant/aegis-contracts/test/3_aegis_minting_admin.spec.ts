import { ethers } from 'hardhat'
import { loadFixture } from '@nomicfoundation/hardhat-network-helpers'
import { expect } from 'chai'

import {
  SETTINGS_MANAGER_ROLE,
  MAX_BPS,
  custodianAccount,
  deployFixture,
  FUNDS_MANAGER_ROLE,
  trustedSignerAccount,
} from './helpers'

describe('AegisMinting', () => {
  describe('#setAegisRewardsAddress', () => {
    describe('success', () => {
      it('should update AegisRewards address', async () => {
        const [owner] = await ethers.getSigners()
        const aegisRewardsAccount = await ethers.Wallet.createRandom()

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.setAegisRewardsAddress(aegisRewardsAccount.address)).to.
          emit(aegisMintingContract, 'SetAegisRewardsAddress').
          withArgs(aegisRewardsAccount.address)

        await expect(aegisMintingContract.aegisRewards()).eventually.to.be.equal(aegisRewardsAccount.address)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have role SETTINGS_MANAGER_ROLE', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const aegisRewardsAccount = await ethers.Wallet.createRandom()

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).setAegisRewardsAddress(aegisRewardsAccount.address)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when new AegisRewards is zero address', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.setAegisRewardsAddress(ethers.ZeroAddress)).to.be.
          revertedWithCustomError(aegisMintingContract, 'ZeroAddress')
      })
    })
  })

  describe('#setInsuranceFundAddress', () => {
    describe('success', () => {
      it('should update InsuranceFund address', async () => {
        const [owner] = await ethers.getSigners()
        const insuranceFundAccount = await ethers.Wallet.createRandom()

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.setInsuranceFundAddress(insuranceFundAccount.address)).to.
          emit(aegisMintingContract, 'SetInsuranceFundAddress').
          withArgs(insuranceFundAccount.address)

        await expect(aegisMintingContract.insuranceFundAddress()).eventually.to.be.equal(insuranceFundAccount.address)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have role SETTINGS_MANAGER_ROLE', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const insuranceFundAccount = await ethers.Wallet.createRandom()

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).setInsuranceFundAddress(insuranceFundAccount.address)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })
    })
  })

  describe('#setIncomeFeeBP', () => {
    describe('success', () => {
      it('should update insurance fund percent', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        const newPercentBP = 3_000
        await expect(aegisMintingContract.setIncomeFeeBP(newPercentBP)).to.
          emit(aegisMintingContract, 'SetIncomeFeeBP').
          withArgs(newPercentBP)

        await expect(aegisMintingContract.incomeFeeBP()).eventually.to.be.equal(newPercentBP)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have role SETTINGS_MANAGER_ROLE', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).setIncomeFeeBP(2000)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when value greater than 50%', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.setIncomeFeeBP((MAX_BPS / 2n) + 1n)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidPercentBP')
      })
    })
  })

  describe('#addSupportedAsset', () => {
    describe('success', () => {
      it('should add new asset address', async () => {
        const { aegisMintingContract } = await loadFixture(deployFixture)
        const newAssetContract = await ethers.deployContract('TestToken', ['Test', 'TST', 18])
        const newAssetAddress = await newAssetContract.getAddress()

        const heartbeat = 86400
        await expect(aegisMintingContract.addSupportedAsset(newAssetAddress, heartbeat)).to.
          emit(aegisMintingContract, 'AssetAdded').
          withArgs(newAssetAddress, heartbeat)

        await expect(aegisMintingContract.isSupportedAsset(newAssetAddress)).eventually.to.be.equal(true)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have DEFAULT_ADMIN_ROLE role', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const newAsset = await ethers.Wallet.createRandom()

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).addSupportedAsset(newAsset.address, 86400)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when new asset address is zero', async () => {
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.addSupportedAsset(ethers.ZeroAddress, 86400)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })

      it('should revert when new asset address is YUSD address', async () => {
        const { aegisMintingContract, yusdAddress } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.addSupportedAsset(yusdAddress, 86400)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })

      it('should revert when adding existing asset address', async () => {
        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.addSupportedAsset(assetAddress, 86400)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })
    })
  })

  describe('#removeSupportedAsset', () => {
    describe('success', () => {
      it('should remove supported asset address', async () => {
        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.removeSupportedAsset(assetAddress)).to.
          emit(aegisMintingContract, 'AssetRemoved').
          withArgs(assetAddress)

        await expect(aegisMintingContract.isSupportedAsset(assetAddress)).eventually.to.be.equal(false)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have DEFAULT_ADMIN_ROLE role', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).removeSupportedAsset(assetAddress)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when removing unknown asset', async () => {
        const { aegisMintingContract } = await loadFixture(deployFixture)

        const unknownAsset = await ethers.Wallet.createRandom()

        await expect(aegisMintingContract.removeSupportedAsset(unknownAsset.address)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })
    })
  })

  describe('#addCustodianAddress', () => {
    describe('success', () => {
      it('should add new custodian address', async () => {
        const { aegisMintingContract } = await loadFixture(deployFixture)
        const newCustodianAccount = await ethers.Wallet.createRandom()

        await expect(aegisMintingContract.addCustodianAddress(newCustodianAccount.address)).to.
          emit(aegisMintingContract, 'CustodianAddressAdded').
          withArgs(newCustodianAccount.address)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have DEFAULT_ADMIN_ROLE role', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const newCustodian = await ethers.Wallet.createRandom()

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).addCustodianAddress(newCustodian.address)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when new custodian address is zero', async () => {
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.addCustodianAddress(ethers.ZeroAddress)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidCustodianAddress')
      })

      it('should revert when new custodian address is YUSD address', async () => {
        const { aegisMintingContract, yusdAddress } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.addCustodianAddress(yusdAddress)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidCustodianAddress')
      })

      it('should revert when adding existing custodian address', async () => {
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.addCustodianAddress(custodianAccount.address)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidCustodianAddress')
      })
    })
  })

  describe('#removeCustodianAddress', () => {
    describe('success', () => {
      it('should remove custodian address', async () => {
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.removeCustodianAddress(custodianAccount.address)).to.
          emit(aegisMintingContract, 'CustodianAddressRemoved').
          withArgs(custodianAccount.address)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have DEFAULT_ADMIN_ROLE role', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).removeCustodianAddress(custodianAccount.address)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when removing unknown custodian', async () => {
        const { aegisMintingContract } = await loadFixture(deployFixture)

        const unknownCustodianAccount = await ethers.Wallet.createRandom()

        await expect(aegisMintingContract.removeCustodianAddress(unknownCustodianAccount.address)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidCustodianAddress')
      })
    })
  })

  describe('#setMintPaused', () => {
    describe('success', () => {
      it('should pause mint', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.setMintPaused(true)).to.
          emit(aegisMintingContract, 'MintPauseChanged').
          withArgs(true)

        await expect(aegisMintingContract.mintPaused()).eventually.to.be.equal(true)
      })

      it('should unpause mint', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        await aegisMintingContract.setMintPaused(true)

        await expect(aegisMintingContract.setMintPaused(false)).to.
          emit(aegisMintingContract, 'MintPauseChanged').
          withArgs(false)

        await expect(aegisMintingContract.mintPaused()).eventually.to.be.equal(false)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have SETTINGS_MANAGER_ROLE role', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).setMintPaused(true)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })
    })
  })

  describe('#setRedeemPaused', () => {
    describe('success', () => {
      it('should pause redeem', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.setRedeemPaused(true)).to.
          emit(aegisMintingContract, 'RedeemPauseChanged').
          withArgs(true)

        await expect(aegisMintingContract.redeemPaused()).eventually.to.be.equal(true)
      })

      it('should unpause redeem', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        await aegisMintingContract.setRedeemPaused(true)

        await expect(aegisMintingContract.setRedeemPaused(false)).to.
          emit(aegisMintingContract, 'RedeemPauseChanged').
          withArgs(false)

        await expect(aegisMintingContract.redeemPaused()).eventually.to.be.equal(false)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have SETTINGS_MANAGER_ROLE role', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).setRedeemPaused(true)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })
    })
  })

  describe('#setMintFeeBP', () => {
    describe('success', () => {
      it('should update mint fee bp', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        const bp = 300
        await expect(aegisMintingContract.setMintFeeBP(bp)).to.
          emit(aegisMintingContract, 'SetMintFeeBP').
          withArgs(bp)

        await expect(aegisMintingContract.mintFeeBP()).eventually.to.be.equal(bp)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have role SETTINGS_MANAGER_ROLE', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).setMintFeeBP(500)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when value greater than 50%', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.setMintFeeBP((MAX_BPS / 2n) + 1n)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidPercentBP')
      })
    })
  })

  describe('#setRedeemFeeBP', () => {
    describe('success', () => {
      it('should update redeem fee bp', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        const bp = 300
        await expect(aegisMintingContract.setRedeemFeeBP(bp)).to.
          emit(aegisMintingContract, 'SetRedeemFeeBP').
          withArgs(bp)

        await expect(aegisMintingContract.redeemFeeBP()).eventually.to.be.equal(bp)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have role SETTINGS_MANAGER_ROLE', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).setRedeemFeeBP(500)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when value greater than 50%', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner)

        await expect(aegisMintingContract.setRedeemFeeBP((MAX_BPS / 2n) + 1n)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidPercentBP')
      })
    })
  })

  describe('#freezeFunds', () => {
    describe('success', () => {
      it('should freeze funds', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner.address)

        const freezeAmount = 1n
        const frozenFundsBefore = await aegisMintingContract.assetFrozenFunds(assetAddress)

        await assetContract.mint(aegisMintingAddress, freezeAmount)

        await expect(aegisMintingContract.freezeFunds(assetAddress, freezeAmount)).to.
          emit(aegisMintingContract, 'FreezeFunds').
          withArgs(assetAddress, freezeAmount)

        await expect(aegisMintingContract.assetFrozenFunds(assetAddress)).to.be.eventually.equal(frozenFundsBefore + freezeAmount)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have role FUNDS_MANAGER_ROLE', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).freezeFunds(ethers.ZeroAddress, 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when freezing unknown asset', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner.address)

        const unknownAsset = await ethers.Wallet.createRandom()

        await expect(aegisMintingContract.freezeFunds(unknownAsset.address, 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })

      it('should revert when freezing amount exceeds contract balance', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner.address)

        await expect(aegisMintingContract.freezeFunds(assetAddress, 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAmount')
      })
    })
  })

  describe('#unfreezeFunds', () => {
    describe('success', () => {
      it('should unfreeze durty funds', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, aegisMintingAddress, assetContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner.address)

        const freezeAmount = 1n
        await assetContract.mint(aegisMintingAddress, freezeAmount)

        await expect(aegisMintingContract.freezeFunds(assetAddress, freezeAmount)).to.be.not.reverted

        const frozenFundsBefore = await aegisMintingContract.assetFrozenFunds(assetAddress)

        await expect(aegisMintingContract.unfreezeFunds(assetAddress, freezeAmount)).to.
          emit(aegisMintingContract, 'UnfreezeFunds').
          withArgs(assetAddress, freezeAmount)

        await expect(aegisMintingContract.assetFrozenFunds(assetAddress)).to.be.eventually.equal(frozenFundsBefore - freezeAmount)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have role FUNDS_MANAGER_ROLE', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).unfreezeFunds(ethers.ZeroAddress, 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when unfreezing unknown asset', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner.address)

        const unknownAsset = await ethers.Wallet.createRandom()

        await expect(aegisMintingContract.unfreezeFunds(unknownAsset.address, 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })

      it('should revert when unfreezing amount excess frozen amount', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(FUNDS_MANAGER_ROLE, owner.address)

        await expect(aegisMintingContract.unfreezeFunds(assetAddress, 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAmount')
      })
    })
  })

  describe('#setMintLimits', () => {
    describe('success', () => {
      it('should set mint limits', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner.address)

        const periodDuration = 60
        const maxMintAmount = ethers.parseEther('1000')

        await expect(aegisMintingContract.setMintLimits(periodDuration, maxMintAmount)).to.
          emit(aegisMintingContract, 'SetMintLimits').
          withArgs(periodDuration, maxMintAmount)

        const mintLimit = await aegisMintingContract.mintLimit()

        expect(mintLimit.maxPeriodAmount).to.be.equal(maxMintAmount)
        expect(mintLimit.periodDuration).to.be.equal(periodDuration)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have role SETTINGS_MANAGER_ROLE', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).setMintLimits(1, 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })
    })
  })

  describe('#setRedeemLimits', () => {
    describe('success', () => {
      it('should set redeem limits', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner.address)

        const periodDuration = 60
        const maxMintAmount = ethers.parseEther('1000')

        await expect(aegisMintingContract.setRedeemLimits(periodDuration, maxMintAmount)).to.
          emit(aegisMintingContract, 'SetRedeemLimits').
          withArgs(periodDuration, maxMintAmount)

        const redeemLimit = await aegisMintingContract.redeemLimit()

        expect(redeemLimit.maxPeriodAmount).to.be.equal(maxMintAmount)
        expect(redeemLimit.periodDuration).to.be.equal(periodDuration)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have role SETTINGS_MANAGER_ROLE', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).setRedeemLimits(1, 1)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })
    })
  })

  describe('#setAegisConfigAddress', () => {
    describe('success', () => {
      it('should change AegisConfig address', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        const aegisConfig = await ethers.deployContract('AegisConfig', [trustedSignerAccount, [], owner])

        await expect(aegisMintingContract.setAegisConfigAddress(aegisConfig)).to.
          emit(aegisMintingContract, 'SetAegisConfigAddress').
          withArgs(aegisConfig)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have role DEFAULT_ADMIN_ROLE', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract } = await loadFixture(deployFixture)

        const someAccount = await ethers.Wallet.createRandom()

        await expect(aegisMintingContract.connect(notOwner).setAegisConfigAddress(someAccount)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when passed address does not support IAegisConfig interface', async () => {
        const { aegisMintingContract } = await loadFixture(deployFixture)

        const someAccount = await ethers.Wallet.createRandom()

        await expect(aegisMintingContract.setAegisConfigAddress(someAccount)).to.be.reverted
      })
    })
  })

  describe('#setChainlinkAssetHeartbeat', () => {
    describe('success', () => {
      it('should update chainlink asset heartbeat', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner.address)

        const heartbeat = 3600
        await expect(aegisMintingContract.setChainlinkAssetHeartbeat(assetAddress, heartbeat)).to.
          emit(aegisMintingContract, 'SetChainlinkAssetHeartbeat').
          withArgs(assetAddress, heartbeat)
      })
    })

    describe('error', () => {
      it('should revert when caller does not have role SETTINGS_MANAGER_ROLE', async () => {
        const signers = await ethers.getSigners()
        const notOwner = signers[1]

        const { aegisMintingContract, assetAddress } = await loadFixture(deployFixture)

        await expect(aegisMintingContract.connect(notOwner).setChainlinkAssetHeartbeat(assetAddress, 86400)).to.be.
          revertedWithCustomError(aegisMintingContract, 'AccessControlUnauthorizedAccount')
      })

      it('should revert when asset is not supported', async () => {
        const [owner] = await ethers.getSigners()
        const { aegisMintingContract } = await loadFixture(deployFixture)

        await aegisMintingContract.grantRole(SETTINGS_MANAGER_ROLE, owner.address)

        const testAssetContract = await ethers.deployContract('TestToken', ['Test', 'TST', 18])

        await expect(aegisMintingContract.setChainlinkAssetHeartbeat(testAssetContract, 86400)).to.be.
          revertedWithCustomError(aegisMintingContract, 'InvalidAssetAddress')
      })
    })
  })
})
