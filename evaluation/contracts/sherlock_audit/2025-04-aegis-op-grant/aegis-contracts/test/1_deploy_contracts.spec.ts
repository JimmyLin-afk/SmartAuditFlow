import { ethers } from 'hardhat'
import { expect } from 'chai'
import { trustedSignerAccount } from './helpers'

describe('deployContracts', () => {
  describe('YUSD', () => {
    describe('success', () => {
      it('should deploy YUSD contract', async () => {
        const signers = await ethers.getSigners()
        const admin = signers[1]

        const factory = await ethers.getContractFactory('YUSD')
        await expect(factory.deploy(admin.address)).to.be.not.reverted
        const contract = await factory.deploy(admin.address)

        await expect(contract.owner()).eventually.to.be.equal(admin.address)
        await expect(contract.name()).eventually.to.be.equal('YUSD')
        await expect(contract.symbol()).eventually.to.be.equal('YUSD')
      })
    })

    describe('error', () => {
      it('should revert when admin address is zero', async () => {
        const factory = await ethers.getContractFactory('YUSD')
        await expect(factory.deploy(ethers.ZeroAddress)).to.be.reverted
      })
    })
  })

  describe('AegisRewards', () => {
    describe('success', () => {
      it('should deploy AegisRewards contract', async () => {
        const [admin] = await ethers.getSigners()

        const yusd = await ethers.deployContract('YUSD', [admin.address])
        const yusdAddress = await yusd.getAddress()

        const aegisConfig = await ethers.deployContract('AegisConfig', [trustedSignerAccount, [], admin])

        const factory = await ethers.getContractFactory('AegisRewards')
        await expect(factory.deploy(
          yusdAddress,
          aegisConfig,
          admin.address,
        )).to.be.not.reverted
        const contract = await factory.deploy(
          yusdAddress,
          aegisConfig,
          admin.address,
        )

        await expect(contract.yusd()).eventually.to.be.equal(yusdAddress)
      })
    })

    describe('error', () => {
      it('should revert when YUSD address is zero', async () => {
        const admin = await ethers.Wallet.createRandom()

        const factory = await ethers.getContractFactory('AegisRewards')

        await expect(factory.deploy(
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          admin.address,
        )).to.be.reverted
      })

      it('should revert when AegisConfig address is zero', async () => {
        const [admin] = await ethers.getSigners()

        const yusd = await ethers.deployContract('YUSD', [admin.address])

        const factory = await ethers.getContractFactory('AegisRewards')
        await expect(factory.deploy(
          await yusd.getAddress(),
          ethers.ZeroAddress,
          admin.address,
        )).to.be.reverted
      })

      it('should revert when admin address is zero', async () => {
        const [admin, some] = await ethers.getSigners()

        const yusd = await ethers.deployContract('YUSD', [admin.address])

        const factory = await ethers.getContractFactory('AegisRewards')
        await expect(factory.deploy(
          await yusd.getAddress(),
          some,
          ethers.ZeroAddress,
        )).to.be.reverted
      })
    })
  })

  describe('AegisMinting', () => {
    describe('success', () => {
      it('should deploy AegisMinting contract', async () => {
        const admin = await ethers.Wallet.createRandom()
        const yusd = await ethers.deployContract('YUSD', [admin.address])
        const yusdAddress = await yusd.getAddress()

        const aegisRewards = await ethers.Wallet.createRandom()
        const insuranceFund = await ethers.Wallet.createRandom()
        const aegisConfig = await ethers.deployContract('AegisConfig', [trustedSignerAccount, [], admin])

        const testAssetContract = await ethers.deployContract('TestToken', ['Test', 'TST', 18])
        const testAssetAddress = await testAssetContract.getAddress()

        const custodian = await ethers.Wallet.createRandom()

        const factory = await ethers.getContractFactory('AegisMinting')
        await expect(factory.deploy(
          yusdAddress,
          aegisConfig,
          aegisRewards.address,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          insuranceFund.address,
          [testAssetAddress],
          [86400],
          [custodian.address],
          admin.address,
        )).to.be.not.reverted
      })
    })

    describe('error', () => {
      it('should revert when YUSD address is zero', async () => {
        const aegisRewards = await ethers.Wallet.createRandom()
        const insuranceFund = await ethers.Wallet.createRandom()
        const admin = await ethers.Wallet.createRandom()

        const factory = await ethers.getContractFactory('AegisMinting')
        await expect(factory.deploy(
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          aegisRewards.address,
          insuranceFund.address,
          [],
          [],
          [],
          admin.address,
        )).to.be.reverted
      })

      it('should revert when AegisRewards address is zero', async () => {
        const admin = await ethers.Wallet.createRandom()
        const yusd = await ethers.deployContract('YUSD', [admin.address])
        const yusdAddress = await yusd.getAddress()

        const insuranceFund = await ethers.Wallet.createRandom()

        const factory = await ethers.getContractFactory('AegisMinting')
        await expect(factory.deploy(
          yusdAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          insuranceFund.address,
          [],
          [],
          [],
          admin.address,
        )).to.be.reverted
      })

      it('should revert when AegisConfig address is zero', async () => {
        const admin = await ethers.Wallet.createRandom()
        const yusd = await ethers.deployContract('YUSD', [admin.address])
        const yusdAddress = await yusd.getAddress()

        const aegisRewards = await ethers.Wallet.createRandom()
        const insuranceFund = await ethers.Wallet.createRandom()

        const factory = await ethers.getContractFactory('AegisMinting')
        await expect(factory.deploy(
          yusdAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          aegisRewards.address,
          insuranceFund.address,
          [],
          [],
          [],
          admin.address,
        )).to.be.reverted
      })

      it('should revert when admin address is zero', async () => {
        const admin = await ethers.Wallet.createRandom()
        const yusd = await ethers.deployContract('YUSD', [admin.address])
        const yusdAddress = await yusd.getAddress()

        const aegisRewards = await ethers.Wallet.createRandom()
        const insuranceFund = await ethers.Wallet.createRandom()

        const factory = await ethers.getContractFactory('AegisMinting')
        await expect(factory.deploy(
          yusdAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          aegisRewards.address,
          insuranceFund.address,
          [],
          [],
          [],
          ethers.ZeroAddress,
        )).to.be.reverted
      })

      it('should revert when one of asset addresses is zero', async () => {
        const admin = await ethers.Wallet.createRandom()
        const yusd = await ethers.deployContract('YUSD', [admin.address])
        const yusdAddress = await yusd.getAddress()

        const aegisRewards = await ethers.Wallet.createRandom()
        const insuranceFund = await ethers.Wallet.createRandom()

        const factory = await ethers.getContractFactory('AegisMinting')
        await expect(factory.deploy(
          yusdAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          aegisRewards.address,
          insuranceFund.address,
          [ethers.ZeroAddress],
          [86400],
          [admin.address],
          admin.address,
        )).to.be.reverted
      })

      it('should revert when one of asset addresses is YUSD address', async () => {
        const admin = await ethers.Wallet.createRandom()
        const yusd = await ethers.deployContract('YUSD', [admin.address])
        const yusdAddress = await yusd.getAddress()

        const aegisRewards = await ethers.Wallet.createRandom()
        const insuranceFund = await ethers.Wallet.createRandom()

        const factory = await ethers.getContractFactory('AegisMinting')
        await expect(factory.deploy(
          yusdAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          aegisRewards.address,
          insuranceFund.address,
          [yusdAddress],
          [86400],
          [admin.address],
          admin.address,
        )).to.be.reverted
      })

      it('should revert when one of custodian addresses is zero', async () => {
        const admin = await ethers.Wallet.createRandom()
        const yusd = await ethers.deployContract('YUSD', [admin.address])
        const yusdAddress = await yusd.getAddress()

        const aegisRewards = await ethers.Wallet.createRandom()
        const insuranceFund = await ethers.Wallet.createRandom()

        const testAssetContract = await ethers.deployContract('TestToken', ['Test', 'TST', 18])
        const testAssetAddress = await testAssetContract.getAddress()

        const factory = await ethers.getContractFactory('AegisMinting')
        await expect(factory.deploy(
          yusdAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          aegisRewards.address,
          insuranceFund.address,
          [testAssetAddress],
          [86400],
          [ethers.ZeroAddress],
          admin.address,
        )).to.be.reverted
      })

      it('should revert when one of custodian addresses is YUSD address', async () => {
        const admin = await ethers.Wallet.createRandom()
        const yusd = await ethers.deployContract('YUSD', [admin.address])
        const yusdAddress = await yusd.getAddress()

        const aegisRewards = await ethers.Wallet.createRandom()
        const insuranceFund = await ethers.Wallet.createRandom()

        const testAssetContract = await ethers.deployContract('TestToken', ['Test', 'TST', 18])
        const testAssetAddress = await testAssetContract.getAddress()

        const factory = await ethers.getContractFactory('AegisMinting')
        await expect(factory.deploy(
          yusdAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          ethers.ZeroAddress,
          aegisRewards.address,
          insuranceFund.address,
          [testAssetAddress],
          [86400],
          [yusdAddress],
          admin.address,
        )).to.be.reverted
      })
    })
  })
})