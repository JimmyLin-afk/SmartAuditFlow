import { ethers } from 'hardhat'
import { expect } from 'chai'

describe('AegisConfig', () => {
  describe('#setTrustedSigner', () => {
    describe('success', () => {
      it('should update trusted signer', async () => {
        const [owner] = await ethers.getSigners()
        const signer = await ethers.Wallet.createRandom()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract.setTrustedSigner(signer)).to.
          emit(contract, 'SetTrustedSigner').
          withArgs(signer)
      })
    })

    describe('error', () => {
      it('should revert when called by not an owner', async () => {
        const [owner, notOwner] = await ethers.getSigners()

        const signer = await ethers.Wallet.createRandom()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract.connect(notOwner).setTrustedSigner(signer)).to.be.
          revertedWithCustomError(contract, 'OwnableUnauthorizedAccount')
      })

      it('should revert when new signer is zero address', async () => {
        const [owner] = await ethers.getSigners()
        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract.setTrustedSigner(ethers.ZeroAddress)).to.be.
          revertedWithCustomError(contract, 'ZeroAddress')
      })
    })
  })

  describe('#setOperator', () => {
    describe('success', () => {
      it('should add new operator', async () => {
        const [owner, operator] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract.setOperator(operator, true)).to.
          emit(contract, 'SetOperator').
          withArgs(operator, true)
      })

      it('should remove existing operator', async () => {
        const [owner, operator] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract.setOperator(operator, true)).to.be.not.reverted

        await expect(contract.setOperator(operator, false)).to.
          emit(contract, 'SetOperator').
          withArgs(operator, false)
      })
    })

    describe('error', () => {
      it('should revert when caller is not an owner', async () => {
        const [owner, notOwner] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract.connect(notOwner).setOperator(notOwner, true)).to.be.
          revertedWithCustomError(contract, 'OwnableUnauthorizedAccount')
      })
    })
  })

  describe('#disableWhitelist', () => {
    describe('success', () => {
      it('should disable whitelist', async () => {
        const [owner] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract.disableWhitelist()).to.
          emit(contract, 'WhitelistDisabled')

        await expect(contract.whitelistEnabled()).to.be.eventually.false
      })
    })

    describe('error', () => {
      it('should revert when caller is not an owner', async () => {
        const [owner, notOwner] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract.connect(notOwner).disableWhitelist()).to.be.
          revertedWithCustomError(contract, 'OwnableUnauthorizedAccount')
      })
    })
  })

  describe('#enableWhitelist', () => {
    describe('success', () => {
      it('should enable whitelist', async () => {
        const [owner] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await contract.disableWhitelist()

        await expect(contract.enableWhitelist()).to.
          emit(contract, 'WhitelistEnabled')

        await expect(contract.whitelistEnabled()).to.be.eventually.true
      })
    })

    describe('error', () => {
      it('should revert when caller is not an owner', async () => {
        const [owner, notOwner] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract.connect(notOwner).enableWhitelist()).to.be.
          revertedWithCustomError(contract, 'OwnableUnauthorizedAccount')
      })
    })
  })

  describe('#whitelistAddress (Single)', () => {
    describe('success', () => {
      it('should add address to whitelist', async () => {
        const [owner, user] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract['whitelistAddress(address,bool)'](user, true)).to.
          emit(contract, 'WhitelistAddress').
          withArgs(user, true)

        await expect(contract.isWhitelisted(user)).to.be.eventually.true
      })

      it('should remove address from whitelist', async () => {
        const [owner, user] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await contract['whitelistAddress(address,bool)'](user, true)

        await expect(contract['whitelistAddress(address,bool)'](user, false)).to.
          emit(contract, 'WhitelistAddress').
          withArgs(user, false)

        await expect(contract.isWhitelisted(user)).to.be.eventually.false
      })
    })

    describe('error', () => {
      it('should revert when caller is not an operator', async () => {
        const [owner, caller] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract.connect(caller)['whitelistAddress(address,bool)'](caller, true)).to.be.revertedWithCustomError(contract, 'AccessForbidden')
      })
    })
  })

  describe('#whitelistAddress (Batch)', () => {
    describe('success', () => {
      it('should update addresses in whitelist', async () => {
        const [owner] = await ethers.getSigners()

        const user1 = ethers.Wallet.createRandom()
        const user2 = ethers.Wallet.createRandom()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract['whitelistAddress(address[],bool[])']([user1, user2], [true, false])).to.
          emit(contract, 'WhitelistAddressBatch').
          withArgs([user1, user2], [true, false])

        await expect(contract.isWhitelisted(user1)).to.be.eventually.true
        await expect(contract.isWhitelisted(user2)).to.be.eventually.false
      })
    })

    describe('error', () => {
      it('should revert when caller is not an operator', async () => {
        const [owner, caller] = await ethers.getSigners()

        const user1 = ethers.Wallet.createRandom()
        const user2 = ethers.Wallet.createRandom()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract.connect(caller)['whitelistAddress(address[],bool[])']([user1, user2], [true, true])).to.be.revertedWithCustomError(contract, 'AccessForbidden')
      })

      it('should revert when arguments length mismatch', async () => {
        const [owner] = await ethers.getSigners()

        const user1 = ethers.Wallet.createRandom()
        const user2 = ethers.Wallet.createRandom()

        const contract = await ethers.deployContract('AegisConfig', [owner, [owner], owner])

        await expect(contract['whitelistAddress(address[],bool[])']([user1, user2], [true])).to.be.revertedWithCustomError(contract, 'InvalidArguments')
      })
    })
  })
})