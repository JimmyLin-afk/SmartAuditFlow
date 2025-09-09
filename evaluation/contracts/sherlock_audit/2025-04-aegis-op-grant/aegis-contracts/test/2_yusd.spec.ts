import { ethers } from 'hardhat'
import { expect } from 'chai'

describe('YUSD', () => {
  describe('#setMinter', () => {
    describe('success', () => {
      it('should set new minter', async () => {
        const [owner] = await ethers.getSigners()
        const minter = await ethers.Wallet.createRandom()

        const contract = await ethers.deployContract('YUSD', [owner.address])

        await expect(contract.setMinter(minter.address)).to.
          emit(contract, 'SetMinter').
          withArgs(minter.address, ethers.ZeroAddress)

        await expect(contract.minter()).eventually.to.be.equal(minter.address)
      })
    })

    describe('error', () => {
      it('should revert when called by not an owner', async () => {
        const [owner, notOwner] = await ethers.getSigners()
        const minter = await ethers.Wallet.createRandom()

        const contract = await ethers.deployContract('YUSD', [owner.address])

        await expect(contract.connect(notOwner).setMinter(minter.address)).to.be.revertedWithCustomError(contract, 'OwnableUnauthorizedAccount')
      })
    })
  })

  describe('#mint', () => {
    describe('success', () => {
      it('should mint 1 YUSD token', async () => {
        const [owner, minter] = await ethers.getSigners()

        const contract = await ethers.deployContract('YUSD', [owner.address])
        await contract.setMinter(minter.address)

        const amount = ethers.parseEther('1')
        await expect(contract.connect(minter).mint(owner.address, amount)).to.
          emit(contract, 'Transfer').
          withArgs(ethers.ZeroAddress, owner.address, amount)

        await expect(contract.balanceOf(owner.address)).eventually.to.be.equal(amount)
      })
    })

    describe('error', () => {
      it('should revert when called by not a minter', async () => {
        const [owner, minter] = await ethers.getSigners()

        const contract = await ethers.deployContract('YUSD', [owner.address])
        await contract.setMinter(minter.address)

        await expect(contract.connect(owner).mint(owner.address, ethers.parseEther('1'))).to.be.revertedWithCustomError(contract, 'OnlyMinter')
      })
    })
  })

  describe('#addBlacklist', () => {
    describe('success', () => {
      it('should add user to a blacklist and revert transfer from address', async () => {
        const [owner, user] = await ethers.getSigners()

        const contract = await ethers.deployContract('YUSD', [owner.address])
        await contract.setMinter(owner)

        const amount = ethers.parseEther('1')
        await contract.mint(user, amount)

        await expect(contract.addBlackList(user)).to.
          emit(contract, 'AddedBlackList').
          withArgs(user)

        await expect(contract.connect(user).transfer(owner, amount)).to.be.revertedWithCustomError(contract, 'Blacklisted')
      })

      it('should add user to a blacklist and revert transfer to address', async () => {
        const [owner, user] = await ethers.getSigners()

        const contract = await ethers.deployContract('YUSD', [owner.address])
        await contract.setMinter(owner)

        const amount = ethers.parseEther('1')
        await contract.mint(owner, amount)

        await expect(contract.addBlackList(user)).to.
          emit(contract, 'AddedBlackList').
          withArgs(user)

        await expect(contract.connect(owner).transfer(user, amount)).to.be.revertedWithCustomError(contract, 'Blacklisted')
      })
    })

    describe('error', () => {
      it('should revert when caller is not an owner', async () => {
        const [owner, notOwner] = await ethers.getSigners()
        const user = await ethers.Wallet.createRandom()

        const contract = await ethers.deployContract('YUSD', [owner.address])

        await expect(contract.connect(notOwner).addBlackList(user)).to.be.revertedWithCustomError(contract, 'OwnableUnauthorizedAccount')
      })
    })
  })

  describe('#removeBlacklist', () => {
    describe('success', () => {
      it('should remove user from a blacklist and allow from transfer', async () => {
        const [owner, user] = await ethers.getSigners()

        const contract = await ethers.deployContract('YUSD', [owner.address])
        await contract.setMinter(owner)

        const amount = ethers.parseEther('1')
        await contract.mint(user, amount)

        await expect(contract.addBlackList(user)).to.be.not.reverted

        await expect(contract.removeBlackList(user)).to.
          emit(contract, 'RemovedBlackList').
          withArgs(user)

        await expect(contract.connect(user).transfer(owner, amount)).to.
          emit(contract, 'Transfer').
          withArgs(user, owner, amount)
      })

      it('should remove user from a blacklist and allow to transfer', async () => {
        const [owner, user] = await ethers.getSigners()

        const contract = await ethers.deployContract('YUSD', [owner.address])
        await contract.setMinter(owner)

        const amount = ethers.parseEther('1')
        await contract.mint(owner, amount)

        await expect(contract.addBlackList(user)).to.be.not.reverted

        await expect(contract.removeBlackList(user)).to.
          emit(contract, 'RemovedBlackList').
          withArgs(user)

        await expect(contract.connect(owner).transfer(user, amount)).to.
          emit(contract, 'Transfer').
          withArgs(owner, user, amount)
      })
    })

    describe('error', () => {
      it('should revert when caller is not an owner', async () => {
        const [owner, notOwner] = await ethers.getSigners()
        const user = await ethers.Wallet.createRandom()

        const contract = await ethers.deployContract('YUSD', [owner.address])

        await expect(contract.connect(notOwner).removeBlackList(user)).to.be.revertedWithCustomError(contract, 'OwnableUnauthorizedAccount')
      })
    })
  })
})
