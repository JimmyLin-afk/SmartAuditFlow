import { ethers } from 'hardhat'
import { time } from '@nomicfoundation/hardhat-network-helpers'
import { expect } from 'chai'

describe('#AegisOracle', () => {
  describe('#setOperator', () => {
    describe('success', () => {
      it('should add new operator', async () => {
        const [owner, operator] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisOracle', [[owner], owner])

        await expect(contract.setOperator(operator, true)).to.
          emit(contract, 'SetOperator').
          withArgs(operator, true)
      })

      it('should remove existing operator', async () => {
        const [owner, operator] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisOracle', [[owner], owner])

        await expect(contract.setOperator(operator, true)).to.be.not.reverted

        await expect(contract.setOperator(operator, false)).to.
          emit(contract, 'SetOperator').
          withArgs(operator, false)
      })
    })

    describe('error', () => {
      it('should revert when caller is not an owner', async () => {
        const [owner, notOwner] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisOracle', [[owner], owner])

        await expect(contract.connect(notOwner).setOperator(notOwner, true)).to.be.
          revertedWithCustomError(contract, 'OwnableUnauthorizedAccount')
      })
    })
  })

  describe('#updateYUSDPrice', () => {
    describe('success', () => {
      it('should update yusd price', async () => {
        const [owner] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisOracle', [[owner], owner])

        const price = 99963000n
        const timestamp = (await time.latest()) + 1

        await expect(contract.updateYUSDPrice(price)).to.
          emit(contract, 'UpdateYUSDPrice').
          withArgs(price, timestamp)

        await expect(contract.yusdUSDPrice()).to.be.eventually.equal(price)
        await expect(contract.lastUpdateTimestamp()).to.be.eventually.equal(timestamp)
      })
    })

    describe('error', () => {
      it('should revert when caller is not an operator', async () => {
        const [owner] = await ethers.getSigners()

        const contract = await ethers.deployContract('AegisOracle', [[], owner])

        await expect(contract.updateYUSDPrice(99963000)).to.be.
          revertedWithCustomError(contract, 'AccessForbidden')
      })
    })
  })
})
