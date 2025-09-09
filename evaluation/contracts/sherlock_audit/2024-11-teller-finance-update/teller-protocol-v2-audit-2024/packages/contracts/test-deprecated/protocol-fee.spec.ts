import chai from 'chai'
import chaiAsPromised from 'chai-as-promised'
import hre from 'hardhat'
import { deploy } from 'helpers/deploy-helpers'
import { ProtocolFeeMock } from 'types/typechain'

chai.should()
chai.use(chaiAsPromised)

const { getNamedSigner, deployments } = hre

const setup = deployments.createFixture(async () => {
  const contract = await deploy<ProtocolFeeMock>({
    contract: 'ProtocolFee',
    mock: true,
    hre,
  })

  await contract.initialize(100)

  return { contract }
})

describe('ProtocolFee', () => {
  let contract: ProtocolFeeMock

  beforeEach(async () => {
    const result = await setup()
    contract = result.contract
  })

  describe('__ProtocolFee_init', () => {
    it('should have initialized the contract with an owner and fee', async () => {
      const deployer = await getNamedSigner('deployer')

      await contract
        .owner()
        .should.eventually.become(await deployer.getAddress())

      await contract.setProtocolFeeCalled().should.eventually.be.true
    })
  })

  describe('setProtocolFee', () => {
    it('should fail if the caller is not the owner', async () => {
      const rando = await getNamedSigner('rando')
      await contract
        .connect(rando)
        .setProtocolFee(0)
        .should.be.revertedWith('Ownable: caller is not the owner')
    })

    it('should be able to update the protocol fee', async () => {
      const oldFee = await contract.protocolFee()
      const newFee = Number(oldFee) + 100
      await contract
        .setProtocolFee(newFee)
        .should.emit(contract, 'ProtocolFeeSet')
        .withArgs(newFee, oldFee)
    })

    it('should not do anything if the fee is the same', async () => {
      const oldFee = await contract.protocolFee()
      await contract
        .setProtocolFee(oldFee)
        .should.not.emit(contract, 'ProtocolFeeSet')
    })
  })
})
