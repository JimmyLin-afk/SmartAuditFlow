import { loadFixture } from '@nomicfoundation/hardhat-toolbox/network-helpers'
import { anyValue } from '@nomicfoundation/hardhat-chai-matchers/withArgs'
import { expect } from 'chai'
import { ethers } from 'hardhat'

import { VaultFactory } from '../typechain-types'
import lidoVaultArtifact from '../artifacts/contracts/LidoVault.sol/LidoVault.json'
import { DEFAULTS, ONE_ADDRESS, SIDE, ZERO_ADDRESS } from './helpers'

context('VaultFactory', () => {
  const vaultBytecode = lidoVaultArtifact.bytecode
  const vaultDeployedBytecode = lidoVaultArtifact.deployedBytecode

  async function deployVaultFactoryFixture() {
    let VaultFactoryFactory = await ethers.getContractFactory('VaultFactory')

    let deployer
    let addr1
    let addr2
    let addr3
    let addr4
    let addrs
    ;[deployer, addr1, addr2, addr3, addr4, ...addrs] = await ethers.getSigners()
    
    const protocolFeeBps = DEFAULTS.protocolFeeBps; // Default value
    const earlyExitFeeBps = DEFAULTS.earlyExitFeeBps; // Default value
 
    let vaultFactory: VaultFactory = (await VaultFactoryFactory.deploy(
      protocolFeeBps,
      earlyExitFeeBps
    )) as any
    
    return {
      vaultFactory,
      deployer,
      addr1,
      addr2,
      addr3,
      addr4,
      addrs,
    }
  }

  async function addVaultCloneFixture() {
    const vaultFactoryState = await deployVaultFactoryFixture()

    const { vaultFactory } = vaultFactoryState

    return { ...vaultFactoryState }
  }

  async function createLidoVaultFixture() {
    const vaultFactoryState = await deployVaultFactoryFixture()

    const { vaultFactory } = vaultFactoryState

    await (
      await vaultFactory.createVault(
        DEFAULTS.fixedSideCapacity,
        DEFAULTS.durationSeconds,
        DEFAULTS.variableSideCapacity
      )
    ).wait()
    const vaultInfo = await vaultFactory.vaultInfo(1)
    const vaultContract = await ethers.getContractAt('LidoVault', vaultInfo.addr)

    return { ...vaultFactoryState, vaultContract }
  }

  context('Deployment', () => {
    it('deployment Fee Bps error', async () => {
      const ContractVaultFactoryFactory = await ethers.getContractFactory('VaultFactory')
      const earlyExitFeeBps = DEFAULTS.earlyExitFeeBps
      const protocolFeeBps = 10_001

      await expect(ContractVaultFactoryFactory.deploy(
        protocolFeeBps,
        earlyExitFeeBps
      )).to.be.revertedWith('IPB')
    })

    it('is deployed', async () => {
      const { vaultFactory } = await loadFixture(deployVaultFactoryFixture)
      expect(await vaultFactory.getAddress()).to.not.be.undefined
    })

    it('has correct initial state values', async () => {
      const { vaultFactory, deployer } = await loadFixture(deployVaultFactoryFixture)

      expect(await vaultFactory.nextVaultId()).to.eq(1)
      expect(await vaultFactory.protocolFeeBps()).to.eq(DEFAULTS.protocolFeeBps)
      expect(await vaultFactory.protocolFeeReceiver()).to.eq(deployer.address)
      expect(await vaultFactory.earlyExitFeeBps()).to.eq(DEFAULTS.earlyExitFeeBps)
    })
  })

  context('Vault creation', () => {

    context('Lido Vault creation', () => {

      it('creates lido vault', async () => {
        const { vaultFactory, vaultContract, deployer } = await loadFixture(
          createLidoVaultFixture
        )

        expect(await vaultContract.fixedSideCapacity()).to.be.greaterThan(0)
        expect(await vaultContract.variableSideCapacity()).to.be.greaterThan(0)
        expect(await vaultContract.id()).to.be.equal(1)
        expect(await vaultContract.duration()).to.be.equal(DEFAULTS.durationSeconds)
        expect(await vaultContract.fixedSideCapacity()).to.be.equal(
          DEFAULTS.fixedSideCapacity
        )
        expect(await vaultContract.variableSideCapacity()).to.be.equal(
          DEFAULTS.variableSideCapacity
        )

        expect(await vaultContract.minimumDepositAmount()).to.be.equal(DEFAULTS.minimumDepositAmount)
        expect(await vaultContract.minimumFixedDepositBps()).to.be.equal(DEFAULTS.minimumFixedDepositBps)
        
        expect(await vaultContract.earlyExitFeeBps()).to.be.equal(DEFAULTS.earlyExitFeeBps)
        expect(await vaultContract.protocolFeeBps()).to.be.equal(DEFAULTS.protocolFeeBps)
        expect(await vaultContract.protocolFeeReceiver()).to.be.equal(
          await vaultFactory.protocolFeeReceiver()
        )
      })
    })

    it('creates a vault', async () => {
      const { vaultFactory, deployer } = await loadFixture(addVaultCloneFixture)

      const createVaultTrx = await vaultFactory.createVault(
        DEFAULTS.fixedSideCapacity,
        DEFAULTS.durationSeconds,
        DEFAULTS.variableSideCapacity
      )

      await expect(createVaultTrx)
        .to.emit(vaultFactory, 'VaultCreated')
        .withArgs(
          1,
          DEFAULTS.durationSeconds,
          DEFAULTS.fixedSideCapacity,
          DEFAULTS.variableSideCapacity,
          DEFAULTS.earlyExitFeeBps,
          DEFAULTS.protocolFeeBps,
          deployer.address,
          deployer.address,
          anyValue
        )

      const nextVaultId = await vaultFactory.nextVaultId()
      expect(nextVaultId).to.equal(2)

      const vaultInfo = await vaultFactory.vaultInfo(1)
      expect(vaultInfo.creator).to.equal(deployer.address)
    })
  })

  context('wasDeployedByFactory', () => {
    it('returns true when querying a factory-deployed vault', async () => {
      const { vaultFactory, vaultContract } = await loadFixture(createLidoVaultFixture)

      const wasDeployedByFactory = await vaultFactory.wasDeployedByFactory(
        await vaultContract.getAddress()
      )
      expect(wasDeployedByFactory).to.be.true
    })

    it('returns false when querying an irrelevant address', async () => {
      const { vaultFactory } = await loadFixture(deployVaultFactoryFixture)

      const wasDeployedByFactory = await vaultFactory.wasDeployedByFactory(
        '0x0000000000000000000000000000000000000001'
      )
      expect(wasDeployedByFactory).to.be.false
    })
  })

  context('setProtocolFeeBps', () => {
    it('error sets protocol fee bps', async () => {
      const { vaultFactory, deployer } = await loadFixture(deployVaultFactoryFixture)
      const newProtocolFeeBps = 10_001

      expect(vaultFactory.setProtocolFeeBps(newProtocolFeeBps)).to.be.revertedWith('IPB')
    })

    it('sets protocol fee bps', async () => {
      const { vaultFactory, deployer } = await loadFixture(deployVaultFactoryFixture)
      const newProtocolFeeBps = 1000

      await vaultFactory.setProtocolFeeBps(newProtocolFeeBps)
      expect(await vaultFactory.protocolFeeBps()).to.equal(newProtocolFeeBps)
    })

    it('emits SetProtocolFeeBps event', async () => {
      const { vaultFactory, deployer } = await loadFixture(deployVaultFactoryFixture)
      const newProtocolFeeBps = 1000

      await expect(vaultFactory.setProtocolFeeBps(newProtocolFeeBps))
        .to.emit(vaultFactory, 'SetProtocolFeeBps')
        .withArgs(newProtocolFeeBps)
    })
  })

  context('setProtocolFeeReceiver', () => {
    it('sets protocol fee receiver', async () => {
      const { vaultFactory, deployer, addr1 } = await loadFixture(deployVaultFactoryFixture)

      await vaultFactory.setProtocolFeeReceiver(addr1.address)
      expect(await vaultFactory.protocolFeeReceiver()).to.equal(addr1.address)
    })

    it('emits SetProtocolFeeReceiver event', async () => {
      const { vaultFactory, deployer, addr1 } = await loadFixture(deployVaultFactoryFixture)

      await expect(vaultFactory.setProtocolFeeReceiver(addr1.address))
        .to.emit(vaultFactory, 'SetProtocolFeeReceiver')
        .withArgs(addr1.address)
    })
  })

  context('setEarlyExitFeeBps', () => {
    it('sets early exit fee bps', async () => {
      const { vaultFactory, deployer } = await loadFixture(deployVaultFactoryFixture)
      const newEarlyExitFeeBps = 500

      await vaultFactory.setEarlyExitFeeBps(newEarlyExitFeeBps)
      expect(await vaultFactory.earlyExitFeeBps()).to.equal(newEarlyExitFeeBps)
    })

    it('emits SetEarlyExitFeeBps event', async () => {
      const { vaultFactory, deployer } = await loadFixture(deployVaultFactoryFixture)
      const newEarlyExitFeeBps = 500

      await expect(vaultFactory.setEarlyExitFeeBps(newEarlyExitFeeBps))
        .to.emit(vaultFactory, 'SetEarlyExitFeeBps')
        .withArgs(newEarlyExitFeeBps)
    })
  })
})
