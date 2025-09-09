import { time, loadFixture } from '@nomicfoundation/hardhat-toolbox/network-helpers'
import { anyValue, anyUint } from '@nomicfoundation/hardhat-chai-matchers/withArgs'
import { assert, expect } from 'chai'
import { AddressLike, BigNumberish, ContractTransactionReceipt, parseEther, formatEther } from 'ethers'
import { ethers } from 'hardhat'

import { LidoVault, VaultFactory } from '../typechain-types'
import { ContractMethodArgs } from '../typechain-types/common'
import { ILidoVaultInitializer } from '../typechain-types/contracts/LidoVault'
import {
  compareWithBpsTolerance,
  decodeLidoErrorData,
  finalizeLidoWithdrawalRequests,
  getLidoContract,
  getWithdrawalQueueERC721Contract,
  submitOracleReport,
  BIG_INT_ZERO,
  BIG_INT_ONE,
  BIG_INT_10K,
  DEFAULTS,
  SIDE,
  ONE_ADDRESS,
  ZERO_ADDRESS,
  setupMockLidoContracts,
} from './helpers'

describe('LidoVault', function () {
  function calculateGasFees(receipt: ContractTransactionReceipt | null) {
    const gasFees = receipt ? receipt?.gasUsed * receipt.gasPrice : BIG_INT_ZERO
    return gasFees
  }

  async function getTimeState(lidoVault: LidoVault, requestTimestamp?: bigint) {
    const currentTimestamp = requestTimestamp ?? BigInt(await time.latest())
    const endTime = await lidoVault.endTime()
    const duration = await lidoVault.duration()
    const startTime = endTime - duration
    const elapsedTime = currentTimestamp - startTime
    const remainingTime = endTime - currentTimestamp

    return {
      currentTimestamp,
      endTime,
      startTime,
      duration,
      elapsedTime,
      remainingTime,
    }
  }

  function calculateFixedScaledEarlyExitFee({
    remainingTime,
    duration,
    upfrontPremium,
  }: {
    remainingTime: bigint
    duration: bigint
    upfrontPremium: bigint
  }) {
    const ONE_E18 = parseEther('1')
    const remainingProportion = (remainingTime * ONE_E18) / duration
    
    const scalingFeeBps = ( BigInt(1) + BigInt(DEFAULTS.earlyExitFeeBps)) * remainingProportion / ONE_E18
    const scaledEarlyExitFee = (upfrontPremium * scalingFeeBps) / BIG_INT_10K
    
    return scaledEarlyExitFee
  }

  async function calculateFees({
    lidoVault,
    upfrontPremium,
    timestamp,
  }: {
    lidoVault: LidoVault
    upfrontPremium: bigint
    timestamp: bigint
  }) {
    const { duration, remainingTime, elapsedTime } = await getTimeState(lidoVault, timestamp)
    const scaledEarlyExitFee = calculateFixedScaledEarlyExitFee({
      remainingTime,
      duration,
      upfrontPremium,
    })
    
    const payBackAmount = upfrontPremium - (upfrontPremium * elapsedTime) / duration
    
    return { payBackAmount, scaledEarlyExitFee }
  }

  async function expectVaultWithdrawalArraysAreEqual({
    expected,
    userAddress,
    map,
  }: {
    expected: any[]
    userAddress: string
    map: (
      ...args: ContractMethodArgs<[arg0: AddressLike, arg1: BigNumberish], 'view'>
    ) => Promise<bigint>
  }) {
    for (let i = 0; i < expected.length; i++) {
      const actualItem = await map(userAddress, i)
      expect(actualItem).to.equal(expected[i])
    }
  }

  async function expectVaultNotStartedWithdrawalArraysAreEqual(
    lidoVault: LidoVault,
    expected: any[],
    userAddress: string
  ) {
    await expectVaultWithdrawalArraysAreEqual({
      expected,
      userAddress,
      map: lidoVault.fixedToVaultNotStartedWithdrawalRequestIds,
    })
  }

  async function expectVaultOngoingFixedWithdrawalArraysAreEqual(
    lidoVault: LidoVault,
    expected: any[],
    userAddress: string,
    expectedTimestamp: bigint = BIG_INT_ZERO
  ) {
    const actualLength = (await lidoVault.getFixedOngoingWithdrawalRequestIds(userAddress)).length
    if (expected.length !== actualLength) {
      assert.fail(
        expected.length,
        actualLength,
        'Array lengths do not match. Expected ' + expected.length + ' Actual ' + actualLength
      )
    }

    const requestIds = await lidoVault.getFixedOngoingWithdrawalRequestIds(userAddress)
    const timestamp = await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(userAddress)

    if (expectedTimestamp > 0) {
      expect(timestamp).to.be.greaterThan(expectedTimestamp)
    } else {
      expect(timestamp).to.equal(0)
    }
    for (let i = 0; i < expected.length; i++) {
      const actualItem = requestIds[i]
      expect(actualItem).to.equal(expected[i])
    }
  }

  async function expectVaultOngoingVariableWithdrawalArraysAreEqual(
    lidoVault: LidoVault,
    expected: any[],
    userAddress: string
  ) {
    await expectVaultWithdrawalArraysAreEqual({
      expected,
      userAddress,
      map: lidoVault.variableToVaultOngoingWithdrawalRequestIds,
    })
  }

  async function expectVaultEndedWithdrawalArraysAreEqual(lidoVault: LidoVault, expected: any[]) {
    for (let i = 0; i < expected.length; i++) {
      const actualItem = await lidoVault.vaultEndedWithdrawalRequestIds(i)
      expect(actualItem).to.equal(expected[i])
    }
  }

  async function expectBalancesAddrAndWithdrawalArraysAreEqual(lidoVault: LidoVault, lidoWithdrawalQueueContract: any[], accountAddress: any) {
    // generate earnings
    await submitOracleReport()

    const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

    const balanceBefore = await ethers.provider.getBalance(accountAddress)
    const receipt = await (await lidoVault.connect(accountAddress).withdraw(SIDE.VARIABLE)).wait()
    const gasFees = calculateGasFees(receipt)
    const balanceAfter = await ethers.provider.getBalance(accountAddress)

    const requestIds = [lidoLastRequestId + BIG_INT_ONE]
    await expectVaultOngoingVariableWithdrawalArraysAreEqual(
      lidoVault,
      requestIds,
      accountAddress.address
    )

    expect(balanceBefore).to.be.equal(balanceAfter + gasFees)
  }

  async function expectBalancesAddrAreEqual(lidoVault: LidoVault, accountAddress: any) {
    const balanceBefore = await ethers.provider.getBalance(accountAddress)
    const receipt = await (await lidoVault.connect(accountAddress).withdraw(SIDE.VARIABLE)).wait()
    const gasFees = calculateGasFees(receipt)
    const balanceAfter = await ethers.provider.getBalance(accountAddress)

    expect(balanceBefore).to.be.equal(balanceAfter + gasFees)
  }

  let nextVaultId = 0

  async function deployVaultFactoryFixture() {
    let VaultFactoryFactory = await ethers.getContractFactory('VaultFactory')

    let deployer
    let addr1
    let addr2
    let addr3
    let addr4
    let addrs
    ;[deployer, addr1, addr2, addr3, addr4, ...addrs] = await ethers.getSigners()

    const protocolFeeBps = DEFAULTS.protocolFeeBps;
    const earlyExitFeeBps = DEFAULTS.earlyExitFeeBps;

    let vaultFactory: VaultFactory = (await VaultFactoryFactory.deploy(
      protocolFeeBps,
      earlyExitFeeBps
    )) as any

    let vaultAddress = await vaultFactory.vaultContract();

    console.log('vaultInfo', vaultAddress)
    const vaultContract: LidoVault = await ethers.getContractAt('LidoVault', vaultAddress)

    return {
      vaultFactory,
      deployer,
      vaultContract,
    }
  }

  async function deployVault({
    durationSeconds = DEFAULTS.durationSeconds,
    fixedSideCapacity = DEFAULTS.fixedSideCapacity,
    variableSideCapacity = fixedSideCapacity * BigInt(DEFAULTS.fixedPremiumBps) / BigInt(10000),
    earlyExitFeeBps = DEFAULTS.earlyExitFeeBps,
    protocolFeeBps = DEFAULTS.protocolFeeBps,
    protocolFeeReceiver,
    admin,
  }: {
    durationSeconds?: number
    fixedSideCapacity?: BigInt
    variableSideCapacity?: BigInt
    earlyExitFeeBps?: number
    protocolFeeBps?: number
    protocolFeeReceiver?: string
    admin?: string
  }) {
    let LidoVaultFactory = await ethers.getContractFactory('LidoVault')

    let deployer
    let addr1
    let addr2
    let addr3
    let addr4
    let addrs
    ;[deployer, addr1, addr2, addr3, addr4, ...addrs] = await ethers.getSigners()

    const feeReceiver = protocolFeeReceiver ?? deployer

    const vaultId = ++nextVaultId
    const lidoVault: LidoVault = (await LidoVaultFactory.deploy(false)) as any

    const lidoVaultAddress = await lidoVault.getAddress()
    
    await lidoVault.initialize({
      vaultId,
      duration: durationSeconds,
      fixedSideCapacity,
      variableSideCapacity,
      earlyExitFeeBps,
      protocolFeeBps,
      protocolFeeReceiver: feeReceiver,
      admin: admin ?? deployer
    } as ILidoVaultInitializer.InitializationParamsStruct)

    const lidoContract = await getLidoContract(deployer)
    const lidoWithdrawalQueueContract = getWithdrawalQueueERC721Contract(deployer)

    return {
      lidoVault,
      deployer,
      addr1,
      addr2,
      addr3,
      addr4,
      addrs,
      vaultId,
      protocolFeeReceiver: feeReceiver,
      lidoVaultAddress,
      lidoContract,
      lidoWithdrawalQueueContract,
    }
  }
  

  async function deployVaultWithMockedLido(args: {
    durationSeconds?: number
    fixedSideCapacity?: BigInt
    variableSideCapacity?: BigInt
    earlyExitFeeBps?: number
    protocolFeeBps?: number
    protocolFeeReceiver?: string
    admin?: string
  }) {
    const state = await deployVault(args)
    
    const { lidoVault } = state

    const { lidoMock, lidoWithdrawalQueueMock } = await setupMockLidoContracts(lidoVault)

    return {
      ...state,
      lidoMock,
      lidoWithdrawalQueueMock,
    }
  }

  const deployLidoVaultFixture = () => deployVault({})
  const deployLidoVaultWithMockedLidoFixture = () =>
    deployVaultWithMockedLido({ variableSideCapacity: parseEther('20'), protocolFeeBps: 0 })

  describe('Deployment', function () {
    it('Should set the correct variable side capacity', async function () {
      const { lidoVault } = await loadFixture(deployLidoVaultFixture)

      expect(await lidoVault.variableSideCapacity()).to.equal(parseEther('30')) // 3% of 1000
    })

    it('Smart Contract initialized by the LidoFactory', async function () {
      const { vaultContract, deployer } = await loadFixture(deployVaultFactoryFixture)

      await expect(
        vaultContract.connect(deployer).initialize({
          vaultId: 0,
          duration: DEFAULTS.durationSeconds,
          fixedSideCapacity: DEFAULTS.fixedSideCapacity,
          variableSideCapacity: DEFAULTS.variableSideCapacity,
          earlyExitFeeBps: DEFAULTS.earlyExitFeeBps,
          protocolFeeBps: DEFAULTS.protocolFeeBps,
          protocolFeeReceiver: deployer.address,
        })
      ).to.be.revertedWith('MBF')
    })
  })
  
  describe('Initialization', function () {
    it('Should emit the correct initialization event (events)', async function () {
      let LidoVaultFactory = await ethers.getContractFactory('LidoVault')

      let deployer
      ;[deployer] = await ethers.getSigners()

      const lidoVault: LidoVault = (await LidoVaultFactory.deploy(false)) as any

      const lidoVaultAddress = await lidoVault.getAddress()

      const initParams = {
        vaultId: 1,
        duration: DEFAULTS.durationSeconds,
        fixedSideCapacity: DEFAULTS.fixedSideCapacity,
        variableSideCapacity: DEFAULTS.variableSideCapacity,
        earlyExitFeeBps: DEFAULTS.earlyExitFeeBps,
        protocolFeeBps: DEFAULTS.protocolFeeBps,
        protocolFeeReceiver: deployer.address,
      }
      const initTrx = await lidoVault.initialize(initParams)

      await expect(initTrx)
        .to.emit(lidoVault, 'VaultInitialized')
        .withArgs(
          initParams.vaultId,
          initParams.duration,
          parseEther('30'),
          initParams.fixedSideCapacity,
          initParams.earlyExitFeeBps,
          initParams.protocolFeeBps,
          initParams.protocolFeeReceiver,
         )
    })
    
    it('Fails to initialize if already initialized', async function () {
      // already initialized in fixture
      const { lidoVault, deployer } = await loadFixture(deployLidoVaultFixture)
      
      await expect(
        lidoVault.connect(deployer).initialize({
          vaultId: 0,
          duration: DEFAULTS.durationSeconds,
          fixedSideCapacity: DEFAULTS.fixedSideCapacity,
          variableSideCapacity: DEFAULTS.variableSideCapacity,
          earlyExitFeeBps: DEFAULTS.earlyExitFeeBps,
          protocolFeeBps: DEFAULTS.protocolFeeBps,
          protocolFeeReceiver: deployer.address,
        })
      ).to.be.revertedWith('ORO')
    
    })

   it('Should revert when sending funds to LidoVault', async function () {
     const { lidoVaultAddress, addr1 } = await loadFixture(deployLidoVaultFixture)
      const amountToSend = parseEther('1.0');

      await expect(
        addr1.sendTransaction({
          to: lidoVaultAddress,
          value: amountToSend,
        })
      ).to.be.revertedWith('LRC');
    });

    it('Fails to initialize if parameters are not set', async function () {
      let LidoVaultFactory = await ethers.getContractFactory('LidoVault')

      let deployer
      ;[deployer] = await ethers.getSigners()

      const lidoVault: LidoVault = (await LidoVaultFactory.deploy(false)) as any

      await expect(
        lidoVault.connect(deployer).initialize({
          vaultId: 0,
          duration: DEFAULTS.durationSeconds,
          fixedSideCapacity: DEFAULTS.fixedSideCapacity,
          variableSideCapacity: DEFAULTS.variableSideCapacity,
          earlyExitFeeBps: DEFAULTS.earlyExitFeeBps,
          protocolFeeBps: DEFAULTS.protocolFeeBps,
          protocolFeeReceiver: deployer.address,
        })
      ).to.be.revertedWith('NEI')

      await expect(
        lidoVault.connect(deployer).initialize({
          vaultId: 2,
          duration: 0,
          fixedSideCapacity: DEFAULTS.fixedSideCapacity,
          variableSideCapacity: DEFAULTS.variableSideCapacity,
          earlyExitFeeBps: DEFAULTS.earlyExitFeeBps,
          protocolFeeBps: DEFAULTS.protocolFeeBps,
          protocolFeeReceiver: deployer.address,
        })
      ).to.be.revertedWith('NEI')

      await expect(
        lidoVault.connect(deployer).initialize({
          vaultId: 2,
          duration: DEFAULTS.durationSeconds,
          fixedSideCapacity: 0,
          variableSideCapacity: DEFAULTS.variableSideCapacity,
          earlyExitFeeBps: DEFAULTS.earlyExitFeeBps,
          protocolFeeBps: DEFAULTS.protocolFeeBps,
          protocolFeeReceiver: deployer.address,
        })
      ).to.be.revertedWith('NEI')

      await expect(
        lidoVault.connect(deployer).initialize({
          vaultId: 2,
          duration: DEFAULTS.durationSeconds,
          fixedSideCapacity: DEFAULTS.fixedSideCapacity,
          variableSideCapacity: 0,
          earlyExitFeeBps: DEFAULTS.earlyExitFeeBps,
          protocolFeeBps: DEFAULTS.protocolFeeBps,
          protocolFeeReceiver: deployer.address,
        })
      ).to.be.revertedWith('NEI')

      await expect(
        lidoVault.connect(deployer).initialize({
          vaultId: 2,
          duration: DEFAULTS.durationSeconds,
          fixedSideCapacity: DEFAULTS.fixedSideCapacity,
          variableSideCapacity: DEFAULTS.variableSideCapacity,
          earlyExitFeeBps: 0,
          protocolFeeBps: DEFAULTS.protocolFeeBps,
          protocolFeeReceiver: deployer.address,
        })
      ).to.be.revertedWith('NEI')

      await expect(
        lidoVault.connect(deployer).initialize({
          vaultId: 2,
          duration: DEFAULTS.durationSeconds,
          fixedSideCapacity: DEFAULTS.fixedSideCapacity,
          variableSideCapacity: DEFAULTS.variableSideCapacity,
          earlyExitFeeBps: DEFAULTS.earlyExitFeeBps,
          protocolFeeBps: DEFAULTS.protocolFeeBps,
          protocolFeeReceiver: ZERO_ADDRESS,
        })
      ).to.be.revertedWith('NEI')

    })

    it('Fails to initialize if the calculated fixed minimum deposit is less than the overall minimum deposit amount', async function () {
      let LidoVaultFactory = await ethers.getContractFactory('LidoVault')

      let deployer;
      [deployer] = await ethers.getSigners()

      const lidoVault: LidoVault = (await LidoVaultFactory.deploy(false)) as any

      await expect(
        lidoVault.connect(deployer).initialize({
          vaultId: 2,
          duration: DEFAULTS.durationSeconds,
          fixedSideCapacity: parseEther('0.05'),
          variableSideCapacity: DEFAULTS.variableSideCapacity,
          earlyExitFeeBps: DEFAULTS.earlyExitFeeBps,
          protocolFeeBps: DEFAULTS.protocolFeeBps,
          protocolFeeReceiver: deployer.address,
        })
      ).to.be.revertedWith('IFC')
    })

    it('Can initialize if the calculated fixed minimum deposit is exactly the overall minimum deposit amount', async function () {
      let LidoVaultFactory = await ethers.getContractFactory('LidoVault')

      let deployer
      let addr1
      [deployer, addr1] = await ethers.getSigners()

      const lidoVault: LidoVault = (await LidoVaultFactory.deploy(false)) as any

      const lidoVaultAddress = await lidoVault.getAddress()

      await lidoVault.connect(deployer).initialize({
        vaultId: 2,
        duration: DEFAULTS.durationSeconds,
        fixedSideCapacity: parseEther('0.2'), // 5% of 0.2 = 0.01 aka the min deposit amount
        variableSideCapacity: DEFAULTS.variableSideCapacity,
        earlyExitFeeBps: DEFAULTS.earlyExitFeeBps,
        protocolFeeBps: DEFAULTS.protocolFeeBps,
        protocolFeeReceiver: deployer.address,
      })

      expect(await lidoVault.fixedSideCapacity()).to.be.greaterThan(0)
    })

    it('Fails to deposit if not initialized', async function () {
      let LidoVaultFactory = await ethers.getContractFactory('LidoVault')

      let deployer
      ;[deployer] = await ethers.getSigners()

      const lidoVault: LidoVault = (await LidoVaultFactory.deploy(false)) as any

      await expect(
        lidoVault.connect(deployer).deposit(SIDE.FIXED, { value: parseEther('200') })
      ).to.be.revertedWith('NI')
    })
  })

  describe('Vault not Started', function () {
    it('isStarted should be set to false', async function () {
      const { lidoVault } = await loadFixture(deployLidoVaultFixture)

      expect(await lidoVault.isStarted()).to.equal(false)
    })

    it('isEnded should be set to false', async function () {
      const { lidoVault } = await loadFixture(deployLidoVaultFixture)

      expect(await lidoVault.isEnded()).to.equal(false)
    })

    describe('Deposit', function () {
      it('Should fail on invalid side', async function () {
        const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

        await expect(lidoVault.connect(addr1).deposit(3)).to.be.revertedWith('IS')
      })

      it('Should fail if no ETH value is sent', async function () {
        const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

        await expect(lidoVault.connect(addr1).deposit(SIDE.VARIABLE)).to.be.revertedWith('MDA')
      })

      describe('Fixed Side', function () {
        it('Should fail if ETH value is less than minimum for fixed', async function () {
          const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

          await expect(
            lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: parseEther('0.0001') })
          ).to.be.revertedWith('MDA')
        })

        it('Should fail if ETH value causes remaining capacity to be less than minimum for fixed', async function () {
          const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

          const fixedDepositMin =
            (DEFAULTS.fixedSideCapacity * BigInt(DEFAULTS.minimumFixedDepositBps)) / BIG_INT_10K

          await expect(
            lidoVault.connect(addr1).deposit(SIDE.FIXED, {
              value: DEFAULTS.fixedSideCapacity - (fixedDepositMin - BIG_INT_ONE),
            })
          ).to.be.revertedWith('RC')
        })

        it('Should fail if ETH value causes remaining capacity to be less than minimum for fixed on next deposit', async function () {
          const { lidoVault, addr1, addr2 } = await loadFixture(deployLidoVaultFixture)

          const fixedDepositMin =
            (DEFAULTS.fixedSideCapacity * BigInt(DEFAULTS.minimumFixedDepositBps)) / BIG_INT_10K
          await lidoVault.connect(addr2).deposit(SIDE.FIXED, {
            value: DEFAULTS.fixedSideCapacity - fixedDepositMin * BigInt(2),
          })

          await expect(
            lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDepositMin + BIG_INT_ONE })
          ).to.be.revertedWith('RC')
        })

        it('Should fail on deposit greater than the fixed capacity', async function () {
          const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

          const depositAmount = parseEther('2000')

          await expect(
            lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: depositAmount })
          ).to.be.revertedWith('OED')
        })

        it('Should fail if fixed capacity is already reached', async function () {
          const { lidoVault, addr1, addr2 } = await loadFixture(deployLidoVaultFixture)

          await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: DEFAULTS.fixedSideCapacity })

          await expect(
            lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: parseEther('200') })
          ).to.be.revertedWith('OED')
        })

        it('Should fail if deposit exceeds fixed capacity', async function () {
          const { lidoVault, addr1, addr2 } = await loadFixture(deployLidoVaultFixture)

          await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: parseEther('900') })

          await expect(
            lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: parseEther('200') })
          ).to.be.revertedWith('OED')
        })

        it('Should fail to deposit if amount is less than calculated fixed deposit min', async function () {
          const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

          const fixedDepositMin =
            (DEFAULTS.fixedSideCapacity * BigInt(DEFAULTS.minimumFixedDepositBps)) / BIG_INT_10K

          await expect(
            lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDepositMin - BIG_INT_ONE })
          ).to.be.revertedWith('MFD')
        })

        it('Should emit the correct fixed deposit event (events)', async function () {
          const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

          const depositAmount = parseEther('200')
          const depositTrx = await lidoVault
            .connect(addr1)
            .deposit(SIDE.FIXED, { value: depositAmount })

          await expect(depositTrx)
            .to.emit(lidoVault, 'FixedFundsDeposited')
            .withArgs(depositAmount, anyUint, addr1.address)
        })

        describe('Single Fixed Participant', function () {
          it('Should deposit correctly', async function () {
            const { lidoVault, addr1 } = await loadFixture(
              deployLidoVaultFixture
            )

            const depositAmount = parseEther('200')
            const claimTokensBefore = await lidoVault.fixedClaimTokenTotalSupply()
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: depositAmount })
            ).wait()
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(depositAmount).to.be.equal(await lidoVault.fixedETHDepositToken(addr1.address))
            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.greaterThan(claimTokensBefore)
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(depositAmount)
            expect(await lidoVault.fixedETHDepositTokenTotalSupply()).to.equal(depositAmount)
            expect(balanceAfter).to.equal(balanceBefore - depositAmount - calculateGasFees(receipt))
          })

          it('Should deposit multiple times correctly', async function () {
            const { lidoVault, addr1 } = await loadFixture(
              deployLidoVaultFixture
            )

            const depositAmount1 = parseEther('200')
            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: depositAmount1 })
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(depositAmount1)

            const depositAmount2 = parseEther('50')
            const claimTokensBefore = await lidoVault.fixedClaimTokenTotalSupply()
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: depositAmount2 })
            ).wait()
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(depositAmount1 + depositAmount2).to.be.equal(
              await lidoVault.fixedETHDepositToken(addr1.address)
            )
            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.greaterThan(claimTokensBefore)
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(parseEther('250'))
            expect(await lidoVault.fixedETHDepositTokenTotalSupply()).to.equal(parseEther('250'))
            expect(balanceAfter).to.equal(
              balanceBefore - depositAmount2 - calculateGasFees(receipt)
            )
          })

          it('Should deposit if amount is exactly than calculated fixed deposit min', async function () {
            const { lidoVault, addr1 } = await loadFixture(
              deployLidoVaultFixture
            )

            const fixedDepositMin =
              (DEFAULTS.fixedSideCapacity * BigInt(DEFAULTS.minimumFixedDepositBps)) / BIG_INT_10K

            const claimTokensBefore = await lidoVault.fixedClaimTokenTotalSupply()
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDepositMin })
            ).wait()
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(fixedDepositMin).to.be.equal(await lidoVault.fixedETHDepositToken(addr1.address))
            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.greaterThan(claimTokensBefore)
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDepositMin)
            expect(await lidoVault.fixedETHDepositTokenTotalSupply()).to.equal(fixedDepositMin)
            expect(balanceAfter).to.equal(
              balanceBefore - fixedDepositMin - calculateGasFees(receipt)
            )
          })

          it('Should deposit if amount is greater than calculated fixed deposit min', async function () {
            const { lidoVault, addr1 } = await loadFixture(
              deployLidoVaultFixture
            )

            const fixedDepositMin =
              (DEFAULTS.fixedSideCapacity * BigInt(DEFAULTS.minimumFixedDepositBps)) / BIG_INT_10K
            const fixedDeposit = fixedDepositMin + BIG_INT_ONE

            const claimTokensBefore = await lidoVault.fixedClaimTokenTotalSupply()
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
            ).wait()
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(fixedDeposit).to.be.equal(await lidoVault.fixedETHDepositToken(addr1.address))
            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.greaterThan(claimTokensBefore)
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDeposit)
            expect(await lidoVault.fixedETHDepositTokenTotalSupply()).to.equal(fixedDeposit)
            expect(balanceAfter).to.equal(balanceBefore - fixedDeposit - calculateGasFees(receipt))
          })
        })

        describe('Multiple Fixed Participants', function () {
          it('Should be able to have multiple participants deposit correctly (multiple-participants)', async function () {
            const { lidoVault, addr1, addr2 } =
              await loadFixture(deployLidoVaultFixture)

            const depositAmount1 = parseEther('200')
            const stETHBefore1 = await lidoVault.fixedClaimTokenTotalSupply()
            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: depositAmount1 })

            expect(depositAmount1).to.be.equal(await lidoVault.fixedETHDepositToken(addr1.address))
            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.greaterThan(stETHBefore1)

            const depositAmount2 = parseEther('300')
            const stETHBefore2 = await lidoVault.fixedClaimTokenTotalSupply()
            await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: depositAmount2 })

            expect(depositAmount2).to.be.equal(await lidoVault.fixedETHDepositToken(addr2.address))
            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.greaterThan(stETHBefore2)
          })
        })
      })

      describe('Variable Side', function () {
        it('Should fail if ETH value is less than minimum for variable', async function () {
          const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

          await expect(
            lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: parseEther('0.0001') })
          ).to.be.revertedWith('MDA')
        })

        it('Should fail if ETH value causes remaining capacity to be less than minimum for variable', async function () {
          const { lidoVault, addr1, addr2 } = await loadFixture(deployLidoVaultFixture)

          await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, {
            value: parseEther('30') - DEFAULTS.minimumDepositAmount * BigInt(2),
          })

          await expect(
            lidoVault
              .connect(addr1)
              .deposit(SIDE.VARIABLE, { value: DEFAULTS.minimumDepositAmount + BIG_INT_ONE })
          ).to.be.revertedWith('RC')
        })

        it('Should fail on deposit greater than the variable capacity', async function () {
          const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

          const depositAmount = parseEther('500')

          await expect(
            lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: depositAmount })
          ).to.be.revertedWith('OED')
        })

        it('Should fail if variable capacity is already reached', async function () {
          const { lidoVault, addr1, addr2 } = await loadFixture(deployLidoVaultFixture)

          await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: parseEther('30') })

          await expect(
            lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: parseEther('20') })
          ).to.be.revertedWith('OED')
        })

        it('Should fail if deposit exceeds variable capacity', async function () {
          const { lidoVault, addr1, addr2 } = await loadFixture(deployLidoVaultFixture)

          await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: parseEther('25') })

          await expect(
            lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: parseEther('6') })
          ).to.be.revertedWith('OED')
        })

        it('Should emit the correct variable deposit event (events)', async function () {
          const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

          const depositAmount = parseEther('20')
          const depositTrx = await lidoVault
            .connect(addr1)
            .deposit(SIDE.VARIABLE, { value: depositAmount })

          await expect(depositTrx)
            .to.emit(lidoVault, 'VariableFundsDeposited')
            .withArgs(depositAmount, addr1.address)
        })

        describe('Single Variable Participant', function () {
          it('Should deposit correctly', async function () {
            const { lidoVault, addr1 } = await loadFixture(
              deployLidoVaultFixture
            )

            const depositAmount = parseEther('20')
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: depositAmount })
            ).wait()
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(await lidoVault.variableBearerToken(addr1.address)).to.equal(depositAmount)
            expect(await lidoVault.variableBearerTokenTotalSupply()).to.equal(depositAmount)
            expect(balanceAfter).to.equal(balanceBefore - depositAmount - calculateGasFees(receipt))
          })

          it('Should deposit multiple times correctly', async function () {
            const { lidoVault, addr1 } = await loadFixture(
              deployLidoVaultFixture
            )

            const depositAmount1 = parseEther('20')
            await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: depositAmount1 })
            expect(await lidoVault.variableBearerToken(addr1.address)).to.equal(depositAmount1)

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const depositAmount2 = parseEther('5')
            const receipt = await (
              await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: depositAmount2 })
            ).wait()
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(await lidoVault.variableBearerToken(addr1.address)).to.equal(parseEther('25'))
            expect(await lidoVault.variableBearerTokenTotalSupply()).to.equal(parseEther('25'))
            expect(balanceAfter).to.equal(
              balanceBefore - depositAmount2 - calculateGasFees(receipt)
            )
          })
        })

        describe('Multiple Variable Participants', function () {
          it('Should be able to have multiple participants deposit correctly (multiple-participants)', async function () {
            const { lidoVault, addr1, addr2 } = await loadFixture(
              deployLidoVaultFixture
            )

            const depositAmount1 = parseEther('20')
            await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: depositAmount1 })
            expect(await lidoVault.variableBearerToken(addr1.address)).to.equal(depositAmount1)

            const depositAmount2 = parseEther('10')
            await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: depositAmount2 })
            expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(depositAmount2)
          })
        })
      })

      describe('Fixed And Variable Side', function () {
        it('Should be able to deposit as normal (fixed-and-variable-side)', async function () {
          const { lidoVault, addr1 } =
            await loadFixture(deployLidoVaultFixture)

          const fixedDepositAmount = parseEther('200')
          await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDepositAmount })
          expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDepositAmount)
          expect(await lidoVault.fixedClaimToken(addr1.address)).to.be.greaterThan(0)

          const variableDepositAmount = parseEther('20')
          await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: variableDepositAmount })
          expect(await lidoVault.variableBearerToken(addr1.address)).to.equal(variableDepositAmount)
        })
      })
    })

    describe('Claim', function () {
      it('Should fail before vault start', async function () {
        const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

        await expect(lidoVault.connect(addr1).claimFixedPremium()).to.be.revertedWith('CBS')
      })
    })

    describe('Withdraw', function () {
      it('Should fail on invalid side', async function () {
        const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

        await expect(lidoVault.connect(addr1).withdraw(3)).to.be.revertedWith('IS')
      })

      it('Should fail to finalize an ongoing vault withdrawal', async function () {
        const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

        await expect(
          lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
        ).to.be.revertedWith('WNR')
      })

      it('Should fail to finalize the vault ended withdrawal', async function () {
        const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

        await expect(
          lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
        ).to.be.revertedWith('WNR')
      })

      it('Should fail to finalize the vault ended withdrawal with an invalid side', async function () {
        const { lidoVault, addr1 } = await loadFixture(deployLidoVaultFixture)

        await expect(lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(5)).to.be.revertedWith(
          'IS'
        )
      })

      describe('Fixed Side', function () {
        const fixedDeposit = DEFAULTS.fixedSideCapacity

        async function depositFixedVaultFixture() {
          const state = await deployVault({})

          const { lidoVault, addr1 } = state

          await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })

          return state
        }

        it('Should emit the correct lido withdrawal requested event (events)', async function () {
          const { lidoVault, addr1 } = await loadFixture(depositFixedVaultFixture)

          const withdrawTrx = await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

          await expect(withdrawTrx)
            .to.emit(lidoVault, 'LidoWithdrawalRequested')
            .withArgs(addr1.address, anyValue, SIDE.FIXED, false, false)
        })

        it('Should emit the correct fixed withdrawal events on withdrawal finalization (events)', async function () {
          const { lidoVault, addr1, lidoWithdrawalQueueContract } = await loadFixture(
            depositFixedVaultFixture
          )

          const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
          const requestIds = [lidoLastRequestId + BIG_INT_ONE]
          await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
          await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

          const finalizedTrx = await lidoVault
            .connect(addr1)
            .finalizeVaultNotStartedFixedWithdrawals()

          await expect(finalizedTrx)
            .to.emit(lidoVault, 'LidoWithdrawalFinalized')
            .withArgs(addr1.address, anyValue, SIDE.FIXED, false, false)
          await expect(finalizedTrx)
            .to.emit(lidoVault, 'FixedFundsWithdrawn')
            .withArgs(anyUint, addr1.address, false, false)
        })

        describe('Single Fixed Participant', function () {
          it('Should withdraw correctly', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1,
              fixedETHDepositToken,
              fixedClaimToken,
            } = await loadFixture(depositFixedVaultFixture)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )

            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[0]
            )

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit - gasFees, 1)
            await expectVaultNotStartedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)
          })

          it('Should withdraw correctly and be able to deposit again', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1
            } = await loadFixture(depositFixedVaultFixture)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
            await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDeposit)
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.be.greaterThan(0)
            expect(balanceAfter).to.be.equal(balanceBefore - fixedDeposit - gasFees)
          })

          it('Should fail to withdraw again if user already has another withdrawal in progress', async function () {
            const { lidoVault, lidoWithdrawalQueueContract, addr1 } = await loadFixture(
              depositFixedVaultFixture
            )

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )

            await expect(lidoVault.connect(addr1).withdraw(SIDE.FIXED)).to.be.revertedWith('WAR')
          })

          it('Withdrawals should not count towards hitting the fixed capacity', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1
            } = await loadFixture(depositFixedVaultFixture)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )

            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)

            // hit variable capacity
            await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: parseEther('30') })
            expect(await lidoVault.variableBearerTokenTotalSupply()).to.equal(parseEther('30'))

            expect(await lidoVault.isStarted()).to.equal(false)
          })

          async function maxWithdrawalAmountVaultFixture() {
            const state = await deployVault({ fixedSideCapacity: parseEther('1500') })

            return state
          }

          it('Should calculate the withdrawal requests correctly if amount withdrawn is just over the max withdrawal amount', async function () {
            const { lidoVault, lidoWithdrawalQueueContract, addr1 } = await loadFixture(
              maxWithdrawalAmountVaultFixture
            )

            // the max withdrawal amount is 1000 ETH and the min is 100 wei
            const depositAmount = parseEther('1000') + BigInt(5)
            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: depositAmount })

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
            const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )
          })

          it('Should fail to withdraw if address has no claim tokens', async function () {
            const { lidoVault, addr3, fixedClaimToken } = await loadFixture(
              depositFixedVaultFixture
            )

            expect(await lidoVault.fixedClaimToken(addr3.address)).to.equal(0)
            await expect(lidoVault.connect(addr3).withdraw(SIDE.FIXED)).to.be.revertedWith('NCT')
          })

          it('Should fail to withdraw if lido withdrawal request has not been finalized', async function () {
            const { lidoVault, lidoWithdrawalQueueContract, addr1 } = await loadFixture(
              depositFixedVaultFixture
            )

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )

            try {
              await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
              assert.fail('Lido error not thrown')
            } catch (error: any) {
              const lidoError = decodeLidoErrorData(error)
              expect(lidoError?.name).to.equal('RequestNotFoundOrNotFinalized')
            }
          })

          it('Should fail to withdraw if lido withdrawal request if request has already been finalized', async function () {
            const { lidoVault, lidoWithdrawalQueueContract, addr1 } = await loadFixture(
              depositFixedVaultFixture
            )

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )

            await finalizeLidoWithdrawalRequests(requestIds, parseEther('1000'))
            await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()

            await expect(
              lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
            ).to.be.revertedWith('WNR')
          })

          it('Should fail to finalize withdraw if user has not requested a withdrawal', async function () {
            const { lidoVault, addr1 } = await loadFixture(depositFixedVaultFixture)

            await expect(
              lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
            ).to.be.revertedWith('WNR')
          })

          it('Should get all of the staking earnings on their deposit', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              lidoContract,
              addr1
            } = await loadFixture(depositFixedVaultFixture)

            await submitOracleReport()

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            // how much their initial deposited shares are worth in ETH
            const ethAmount = await lidoContract.getPooledEthByShares(
              await lidoVault.fixedClaimToken(addr1.address)
            )
            expect(ethAmount).to.be.greaterThan(fixedDeposit)

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
            const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )

            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[1]
            )

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(balanceAfter - balanceBefore).to.be.equal(ethAmount - gasFees)
            await expectVaultNotStartedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)

            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.equal(0)
            // no protocol fee should have been applied since the vault has not started
            const protocolFee = await lidoVault.appliedProtocolFee()
            expect(protocolFee).to.be.equal(0)
          })
          

          it('Calculates the correct withdrawal amounts if staked exactly the max withdrawal amount', async function () {
          
            const { lidoVault, lidoMock, addr1, lidoWithdrawalQueueMock } =
              await loadFixture(deployLidoVaultWithMockedLidoFixture)

            const lidoLastRequestId = BIG_INT_ZERO
            
            const maxAmount = await lidoVault.maxStETHWithdrawalAmount()

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: maxAmount })

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
            
            expect(await lidoWithdrawalQueueMock.withdrawalAmounts(1)).to.be.equal(maxAmount)
            // no 2nd request should be 0 - since its not set
            expect(await lidoWithdrawalQueueMock.withdrawalAmounts(2)).to.be.equal(0)            
 
          })

          it('Should get less than their initial deposit in the case of staking balance decreasing (staking-loss)', async function () {
            const { lidoVault, lidoMock, addr1 } =
              await loadFixture(deployLidoVaultWithMockedLidoFixture)

            const lidoLastRequestId = BIG_INT_ZERO

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })

            const shares = await lidoVault.fixedClaimToken(addr1.address)
            expect(shares).to.equal(parseEther('1000'))

            const stakingStETHBalanceBefore = await lidoMock.getPooledEthByShares(shares)

            // mock staking loss - 100 ETH
            const stakingEarnings = parseEther('100')
            await lidoMock.subtractStakingEarnings(stakingEarnings)

            const stakingStETHBalanceAfter = await lidoMock.getPooledEthByShares(shares)
            // eth = _sharesAmount.mulDiv(totalPooledEther, totalShares) = (1000 * (2000 - 100)) / 2000 = 950 - 50 ETH loss from original deposit
            const stakingLoss = parseEther('50')
            expect(stakingStETHBalanceBefore - stakingStETHBalanceAfter).to.equal(stakingLoss)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.equal(0)

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(balanceAfter - balanceBefore).to.be.equal(fixedDeposit - stakingLoss - gasFees)
            await expectVaultNotStartedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)

            // no protocol fee should have been applied since the vault has not started
            const protocolFee = await lidoVault.appliedProtocolFee()
            expect(protocolFee).to.be.equal(0)
          })
        })

        describe('Multiple Fixed Participants', function () {
          it('Should withdraw correctly (multiple-participants)', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1,
              addr2,
              fixedETHDepositToken,
              fixedClaimToken,
            } = await loadFixture(deployLidoVaultFixture)

            const fixedDeposit1 = parseEther('600')
            const fixedDeposit2 = parseEther('400')
            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
            await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            // withdraw #1
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )

            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[0]
            )

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit1 - gasFees, 1)
            await expectVaultNotStartedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)

            // withdraw #2
            await lidoVault.connect(addr2).withdraw(SIDE.FIXED)

            const requestIds2 = [lidoLastRequestId + BigInt(2)]
            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds2,
              addr2.address
            )

            expect(await lidoVault.fixedETHDepositToken(addr2.address)).to.equal(0)
            expect(await lidoVault.fixedClaimToken(addr2.address)).to.equal(0)

            await finalizeLidoWithdrawalRequests(requestIds2, DEFAULTS.fixedSideCapacity)

            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds2[0]
            )

            const balanceBefore2 = await ethers.provider.getBalance(addr2)
            const receipt2 = await (
              await lidoVault.connect(addr2).finalizeVaultNotStartedFixedWithdrawals()
            ).wait()
            const gasFees2 = calculateGasFees(receipt2)
            const balanceAfter2 = await ethers.provider.getBalance(addr2)

            expect(balanceAfter2 - balanceBefore2).to.equalWithTolerance(
              fixedDeposit2 - gasFees2,
              2
            )
            await expectVaultNotStartedWithdrawalArraysAreEqual(lidoVault, [], addr2.address)

            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.equal(0)
          })

          it('Should get their portion of the staking earnings and their initial deposit (multiple-participants)', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              lidoContract,
              addr1,
              addr2
            } = await loadFixture(deployLidoVaultFixture)

            const fixedDeposit1 = parseEther('600')
            const fixedDeposit2 = parseEther('400')
            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
            await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })

            await submitOracleReport()

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            // withdraw #1
            // how much their initial deposited shares are worth in ETH
            const ethAmount = await lidoContract.getPooledEthByShares(
              await lidoVault.fixedClaimToken(addr1.address)
            )
            expect(ethAmount).greaterThan(fixedDeposit1)

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )

            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[0]
            )

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(balanceAfter - balanceBefore).to.be.equal(ethAmount - gasFees)
            await expectVaultNotStartedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)

            // withdraw #2
            const ethAmount2 = await lidoContract.getPooledEthByShares(
              await lidoVault.fixedClaimToken(addr2.address)
            )
            expect(ethAmount2).greaterThan(fixedDeposit2)

            await lidoVault.connect(addr2).withdraw(SIDE.FIXED)

            const requestIds2 = [lidoLastRequestId + BigInt(2)]
            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds2,
              addr2.address
            )

            expect(await lidoVault.fixedETHDepositToken(addr2.address)).to.equal(0)
            expect(await lidoVault.fixedClaimToken(addr2.address)).to.equal(0)

            await finalizeLidoWithdrawalRequests(requestIds2, DEFAULTS.fixedSideCapacity)

            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds2[0]
            )

            const balanceBefore2 = await ethers.provider.getBalance(addr2)
            const receipt2 = await (
              await lidoVault.connect(addr2).finalizeVaultNotStartedFixedWithdrawals()
            ).wait()
            const gasFees2 = calculateGasFees(receipt2)
            const balanceAfter2 = await ethers.provider.getBalance(addr2)

            expect(balanceAfter2 - balanceBefore2).to.equalWithTolerance(ethAmount2 - gasFees2, 1)
            await expectVaultNotStartedWithdrawalArraysAreEqual(lidoVault, [], addr2.address)

            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.equal(0)
            // no protocol fee should have been applied since the vault has not started
            const protocolFee = await lidoVault.appliedProtocolFee()
            expect(protocolFee).to.be.equal(0)
          })

          it('Should get less than their initial deposit in the case of staking balance decreasing (multiple-participants, staking-loss)', async function () {
            const { lidoVault, lidoMock, addr1, addr2 } =
              await loadFixture(deployLidoVaultWithMockedLidoFixture)

            const lidoLastRequestId = BIG_INT_ZERO

            const fixedDeposit1 = parseEther('600')
            const fixedDeposit2 = parseEther('400')
            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
            await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })

            const shares1 = await lidoVault.fixedClaimToken(addr1.address)
            const shares2 = await lidoVault.fixedClaimToken(addr2.address)
            expect(shares1).to.equal(parseEther('600'))
            expect(shares2).to.equal(parseEther('400'))

            const stakingStETHBalanceBefore1 = await lidoMock.getPooledEthByShares(shares1)
            const stakingStETHBalanceBefore2 = await lidoMock.getPooledEthByShares(shares2)

            // mock staking loss - 100 ETH
            const stakingEarnings = parseEther('100')
            await lidoMock.subtractStakingEarnings(stakingEarnings)

            const stakingStETHBalanceAfter1 = await lidoMock.getPooledEthByShares(shares1)
            const stakingStETHBalanceAfter2 = await lidoMock.getPooledEthByShares(shares2)
            //  eth = _sharesAmount.mulDiv(totalPooledEther, totalShares) = (600 * (2000 - 100)) / 2000 = 570 - 30 ETH loss from original deposit
            const stakingLoss1 = parseEther('30')
            expect(stakingStETHBalanceBefore1 - stakingStETHBalanceAfter1).to.equal(stakingLoss1)
            //  eth = _sharesAmount.mulDiv(totalPooledEther, totalShares) = (400 * (2000 - 100)) / 2000 = 380 - 20 ETH loss from original deposit
            const stakingLoss2 = parseEther('20')
            expect(stakingStETHBalanceBefore2 - stakingStETHBalanceAfter2).to.equal(stakingLoss2)

            // addr1 withdraws
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.equal(parseEther('400')) // just the addr2 claim tokens

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(balanceAfter - balanceBefore).to.be.equal(fixedDeposit1 - stakingLoss1 - gasFees)
            await expectVaultNotStartedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)

            // no protocol fee should have been applied since the vault has not started
            const protocolFee = await lidoVault.appliedProtocolFee()
            expect(protocolFee).to.be.equal(0)

            // addr2 withdraws
            const requestIds2 = [lidoLastRequestId + BigInt(2)]
            await lidoVault.connect(addr2).withdraw(SIDE.FIXED)

            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds2,
              addr2.address
            )
            expect(await lidoVault.fixedETHDepositToken(addr2.address)).to.equal(0)
            expect(await lidoVault.fixedClaimToken(addr2.address)).to.equal(0)
            expect(await lidoVault.fixedClaimTokenTotalSupply()).to.equal(0) // just the addr2 claim tokens

            const balanceBefore2 = await ethers.provider.getBalance(addr2)
            const receipt2 = await (
              await lidoVault.connect(addr2).finalizeVaultNotStartedFixedWithdrawals()
            ).wait()
            const gasFees2 = calculateGasFees(receipt2)
            const balanceAfter2 = await ethers.provider.getBalance(addr2)

            expect(balanceAfter2 - balanceBefore2).to.be.equal(
              fixedDeposit2 - stakingLoss2 - gasFees2
            )
            await expectVaultNotStartedWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
          })
        })
      })

      describe('Variable Side', function () {
        const variableDeposit = parseEther('10')

        async function depositVariableVaultFixture() {
          const state = await deployVault({})

          const { lidoVault, addr1 } = state

          await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: variableDeposit })

          return state
        }

        it('Should fail to withdraw if address has no bearer tokens', async function () {
          const { lidoVault, addr3, variableBearerToken } = await loadFixture(
            depositVariableVaultFixture
          )

          expect(await lidoVault.variableBearerToken(addr3.address)).to.equal(0)
          await expect(lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)).to.be.revertedWith('NBT')
        })

        it('Should emit the correct variable withdrawn event (events)', async function () {
          const { lidoVault, addr1 } = await loadFixture(depositVariableVaultFixture)

          const withdrawTrx = await lidoVault.connect(addr1).withdraw(SIDE.VARIABLE)

          await expect(withdrawTrx)
            .to.emit(lidoVault, 'VariableFundsWithdrawn')
            .withArgs(variableDeposit, addr1.address, false, false)
        })

        describe('Single Variable Participant', function () {
          it('Should withdraw correctly', async function () {
            const { lidoVault, addr1, variableBearerToken } = await loadFixture(
              depositVariableVaultFixture
            )

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (await lidoVault.connect(addr1).withdraw(SIDE.VARIABLE)).wait()

            expect(await lidoVault.variableBearerToken(addr1.address)).to.equal(0)
            expect(await lidoVault.variableBearerTokenTotalSupply()).to.equal(0)
            expect(await ethers.provider.getBalance(addr1)).to.equal(
              balanceBefore + parseEther('10') - calculateGasFees(receipt)
            )
          })

          it('Should withdraw correctly and be able to deposit again', async function () {
            const { lidoVault, addr1 } = await loadFixture(
              depositVariableVaultFixture
            )

            await lidoVault.connect(addr1).withdraw(SIDE.VARIABLE)
            expect(await lidoVault.variableBearerToken(addr1.address)).to.equal(0)

            const depositAmount = parseEther('20')
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: depositAmount })
            ).wait()

            expect(await lidoVault.variableBearerToken(addr1.address)).to.equal(depositAmount)
            expect(await ethers.provider.getBalance(addr1)).to.equal(
              balanceBefore - depositAmount - calculateGasFees(receipt)
            )
          })
        })

        describe('Multiple Variable Participants', function () {
          it('Should withdraw correctly (multiple-participants)', async function () {
            const { lidoVault, addr1, addr2, variableBearerToken } = await loadFixture(
              deployLidoVaultFixture
            )

            const variableDeposit1 = parseEther('10')
            const variableDeposit2 = parseEther('20')

            await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: variableDeposit1 })
            await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })

            // withdraw #1
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (await lidoVault.connect(addr1).withdraw(SIDE.VARIABLE)).wait()

            expect(await lidoVault.variableBearerToken(addr1.address)).to.equal(0)
            expect(await lidoVault.variableBearerTokenTotalSupply()).to.equal(
              await lidoVault.variableBearerToken(addr2.address)
            )
            expect(await ethers.provider.getBalance(addr1)).to.equal(
              balanceBefore + variableDeposit1 - calculateGasFees(receipt)
            )

            // withdraw #2
            const balanceBefore2 = await ethers.provider.getBalance(addr2)
            const receipt2 = await (await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).wait()

            expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
            expect(await lidoVault.variableBearerTokenTotalSupply()).to.equal(0)
            expect(await ethers.provider.getBalance(addr2)).to.equal(
              balanceBefore2 + variableDeposit2 - calculateGasFees(receipt2)
            )
          })
        })
      })
    })
  })

  describe('Vault Started', function () {
    const fixedDeposit = DEFAULTS.fixedSideCapacity
    const variableDeposit = parseEther('30')

    async function startVaultFixture() {
      const state = await deployVault({})

      const { lidoVault, addr1, addr2 } = state

      await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
      await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })

      return state
    }

    it('fixedClaimToken should be minted', async function () {
      const { lidoVault } = await loadFixture(startVaultFixture)

      expect(await lidoVault.fixedClaimTokenTotalSupply()).to.greaterThan(0)
    })

    it('fixedETHDepositToken should be set to the fixed capacity', async function () {
      const { lidoVault } = await loadFixture(startVaultFixture)

      expect(await lidoVault.fixedETHDepositTokenTotalSupply()).to.equal(fixedDeposit)
    })

    it('isStarted should be set to true', async function () {
      const { lidoVault } = await loadFixture(startVaultFixture)

      expect(await lidoVault.isStarted()).to.equal(true)
    })

    it('isEnded should be set to false', async function () {
      const { lidoVault } = await loadFixture(startVaultFixture)

      expect(await lidoVault.isEnded()).to.equal(false)
    })

    it('endTime should be set correctly', async function () {
      const { lidoVault } = await loadFixture(startVaultFixture)
      const expectedEndTime = (await time.latest()) + DEFAULTS.durationSeconds

      expect(await lidoVault.endTime()).to.equal(expectedEndTime)
    })

    it('Deposits should fail', async function () {
      const { lidoVault, addr1, addr2 } = await loadFixture(startVaultFixture)

      await expect(
        lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: parseEther('20') })
      ).to.be.revertedWith('DAS')
      await expect(
        lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: parseEther('200') })
      ).to.be.revertedWith('DAS')
    })

    it('Should emit the correct vault started event on variable deposit trigger (events)', async function () {
      const { lidoVault, addr1, addr2 } = await loadFixture(deployLidoVaultFixture)

      await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
      const depositTrx = await lidoVault
        .connect(addr2)
        .deposit(SIDE.VARIABLE, { value: variableDeposit })

      await expect(depositTrx).to.emit(lidoVault, 'VaultStarted').withArgs(anyUint, addr2.address)
    })

    it('Should emit the correct vault started event on fixed deposit trigger (events)', async function () {
      const { lidoVault, addr1, addr2 } = await loadFixture(deployLidoVaultFixture)

      await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })
      const depositTrx = await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })

      await expect(depositTrx).to.emit(lidoVault, 'VaultStarted').withArgs(anyUint, addr1.address)
    })

    describe('Claim', function () {
      it('Should fail if address has no claim tokens', async function () {
        const { lidoVault, addr3, fixedClaimToken } = await loadFixture(startVaultFixture)

        expect(await lidoVault.fixedClaimToken(addr3.address)).to.equal(0)

        await expect(lidoVault.connect(addr3).claimFixedPremium()).to.be.revertedWith('NCT')
        expect(await lidoVault.userToFixedUpfrontPremium(addr3.address)).to.equal(0)
      })

      it('Should emit the correct fixed premium claimed event (events)', async function () {
        const { lidoVault, addr1 } = await loadFixture(startVaultFixture)

        const claimTrx = await lidoVault.connect(addr1).claimFixedPremium()

        await expect(claimTrx)
          .to.emit(lidoVault, 'FixedPremiumClaimed')
          .withArgs(variableDeposit, anyUint, addr1.address)
      })

      describe('Single Fixed Participant', function () {
        it('Should claim fixed premium correctly', async function () {
          const { lidoVault, addr1 } =
            await loadFixture(startVaultFixture)

          const balanceBefore = await ethers.provider.getBalance(addr1)
          const claimBalance = await lidoVault.fixedClaimToken(addr1.address)
          const receipt = await (await lidoVault.connect(addr1).claimFixedPremium()).wait()

          expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDeposit)
          expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(claimBalance)
          expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)
          expect(await ethers.provider.getBalance(addr1)).to.equal(
            balanceBefore + variableDeposit - calculateGasFees(receipt) // gets all of the variable deposit since they are the only fixed depositor
          )
          expect(await lidoVault.userToFixedUpfrontPremium(addr1.address)).to.equal(variableDeposit)
        })
      })

      describe('Multiple Fixed Participants', function () {
        it('Should claim fixed premium correctly (multiple-participants)', async function () {
          const {
            lidoVault,
            addr1,
            addr2,
            addr3
          } = await loadFixture(deployLidoVaultFixture)

          await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit })

          const fixedDeposit1 = parseEther('600')
          const fixedDeposit2 = parseEther('400')
          await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
          await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })

          expect(await lidoVault.isStarted()).to.equal(true)

          const balanceBefore1 = await ethers.provider.getBalance(addr1)
          const claimBalance1 = await lidoVault.fixedClaimToken(addr1.address)
          const receipt1 = await (await lidoVault.connect(addr1).claimFixedPremium()).wait()
          const upfrontPremium1 = (variableDeposit * BigInt(6)) / BigInt(10)
          expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDeposit1)
          expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(claimBalance1)
          expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)
          expect(await ethers.provider.getBalance(addr1)).to.equal(
            balanceBefore1 + upfrontPremium1 - calculateGasFees(receipt1)
          )
          expect(await lidoVault.userToFixedUpfrontPremium(addr1.address)).to.equal(upfrontPremium1)

          const balanceBefore2 = await ethers.provider.getBalance(addr2)
          const claimBalance2 = await lidoVault.fixedClaimToken(addr2.address)
          const receipt2 = await (await lidoVault.connect(addr2).claimFixedPremium()).wait()
          const upfrontPremium2 = (variableDeposit * BigInt(4)) / BigInt(10)
          expect(await lidoVault.fixedETHDepositToken(addr2.address)).to.equal(fixedDeposit2)
          expect(await lidoVault.fixedBearerToken(addr2.address)).to.equal(claimBalance2)
          expect(await lidoVault.fixedClaimToken(addr2.address)).to.equal(0)
          expect(await ethers.provider.getBalance(addr2)).to.equal(
            balanceBefore2 + upfrontPremium2 - calculateGasFees(receipt2)
          )
          expect(await lidoVault.userToFixedUpfrontPremium(addr2.address)).to.equal(upfrontPremium2)
        })

        it('Should claim fixed premium correctly if someone has already withdrawn (multiple-participants)', async function () {
          const {
            lidoVault,
            addr1,
            addr2,
            addr3
          } = await loadFixture(deployLidoVaultFixture)

          await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit })

          const fixedDeposit1 = parseEther('600')
          const fixedDeposit2 = parseEther('400')
          await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
          await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })

          expect(await lidoVault.isStarted()).to.equal(true)

          // withdraw
          await lidoVault.connect(addr2).claimFixedPremium()
          expect(await lidoVault.fixedBearerToken(addr2)).to.be.greaterThan(0)
          await lidoVault.connect(addr2).withdraw(SIDE.FIXED)
          expect(await lidoVault.fixedBearerToken(addr2)).to.be.eq(0)

          // try to claim
          const balanceBefore1 = await ethers.provider.getBalance(addr1)
          const claimBalance1 = await lidoVault.fixedClaimToken(addr1.address)
          const receipt1 = await (await lidoVault.connect(addr1).claimFixedPremium()).wait()
          const upfrontPremium1 = (variableDeposit * BigInt(6)) / BigInt(10)
          expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDeposit1)
          expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(claimBalance1)
          expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)
          expect(await ethers.provider.getBalance(addr1)).to.equal(
            balanceBefore1 + upfrontPremium1 - calculateGasFees(receipt1)
          )
          expect(await lidoVault.userToFixedUpfrontPremium(addr1.address)).to.equal(upfrontPremium1)
        })

        it('Should claim fixed premium correctly if deposited with different staking bonus (multiple-participants)', async function () {
          const {
            lidoVault,
            addr1,
            addr2,
            addr3,
            lidoVaultAddress
          } = await loadFixture(deployLidoVaultFixture)

          const { lidoMock, lidoWithdrawalQueueMock } = await setupMockLidoContracts(lidoVault)
          await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit })

          const fixedDeposit1 = parseEther('400')
          const fixedDeposit2 = parseEther('600')
          const stakingEarnings = parseEther('200')
          await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })

          await lidoMock.addStakingEarningsForTargetETH(
            fixedDeposit1 + stakingEarnings,
            lidoVaultAddress
          )
          await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })

          expect(await lidoVault.isStarted()).to.equal(true)
          console.log(await lidoVault.fixedClaimToken(addr1.address), await lidoVault.fixedClaimToken(addr2.address),
          await lidoVault.fixedSidestETHOnStartCapacity(), await lidoVault.stakingBalance(), 
          await lidoVault.fixedClaimTokenTotalSupply())
          const balanceBefore1 = await ethers.provider.getBalance(addr1)
          const claimBalance1 = await lidoVault.fixedClaimToken(addr1.address)
          const receipt1 = await (await lidoVault.connect(addr1).claimFixedPremium()).wait()
          const upfrontPremium1 = (variableDeposit * BigInt(5)) / BigInt(10)
          console.log(balanceBefore1, await ethers.provider.getBalance(addr1))
          expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDeposit1)
          expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(claimBalance1)
          expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)
          expect(await ethers.provider.getBalance(addr1)).to.equal(
            balanceBefore1 + upfrontPremium1 - calculateGasFees(receipt1)
          )
          expect(await lidoVault.userToFixedUpfrontPremium(addr1.address)).to.equal(upfrontPremium1)

          const balanceBefore2 = await ethers.provider.getBalance(addr2)
          const claimBalance2 = await lidoVault.fixedClaimToken(addr2.address)
          const receipt2 = await (await lidoVault.connect(addr2).claimFixedPremium()).wait()
          const upfrontPremium2 = (variableDeposit * BigInt(5)) / BigInt(10)
          expect(await lidoVault.fixedETHDepositToken(addr2.address)).to.equal(fixedDeposit2)
          expect(await lidoVault.fixedBearerToken(addr2.address)).to.equal(claimBalance2)
          expect(await lidoVault.fixedClaimToken(addr2.address)).to.equal(0)
          expect(await ethers.provider.getBalance(addr2)).to.equal(
            balanceBefore2 + upfrontPremium2 - calculateGasFees(receipt2)
          )
          expect(await lidoVault.userToFixedUpfrontPremium(addr2.address)).to.equal(upfrontPremium2)
        })
      })
    })

    describe('Withdraw', function () {
      async function startVaultAndClaimFixture() {
        const state = await loadFixture(deployLidoVaultFixture)

        const { lidoVault, addr1, addr2 } = state

        await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
        await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })

        await lidoVault.connect(addr1).claimFixedPremium()

        return state
      }

      describe('Fixed Side', function () {
        it('Should fail to withdraw if address has no bearer tokens (vault-started)', async function () {
          const { lidoVault, addr3 } = await loadFixture(
            startVaultFixture
          )

          expect(await lidoVault.fixedBearerToken(addr3.address)).to.equal(0)
          expect(await lidoVault.fixedETHDepositToken(addr3.address)).to.equal(0)
          await expect(lidoVault.connect(addr3).withdraw(SIDE.FIXED)).to.be.revertedWith('NBT')
        })

        it('Should fail to withdraw if address has not claimed their bearer tokens (vault-started)', async function () {
          const { lidoVault, addr1 } = await loadFixture(startVaultFixture)

          expect(await lidoVault.fixedClaimToken(addr1.address)).to.be.greaterThan(0)
          await expect(lidoVault.connect(addr1).withdraw(SIDE.FIXED)).to.be.revertedWith('NBT')
        })

        it('Should fail to finalize withdraw if user has not requested a withdrawal (vault-started)', async function () {
          const { lidoVault, addr1 } = await loadFixture(startVaultAndClaimFixture)

          await expect(
            lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
          ).to.be.revertedWith('WNR')
        })

        it('Should fail to finalize vault not started withdraw if user has not requested a withdrawal (vault-started)', async function () {
          const { lidoVault, addr1 } = await loadFixture(startVaultAndClaimFixture)

          await expect(
            lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
          ).to.be.revertedWith('WNR')
        })

        it('Should fail to finalize vault ended withdraw (vault-started)', async function () {
          const { lidoVault, addr1 } = await loadFixture(startVaultAndClaimFixture)

          await expect(
            lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
          ).to.be.revertedWith('WNR')
        })

        it('Should fail to finalize vault not started withdraw if user only requested (vault-started)', async function () {
          const { lidoVault, addr1 } = await loadFixture(startVaultAndClaimFixture)

          await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

          await expect(
            lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
          ).to.be.revertedWith('WNR')
        })

        it('Should emit the correct lido withdrawal requested event (vault-started, events)', async function () {
          const { lidoVault, addr1 } = await loadFixture(startVaultAndClaimFixture)

          const withdrawTrx = await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

          await expect(withdrawTrx)
            .to.emit(lidoVault, 'LidoWithdrawalRequested')
            .withArgs(addr1.address, anyValue, SIDE.FIXED, true, false)
        })

        it('Should emit the correct fixed withdrawal events on withdrawal finalization (vault-started, events)', async function () {
          const { lidoVault, addr1, lidoWithdrawalQueueContract } = await loadFixture(
            startVaultAndClaimFixture
          )

          const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
          const requestIds = [lidoLastRequestId + BIG_INT_ONE]
          await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
          await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

          const finalizedTrx = await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()

          await expect(finalizedTrx)
            .to.emit(lidoVault, 'LidoWithdrawalFinalized')
            .withArgs(addr1.address, anyValue, SIDE.FIXED, true, false)
          await expect(finalizedTrx)
            .to.emit(lidoVault, 'FixedFundsWithdrawn')
            .withArgs(anyUint, addr1.address, true, false)
        })

        describe('Single Fixed Participant', function () {
          it('Should withdraw correctly', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1
            } = await loadFixture(startVaultAndClaimFixture)

            const { currentTimestamp } = await getTimeState(lidoVault)
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultOngoingFixedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address,
              currentTimestamp
            )

            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[0]
            )

            await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
            expect(balanceBefore).to.be.lessThan(await ethers.provider.getBalance(addr1))
            await expectVaultOngoingFixedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
          })

          const deployLidoVaultHighFixedPremiumBpsFixture = () =>
            deployVault({ variableSideCapacity: parseEther('900'), earlyExitFeeBps: 2000 }) // 90% & 20%

          it('Should withdraw correctly even if early exit fees exceeds amount withdrawn from Lido', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1,
              addr2
            } = await loadFixture(deployLidoVaultHighFixedPremiumBpsFixture)

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
            await lidoVault
              .connect(addr2)
              .deposit(SIDE.VARIABLE, { value: (fixedDeposit * BigInt(90)) / BigInt(100) })

            await lidoVault.connect(addr1).claimFixedPremium()

            expect(await lidoVault.isStarted()).to.be.true

            const { currentTimestamp } = await getTimeState(lidoVault)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultOngoingFixedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address,
              currentTimestamp
            )

            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[0]
            )

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)

            // should have received nothing
            expect(balanceBefore).to.be.equal((await ethers.provider.getBalance(addr1)) + gasFees)
            await expectVaultOngoingFixedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
          })

          it('Should withdraw correctly if they requested a withdraw prior to vault start (vault-started)', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1,
              addr2,
              addr3
            } = await loadFixture(deployLidoVaultFixture)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

            await lidoVault.connect(addr3).deposit(SIDE.FIXED, { value: fixedDeposit })
            await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })
            await lidoVault.connect(addr3).claimFixedPremium()

            expect(await lidoVault.isStarted()).to.be.equal(true)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[0]
            )

            const fixedBearerTokenTotalSupplyBefore = await lidoVault.fixedBearerTokenTotalSupply()
            const fixedBearerTokenBalanceBefore = await lidoVault.fixedBearerToken(addr3.address)
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            await expectVaultNotStartedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
            expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit - gasFees, 1)
            // current vault state should not have changed
            expect(await lidoVault.fixedETHDepositTokenTotalSupply()).to.equal(fixedDeposit)
            expect(await lidoVault.fixedBearerTokenTotalSupply()).to.be.equal(
              fixedBearerTokenTotalSupplyBefore
            )
            expect(await lidoVault.fixedBearerToken(addr3.address)).to.be.equal(
              fixedBearerTokenBalanceBefore
            )
            expect(await lidoVault.fixedETHDepositToken(addr3.address)).to.be.equal(fixedDeposit)
          })

          it('Should fail to withdraw again if user already had another vault withdrawal in progress before the vault started (vault-started)', async function () {
            const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2, addr3 } =
              await loadFixture(deployLidoVaultFixture)

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )

            await lidoVault.connect(addr3).deposit(SIDE.FIXED, { value: fixedDeposit })
            await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })

            expect(await lidoVault.isStarted()).to.equal(true)

            await expect(lidoVault.connect(addr1).withdraw(SIDE.FIXED)).to.be.revertedWith('WAR')
          })

          it('Should fail to withdraw again if user already has another vault ongoing withdrawal in progress (vault-started)', async function () {
            const { lidoVault, lidoWithdrawalQueueContract, addr1 } = await loadFixture(
              startVaultAndClaimFixture
            )

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            const { currentTimestamp } = await getTimeState(lidoVault)
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultOngoingFixedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address,
              currentTimestamp
            )

            await expect(lidoVault.connect(addr1).withdraw(SIDE.FIXED)).to.be.revertedWith('WAR')
          })

          it('Should fail to withdraw again if user already finalized their withdrawal (vault-started)', async function () {
            const { lidoVault, lidoWithdrawalQueueContract, addr1 } = await loadFixture(
              startVaultAndClaimFixture
            )

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()

            // has no more bearer tokens
            await expect(lidoVault.connect(addr1).withdraw(SIDE.FIXED)).to.be.revertedWith('NBT')
          })

          it('Should fail to withdraw if lido withdrawal request has not been finalized (vault-started)', async function () {
            const { lidoVault, lidoWithdrawalQueueContract, addr1 } = await loadFixture(
              startVaultAndClaimFixture
            )

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            const { currentTimestamp } = await getTimeState(lidoVault)
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultOngoingFixedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address,
              currentTimestamp
            )

            try {
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
              assert.fail('Lido error not thrown')
            } catch (error: any) {
              const lidoError = decodeLidoErrorData(error)
              expect(lidoError?.name).to.equal('RequestNotFoundOrNotFinalized')
            }
          })

          it('Should fail to finalize withdraw if user has already finalized their withdrawal (vault-started)', async function () {
            const { lidoVault, lidoWithdrawalQueueContract, addr1 } = await loadFixture(
              startVaultAndClaimFixture
            )

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()

            await expect(
              lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
            ).to.be.revertedWith('WNR')
          })

          it('Should get less than their initial deposit in the case of staking balance decreasing (vault-started, staking-loss)', async function () {
            const { lidoVault, lidoMock, addr1, addr2, fixedETHDepositToken, fixedBearerToken } =
              await loadFixture(deployLidoVaultWithMockedLidoFixture)

            const lidoLastRequestId = BIG_INT_ZERO

            // start vault
            const variableDeposit = parseEther('20')
            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
            await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })
            await lidoVault.connect(addr1).claimFixedPremium()

            expect(await lidoVault.isStarted()).to.be.equal(true)
            const shares = await lidoVault.fixedBearerToken(addr1.address)
            expect(shares).to.equal(fixedDeposit)

            const stakingStETHBalanceBefore = await lidoMock.getPooledEthByShares(shares)

            // mock staking loss - 100 ETH
            const stakingEarnings = parseEther('100')
            await lidoMock.subtractStakingEarnings(stakingEarnings)

            const stakingStETHBalanceAfter = await lidoMock.getPooledEthByShares(shares)
            // eth = _sharesAmount.mulDiv(totalPooledEther, totalShares) = (1000 * (2000 - 100)) / 2000 = 950 - 50 ETH loss from original deposit
            const stakingLoss = parseEther('50')
            expect(stakingStETHBalanceBefore - stakingStETHBalanceAfter).to.equal(stakingLoss)

            const { currentTimestamp } = await getTimeState(lidoVault)
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            await expectVaultOngoingFixedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address,
              currentTimestamp
            )

            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedBearerTokenTotalSupply()).to.equal(0)

            const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
              lidoVault,
              upfrontPremium: variableDeposit,
              timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
            })

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(balanceAfter - balanceBefore).to.be.equal(
              fixedDeposit - stakingLoss - payBackAmount - scaledEarlyExitFee - gasFees
            )
            await expectVaultOngoingFixedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)

            // no protocol fee should have been applied since the vault has not started
            const protocolFee = await lidoVault.appliedProtocolFee()
            expect(protocolFee).to.be.equal(0)
          })

          describe('Fee Calculations', function () {
            it('Should have correct early exit fee applied if immediately withdraw on vault start (vault-started)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                addr1,
                protocolFeeReceiver,
                variableBearerToken,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(startVaultAndClaimFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

              const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                lidoVault,
                upfrontPremium: variableDeposit,
                timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
              })
              expect(await lidoVault.fixedBearerToken(addr1.address)).to.be.equal(0)

              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              const balanceBefore = await ethers.provider.getBalance(addr1)
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              const receipt = await (
                await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)
              const amountWithdrawn = balanceAfter - balanceBefore

              // almost all of the fixed upfront premium should be paid back
              // no earnings should be withdrawn by the fixed side now that the vault has started
              const protocolFee = BIG_INT_ZERO
              expect(await lidoVault.variableBearerToken(protocolFeeReceiver)).to.equal(0)
              // almost 10% max scaled fee
              const approxScaledEarlyExitFee =
                (variableDeposit * BigInt(DEFAULTS.earlyExitFeeBps)) / BIG_INT_10K
              // sanity check that the scaled fee is almost close to 10%
              compareWithBpsTolerance(
                amountWithdrawn,
                fixedDeposit - (payBackAmount + protocolFee + approxScaledEarlyExitFee) - gasFees,
                1
              )
              // now calculate exact fee amount
              expect(amountWithdrawn).to.equalWithTolerance(
                fixedDeposit - (payBackAmount + protocolFee + scaledEarlyExitFee) - gasFees,
                1
              )
            })

            it('Should have correct early exit fee applied if withdraw after half way through vault term but no earnings were made (vault-started)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                lidoContract,
                lidoVaultAddress,
                addr1,
                variableBearerToken,
                protocolFeeReceiver,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(startVaultAndClaimFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // fast forward to half way through vault duration
              const { duration, endTime } = await getTimeState(lidoVault)
              const halfDuration = duration / BigInt(2)
              await time.increaseTo(endTime - halfDuration)

              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

              const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                lidoVault,
                upfrontPremium: variableDeposit,
                timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
              })
              expect(await lidoVault.fixedBearerToken(addr1.address)).to.be.equal(0)

              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              const balanceBefore = await ethers.provider.getBalance(addr1)
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt = await (
                await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
              const amountWithdrawn = balanceAfter - balanceBefore
              const amountNotWithdrawn = vaultBalanceAfter - vaultBalanceBefore

              const lidoStETHAmountAfter = await lidoContract.balanceOf(lidoVaultAddress)
              // no earnings so stETH balance should now be zero since the fixed depositor withdrew their deposit
              expect(lidoStETHAmountAfter).to.be.equalWithTolerance(BIG_INT_ZERO, 1)

              // almost all of the fixed upfront premium should be paid back
              expect(await lidoVault.feeEarnings()).to.equal(
                payBackAmount + scaledEarlyExitFee,
                'incorrect earnings'
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              expect(amountNotWithdrawn).to.equal(payBackAmount + scaledEarlyExitFee)

              // no earnings should be withdrawn by the fixed side now that the vault has started
              const protocolFee = BIG_INT_ZERO
              expect(await lidoVault.variableBearerToken(protocolFeeReceiver)).to.equal(0)
              expect(amountWithdrawn).to.equalWithTolerance(
                fixedDeposit - (payBackAmount + protocolFee + scaledEarlyExitFee) - gasFees,
                1
              )
            })

            it('Should have correct early exit fee applied if withdraw after half way through vault term but staking earnings have accumulated (vault-started)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                lidoContract,
                addr1,
                lidoVaultAddress,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(startVaultAndClaimFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // fast forward to half way through vault duration
              const { duration, endTime } = await getTimeState(lidoVault)
              const halfDuration = duration / BigInt(2)
              await time.increaseTo(endTime - halfDuration)

              await submitOracleReport()

              const lidoStETHAmountBefore = await lidoContract.balanceOf(lidoVaultAddress)

              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

              const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                lidoVault,
                upfrontPremium: variableDeposit,
                timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
              })
              expect(await lidoVault.fixedBearerToken(addr1.address)).to.be.equal(0)

              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              const balanceBefore = await ethers.provider.getBalance(addr1)
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt = await (
                await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
              const amountWithdrawn = balanceAfter - balanceBefore
              const amountNotWithdrawn = vaultBalanceAfter - vaultBalanceBefore

              const lidoStETHAmountAfter = await lidoContract.balanceOf(lidoVaultAddress)
              // should still have the earnings as the vaults balance
              expect(lidoStETHAmountAfter).to.be.greaterThan(0)
              // difference should be the fixed deposit
              expect(lidoStETHAmountBefore - lidoStETHAmountAfter).to.be.equalWithTolerance(fixedDeposit, 10)

              // no earnings should be withdrawn by the fixed side now that the vault has started
              const protocolFee = await lidoVault.appliedProtocolFee()
              expect(protocolFee).to.be.equal(0)

              expect(await lidoVault.feeEarnings()).to.equal(
                payBackAmount + scaledEarlyExitFee,
                'incorrect vault earnings'
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)

              const fees = payBackAmount + protocolFee + scaledEarlyExitFee
              expect(amountNotWithdrawn).to.equal(fees)
              expect(amountWithdrawn).to.equalWithTolerance(fixedDeposit - (fees + gasFees), 10)
            })

            describe('Large Fixed Deposit', () => {
              const largeFixedDeposit = parseEther('1500')
              const variableDeposit = parseEther('45')

              const deployLidoVaultLargeFixedCapacityFixture = () =>
                deployVault({ fixedSideCapacity: largeFixedDeposit })

              async function startVaultWithLargeFixedCapacity() {
                const state = await loadFixture(deployLidoVaultLargeFixedCapacityFixture)

                const { lidoVault, addr1, addr2 } = state

                await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: largeFixedDeposit })
                await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })

                await lidoVault.connect(addr1).claimFixedPremium()

                return state
              }

              it('Should have correct early exit fee applied if immediately withdraw on vault start (vault-started)', async function () {
                const {
                  lidoVault,
                  lidoWithdrawalQueueContract,
                  addr1,
                  protocolFeeReceiver,
                  variableBearerToken,
                  fixedBearerToken,
                  fixedETHDepositToken,
                } = await loadFixture(startVaultWithLargeFixedCapacity)

//                 const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
// 
//                 await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
// 
//                 const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
//                   lidoVault,
//                   upfrontPremium: variableDeposit,
//                   timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
//                 })
//                 expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.be.equal(0)
//                 expect(await lidoVault.fixedBearerToken(addr1.address)).to.be.equal(0)
// 
//                 const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]

//                 const balanceBefore = await ethers.provider.getBalance(addr1)
//                 await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
//                 const receipt = await (
//                   await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
//                 ).wait()
//                 const gasFees = calculateGasFees(receipt)
//                 const balanceAfter = await ethers.provider.getBalance(addr1)
//                 const amountWithdrawn = balanceAfter - balanceBefore
// 
//                 // almost all of the fixed upfront premium should be paid back
//                 // no earnings should be withdrawn by the fixed side now that the vault has started
//                 const protocolFee = BIG_INT_ZERO
//                 expect(await lidoVault.variableBearerToken(protocolFeeReceiver)).to.equal(0)
//                 // almost 10% max scaled fee
//                 const approxScaledEarlyExitFee =
//                   (variableDeposit * BigInt(DEFAULTS.earlyExitFeeBps)) / BIG_INT_10K
//                 // sanity check that the scaled fee is almost close to 10%
//                 compareWithBpsTolerance(
//                   amountWithdrawn,
//                   largeFixedDeposit -
//                     (payBackAmount + protocolFee + approxScaledEarlyExitFee) -
//                     gasFees,
//                   1
//                 )
// 
//                 // now calculate exact fee amount
//                 expect(amountWithdrawn).to.equalWithTolerance(
//                   largeFixedDeposit - (payBackAmount + protocolFee + scaledEarlyExitFee) - gasFees,
//                   1
//                 )
              })

              it('Should have correct early exit fee applied if withdraw after half way through vault term (vault-started)', async function () {
                const {
                  lidoVault,
                  lidoWithdrawalQueueContract,
                  addr1,
                  lidoVaultAddress,
                  fixedETHDepositToken,
                  fixedBearerToken,
                } = await loadFixture(startVaultWithLargeFixedCapacity)

                const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

                // fast forward to half way through vault duration
                const { duration, endTime } = await getTimeState(lidoVault)
                const halfDuration = duration / BigInt(2)
                await time.increaseTo(endTime - halfDuration)

                await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

                const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                  lidoVault,
                  upfrontPremium: variableDeposit,
                  timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
                })
                expect(await lidoVault.fixedBearerToken(addr1.address)).to.be.equal(0)

                const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
                await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

                const balanceBefore = await ethers.provider.getBalance(addr1)
                const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
                const receipt = await (
                  await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
                ).wait()
                const gasFees = calculateGasFees(receipt)
                const balanceAfter = await ethers.provider.getBalance(addr1)
                const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
                const amountWithdrawn = balanceAfter - balanceBefore
                const amountNotWithdrawn = vaultBalanceAfter - vaultBalanceBefore

                // no earnings should be withdrawn by the fixed side now that the vault has started
                const protocolFee = await lidoVault.appliedProtocolFee()
                expect(protocolFee).to.be.equal(0)

                expect(await lidoVault.feeEarnings()).to.equal(
                  payBackAmount + scaledEarlyExitFee,
                  'incorrect vault earnings'
                )
                expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)

                const fees = payBackAmount + protocolFee + scaledEarlyExitFee
                expect(amountNotWithdrawn).to.equal(fees)
                expect(amountWithdrawn).to.equalWithTolerance(
                  largeFixedDeposit - (fees + gasFees),
                  1
                )
              })
            })
          })
        })

        describe('Multiple Fixed Participants', function () {
          const fixedDeposit1 = parseEther('600')
          const fixedDeposit2 = parseEther('400')
          const upfrontPremium1 = (variableDeposit * BigInt(6)) / BigInt(10)
          const upfrontPremium2 = (variableDeposit * BigInt(4)) / BigInt(10)

          async function multipleParticipantsStartVaultAndClaimFixture() {
            const state = await loadFixture(deployLidoVaultFixture)

            const { lidoVault, addr1, addr2, addr3 } = state

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
            await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })
            await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit })

            await lidoVault.connect(addr1).claimFixedPremium()
            await lidoVault.connect(addr2).claimFixedPremium()

            return state
          }

          it('Should withdraw correctly if they requested a withdraw prior to vault start (vault-started, multiple-participants)', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1,
              addr2,
              addr3,
              addr4,
              fixedBearerToken,
              fixedClaimToken,
              fixedETHDepositToken,
            } = await loadFixture(deployLidoVaultFixture)

            const fixedDeposit1 = parseEther('600')

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]

            // withdraw
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            await expectVaultNotStartedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address
            )
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

            const fixedDeposit2 = parseEther('400')
            const fixedDeposit3 = parseEther('600')

            await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })
            await lidoVault.connect(addr3).deposit(SIDE.FIXED, { value: fixedDeposit3 })
            await lidoVault.connect(addr4).deposit(SIDE.VARIABLE, { value: variableDeposit })
            await lidoVault.connect(addr2).claimFixedPremium()
            await lidoVault.connect(addr3).claimFixedPremium()

            expect(await lidoVault.isStarted()).to.be.equal(true)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[0]
            )

            const fixedBearerTokenTotalSupplyBefore = await lidoVault.fixedBearerTokenTotalSupply()
            const fixedBearerTokenBalanceBefore2 = await lidoVault.fixedBearerToken(addr2.address)
            const fixedBearerTokenBalanceBefore3 = await lidoVault.fixedBearerToken(addr3.address)
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            await expectVaultNotStartedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)
            expect(await lidoVault.fixedClaimToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
            expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit1 - gasFees, 1) // should have no extra fees since they withdrew before vault start
            // current vault state should not have changed
            expect(await lidoVault.fixedETHDepositTokenTotalSupply()).to.equal(fixedDeposit2 + fixedDeposit3)
            expect(await lidoVault.fixedBearerTokenTotalSupply()).to.be.equal(
              fixedBearerTokenTotalSupplyBefore
            )
            expect(await lidoVault.fixedBearerToken(addr2.address)).to.be.equal(
              fixedBearerTokenBalanceBefore2
            )
            expect(await lidoVault.fixedBearerToken(addr3.address)).to.be.equal(
              fixedBearerTokenBalanceBefore3
            )
            expect(await lidoVault.fixedETHDepositToken(addr2.address)).to.be.equal(fixedDeposit2)
            expect(await lidoVault.fixedETHDepositToken(addr3.address)).to.be.equal(fixedDeposit3)
          })

          const deployLidoVaultLongerDurationFixture = () =>
            deployVault({ durationSeconds: DEFAULTS.durationSeconds * 10 })

          it('Should be able to withdraw with many fixed depositors (vault-started, multiple-participants, gas)', async function () {
            const { lidoVault, lidoWithdrawalQueueContract, fixedETHDepositToken, addr1, addrs } =
              await loadFixture(deployLidoVaultLongerDurationFixture)

            const fixedDepositorCount = 20

            // deposit
            const fixedDeposit = DEFAULTS.fixedSideCapacity / BigInt(fixedDepositorCount)
            for (let i = 0; i < fixedDepositorCount; i++) {
              await lidoVault.connect(addrs[i]).deposit(SIDE.FIXED, { value: fixedDeposit })
              expect(await lidoVault.fixedETHDepositToken(addrs[i])).to.equal(fixedDeposit)
            }

            await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: variableDeposit })

            expect(await lidoVault.isStarted()).to.be.true

            // claim
            for (let i = 0; i < fixedDepositorCount; i++) {
              await lidoVault.connect(addrs[i]).claimFixedPremium()
            }

            // withdraw
            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
            for (let i = 0; i < fixedDepositorCount; i++) {
              const { currentTimestamp } = await getTimeState(lidoVault)
              await lidoVault.connect(addrs[i]).withdraw(SIDE.FIXED)
              const requestIds = [lidoLastRequestId + BIG_INT_ONE + BigInt(i)]
              await expectVaultOngoingFixedWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addrs[i].address,
                currentTimestamp
              )
              await finalizeLidoWithdrawalRequests(requestIds, fixedDeposit + BIG_INT_ONE)
            }

            // end vault
            expect(await lidoVault.isEnded()).to.be.false
            const { endTime } = await getTimeState(lidoVault)
            await time.increaseTo(endTime + BIG_INT_ONE)
            expect(await lidoVault.isEnded()).to.be.true

            // vault ended withdraw - should claim ongoing fixed withdrawals
            console.log('vault end withdraw')
            
            await lidoVault.connect(addr1).withdraw(SIDE.VARIABLE)
            console.log('vault end withdraw completed')
            for (let i = 0; i < fixedDepositorCount; i++) {
              await expectVaultOngoingFixedWithdrawalArraysAreEqual(
                lidoVault,
                [],
                addrs[i].address,
                BIG_INT_ZERO
              )
            }
          })

          it('Should have correct early exit fee applied if withdraw after half way through vault term but no earnings were made (vault-started, multiple-participants)', async function () {
            const {
              lidoVault,
              lidoVaultAddress,
              lidoWithdrawalQueueContract,
              lidoContract,
              addr1,
              addr2,
              fixedBearerToken,
              variableBearerToken,
              protocolFeeReceiver,
            } = await loadFixture(multipleParticipantsStartVaultAndClaimFixture)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            // fast forward to half way through vault duration
            const { duration, endTime, currentTimestamp } = await getTimeState(lidoVault)
            const halfDuration = duration / BigInt(2)
            await time.increaseTo(endTime - halfDuration)

            // withdraw #1
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
              lidoVault,
              upfrontPremium: upfrontPremium1,
              timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
            })
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultOngoingFixedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address,
              currentTimestamp
            )
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)
            const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
            const amountWithdrawn = balanceAfter - balanceBefore
            const amountNotWithdrawn = vaultBalanceAfter - vaultBalanceBefore

            const lidoStETHAmountAfter = await lidoContract.balanceOf(lidoVaultAddress)
            expect(lidoStETHAmountAfter).to.equalWithTolerance(fixedDeposit2, 1)

            // almost all of the fixed upfront premium should be paid back
            const feeEarnings = await lidoVault.feeEarnings()
            expect(feeEarnings).to.equal(payBackAmount + scaledEarlyExitFee, 'incorrect earnings')
            expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
            expect(amountNotWithdrawn).to.equal(payBackAmount + scaledEarlyExitFee)

            // no earnings should be withdrawn by the fixed side now that the vault has started
            const protocolFee = BIG_INT_ZERO
            expect(await lidoVault.variableBearerToken(protocolFeeReceiver)).to.equal(0)
            expect(amountWithdrawn).to.equalWithTolerance(
              fixedDeposit1 - (payBackAmount + protocolFee + scaledEarlyExitFee) - gasFees,
              2
            )

            // withdraw #2
            const { currentTimestamp: currentTimestamp2 } = await getTimeState(lidoVault)
            await lidoVault.connect(addr2).withdraw(SIDE.FIXED)

            const { payBackAmount: payBackAmount2, scaledEarlyExitFee: scaledEarlyExitFee2 } =
              await calculateFees({
                lidoVault,
                upfrontPremium: upfrontPremium2,
                timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr2),
              })
            const requestIds2 = [lidoLastRequestId + BigInt(2)]
            await expectVaultOngoingFixedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds2,
              addr2.address,
              currentTimestamp2
            )
            expect(await lidoVault.fixedBearerToken(addr2.address)).to.equal(0)

            await finalizeLidoWithdrawalRequests(requestIds2, DEFAULTS.fixedSideCapacity)

            const balanceBefore2 = await ethers.provider.getBalance(addr2)
            const vaultBalanceBefore2 = await ethers.provider.getBalance(lidoVaultAddress)
            const receipt2 = await (
              await lidoVault.connect(addr2).finalizeVaultOngoingFixedWithdrawals()
            ).wait()
            const gasFees2 = calculateGasFees(receipt2)
            const balanceAfter2 = await ethers.provider.getBalance(addr2)
            const vaultBalanceAfter2 = await ethers.provider.getBalance(lidoVaultAddress)
            const amountWithdrawn2 = balanceAfter2 - balanceBefore2
            const amountNotWithdrawn2 = vaultBalanceAfter2 - vaultBalanceBefore2

            const lidoStETHAmountAfter2 = await lidoContract.balanceOf(lidoVaultAddress)
            expect(lidoStETHAmountAfter2).to.be.equalWithTolerance(BIG_INT_ZERO, 1)

            // almost all of the fixed upfront premium should be paid back
            const feeEarnings2 = await lidoVault.feeEarnings()
            expect(feeEarnings2).to.equal(
              payBackAmount2 + scaledEarlyExitFee2 + feeEarnings,
              'incorrect earnings'
            )
            expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
            expect(amountNotWithdrawn2).to.equal(payBackAmount2 + scaledEarlyExitFee2)

            // no earnings should be withdrawn by the fixed side now that the vault has started
            const protocolFee2 = BIG_INT_ZERO
            expect(await lidoVault.variableBearerToken(protocolFeeReceiver)).to.equal(0)
            expect(amountWithdrawn2).to.equal(
              fixedDeposit2 - (payBackAmount2 + protocolFee2 + scaledEarlyExitFee2) - gasFees2
            )
          })

          it('Should have correct early exit fee applied if withdraw after half way through vault term but staking earnings have accumulated (vault-started, multiple-participants)', async function () {
            const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2, lidoVaultAddress } =
              await loadFixture(multipleParticipantsStartVaultAndClaimFixture)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            // fast forward to half way through vault duration
            const { duration, endTime } = await getTimeState(lidoVault)
            const halfDuration = duration / BigInt(2)
            await time.increaseTo(endTime - halfDuration)

            await submitOracleReport()

            // withdraw #1
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
              lidoVault,
              upfrontPremium: upfrontPremium1,
              timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
            })
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)
            const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
            const amountWithdrawn = balanceAfter - balanceBefore
            const amountNotWithdrawn = vaultBalanceAfter - vaultBalanceBefore

            // no earnings should be withdrawn by the fixed side now that the vault has started
            const protocolFee = await lidoVault.appliedProtocolFee()
            expect(protocolFee).to.be.equal(0)

            const vaultEarnings = await lidoVault.feeEarnings()
            expect(vaultEarnings).to.equal(
              payBackAmount + scaledEarlyExitFee,
              'incorrect vault earnings'
            )
            expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)

            const fees = payBackAmount + protocolFee + scaledEarlyExitFee
            expect(amountNotWithdrawn).to.equal(fees)
            expect(amountWithdrawn).to.equalWithTolerance(fixedDeposit1 - (fees + gasFees), 10)

            // withdraw #2
            const lidoLastRequestId2 = await lidoWithdrawalQueueContract.getLastRequestId()
            const requestIds2 = [lidoLastRequestId2 + BIG_INT_ONE]

            await lidoVault.connect(addr2).withdraw(SIDE.FIXED)

            const { payBackAmount: payBackAmount2, scaledEarlyExitFee: scaledEarlyExitFee2 } =
              await calculateFees({
                lidoVault,
                upfrontPremium: upfrontPremium2,
                timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr2),
              })

            await finalizeLidoWithdrawalRequests(requestIds2, DEFAULTS.fixedSideCapacity)

            const balanceBefore2 = await ethers.provider.getBalance(addr2)
            const vaultBalanceBefore2 = await ethers.provider.getBalance(lidoVaultAddress)
            const receipt2 = await (
              await lidoVault.connect(addr2).finalizeVaultOngoingFixedWithdrawals()
            ).wait()
            const balanceAfter2 = await ethers.provider.getBalance(addr2)
            const vaultBalanceAfter2 = await ethers.provider.getBalance(lidoVaultAddress)
            const gasFees2 = calculateGasFees(receipt2)
            const amountWithdrawn2 = balanceAfter2 - balanceBefore2
            const amountNotWithdrawn2 = vaultBalanceAfter2 - vaultBalanceBefore2

            // no earnings should be withdrawn by the fixed side now that the vault has started
            const protocolFee2 = await lidoVault.appliedProtocolFee()
            expect(protocolFee2).to.be.equal(0)

            const vaultEarnings2 = await lidoVault.feeEarnings()
            expect(vaultEarnings2).to.equal(
              payBackAmount2 + scaledEarlyExitFee2 + vaultEarnings,
              'incorrect vault earnings'
            )
            expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)

            const fees2 = payBackAmount2 + protocolFee2 + scaledEarlyExitFee2
            expect(amountNotWithdrawn2).to.equal(fees2)
            expect(amountWithdrawn2).to.equal(fixedDeposit2 - (fees2 + gasFees2))
          })

          it('Should get less than their initial deposit in the case of staking balance decreasing (vault-started, multiple-participants, staking-loss)', async function () {
            const { lidoVault, lidoMock, addr1, addr2, fixedETHDepositToken, fixedBearerToken } =
              await loadFixture(deployLidoVaultWithMockedLidoFixture)

            const lidoLastRequestId = BIG_INT_ZERO

            // start vault
            const fixedDeposit1 = parseEther('600')
            const fixedDeposit2 = parseEther('400')
            const variableDeposit = parseEther('20')
            const upfrontPremium1 = (variableDeposit * BigInt(6)) / BigInt(10)

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
            await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })
            await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })
            await lidoVault.connect(addr1).claimFixedPremium()
            await lidoVault.connect(addr2).claimFixedPremium()

            expect(await lidoVault.isStarted()).to.be.equal(true)
            const shares1 = await lidoVault.fixedBearerToken(addr1.address)
            const shares2 = await lidoVault.fixedBearerToken(addr2.address)
            expect(shares1).to.equal(fixedDeposit1)
            expect(shares2).to.equal(fixedDeposit2)

            const stakingStETHBalanceBefore1 = await lidoMock.getPooledEthByShares(shares1)
            const stakingStETHBalanceBefore2 = await lidoMock.getPooledEthByShares(shares2)

            // mock staking loss - 100 ETH
            const stakingEarnings = parseEther('100')
            await lidoMock.subtractStakingEarnings(stakingEarnings)

            const stakingStETHBalanceAfter1 = await lidoMock.getPooledEthByShares(shares1)
            const stakingStETHBalanceAfter2 = await lidoMock.getPooledEthByShares(shares2)
            expect(stakingStETHBalanceBefore1 - stakingStETHBalanceAfter1).to.equal(
              parseEther('30')
            )
            expect(stakingStETHBalanceBefore2 - stakingStETHBalanceAfter2).to.equal(
              parseEther('20')
            )

            const { currentTimestamp } = await getTimeState(lidoVault)
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            await expectVaultOngoingFixedWithdrawalArraysAreEqual(
              lidoVault,
              requestIds,
              addr1.address,
              currentTimestamp
            )
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
            expect(await lidoVault.fixedBearerTokenTotalSupply()).to.equal(fixedDeposit2) // only addr2 shares

            const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
              lidoVault,
              upfrontPremium: upfrontPremium1,
              timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
            })

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            // eth = _sharesAmount.mulDiv(totalPooledEther, totalShares) = (600 * (2000 - 100)) / 2000 = 570
            expect(balanceAfter - balanceBefore).to.be.equal(
              parseEther('570') - payBackAmount - scaledEarlyExitFee - gasFees
            )
            await expectVaultOngoingFixedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)

            // no protocol fee should have been applied since the vault has not started
            const protocolFee = await lidoVault.appliedProtocolFee()
            expect(protocolFee).to.be.equal(0)
          })

          describe('Prestart deposit acumulates staking bonus', function () { 
            const fixedDeposit1 = parseEther('600')
            const fixedDeposit2 = parseEther('400')
            const stakingEarnings = parseEther('200')
            const upfrontPremium1 = (variableDeposit * BigInt(5)) / BigInt(10)
            const upfrontPremium2 = (variableDeposit * BigInt(5)) / BigInt(10)
            async function multipleParticipantsStartVaultAndClaimFixtureWithStaking() {
              const state = await loadFixture(deployLidoVaultFixture) 
              const { lidoVault, addr1, addr2, addr3, lidoVaultAddress } = state
              const { lidoMock, lidoWithdrawalQueueMock } = await setupMockLidoContracts(lidoVault)
              await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })
             
              await lidoMock.addStakingEarningsForTargetETH(
                fixedDeposit2 + stakingEarnings,
                lidoVaultAddress
              )

              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
              
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit })
  
              await lidoVault.connect(addr1).claimFixedPremium()
              await lidoVault.connect(addr2).claimFixedPremium()
  
              return state
            }
            it('Should be able to withdraw with many fixed depositors (vault-started, multiple-participants, gas)', async function () {
              const { lidoVault, addr1, addrs, lidoVaultAddress } =
                await loadFixture(deployLidoVaultLongerDurationFixture)
              const { lidoMock, lidoWithdrawalQueueMock } = await setupMockLidoContracts(lidoVault)
              const fixedFirstDepositorCount = 10n
              const fixedDepositorCount = 20n
  
              // deposit
              const fixedDeposit = DEFAULTS.fixedSideCapacity / BigInt(fixedDepositorCount)
              for (let i = 0; i < fixedFirstDepositorCount; i++) {
                await lidoVault.connect(addrs[i]).deposit(SIDE.FIXED, { value: fixedDeposit })
                expect(await lidoVault.fixedETHDepositToken(addrs[i])).to.equal(fixedDeposit)
              }
              const stakingEarnings = parseEther('200')
              await lidoMock.addStakingEarningsForTargetETH(
                fixedDeposit * fixedFirstDepositorCount  + stakingEarnings,
                lidoVaultAddress
              )
  
              for (let j = 10; j < fixedDepositorCount; j++) {
                await lidoVault.connect(addrs[j]).deposit(SIDE.FIXED, { value: fixedDeposit })
                expect(await lidoVault.fixedETHDepositToken(addrs[j])).to.equal(fixedDeposit)
              }
  
              await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: variableDeposit })
  
              expect(await lidoVault.isStarted()).to.be.true
  
              // claim
              for (let i = 0; i < fixedDepositorCount; i++) {
                await lidoVault.connect(addrs[i]).claimFixedPremium()
              }

              // Contract should take into account that first 10 fixed investors got staking bonus from early deposit
              const averageUpfrontPremium = (variableDeposit) / fixedDepositorCount
              const prestakingUpfrontPremium = (variableDeposit) * (fixedDeposit + stakingEarnings/10n) / parseEther('1200')
              const afterstakingUpfrontPremium = (variableDeposit) * (fixedDeposit) / parseEther('1200')
              expect(await lidoVault.userToFixedUpfrontPremium(addrs[0])).to.equalWithTolerance(prestakingUpfrontPremium, 1)
              expect(await lidoVault.userToFixedUpfrontPremium(addrs[10])).to.equalWithTolerance(afterstakingUpfrontPremium, 1)
              expect(prestakingUpfrontPremium).to.greaterThan(averageUpfrontPremium)
              // withdraw
              const lidoLastRequestId = BIG_INT_ZERO
              for (let i = 0; i < fixedDepositorCount; i++) {
                const { currentTimestamp } = await getTimeState(lidoVault)
                await lidoVault.connect(addrs[i]).withdraw(SIDE.FIXED)
                const requestIds = [lidoLastRequestId + BIG_INT_ONE + BigInt(i)]
                await expectVaultOngoingFixedWithdrawalArraysAreEqual(
                  lidoVault,
                  requestIds,
                  addrs[i].address,
                  currentTimestamp
                )
              }
  
              // end vault
              expect(await lidoVault.isEnded()).to.be.false
              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)
              expect(await lidoVault.isEnded()).to.be.true
  
              // vault ended withdraw - should claim ongoing fixed withdrawals
              console.log('vault end withdraw')
              await lidoVault.connect(addr1).withdraw(SIDE.VARIABLE)
              console.log('vault end withdraw completed')
              for (let i = 0; i < fixedDepositorCount; i++) {
                await expectVaultOngoingFixedWithdrawalArraysAreEqual(
                  lidoVault,
                  [],
                  addrs[i].address,
                  BIG_INT_ZERO
                )
              }
            })
            it('Should have correct early exit fee applied if withdraw after half way through vault term but staking earnings have accumulated (vault-started, multiple-participants)', async function () {
              const { lidoVault, addr1, addr2, lidoVaultAddress } =
                await loadFixture(multipleParticipantsStartVaultAndClaimFixtureWithStaking)
   
              // fast forward to half way through vault duration
              const { duration, endTime } = await getTimeState(lidoVault)
              const halfDuration = duration / BigInt(2)
              await time.increaseTo(endTime - halfDuration)
  
  
              // withdraw #1
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
  
              const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                lidoVault,
                upfrontPremium: upfrontPremium1,
                timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
              })

              const balanceBefore = await ethers.provider.getBalance(addr1)
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt = await (
                await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
              const amountWithdrawn = balanceAfter - balanceBefore
              const amountNotWithdrawn = vaultBalanceAfter - vaultBalanceBefore
  
              // no earnings should be withdrawn by the fixed side now that the vault has started
              const protocolFee = await lidoVault.appliedProtocolFee()
              expect(protocolFee).to.be.equal(0)
  
              const vaultEarnings = await lidoVault.feeEarnings()
              expect(vaultEarnings).to.equal(
                payBackAmount + scaledEarlyExitFee,
                'incorrect vault earnings'
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
  
              const fees = payBackAmount + protocolFee + scaledEarlyExitFee
              expect(amountNotWithdrawn).to.equal(fees)
              expect(amountWithdrawn).to.equalWithTolerance(fixedDeposit1 - (fees + gasFees), 10)
  
              // withdraw #2

              await lidoVault.connect(addr2).withdraw(SIDE.FIXED)
  
              const { payBackAmount: payBackAmount2, scaledEarlyExitFee: scaledEarlyExitFee2 } =
                await calculateFees({
                  lidoVault,
                  upfrontPremium: upfrontPremium2,
                  timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr2),
                })
  
  
              const balanceBefore2 = await ethers.provider.getBalance(addr2)
              const vaultBalanceBefore2 = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt2 = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingFixedWithdrawals()
              ).wait()
              const balanceAfter2 = await ethers.provider.getBalance(addr2)
              const vaultBalanceAfter2 = await ethers.provider.getBalance(lidoVaultAddress)
              const gasFees2 = calculateGasFees(receipt2)
              const amountWithdrawn2 = balanceAfter2 - balanceBefore2
              const amountNotWithdrawn2 = vaultBalanceAfter2 - vaultBalanceBefore2
  
              // no earnings should be withdrawn by the fixed side now that the vault has started
              const protocolFee2 = await lidoVault.appliedProtocolFee()
              expect(protocolFee2).to.be.equal(0)
  
              const vaultEarnings2 = await lidoVault.feeEarnings()
              expect(vaultEarnings2).to.equal(
                payBackAmount2 + scaledEarlyExitFee2 + vaultEarnings,
                'incorrect vault earnings'
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
  
              const fees2 = payBackAmount2 + protocolFee2 + scaledEarlyExitFee2
              expect(amountNotWithdrawn2).to.equal(fees2)
              // All staking earnings should have gone to addr2
              expect(amountWithdrawn2).to.equal(fixedDeposit2 + stakingEarnings - (fees2 + gasFees2))
            })
            it('Should have correct early exit fee applied if withdraw after half way through vault term but staking earnings have accumulated (vault-started, multiple-participants)', async function () {
              const { lidoVault, addr1, addr2, lidoVaultAddress } =
                await loadFixture(multipleParticipantsStartVaultAndClaimFixtureWithStaking)
  

              // fast forward to half way through vault duration
              const { duration, endTime } = await getTimeState(lidoVault)
              const halfDuration = duration / BigInt(2)
              await time.increaseTo(endTime - halfDuration)
  
              // withdraw #1
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
  
              const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                lidoVault,
                upfrontPremium: upfrontPremium1,
                timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
              })
  
              const balanceBefore = await ethers.provider.getBalance(addr1)
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt = await (
                await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
              const amountWithdrawn = balanceAfter - balanceBefore
              const amountNotWithdrawn = vaultBalanceAfter - vaultBalanceBefore
  
              // no earnings should be withdrawn by the fixed side now that the vault has started
              const protocolFee = await lidoVault.appliedProtocolFee()
              expect(protocolFee).to.be.equal(0)
  
              const vaultEarnings = await lidoVault.feeEarnings()
              expect(vaultEarnings).to.equal(
                payBackAmount + scaledEarlyExitFee,
                'incorrect vault earnings'
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
  
              const fees = payBackAmount + protocolFee + scaledEarlyExitFee
              expect(amountNotWithdrawn).to.equal(fees)
              expect(amountWithdrawn).to.equalWithTolerance(fixedDeposit1 - (fees + gasFees), 10)
  
              // withdraw #2

              await lidoVault.connect(addr2).withdraw(SIDE.FIXED)
  
              const { payBackAmount: payBackAmount2, scaledEarlyExitFee: scaledEarlyExitFee2 } =
                await calculateFees({
                  lidoVault,
                  upfrontPremium: upfrontPremium2,
                  timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr2),
                })
  
  
              const balanceBefore2 = await ethers.provider.getBalance(addr2)
              const vaultBalanceBefore2 = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt2 = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingFixedWithdrawals()
              ).wait()
              const balanceAfter2 = await ethers.provider.getBalance(addr2)
              const vaultBalanceAfter2 = await ethers.provider.getBalance(lidoVaultAddress)
              const gasFees2 = calculateGasFees(receipt2)
              const amountWithdrawn2 = balanceAfter2 - balanceBefore2
              const amountNotWithdrawn2 = vaultBalanceAfter2 - vaultBalanceBefore2
  
              // no earnings should be withdrawn by the fixed side now that the vault has started
              const protocolFee2 = await lidoVault.appliedProtocolFee()
              expect(protocolFee2).to.be.equal(0)
  
              const vaultEarnings2 = await lidoVault.feeEarnings()
              expect(vaultEarnings2).to.equal(
                payBackAmount2 + scaledEarlyExitFee2 + vaultEarnings,
                'incorrect vault earnings'
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
  
              const fees2 = payBackAmount2 + protocolFee2 + scaledEarlyExitFee2
              expect(amountNotWithdrawn2).to.equal(fees2)
              // All staking earnings should have gone to addr2
              expect(amountWithdrawn2).to.equal(fixedDeposit2 + stakingEarnings - (fees2 + gasFees2))
            })

          })
        })
      })

      describe('Variable Side', function () {

        it('Should fail to withdraw if no earnings have been made or no early fixed withdrawals have been executed (vault-started)', async function () {
          const {
            lidoVault,
            lidoWithdrawalQueueContract,
            addr2,
          } = await loadFixture(
            startVaultAndClaimFixture
          )
          await expectBalancesAddrAndWithdrawalArraysAreEqual(lidoVault, lidoWithdrawalQueueContract, addr2)
        })

        it('Should fail to withdraw again if user already has another withdrawal in progress (vault-started)', async function () {
          const {
            lidoVault,
            lidoWithdrawalQueueContract,
            addr2,
            lidoContract,
            lidoVaultAddress
          } = await loadFixture(
            startVaultAndClaimFixture
          )
          const lidoETHAmount1 = (await lidoContract.balanceOf(addr2)) as bigint
          const lidoETHAmount2 = (await lidoContract.balanceOf(lidoVaultAddress)) as bigint
          await expectBalancesAddrAndWithdrawalArraysAreEqual(lidoVault, lidoWithdrawalQueueContract, addr2)
        })

        it('Should fail to withdraw if lido withdrawal request if request has already been finalized (vault-started)', async function () {
          const { lidoVault, lidoWithdrawalQueueContract, addr2 } = await loadFixture(
            startVaultAndClaimFixture
          )

          // generate earnings
          await submitOracleReport()

          const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
          const requestIds = [lidoLastRequestId + BIG_INT_ONE]

          await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

          await expectVaultOngoingVariableWithdrawalArraysAreEqual(
            lidoVault,
            requestIds,
            addr2.address
          )

          await finalizeLidoWithdrawalRequests(requestIds, parseEther('1000'))
          await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()

          await expect(
            lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
          ).to.be.revertedWith('WNR')
        })

        describe('Fee Receiver', function () {
          it('Should fail to withdraw if no protocolFee has accumulated from withdraws (vault-started)', async function () {
            // const { lidoVault, deployer } = await loadFixture(startVaultAndClaimFixture)
            //
            // await expect(lidoVault.connect(deployer).withdraw(SIDE.VARIABLE)).to.be.revertedWith(
            //   'WNM'
            // )
           const {
              lidoVault,
              deployer,
            } = await loadFixture(
              startVaultAndClaimFixture
            )

            await expectBalancesAddrAreEqual(lidoVault, deployer)
          })

          it('Should withdraw appliedProtocolFee (vault-started)', async function () {
            const {
              lidoVault,
              lidoVaultAddress,
              lidoWithdrawalQueueContract,
              lidoContract,
              deployer,
              addr2,
            } = await loadFixture(startVaultAndClaimFixture)

            await submitOracleReport()

            const lidoETHAmount = await lidoContract.balanceOf(lidoVaultAddress)
            const earnings = lidoETHAmount - fixedDeposit
            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
            await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()

            const appliedProtocolFee = await lidoVault.appliedProtocolFee()
            expect(appliedProtocolFee).to.be.equal(
              (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
            )

            const balanceBefore = await ethers.provider.getBalance(deployer)
            const receipt = await (await lidoVault.connect(deployer).withdraw(SIDE.VARIABLE)).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(deployer)
            const amountWithdrawn = balanceAfter - balanceBefore

            expect(amountWithdrawn).to.be.equal(appliedProtocolFee - gasFees)
          })
        })

        describe('Lido Negative Rebase', () => {
          it('Should throw a WNM error if there is a lido negative rebase and they have previously withdrawn (vault-started)', async function () {
            const { lidoVault, lidoMock, lidoVaultAddress, lidoWithdrawalQueueContract, addr1, addr2, addr3 } =
              await loadFixture(deployLidoVaultWithMockedLidoFixture)

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
            await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: parseEther('10') })
            await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: parseEther('10') })
            expect(await lidoVault.isStarted()).to.be.true

            await lidoVault.connect(addr1).claimFixedPremium()

            // mock staking earnings - 100 ETH
            const stakingEarnings = parseEther('100')
            await lidoMock.addStakingEarningsForTargetETH(
              fixedDeposit + stakingEarnings,
              lidoVaultAddress
            )

            // first withdraw
            let balanceBefore = await ethers.provider.getBalance(addr2)
            await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
            await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
            expect(await ethers.provider.getBalance(addr2)).to.be.greaterThan(balanceBefore)

            // mock staking loss
            await lidoMock.subtractStakingEarnings(parseEther('5'))

            // second withdraw fails since there is 0 ETH to withdraw
            // await expect(lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).to.be.revertedWith('WNM')
            await expectBalancesAddrAreEqual(lidoVault, addr2)
            console.log('Complete Second withdraw')
          })
        })

        describe('Only Fixed Early Exit Fees', function () {
          describe('Single Variable Participant', function () {
            async function startVaultAndFixedEarlyWithdrawFixture() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2 } = state

              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })

              await lidoVault.connect(addr1).claimFixedPremium()

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // early fixed withdrawal
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              const fixedRequestIds = [lidoLastRequestId + BIG_INT_ONE]
              await finalizeLidoWithdrawalRequests(fixedRequestIds, DEFAULTS.fixedSideCapacity)
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()

              return state
            }

            it('Should emit the correct variable withdrawn event if there are only fixed fees (vault-started, events)', async function () {
              const { lidoVault, addr2 } = await loadFixture(startVaultAndFixedEarlyWithdrawFixture)

              const withdrawTrx = await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              await expect(withdrawTrx)
                .to.emit(lidoVault, 'VariableFundsWithdrawn')
                .withArgs(anyUint, addr2.address, true, false)
            })

            it('Should have correct share of the fee earnings (vault-started)', async function () {
              const { lidoVault, addr2, variableBearerToken } = await loadFixture(
                startVaultAndFixedEarlyWithdrawFixture
              )

              // early exit fee + payback amount should have been applied
              const feeEarnings = await lidoVault.feeEarnings()
              expect(feeEarnings).to.be.greaterThan(0)

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              expect(bearerTokenBalance).to.equal(variableDeposit)

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const amountWithdrawn = balanceAfter - balanceBefore

              // should not have any withdraw state
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                BIG_INT_ZERO
              )
              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                feeEarnings
              )

              expect(await lidoVault.feeEarnings()).to.be.equal(0)
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(feeEarnings)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              // will be entire amount of feeEarnings since there is only 1 variable depositor
              expect(amountWithdrawn).to.be.equal(feeEarnings - gasFees)
            })

            it('Should fail to withdraw fee earnings again if they already withdrew (vault-started)', async function () {
              const { lidoVault, addr2 } = await loadFixture(startVaultAndFixedEarlyWithdrawFixture)
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
          //
      //   await expect(lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).to.be.revertedWith(
      //     'WNM'
      //   )
              await expectBalancesAddrAreEqual(lidoVault, addr2)
            })
          })

          describe('Multiple Variable Participants', function () {
            const variableDeposit2 = parseEther('10')
            const variableDeposit3 = parseEther('20')

            async function startVaultAndFixedEarlyWithdrawFixture() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2, addr3 } = state

              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })

              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })

              await lidoVault.connect(addr1).claimFixedPremium()

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // early fixed withdrawal
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              const fixedRequestIds = [lidoLastRequestId + BIG_INT_ONE]
              await finalizeLidoWithdrawalRequests(fixedRequestIds, DEFAULTS.fixedSideCapacity)
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()

              return state
            }

            it('Should have correct share of the fee earnings (vault-started, multiple-participants)', async function () {
              const { lidoVault, addr2, addr3, variableBearerToken } = await loadFixture(
                startVaultAndFixedEarlyWithdrawFixture
              )

              // early exit fee + payback amount should have been applied
              const feeEarnings = await lidoVault.feeEarnings()
              expect(feeEarnings).to.be.greaterThan(0)

              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()

              // withdraw #1
              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              expect(bearerTokenBalance).to.equal(variableDeposit2)

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const amountWithdrawn = balanceAfter - balanceBefore

              // should not have any withdraw state
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                BIG_INT_ZERO
              )

              const feeEarningsShare =
                (bearerTokenBalance * feeEarnings) / variableBearerTokenTotalSupply
              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                feeEarningsShare
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(feeEarningsShare)
              expect(await lidoVault.feeEarnings()).to.be.equal(feeEarnings - feeEarningsShare)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equal(feeEarningsShare - gasFees)

              // withdraw #2
              const bearerTokenBalance3 = await lidoVault.variableBearerToken(addr3.address)
              expect(bearerTokenBalance3).to.equal(variableDeposit3)

              const balanceBefore3 = await ethers.provider.getBalance(addr3)
              const receipt3 = await (await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)).wait()
              const gasFees3 = calculateGasFees(receipt3)
              const balanceAfter3 = await ethers.provider.getBalance(addr3)
              const amountWithdrawn3 = balanceAfter3 - balanceBefore3

              // should not have any withdraw state
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                BIG_INT_ZERO
              )

              const feeEarningsShare3 =
                (bearerTokenBalance3 * feeEarnings) / variableBearerTokenTotalSupply

              expect(await lidoVault.variableToWithdrawnFees(addr3.address)).to.be.equal(
                feeEarningsShare3
              )

              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(
                feeEarningsShare + feeEarningsShare3
              )
              expect(await lidoVault.feeEarnings()).to.be.equal(
                feeEarnings - (feeEarningsShare + feeEarningsShare3)
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr3.address)
              expect(amountWithdrawn3).to.be.equal(feeEarningsShare3 - gasFees3)
            })

            it('Should fail to withdraw fee earnings again if they already withdrew (vault-started, multiple-participants)', async function () {
              const { lidoVault, addr2 } = await loadFixture(startVaultAndFixedEarlyWithdrawFixture)
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
              await expectBalancesAddrAreEqual(lidoVault, addr2)
            })
          })
        })

        describe('Lido Staking Earnings Generated', function () {
          describe('Single Variable Participant', function () {
            async function startVaultAndGenerateEarningsFixture() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, addr1, addr2 } = state

              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })

              await lidoVault.connect(addr1).claimFixedPremium()

              await submitOracleReport()

              return state
            }

            it('Should emit the correct lido withdrawal requested event if staking earnings have been generated (vault-started, events)', async function () {
              const { lidoVault, addr2 } = await loadFixture(startVaultAndGenerateEarningsFixture)

              const withdrawTrx = await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              await expect(withdrawTrx)
                .to.emit(lidoVault, 'LidoWithdrawalRequested')
                .withArgs(addr2.address, anyValue, SIDE.VARIABLE, true, false)
            })


            it('Should fail occurs because the caller of the transaction is not the Protocol Fee Receiver (vault-started)', async function () {
              const { lidoVault, addr2, lidoWithdrawalQueueContract } = await loadFixture(
                startVaultAndGenerateEarningsFixture
              )

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              await expect(lidoVault.connect(addr2)
                .feeReceiverFinalizeVaultOngoingVariableWithdrawals(addr2)).to.be.revertedWith(
                'IFR'
              )
            })

            it('Should emit the correct variable events if staking earnings have been generated and Protocol Fee Receiver finalize their withdrawal (vault-started, events)', async function () {
              const { lidoVault, addr2, lidoWithdrawalQueueContract, protocolFeeReceiver } = await loadFixture(
                startVaultAndGenerateEarningsFixture
              )

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              const appliedProtocolFeeBefore = await lidoVault.appliedProtocolFee()
              await lidoVault
                .connect(protocolFeeReceiver)
                .feeReceiverFinalizeVaultOngoingVariableWithdrawals(addr2)
              const appliedProtocolFeeAfter = await lidoVault.appliedProtocolFee()

              expect(await lidoVault.variableToPendingWithdrawalAmount(addr2.address)).to.greaterThan(0)

              await expect(appliedProtocolFeeAfter).to.be.greaterThan(appliedProtocolFeeBefore)
            })

            it('Should emit the correct variable events if staking earnings have been generated and they finalize their withdrawal (vault-started, events)', async function () {
              const { lidoVault, addr2, lidoWithdrawalQueueContract } = await loadFixture(
                startVaultAndGenerateEarningsFixture
              )

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              const finalizedTrx = await lidoVault
                .connect(addr2)
                .finalizeVaultOngoingVariableWithdrawals()

              await expect(finalizedTrx)
                .to.emit(lidoVault, 'LidoWithdrawalFinalized')
                .withArgs(addr2.address, anyValue, SIDE.VARIABLE, true, false)
              await expect(finalizedTrx)
                .to.emit(lidoVault, 'VariableFundsWithdrawn')
                .withArgs(anyValue, addr2.address, true, false)
            })

            it('Should fail to withdraw if address has no bearer tokens (vault-started)', async function () {
              const { lidoVault, addr3, variableBearerToken } = await loadFixture(
                startVaultAndGenerateEarningsFixture
              )

              expect(await lidoVault.variableBearerToken(addr3.address)).to.equal(0)
              await expect(lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)).to.be.revertedWith(
                'NBT'
              )
            })

            it('Should fail to finalize withdraw if address has no ongoing request (vault-started)', async function () {
              const { lidoVault, addr3 } = await loadFixture(startVaultAndGenerateEarningsFixture)

              await expect(
                lidoVault.connect(addr3).finalizeVaultOngoingVariableWithdrawals()
              ).to.be.revertedWith('WNR')
            })

            it('Should have correct share of the staking earnings withdrawn if user immediately withdraws (vault-started)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                lidoContract,
                addr2,
                lidoVaultAddress,
                variableBearerToken,
              } = await loadFixture(startVaultAndGenerateEarningsFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              expect(bearerTokenBalance).to.equal(variableDeposit)

              const lidoETHAmount = await lidoContract.balanceOf(lidoVaultAddress)

              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              const earnings = lidoETHAmount - fixedDeposit
              const protocolFee = (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
              // sanity checks
              expect(earnings).to.be.greaterThan(BIG_INT_ZERO)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equalWithTolerance(
                earnings - protocolFee, 10
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equalWithTolerance(earnings - protocolFee, 10) // gets all the earnings since they are the only variable depositor
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr2.address
              )

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
              const amountWithdrawn = balanceAfter - balanceBefore
              const amountNotWithdrawn = vaultBalanceAfter - vaultBalanceBefore
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()

              // since some earnings were made protocol fee should have been applied
              expect(await lidoVault.appliedProtocolFee()).to.be.equal(protocolFee)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equalWithTolerance(
                earnings - protocolFee, 10
              )
              // no early fixed withdrawals occurred
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equalWithTolerance(
                (bearerTokenBalance * earnings) / variableBearerTokenTotalSupply -
                  (gasFees + protocolFee), 10
              )
              expect(amountNotWithdrawn).to.be.equal(protocolFee)
            })

            it('Should fail to withdraw if they have just withdrawn and no earnings have accumulated (vault-started)', async function () {
              const { lidoVault, lidoWithdrawalQueueContract, addr2 } = await loadFixture(
                startVaultAndGenerateEarningsFixture
              )

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()

              await expectBalancesAddrAreEqual(lidoVault, addr2)
            })

            it('Should have correct share of the staking earnings withdrawn if user withdraws half way through the duration (vault-started)', async function () {
              const {
                lidoVault,
                lidoVaultAddress,
                lidoWithdrawalQueueContract,
                lidoContract,
                addr2,
                variableBearerToken,
              } = await loadFixture(startVaultAndGenerateEarningsFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
              expect(bearerTokenBalance).to.equal(variableDeposit)

              const lidoETHAmount = await lidoContract.balanceOf(lidoVaultAddress)
              // fast forward
              const { duration, endTime } = await getTimeState(lidoVault)
              const halfDuration = duration / BigInt(2)
              await time.increaseTo(endTime - halfDuration)

              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              const earnings = lidoETHAmount - fixedDeposit
              const protocolFee = (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K

              // sanity checks
              expect(earnings).to.be.greaterThan(BIG_INT_ZERO)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equalWithTolerance(
                earnings - protocolFee, 10
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equalWithTolerance(earnings - protocolFee, 10) // gets all the earnings since they are the only variable depositor
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr2.address
              )

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
              const amountWithdrawn = balanceAfter - balanceBefore
              const amountNotWithdrawn = vaultBalanceAfter - vaultBalanceBefore

              // since some earnings were made protocol fee should have been applied
              expect(await lidoVault.appliedProtocolFee()).to.be.equal(protocolFee)

              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                BIG_INT_ZERO
              )
              // no early fixed withdrawals occurred
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equalWithTolerance(
                (bearerTokenBalance * earnings) / variableBearerTokenTotalSupply -
                  (gasFees + protocolFee), 10
              )
              expect(amountNotWithdrawn).to.be.equal(protocolFee)
            })

            it('Should have correct share of the staking earnings withdrawn if user withdraws after a fixed early withdrawal (vault-started)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                lidoContract,
                addr1,
                addr2,
                lidoVaultAddress,
                variableBearerToken,
                fixedBearerToken,
              } = await loadFixture(startVaultAndGenerateEarningsFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // early fixed withdrawal
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              const fixedRequestIds = [lidoLastRequestId + BIG_INT_ONE]
              await finalizeLidoWithdrawalRequests(fixedRequestIds, DEFAULTS.fixedSideCapacity)
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
              expect(await lidoVault.fixedBearerTokenTotalSupply()).to.equal(0)

              // early exit fee + payback amount should have been applied
              const feeEarnings = await lidoVault.feeEarnings()
              expect(feeEarnings).to.be.greaterThan(0)

              const variableRequestIds = [lidoLastRequestId + BigInt(2)]

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              expect(bearerTokenBalance).to.equal(variableDeposit)

              const lidoETHAmount = await lidoContract.balanceOf(lidoVaultAddress)
              const protocolFee = (lidoETHAmount * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
              // should still have the earnings as the vaults balance
              expect(lidoETHAmount).to.be.greaterThan(0)
              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt0 =  await (await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).wait()
              const gasFees0 = calculateGasFees(receipt0)

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                lidoETHAmount - protocolFee
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equal(
                lidoETHAmount - protocolFee
              ) // gets all the earnings since they are the only variable depositor
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                variableRequestIds,
                addr2.address
              )

              await finalizeLidoWithdrawalRequests(variableRequestIds, DEFAULTS.fixedSideCapacity)

              
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const amountWithdrawn = balanceAfter - balanceBefore 

              // since some earnings were made protocol fee should have been applied
              expect(await lidoVault.appliedProtocolFee()).to.be.equal(protocolFee)

              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                feeEarnings
              )
              expect(await lidoVault.feeEarnings()).to.be.equal(0)
              // will be entire amount of feeEarnings since there is only 1 variable depositor
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(feeEarnings)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equalWithTolerance(
                feeEarnings + lidoETHAmount - (gasFees + gasFees0 + protocolFee), 10
              )
            })

            it('Should have correct share of the staking earnings if they withdraw multiple times (vault-started, variable-perpetual-withdrawal)', async function () {
              const {
                lidoVault,
                lidoMock,
                addr1,
                addr2,
                variableBearerToken,
                fixedBearerToken,
                lidoVaultAddress,
              } = await loadFixture(deployLidoVaultWithMockedLidoFixture)

              const variableDeposit = parseEther('20')

              // vault started
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })
              await lidoVault.connect(addr1).claimFixedPremium()
              expect(await lidoVault.isStarted()).to.equal(true)

              const lidoShares = await lidoVault.fixedBearerToken(addr1.address)
              expect(lidoShares).to.equal(parseEther('1000'))

              // mock staking earnings - 100 ETH
              const lidoLastRequestId = BIG_INT_ZERO
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              const stakingEarnings = parseEther('100')
              await lidoMock.addStakingEarningsForTargetETH(
                fixedDeposit + stakingEarnings,
                lidoVaultAddress
              )

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              expect(bearerTokenBalance).to.equal(variableDeposit)

              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                stakingEarnings
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equal(stakingEarnings) // gets all the earnings since they are the only variable depositor
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr2.address
              )

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const amountWithdrawn = balanceAfter - balanceBefore

              expect(await lidoVault.appliedProtocolFee()).to.be.equal(0)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                stakingEarnings
              )
              // no early fixed withdrawals occurred
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equalWithTolerance(stakingEarnings - gasFees, 10)

              // more staking earnings
              const requestIds2 = [lidoLastRequestId + BigInt(2)]
              const stakingEarnings2 = parseEther('50')
              await lidoMock.addStakingEarningsForTargetETH(
                fixedDeposit + stakingEarnings2,
                lidoVaultAddress
              )

              // withdraw again
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              const totalStakingEarnings = parseEther('150')

              expect(
                await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)
              ).to.be.equalWithTolerance(totalStakingEarnings, 1)
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equalWithTolerance(
                parseEther('150'),
                1
              ) // gets all the earnings since they are the only variable depositor
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds2,
                addr2.address
              )

              const balanceBefore2 = await ethers.provider.getBalance(addr2)
              const receipt2 = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees2 = calculateGasFees(receipt2)
              const balanceAfter2 = await ethers.provider.getBalance(addr2)
              const amountWithdrawn2 = balanceAfter2 - balanceBefore2

              expect(await lidoVault.appliedProtocolFee()).to.be.equal(0)
              expect(
                await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)
              ).to.be.equalWithTolerance(totalStakingEarnings, 1)
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn2).to.be.equalWithTolerance(stakingEarnings2 - gasFees2, 10)
            })
          })

          describe('Multiple Variable Participants', function () {
            const variableDeposit2 = parseEther('10')
            const variableDeposit3 = parseEther('20')

            async function startVaultAndGenerateEarningsFixture() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, addr1, addr2, addr3 } = state

              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })

              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })

              await lidoVault.connect(addr1).claimFixedPremium()

              await submitOracleReport()

              return state
            }

            it('Should have correct ETH staking earnings withdrawn (vault-started, multiple-participants, variable-perpetual-withdrawal)', async function () {
              const { lidoVault, addr1, addr2, addr3, addr4, lidoMock, lidoVaultAddress } =
                await loadFixture(deployLidoVaultWithMockedLidoFixture)

              const variableDeposit2 = parseEther('10')
              const variableDeposit3 = parseEther('5')
              const variableDeposit4 = parseEther('5')

              // vault started
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })
              await lidoVault.connect(addr4).deposit(SIDE.VARIABLE, { value: variableDeposit4 })
              await lidoVault.connect(addr1).claimFixedPremium()
              expect(await lidoVault.isStarted()).to.equal(true)

              // mock staking earnings - 100 ETH
              const lidoLastRequestId = BIG_INT_ZERO
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              const stakingEarnings = parseEther('100')
              await lidoMock.addStakingEarningsForTargetETH(
                fixedDeposit + stakingEarnings,
                lidoVaultAddress
              )

              // withdraw #1
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr2.address
              )
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                parseEther('50') // 100 earnings / 2 since they own 50% of the bearer token supply
              )

              // withdraw #2
              const requestIds2 = [lidoLastRequestId + BigInt(2)]
              await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)

              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds2,
                addr3.address
              )

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)).to.be.equalWithTolerance(
                parseEther('25'), // 100 earnings / 4 since they own 25% of the bearer token supply
                10
              )
            })

            const TEN_PERCENT_PROTOCOL_FEE_BPS = 1000
            const deployLidoVaultWithMockedLidoCustomProtocolFeeFixture = () =>
              deployVaultWithMockedLido({
                variableSideCapacity: parseEther('20'),
                protocolFeeBps: TEN_PERCENT_PROTOCOL_FEE_BPS,
              })

            it('Should have correct share of the ETH staking earnings withdrawn on multiple withdrawals (vault-started, multiple-participants, variable-perpetual-withdrawal)', async function () {
              const { lidoVault, addr1, addr2, addr3, addr4, lidoMock, lidoVaultAddress, lidoContract } =
                await loadFixture(deployLidoVaultWithMockedLidoCustomProtocolFeeFixture)

              const variableDeposit2 = parseEther('10')
              const variableDeposit3 = parseEther('5')
              const variableDeposit4 = parseEther('5')

              // vault started
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })
              await lidoVault.connect(addr4).deposit(SIDE.VARIABLE, { value: variableDeposit4 })
              await lidoVault.connect(addr1).claimFixedPremium()
              expect(await lidoVault.isStarted()).to.equal(true)

              // mock staking earnings - 100 ETH
              const lidoLastRequestId = BIG_INT_ZERO
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              const stakingEarnings = parseEther('100')
              await lidoMock.addStakingEarningsForTargetETH(
                fixedDeposit + stakingEarnings,
                lidoVaultAddress
              )
              console.log(await lidoVault.stakingBalance())
              console.log(await lidoVault.stakingShares())
              // addr2 withdraw #1
              const earningsShareAddr2 = parseEther('50') // 100 earnings / 2 since they own 50% of the bearer token supply
              const protocolFeeShareAddr2 =
                (earningsShareAddr2 * BigInt(TEN_PERCENT_PROTOCOL_FEE_BPS)) / BIG_INT_10K
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr2.address
              )
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                earningsShareAddr2 - protocolFeeShareAddr2
              )

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const amountWithdrawn = balanceAfter - balanceBefore

              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                BIG_INT_ZERO
              )
              // no early fixed withdrawals occurred
              expect(await lidoVault.feeEarnings()).to.be.equal(0)
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equal(
                earningsShareAddr2 - protocolFeeShareAddr2 - gasFees
              )

              // addr3 withdraw #1
              const earningsShareAddr3 = parseEther('25') // 100 earnings / 4 since they own 25% of the bearer token supply
              const protocolFeeShareAddr3 =
                (earningsShareAddr3 * BigInt(TEN_PERCENT_PROTOCOL_FEE_BPS)) / BIG_INT_10K
              const requestIds2 = [lidoLastRequestId + BigInt(2)]
              await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)

              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds2,
                addr3.address
              )

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)).to.be.equal(
                earningsShareAddr3 - protocolFeeShareAddr3
              )

              const balanceBefore3 = await ethers.provider.getBalance(addr3)
              const receipt3 = await (
                await lidoVault.connect(addr3).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees3 = calculateGasFees(receipt3)
              const balanceAfter3 = await ethers.provider.getBalance(addr3)
              const amountWithdrawn3 = balanceAfter3 - balanceBefore3

              expect(await lidoVault.variableToWithdrawnFees(addr3.address)).to.be.equal(
                BIG_INT_ZERO
              )
              expect(await lidoVault.feeEarnings()).to.be.equal(0)
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr3.address)
              expect(amountWithdrawn3).to.be.equal(
                earningsShareAddr3 - protocolFeeShareAddr3 - gasFees3
              )

              // mock more staking earnings - 50 ETH
              const stakingEarnings2 = parseEther('50')
              await lidoMock.addStakingEarningsForTargetETH(
                fixedDeposit +
                  stakingEarnings +
                  stakingEarnings2 -
                  (earningsShareAddr2 + earningsShareAddr3), // subtract already withdrawn funds
                lidoVaultAddress
              )

              
              // addr2 withdraw again
              const earningsShareAddr2Again = (await lidoContract.getPooledEthByShares(await lidoVault.stakingShares() + 
              await lidoVault.withdrawnStakingEarningsInStakes()) -  await lidoVault.fixedETHDepositTokenTotalSupply()) / 2n  - 
              await lidoContract.getPooledEthByShares(await lidoVault.variableToWithdrawnStakingEarningsInShares(addr2.address))
              const protocolFeeShareAddr2Again =
                (earningsShareAddr2Again * BigInt(TEN_PERCENT_PROTOCOL_FEE_BPS)) / BIG_INT_10K
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                 earningsShareAddr2Again - protocolFeeShareAddr2Again + earningsShareAddr2 - protocolFeeShareAddr2 
              )

              const balanceBefore2 = await ethers.provider.getBalance(addr2)
              const receipt2 = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees2 = calculateGasFees(receipt2)
              const balanceAfter2 = await ethers.provider.getBalance(addr2)
              const amountWithdrawn2 = balanceAfter2 - balanceBefore2

              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                BIG_INT_ZERO
              )
              expect(await lidoVault.feeEarnings()).to.be.equal(0)
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn2).to.be.equal(
                earningsShareAddr2Again - protocolFeeShareAddr2Again - gasFees2
              )

              // addr3 withdraw again
              const earningsShareAddr3Again = (await lidoContract.getPooledEthByShares(await lidoVault.stakingShares() + 
              await lidoVault.withdrawnStakingEarningsInStakes()) -  await lidoVault.fixedETHDepositTokenTotalSupply()) / 4n  - 
              await lidoContract.getPooledEthByShares(await lidoVault.variableToWithdrawnStakingEarningsInShares(addr3.address))
              const protocolFeeShareAddr3Again =
                (earningsShareAddr3Again * BigInt(TEN_PERCENT_PROTOCOL_FEE_BPS)) / BIG_INT_10K
              await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)).to.be.equal(
                earningsShareAddr3Again - protocolFeeShareAddr3Again + earningsShareAddr3 - protocolFeeShareAddr3
              )

              const balanceBefore3Again = await ethers.provider.getBalance(addr3)
              const receipt3Again = await (
                await lidoVault.connect(addr3).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees3Again = calculateGasFees(receipt3Again)
              const balanceAfter3Again = await ethers.provider.getBalance(addr3)
              const amountWithdrawn3Again = balanceAfter3Again - balanceBefore3Again

              expect(await lidoVault.variableToWithdrawnFees(addr3.address)).to.be.equal(
                BIG_INT_ZERO
              )
              expect(await lidoVault.feeEarnings()).to.be.equal(0)
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr3.address)
              expect(amountWithdrawn3Again).to.be.equal(
                earningsShareAddr3Again - protocolFeeShareAddr3Again - gasFees3Again
              )

              // addr4 withdraws
              const earningsShareAddr4 = (await lidoContract.getPooledEthByShares(await lidoVault.stakingShares() + 
              await lidoVault.withdrawnStakingEarningsInStakes()) -  await lidoVault.fixedETHDepositTokenTotalSupply()) / 4n  - 
              await lidoContract.getPooledEthByShares(await lidoVault.variableToWithdrawnStakingEarningsInShares(addr4.address))
              const protocolFeeShareAddr4 =
                (earningsShareAddr4 * BigInt(TEN_PERCENT_PROTOCOL_FEE_BPS)) / BIG_INT_10K
              await lidoVault.connect(addr4).withdraw(SIDE.VARIABLE)

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr4.address)).to.be.equal(
                earningsShareAddr4 - protocolFeeShareAddr4
              )

              const balanceBefore4 = await ethers.provider.getBalance(addr4)
              const receipt4 = await (
                await lidoVault.connect(addr4).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees4 = calculateGasFees(receipt4)
              const balanceAfter4 = await ethers.provider.getBalance(addr4)
              const amountWithdrawn4 = balanceAfter4 - balanceBefore4

              expect(await lidoVault.variableToWithdrawnFees(addr4.address)).to.be.equal(
                BIG_INT_ZERO
              )
              expect(await lidoVault.feeEarnings()).to.be.equal(0)
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr4.address)
              expect(amountWithdrawn4).to.be.equal(
                earningsShareAddr4 - protocolFeeShareAddr4 - gasFees4
              )
            })

            it('Should have correct share of the staking earnings withdrawn if users immediately withdraw (vault-started, multiple-participants, variable-perpetual-withdrawal)', async function () {
              const {
                lidoVault,
                lidoVaultAddress,
                lidoWithdrawalQueueContract,
                lidoContract,
                addr2,
                addr3,
                variableBearerToken,
              } = await loadFixture(startVaultAndGenerateEarningsFixture)

              const lidoETHAmount = await lidoContract.balanceOf(lidoVaultAddress)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              // withdraw #1
              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
              expect(bearerTokenBalance).to.equal(variableDeposit2)

              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              const earnings = lidoETHAmount - fixedDeposit
              const earningsShare = (bearerTokenBalance * earnings) / variableBearerTokenTotalSupply
              const protocolFeeShare =
                (earningsShare * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K

              // sanity checks
              expect(earnings).to.be.greaterThan(BIG_INT_ZERO)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equalWithTolerance(
                earningsShare - protocolFeeShare, 10
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equalWithTolerance(
                earningsShare - protocolFeeShare, 10
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr2.address
              )

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
              const amountWithdrawn = balanceAfter - balanceBefore
              const amountNotWithdrawn = vaultBalanceAfter - vaultBalanceBefore

              // since some earnings were made protocol fee should have been applied
              const protocolFee = await lidoVault.appliedProtocolFee()
              expect(protocolFee).to.be.equal(
                (earningsShare * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
              )

              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                BIG_INT_ZERO
              )
              // no early fixed withdrawals occurred
              expect(await lidoVault.feeEarnings()).to.be.equal(0)
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equalWithTolerance(earningsShare - (gasFees + protocolFee), 10)
              expect(amountNotWithdrawn).to.be.equal(protocolFee)

              // withdraw #2
              const lidoETHAmount3 = await lidoContract.balanceOf(lidoVaultAddress)
              const requestIds3 = [lidoLastRequestId + BigInt(2)]

              const bearerTokenBalance3 = await lidoVault.variableBearerToken(addr3.address)
              expect(bearerTokenBalance3).to.equal(variableDeposit3)

              await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)

              const earnings3 = lidoETHAmount3 + earningsShare - fixedDeposit
              const earningsShare3 =
                (bearerTokenBalance3 * earnings) / variableBearerTokenTotalSupply
              const protocolFeeShare3 =
                (earningsShare3 * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K

              expect(earnings3).to.equalWithTolerance(lidoETHAmount - fixedDeposit, 10)
              expect(
                await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)
              ).to.equalWithTolerance(earningsShare3 - protocolFeeShare3, 10)
              expect(await lidoVault.withdrawnStakingEarnings()).to.equalWithTolerance(
                earningsShare + earningsShare3 - protocolFeeShare - protocolFeeShare3,
                10
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds3,
                addr3.address
              )

              await finalizeLidoWithdrawalRequests(requestIds3, DEFAULTS.fixedSideCapacity)

              const balanceBefore3 = await ethers.provider.getBalance(addr3)
              const vaultBalanceBefore3 = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt3 = await (
                await lidoVault.connect(addr3).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees3 = calculateGasFees(receipt3)
              const balanceAfter3 = await ethers.provider.getBalance(addr3)
              const vaultBalanceAfter3 = await ethers.provider.getBalance(lidoVaultAddress)
              const amountWithdrawn3 = balanceAfter3 - balanceBefore3
              const amountNotWithdrawn3 = vaultBalanceAfter3 - vaultBalanceBefore3

              // since some earnings were made protocol fee should have been applied
              const protocolFee3 = (await lidoVault.appliedProtocolFee()) - protocolFee
              expect(protocolFee3).to.be.equal(
                (earningsShare3 * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
              )

              expect(await lidoVault.variableToWithdrawnFees(addr3.address)).to.be.equal(
                BIG_INT_ZERO
              )
              // no early fixed withdrawals occurred
              expect(await lidoVault.feeEarnings()).to.be.equal(0)
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr3.address)
              expect(amountWithdrawn3).to.equalWithTolerance(
                earningsShare3 - (gasFees3 + protocolFee3),
                10
              )
              expect(amountNotWithdrawn3).to.be.equal(protocolFee3)
            })

            it('Should have correct share of the staking earnings if they withdraw multiple times (vault-started, multiple-participants, variable-perpetual-withdrawal)', async function () {
              const {
                lidoVault,
                lidoMock,
                addr1,
                addr2,
                addr3,
                addr4,
                variableBearerToken,
                lidoVaultAddress,
              } = await loadFixture(deployLidoVaultWithMockedLidoFixture)

              const variableDeposit2 = parseEther('10')
              const variableDeposit3 = parseEther('5')
              const variableDeposit4 = parseEther('5')

              // vault started
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })
              await lidoVault.connect(addr4).deposit(SIDE.VARIABLE, { value: variableDeposit4 })
              await lidoVault.connect(addr1).claimFixedPremium()
              expect(await lidoVault.isStarted()).to.equal(true)

              // mock staking earnings - 100 ETH
              const lidoLastRequestId = BIG_INT_ZERO
              const stakingEarnings = parseEther('100')
              await lidoMock.addStakingEarningsForTargetETH(
                fixedDeposit + stakingEarnings,
                lidoVaultAddress
              )

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              expect(bearerTokenBalance).to.equal(variableDeposit2)

              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                parseEther('50') // 100 / 2
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equal(parseEther('50'))
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                [lidoLastRequestId + BIG_INT_ONE],
                addr2.address
              )

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const amountWithdrawn = balanceAfter - balanceBefore

              expect(await lidoVault.appliedProtocolFee()).to.be.equal(0)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                parseEther('50')
              )
              // no early fixed withdrawals occurred
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equalWithTolerance(parseEther('50') - gasFees, 10)

              // addr3 withdraws
              await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)).to.be.equalWithTolerance(
                parseEther('25'), // 100 * 0.25
                10
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equalWithTolerance(parseEther('75'), 10)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                [lidoLastRequestId + BigInt(2)],
                addr3.address
              )

              const balanceBefore3 = await ethers.provider.getBalance(addr3)
              const receipt3 = await (
                await lidoVault.connect(addr3).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees3 = calculateGasFees(receipt3)
              const balanceAfter3 = await ethers.provider.getBalance(addr3)
              const amountWithdrawn3 = balanceAfter3 - balanceBefore3

              expect(await lidoVault.appliedProtocolFee()).to.be.equal(0)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)).to.be.equalWithTolerance(
                parseEther('25'),
                10
              )
              // no early fixed withdrawals occurred
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr3.address)
              expect(amountWithdrawn3).to.be.equalWithTolerance(parseEther('25') - gasFees3, 10)

              // more staking earnings
              const stakingEarnings2 = parseEther('50')
              await lidoMock.addStakingEarningsForTargetETH(
                fixedDeposit + stakingEarnings + stakingEarnings2 - parseEther('75'), // subtract the 75 that was already withdrawn
                lidoVaultAddress
              )

              // addr2 withdraws again
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
              expect(
                await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)
              ).to.be.lessThan(
                parseEther('75') // original 50 + 25 (50 / 2) - loses from early withdraw
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.lessThan(
                parseEther('100') // 100 ETH - loses from early withdraw
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                [lidoLastRequestId + BigInt(3)],
                addr2.address
              )

              const balanceBefore2 = await ethers.provider.getBalance(addr2)
              const receipt2 = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees2 = calculateGasFees(receipt2)
              const balanceAfter2 = await ethers.provider.getBalance(addr2)
              const amountWithdrawn2 = balanceAfter2 - balanceBefore2

              expect(await lidoVault.appliedProtocolFee()).to.be.equal(0)
              expect(
                await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)
              ).to.be.lessThan(parseEther('75'))
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn2).to.be.lessThan(parseEther('25') - gasFees2)


              // addr4 withdraws
              await lidoVault.connect(addr4).withdraw(SIDE.VARIABLE)

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr4.address)).to.be.greaterThan(
                parseEther('37.5') // 150 * 0.25 + lost income due to early withdraws from other variable. 
              )

              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                [lidoLastRequestId + BigInt(4)],
                addr4.address
              )

              const balanceBefore4 = await ethers.provider.getBalance(addr4)
              const receipt4 = await (
                await lidoVault.connect(addr4).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees4 = calculateGasFees(receipt4)
              const balanceAfter4 = await ethers.provider.getBalance(addr4)
              const amountWithdrawn4 = balanceAfter4 - balanceBefore4

              expect(await lidoVault.appliedProtocolFee()).to.be.equal(0)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr4.address)).to.be.greaterThan(
                parseEther('37.5')
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr4.address)
              expect(amountWithdrawn4).to.be.greaterThan(parseEther('37.5') - gasFees4)


              // addr3 withdraws
              await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)).to.be.lessThan(
                parseEther('37.5') // 150 * 0.25 - lost income due to early withdraws. 
              )

              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                [lidoLastRequestId + BigInt(5)],
                addr3.address
              )

              const balanceBefore3again = await ethers.provider.getBalance(addr3)
              const receipt3again = await (
                await lidoVault.connect(addr3).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees3again = calculateGasFees(receipt3again)
              const balanceAfter3again = await ethers.provider.getBalance(addr3)
              const amountWithdrawn3again = balanceAfter3again - balanceBefore3again

              expect(await lidoVault.appliedProtocolFee()).to.be.equal(0)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)).to.be.lessThan(
                parseEther('37.5')
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr3.address)
              expect(amountWithdrawn3again).to.be.lessThan(parseEther('37.5') - gasFees3again)

              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equalWithTolerance(
              parseEther('150'),10)

              expect(amountWithdrawn + amountWithdrawn2 + amountWithdrawn3 + amountWithdrawn4 + amountWithdrawn3again +
                gasFees + gasFees2 + gasFees3 + gasFees4 + gasFees3again 
              ).to.be.equalWithTolerance(
                parseEther('150'), 10)
            })

            it('Should have correct share of the staking earnings withdrawn if users withdraw after a fixed early withdrawal (vault-started, multiple-participants)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                lidoContract,
                addr1,
                addr2,
                addr3,
                lidoVaultAddress,
                variableBearerToken,
                fixedBearerToken,
              } = await loadFixture(startVaultAndGenerateEarningsFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // early fixed withdrawal
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              const fixedRequestIds = [lidoLastRequestId + BIG_INT_ONE]
              await finalizeLidoWithdrawalRequests(fixedRequestIds, DEFAULTS.fixedSideCapacity)
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
              expect(await lidoVault.fixedBearerTokenTotalSupply()).to.equal(0)

              // early exit fee + payback amount should have been applied
              const feeEarnings = await lidoVault.feeEarnings()
              expect(feeEarnings).to.be.greaterThan(0)

              const variableRequestIds = [lidoLastRequestId + BigInt(2)]

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
              expect(bearerTokenBalance).to.equal(variableDeposit2)

              const lidoETHAmount = await lidoContract.balanceOf(lidoVaultAddress)
              // should still have the earnings as the vaults balance
              expect(lidoETHAmount).to.be.greaterThan(0)
              const balanceBefore = await ethers.provider.getBalance(addr2)
              // withdraw #1
              const receipt0 = await (await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).wait()
              const gasFees0 = calculateGasFees(receipt0)
              const earningsShare = BigInt(
                (bearerTokenBalance * lidoETHAmount) / variableBearerTokenTotalSupply
              )
              const protocolFeeShare =
                (earningsShare * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                earningsShare - protocolFeeShare
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equal(
                earningsShare - protocolFeeShare
              ) // only withdraws their portion of the earnings
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                variableRequestIds,
                addr2.address
              )

              await finalizeLidoWithdrawalRequests(variableRequestIds, DEFAULTS.fixedSideCapacity)

             
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const amountWithdrawn = balanceAfter - balanceBefore

              // since some earnings were made protocol fee should have been applied
              expect(await lidoVault.appliedProtocolFee()).to.be.equal(protocolFeeShare)

              const feeEarningsShare =
                (bearerTokenBalance * feeEarnings) / variableBearerTokenTotalSupply
              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                feeEarningsShare
              )
              expect(await lidoVault.feeEarnings()).to.be.equal(feeEarnings - feeEarningsShare)
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(feeEarningsShare)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equalWithTolerance(
                feeEarningsShare + earningsShare - (gasFees + gasFees0 + protocolFeeShare),
                10
              )

              // withdraw #2
              const lidoETHAmount3 = await lidoContract.balanceOf(lidoVaultAddress)
              const requestIds3 = [lidoLastRequestId + BigInt(3)]

              const bearerTokenBalance3 = await lidoVault.variableBearerToken(addr3.address)
              expect(bearerTokenBalance3).to.equal(variableDeposit3)
              const balanceBefore3 = await ethers.provider.getBalance(addr3)
              const receipt2 = await (await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)).wait()
              const gasFees2 = calculateGasFees(receipt2)
              const earnings3 = lidoETHAmount3 + earningsShare // remaining lido staked amount + addr2 withdrawn amount
              const earningsShare3 = BigInt(
                (bearerTokenBalance3 * earnings3) / variableBearerTokenTotalSupply
              )
              const protocolFeeShare3 =
                (earningsShare3 * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K

              // sanity checks
              expect(earnings3).to.equalWithTolerance(lidoETHAmount, 10)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)).to.be.equalWithTolerance(
                earningsShare3 - protocolFeeShare3, 10
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equalWithTolerance(
                earningsShare + earningsShare3 - protocolFeeShare - protocolFeeShare3, 10
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds3,
                addr3.address
              )

              await finalizeLidoWithdrawalRequests(requestIds3, DEFAULTS.fixedSideCapacity)

              
              const receipt3 = await (
                await lidoVault.connect(addr3).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees3 = calculateGasFees(receipt3)
              const balanceAfter3 = await ethers.provider.getBalance(addr3)
              const amountWithdrawn3 = balanceAfter3 - balanceBefore3

              // since some earnings were made protocol fee should have been applied
              const protocolFee3 = (await lidoVault.appliedProtocolFee()) - protocolFeeShare
              expect(protocolFee3).to.be.equal(protocolFeeShare3)

              const feeEarningsShare3 = BigInt(
                (bearerTokenBalance3 * feeEarnings) / variableBearerTokenTotalSupply
              )
              expect(await lidoVault.variableToWithdrawnFees(addr3.address)).to.be.equal(
                feeEarningsShare3
              )
              expect(await lidoVault.feeEarnings()).to.be.equal(
                feeEarnings - feeEarningsShare - feeEarningsShare3
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(
                feeEarningsShare + feeEarningsShare3
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr3.address)
              expect(amountWithdrawn3).to.be.equalWithTolerance(
                feeEarningsShare3 + earningsShare3 - (gasFees3 + gasFees2 + protocolFee3), 10
              )
            })
          })
        })

        describe('Fixed Early Exit Fees with Lido Staking Earnings', function () {
          describe('Single Variable Participant', function () {
            async function startVaultAndFixedEarlyWithdrawWithStakingEarningsFixture() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2 } = state

              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })

              await lidoVault.connect(addr1).claimFixedPremium()

              await submitOracleReport()

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // early fixed withdrawal
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              const fixedRequestIds = [lidoLastRequestId + BIG_INT_ONE]
              await finalizeLidoWithdrawalRequests(fixedRequestIds, DEFAULTS.fixedSideCapacity)
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()

              return state
            }

            it('Should have correct share of the fees / earnings (vault-started)', async function () {
              const {
                lidoVault,
                lidoVaultAddress,
                lidoWithdrawalQueueContract,
                lidoContract,
                addr2,
                variableBearerToken,
              } = await loadFixture(startVaultAndFixedEarlyWithdrawWithStakingEarningsFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              const lidoETHAmount = (await lidoContract.balanceOf(lidoVaultAddress)) as bigint
              const protocolFee = (lidoETHAmount * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K

              // early exit fee + payback amount should have been applied
              const feeEarnings = await lidoVault.feeEarnings()
              expect(feeEarnings).to.be.greaterThan(0)

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              expect(bearerTokenBalance).to.equal(variableDeposit)

              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt0 = await (
                await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
              ).wait()
              const gasFees0 = calculateGasFees(receipt0)
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                lidoETHAmount - protocolFee
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equal(
                lidoETHAmount - protocolFee
              ) // gets all the earnings since they are the only variable depositor
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr2.address
              )

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              
              
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
              const amountWithdrawn = balanceAfter - balanceBefore
              const vaultBalanceDiff = vaultBalanceAfter - vaultBalanceBefore

              // since some earnings were made protocol fee should have been applied
              expect(await lidoVault.appliedProtocolFee()).to.be.equal(protocolFee)

              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                feeEarnings
              )
              expect(await lidoVault.feeEarnings()).to.be.equal(0)
              // withdrew all the fee earnings
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(feeEarnings)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equalWithTolerance(
                feeEarnings + lidoETHAmount - (gasFees + gasFees0 + protocolFee),  // gets all of feeEarnings + lidoETHAmount since they are the only variable depositor
                10
              )
              expect(vaultBalanceDiff).to.be.equal(protocolFee - feeEarnings)
            })
          })

          describe('Multiple Variable Participants', function () {
            const variableDeposit2 = parseEther('10')
            const variableDeposit3 = parseEther('20')

            async function startVaultAndFixedEarlyWithdrawWithStakingEarningsFixture() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2, addr3 } = state

              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })

              await lidoVault.connect(addr1).claimFixedPremium()

              await submitOracleReport()

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // early fixed withdrawal
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              const fixedRequestIds = [lidoLastRequestId + BIG_INT_ONE]
              await finalizeLidoWithdrawalRequests(fixedRequestIds, DEFAULTS.fixedSideCapacity)
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()

              return state
            }

            it('Should have correct share of the fees / earnings (vault-started, multiple-participants)', async function () {
              const {
                lidoVault,
                lidoVaultAddress,
                lidoWithdrawalQueueContract,
                lidoContract,
                deployer,
                addr2,
                addr3,
                variableBearerToken,
              } = await loadFixture(startVaultAndFixedEarlyWithdrawWithStakingEarningsFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              const lidoETHAmount = (await lidoContract.balanceOf(lidoVaultAddress)) as bigint

              // early exit fee + payback amount should have been applied
              const feeEarnings = await lidoVault.feeEarnings()
              expect(feeEarnings).to.be.greaterThan(0)

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              expect(bearerTokenBalance).to.equal(variableDeposit2)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()

              // withdraw #1
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt0 = await ( await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).wait()
              const gasFees0 = calculateGasFees(receipt0)
              const earningsShare =
                (bearerTokenBalance * lidoETHAmount) / variableBearerTokenTotalSupply
              const protocolFeeShare =
                (earningsShare * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                earningsShare - protocolFeeShare
              )
              expect(await lidoVault.withdrawnStakingEarnings()).to.be.equal(
                earningsShare - protocolFeeShare
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr2.address
              )

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)
              const amountWithdrawn = balanceAfter - balanceBefore
              const vaultBalanceDiff = vaultBalanceAfter - vaultBalanceBefore

              // since some earnings were made protocol fee should have been applied
              expect(await lidoVault.appliedProtocolFee()).to.be.equal(protocolFeeShare)

              const feeEarningsShare =
                (bearerTokenBalance * feeEarnings) / variableBearerTokenTotalSupply
              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                feeEarningsShare
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(feeEarningsShare)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equalWithTolerance(
                feeEarningsShare + earningsShare - (gasFees + gasFees0 + protocolFeeShare),
                10
              )
              expect(vaultBalanceDiff).to.be.equal(protocolFeeShare - feeEarningsShare)

              // withdraw #2
              const requestIds3 = [lidoLastRequestId + BigInt(2)]

              const bearerTokenBalance3 = await lidoVault.variableBearerToken(addr3.address)
              expect(bearerTokenBalance3).to.equal(variableDeposit3)

              const vaultBalanceBefore3 = await ethers.provider.getBalance(lidoVaultAddress)
              const balanceBefore3 = await ethers.provider.getBalance(addr3)
              const receipt2 = await ( await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)).wait()
              const gasFees2 = calculateGasFees(receipt2)
              

              const earningsShare3 =
                (bearerTokenBalance3 * lidoETHAmount) / variableBearerTokenTotalSupply
              const protocolFeeShare3 =
                (earningsShare3 * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K

              expect(
                await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)
              ).to.equalWithTolerance(earningsShare3 - protocolFeeShare3, 1)
              expect(await lidoVault.withdrawnStakingEarnings()).to.equalWithTolerance(
                earningsShare + earningsShare3 - protocolFeeShare - protocolFeeShare3,
                1
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds3,
                addr3.address
              )

              await finalizeLidoWithdrawalRequests(requestIds3, DEFAULTS.fixedSideCapacity)

              
              
              const receipt3 = await (
                await lidoVault.connect(addr3).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees3 = calculateGasFees(receipt3)
              const balanceAfter3 = await ethers.provider.getBalance(addr3)
              const vaultBalanceAfter3 = await ethers.provider.getBalance(lidoVaultAddress)
              const amountWithdrawn3 = balanceAfter3 - balanceBefore3
              const vaultBalanceDiff3 = vaultBalanceAfter3 - vaultBalanceBefore3

              // since some earnings were made protocol fee should have been applied
              expect(await lidoVault.appliedProtocolFee()).to.be.equal(
                protocolFeeShare + protocolFeeShare3
              )

              const feeEarningsShare3 =
                (bearerTokenBalance3 * feeEarnings) / variableBearerTokenTotalSupply
              expect(await lidoVault.variableToWithdrawnFees(addr3.address)).to.be.equal(
                feeEarningsShare3
              )
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(
                feeEarningsShare + feeEarningsShare3
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr3.address)
              expect(amountWithdrawn3).to.equalWithTolerance(
                feeEarningsShare3 + earningsShare3 - (gasFees2 + gasFees3 + protocolFeeShare3),
                1
              )
              expect(vaultBalanceDiff3).to.be.equal(protocolFeeShare3 - feeEarningsShare3)
            })
          })
        })
      })

      describe('Fixed And Variable Side', function () {
        it('Should be able to withdraw as normal (vault-started, fixed-and-variable-side)', async function () {
          const {
            lidoVault,
            lidoWithdrawalQueueContract,
            addr1,
            fixedETHDepositToken,
            fixedBearerToken,
            variableBearerToken,
            lidoContract,
            lidoVaultAddress,
          } = await loadFixture(deployLidoVaultFixture)

          await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
          await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: variableDeposit })

          await lidoVault.connect(addr1).claimFixedPremium()

          expect(await lidoVault.isStarted()).to.equal(true)

          // generate earnings
          await submitOracleReport()

          const { currentTimestamp } = await getTimeState(lidoVault)
          const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

          // fixed side withdraw
          await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

          const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
            lidoVault,
            upfrontPremium: variableDeposit,
            timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
          })
          expect(await lidoVault.fixedBearerToken(addr1.address)).to.be.equal(0)

          const requestIds = [lidoLastRequestId + BIG_INT_ONE]
          await expectVaultOngoingFixedWithdrawalArraysAreEqual(
            lidoVault,
            requestIds,
            addr1.address,
            currentTimestamp
          )

          expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

          const balanceBefore = await ethers.provider.getBalance(addr1)
          await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
          const receipt = await (
            await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
          ).wait()
          const gasFees = calculateGasFees(receipt)
          const balanceAfter = await ethers.provider.getBalance(addr1)
          const amountWithdrawn = balanceAfter - balanceBefore

          expect(amountWithdrawn).to.equalWithTolerance(
            fixedDeposit - (payBackAmount + scaledEarlyExitFee) - gasFees,
            1
          )

          // variable side withdraw
          const lidoETHAmount = await lidoContract.balanceOf(lidoVaultAddress)
          const balanceBeforeVariable = await ethers.provider.getBalance(addr1)
          const receipt0 = await (await lidoVault.connect(addr1).withdraw(SIDE.VARIABLE)).wait()
          const gasFees0 = calculateGasFees(receipt0)
          // fixed side already withdrew do not need to subtract fixedDeposit
          const earnings = lidoETHAmount
          const protocolFee = (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
          const earningsAndFees = lidoETHAmount + (payBackAmount + scaledEarlyExitFee) - protocolFee

          const variableRequestIds = [lidoLastRequestId + BigInt(2)]
          await expectVaultOngoingVariableWithdrawalArraysAreEqual(
            lidoVault,
            variableRequestIds,
            addr1.address
          )

          await finalizeLidoWithdrawalRequests(variableRequestIds, DEFAULTS.fixedSideCapacity)
          const receiptVariable = await (
            await lidoVault.connect(addr1).finalizeVaultOngoingVariableWithdrawals()
          ).wait()
          const gasFeesVariable = calculateGasFees(receiptVariable)
          const balanceAfterVariable = await ethers.provider.getBalance(addr1)
          const amountWithdrawnVariable = balanceAfterVariable - balanceBeforeVariable

          expect(amountWithdrawnVariable).to.equalWithTolerance(earningsAndFees - gasFeesVariable - gasFees0, 10)
        })
      })
    })
  })

  describe('Vault Ended', function () {
    const fixedDeposit = DEFAULTS.fixedSideCapacity
    const variableDeposit = parseEther('30')

    async function endVaultFixture() {
      const state = await loadFixture(deployLidoVaultFixture)

      const { lidoVault, addr1, addr2 } = state

      await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
      await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })

      await lidoVault.connect(addr1).claimFixedPremium()

      const { endTime } = await getTimeState(lidoVault)
      await time.increaseTo(endTime + BIG_INT_ONE)

      return state
    }

    it('isStarted should still be set to true (vault-ended)', async function () {
      const { lidoVault } = await loadFixture(endVaultFixture)

      expect(await lidoVault.isStarted()).to.equal(true)
    })

    it('isEnded should be set to true (vault-ended)', async function () {
      const { lidoVault } = await loadFixture(endVaultFixture)

      expect(await lidoVault.isEnded()).to.equal(true)
    })

    it('Deposits should fail (vault-ended)', async function () {
      const { lidoVault, addr1, addr2 } = await loadFixture(endVaultFixture)

      await expect(
        lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: parseEther('20') })
      ).to.be.revertedWith('DAS')
      await expect(
        lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: parseEther('200') })
      ).to.be.revertedWith('DAS')
    })

    it('Should emit the correct vault ended event (vault-ended, events)', async function () {
      const { lidoVault, addr1 } = await loadFixture(endVaultFixture)

      const withdrawTrx = await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

      await expect(withdrawTrx).to.emit(lidoVault, 'VaultEnded').withArgs(anyUint, addr1.address)
    })

    describe('Withdraw', function () {
      it('Should fail on invalid side (vault-ended)', async function () {
        const { lidoVault, addr1 } = await loadFixture(endVaultFixture)

        await expect(lidoVault.connect(addr1).withdraw(3)).to.be.revertedWith('IS')
      })

      it('Should fail to finalize the vault ended withdrawal with an invalid side (vault-ended)', async function () {
        const { lidoVault, addr1 } = await loadFixture(endVaultFixture)

        await expect(lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(5)).to.be.revertedWith(
          'IS'
        )
      })

      it('Should fail to withdraw if vault ended withdraw requests have not been finalized (vault-ended)', async function () {
        const { lidoVault, addr1 } = await loadFixture(endVaultFixture)

        // kick of vault ended withdraw
        await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

        await expect(lidoVault.connect(addr1).withdraw(SIDE.FIXED)).to.be.revertedWith('WNF')
      })

      describe('Fixed Side', function () {
        it('Should emit the correct fixed events if they are first to withdraw and finalize (vault-ended, events)', async function () {
          const { lidoVault, addr1, lidoWithdrawalQueueContract } = await loadFixture(
            endVaultFixture
          )

          const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
          const requestIds = [lidoLastRequestId + BIG_INT_ONE]

          const withdrawTrx = await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

          await expect(withdrawTrx)
            .to.emit(lidoVault, 'LidoWithdrawalRequested')
            .withArgs(addr1.address, anyValue, SIDE.FIXED, true, true)

          await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

          const finalizeTrx = await lidoVault
            .connect(addr1)
            .finalizeVaultEndedWithdrawals(SIDE.FIXED)

          await expect(finalizeTrx)
            .to.emit(lidoVault, 'LidoWithdrawalFinalized')
            .withArgs(addr1.address, anyValue, SIDE.FIXED, true, true)
          await expect(finalizeTrx)
            .to.emit(lidoVault, 'FixedFundsWithdrawn')
            .withArgs(anyUint, addr1.address, true, true)
        })

        it('Should emit the correct fixed withdrawn event if they are not first to withdraw (vault-ended, events)', async function () {
          const { lidoVault, addr1, addr2, lidoWithdrawalQueueContract } = await loadFixture(
            endVaultFixture
          )

          const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
          const requestIds = [lidoLastRequestId + BIG_INT_ONE]
          await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
          await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
          await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)

          const withdrawTrx = await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

          await expect(withdrawTrx)
            .to.emit(lidoVault, 'FixedFundsWithdrawn')
            .withArgs(anyUint, addr1.address, true, true)
        })

        describe('Single Fixed Participant', function () {
          it('Should withdraw all lido staked ETH then be given back their initial deposit if they are first to withdraw and finalize (vault-ended)', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1,
              fixedBearerToken,
              fixedETHDepositToken,
            } = await loadFixture(endVaultFixture)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

            // should not have withdrawn funds yet
            const bearerTokenBalance = await lidoVault.fixedBearerToken(addr1.address)
            expect(bearerTokenBalance).to.be.greaterThan(0)
            expect(bearerTokenBalance).to.equal(await lidoVault.fixedBearerTokenTotalSupply())
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDeposit)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[0]
            )

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
            expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit - gasFees, 1)
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
          })

          it('Should fail to withdraw if they have already withdrawn (vault-ended)', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1,
              fixedBearerToken,
              fixedETHDepositToken,
            } = await loadFixture(endVaultFixture)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
            await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)

            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

            await expect(lidoVault.connect(addr1).withdraw(SIDE.FIXED)).to.be.revertedWith('NBT')
          })

          it('Should withdraw all lido staked ETH then be given back their initial deposit if they withdraw after variable (vault-ended)', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1,
              addr2,
              fixedBearerToken,
              fixedETHDepositToken,
            } = await loadFixture(endVaultFixture)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[0]
            )
            await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)

            // should not have withdrawn funds yet
            const bearerTokenBalance = await lidoVault.fixedBearerToken(addr1.address)
            expect(bearerTokenBalance).to.be.greaterThan(0)
            expect(bearerTokenBalance).to.equal(await lidoVault.fixedBearerTokenTotalSupply())
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDeposit)

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (await lidoVault.connect(addr1).withdraw(SIDE.FIXED)).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit - gasFees, 1)
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
          })

          it('Should withdraw all lido staked ETH then be given back their initial deposit if variable withdraws first but fixed calls finalization (vault-ended)', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1,
              addr2,
              fixedBearerToken,
              fixedETHDepositToken,
            } = await loadFixture(endVaultFixture)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

            // should not have withdrawn funds yet
            const bearerTokenBalance = await lidoVault.fixedBearerToken(addr1.address)
            expect(bearerTokenBalance).to.be.greaterThan(0)
            expect(bearerTokenBalance).to.equal(await lidoVault.fixedBearerTokenTotalSupply())
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDeposit)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[0]
            )

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
            expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit - gasFees, 1)
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
          })

          it('Should get less than their initial deposit in the case of staking balance decreasing (vault-ended, staking-loss)', async function () {
            const { lidoVault, lidoMock, addr1, addr2, fixedBearerToken } = await loadFixture(
              deployLidoVaultWithMockedLidoFixture
            )
            const lidoLastRequestId = BIG_INT_ZERO

            // start vault
            const variableDeposit = parseEther('20')
            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
            await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })
            await lidoVault.connect(addr1).claimFixedPremium()

            expect(await lidoVault.isStarted()).to.be.equal(true)

            const { endTime } = await getTimeState(lidoVault)
            await time.increaseTo(endTime + BIG_INT_ONE)

            // end vault
            expect(await lidoVault.isEnded()).to.be.equal(true)
            const shares = await lidoVault.fixedBearerToken(addr1.address)
            expect(shares).to.equal(parseEther('1000'))

            const stakingStETHBalanceBefore = await lidoMock.getPooledEthByShares(shares)

            // mock staking loss - 100 ETH
            const stakingEarnings = parseEther('100')
            await lidoMock.subtractStakingEarnings(stakingEarnings)

            const stakingStETHBalanceAfter = await lidoMock.getPooledEthByShares(shares)
            // eth = _sharesAmount.mulDiv(totalPooledEther, totalShares) = (1000 * (2000 - 100)) / 2000 = 950 - 50 ETH loss from original deposit
            const stakingLoss = parseEther('50')
            expect(stakingStETHBalanceBefore - stakingStETHBalanceAfter).to.equal(stakingLoss)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(balanceAfter - balanceBefore).to.be.equal(fixedDeposit - stakingLoss - gasFees)
            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, [])

            // no protocol fee should have been applied since the vault has not started
            const protocolFee = await lidoVault.appliedProtocolFee()
            expect(protocolFee).to.be.equal(0)
          })

          describe('Withdraw requested before vault start', () => {
            async function withdrawBeforeVaultStart() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2, addr3 } = state

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              // deposit
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })

              // withdraw
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              await expectVaultNotStartedWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr1.address
              )
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              // vault started
              await lidoVault.connect(addr3).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })
              await lidoVault.connect(addr3).claimFixedPremium()

              // end vault
              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)
              expect(await lidoVault.isEnded()).to.equal(true)

              return state
            }

            it('Should be able to kickoff vault ended withdrawal but cannot finalize (vault-ended, withdraw-before-vault-started)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                addr1,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(withdrawBeforeVaultStart)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // can kick off vault end withdraw process
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )

              // cant call finalize due to ongoing withdrawal request
              await expect(
                lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
              ).to.be.revertedWith('WAR')
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              // can finalize ongoing withdrawal request
              const balanceBefore = await ethers.provider.getBalance(addr1)
              const receipt = await (
                await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)

              expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit - gasFees, 1) // no extra fees since they withdrew before the vault started
              expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
              expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            })

            it('Should be able to withdraw after someone else finalizes (vault-ended, withdraw-before-vault-started)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                addr1,
                addr2,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(withdrawBeforeVaultStart)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              const requestIds = [lidoLastRequestId + BigInt(1)]
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )

              await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)

              const balanceBefore = await ethers.provider.getBalance(addr1)
              const receipt = await (
                await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)

              expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit - gasFees, 1) // no extra fees since they withdrew before the vault started
              expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
              expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(0)
            })
          })

          describe('Withdraw requested before vault end', () => {
            async function withdrawBeforeVaultEnd() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2 } = state

              // vault started
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })
              await lidoVault.connect(addr1).claimFixedPremium()

              // staking earnings
              await submitOracleReport()

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              // withdraw
              const { currentTimestamp } = await getTimeState(lidoVault)
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              await expectVaultOngoingFixedWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr1.address,
                currentTimestamp
              )
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              // end vault
              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)
              expect(await lidoVault.isEnded()).to.equal(true)

              return state
            }

            it('Should be able to kickoff vault ended withdrawal and finalize (vault-ended, withdraw-before-vault-ended)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                addr1,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(withdrawBeforeVaultEnd)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                lidoVault,
                upfrontPremium: variableDeposit,
                timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
              })

              // can kick off vault end withdraw process
              const requestIds = [lidoLastRequestId + BigInt(1)]
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )

              // should be able to finalize their ongoing withdrawal request and the vault end
              const balanceBefore = await ethers.provider.getBalance(addr1)
              const receipt = await (
                await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              expect(balanceAfter - balanceBefore).to.equalWithTolerance(
                fixedDeposit - (gasFees + payBackAmount + scaledEarlyExitFee), 10
              )
              expect(await lidoVault.fixedToPendingWithdrawalAmount(addr1.address)).to.equal(0)
              await expectVaultOngoingFixedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)
              expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
            })

            it('Should have no errors if multiple people had an ongoing vault withdrawal but only one finalized before vault end (vault-ended, withdraw-before-vault-ended)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                addr1,
                addr2,
                addr3,
                addr4,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(deployLidoVaultFixture)

              // vault started
              await lidoVault
                .connect(addr1)
                .deposit(SIDE.FIXED, { value: fixedDeposit / BigInt(2) })
              await lidoVault
                .connect(addr2)
                .deposit(SIDE.FIXED, { value: fixedDeposit / BigInt(4) })
              await lidoVault
                .connect(addr3)
                .deposit(SIDE.FIXED, { value: fixedDeposit / BigInt(4) })
              await lidoVault.connect(addr4).deposit(SIDE.VARIABLE, { value: variableDeposit })
              await lidoVault.connect(addr1).claimFixedPremium()
              await lidoVault.connect(addr2).claimFixedPremium()
              await lidoVault.connect(addr3).claimFixedPremium()

              // staking earnings
              await submitOracleReport()

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              // withdraw #1
              const { currentTimestamp } = await getTimeState(lidoVault)
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              await expectVaultOngoingFixedWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr1.address,
                currentTimestamp
              )
              await finalizeLidoWithdrawalRequests(
                requestIds,
                DEFAULTS.fixedSideCapacity / BigInt(2)
              )

              const requestIds2 = [lidoLastRequestId + BigInt(2)]

              // withdraw #2
              const { currentTimestamp: currentTimestamp2 } = await getTimeState(lidoVault)
              await lidoVault.connect(addr2).withdraw(SIDE.FIXED)
              await expectVaultOngoingFixedWithdrawalArraysAreEqual(
                lidoVault,
                requestIds2,
                addr2.address,
                currentTimestamp2
              )
              await finalizeLidoWithdrawalRequests(
                requestIds2,
                DEFAULTS.fixedSideCapacity / BigInt(2)
              )
              await lidoVault.connect(addr2).finalizeVaultOngoingFixedWithdrawals()

              // end vault
              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)
              expect(await lidoVault.isEnded()).to.equal(true)

              // can kick off vault end withdraw process
              const requestIds3 = [lidoLastRequestId + BigInt(3)]
              await lidoVault.connect(addr3).withdraw(SIDE.FIXED)

              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds3)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds3, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds3[0]
              )

              // should be able to finalize their ongoing withdrawal request and the vault end
              const balanceBefore = await ethers.provider.getBalance(addr3)
              const receipt = await (
                await lidoVault.connect(addr3).finalizeVaultEndedWithdrawals(SIDE.FIXED)
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr3)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              expect(balanceAfter - balanceBefore).to.equal(fixedDeposit / BigInt(4) - gasFees)
              await expectVaultOngoingFixedWithdrawalArraysAreEqual(lidoVault, [], addr3.address)
              expect(await lidoVault.fixedBearerToken(addr3.address)).to.equal(0)
            })

            it('Should be able to withdraw after someone else finalizes (vault-ended, withdraw-before-vault-ended)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                addr1,
                addr2,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(withdrawBeforeVaultEnd)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                lidoVault,
                upfrontPremium: variableDeposit,
                timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
              })

              // can kick off vault end withdraw process
              const requestIds = [lidoLastRequestId + BigInt(1)]
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )

              await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              expect(await lidoVault.fixedToPendingWithdrawalAmount(addr1.address)).to.equalWithTolerance(
                fixedDeposit - (payBackAmount + scaledEarlyExitFee), 10
              )
              await expectVaultOngoingFixedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)
              expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

              // ongoing withdrawal should have been claimed in finalizeVaultEndedWithdrawals
              await expect(
                lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
              ).to.be.revertedWith('WNR')

              const balanceBefore = await ethers.provider.getBalance(addr1)
              const receipt = await (await lidoVault.connect(addr1).withdraw(SIDE.FIXED)).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)

              expect(balanceAfter - balanceBefore).to.equalWithTolerance(
                fixedDeposit - (gasFees + payBackAmount + scaledEarlyExitFee), 10
              )
              expect(await lidoVault.fixedToPendingWithdrawalAmount(addr1.address)).to.equal(0)
            })
          })
        })

        describe('Multiple Fixed Participants', function () {
          const fixedDeposit1 = parseEther('600')
          const fixedDeposit2 = parseEther('400')
          const stakingEarnings = parseEther('200')
          async function multipleFixedParticipantsEndVaultFixture() {
            const state = await loadFixture(deployLidoVaultFixture)

            const { lidoVault, addr1, addr2, addr3 } = state

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
            await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })
            await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit })

            await lidoVault.connect(addr1).claimFixedPremium()
            await lidoVault.connect(addr2).claimFixedPremium()

            const { endTime } = await getTimeState(lidoVault)
            await time.increaseTo(endTime + BIG_INT_ONE)

            return state
          }

          async function multipleFixedParticipantsAndStakingEndVaultFixture() {
            const state = await loadFixture(deployLidoVaultFixture)

            const { lidoVault, addr1, addr2, addr3, lidoVaultAddress } = state
            const { lidoMock, lidoWithdrawalQueueMock } = await setupMockLidoContracts(lidoVault)
            await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })
             
            await lidoMock.addStakingEarningsForTargetETH(
              fixedDeposit2 + stakingEarnings,
              lidoVaultAddress
            )

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
            
            await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit })

            await lidoVault.connect(addr1).claimFixedPremium()
            await lidoVault.connect(addr2).claimFixedPremium()

            const { endTime } = await getTimeState(lidoVault)
            await time.increaseTo(endTime + BIG_INT_ONE)

            return state
          }

          it('Should withdraw all lido staked ETH then be given back their initial deposit if they are first to withdraw and finalize (vault-ended, multiple-participants)', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1,
              addr2,
              fixedBearerToken,
              fixedETHDepositToken,
            } = await loadFixture(multipleFixedParticipantsEndVaultFixture)

            const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

            // should not have withdrawn funds yet
            const bearerTokenBalance = await lidoVault.fixedBearerToken(addr1.address)
            expect(bearerTokenBalance).to.be.greaterThan(0)
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDeposit1)

            await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
            expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
              requestIds[0]
            )

            // withdraw #1
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)

            expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit1 - gasFees, 2)
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

            // should not have touched addr2 funds
            const bearerTokenBalance2 = await lidoVault.fixedBearerToken(addr2.address)
            expect(bearerTokenBalance2).to.be.greaterThan(0)
            expect(await lidoVault.fixedETHDepositToken(addr2.address)).to.equal(fixedDeposit2)

            // withdraw #2
            const balanceBefore2 = await ethers.provider.getBalance(addr2)
            const receipt2 = await (await lidoVault.connect(addr2).withdraw(SIDE.FIXED)).wait()
            const gasFees2 = calculateGasFees(receipt2)
            const balanceAfter2 = await ethers.provider.getBalance(addr2)

            expect(balanceAfter2 - balanceBefore2).to.equal(fixedDeposit2 - gasFees2)
            expect(await lidoVault.fixedBearerToken(addr2.address)).to.equal(0)
          })

          it('Should withdraw all lido staked ETH then be given back their initial deposit + staking bonus if they are first to withdraw and finalize (vault-ended, multiple-participants)', async function () {
            const {
              lidoVault,
              lidoWithdrawalQueueContract,
              addr1,
              addr2,
              fixedBearerToken,
              fixedETHDepositToken,
            } = await loadFixture(multipleFixedParticipantsAndStakingEndVaultFixture)

            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

            // should not have withdrawn funds yet
            const bearerTokenBalance = await lidoVault.fixedBearerToken(addr1.address)
            expect(bearerTokenBalance).to.be.greaterThan(0)
            expect(await lidoVault.fixedETHDepositToken(addr1.address)).to.equal(fixedDeposit1)

            // withdraw #1
            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)

            expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit1 - gasFees, 2)
            expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

            // should not have touched addr2 funds
            const bearerTokenBalance2 = await lidoVault.fixedBearerToken(addr2.address)
            expect(bearerTokenBalance2).to.be.greaterThan(0)
            expect(await lidoVault.fixedETHDepositToken(addr2.address)).to.equal(fixedDeposit2)

            // withdraw #2
            const balanceBefore2 = await ethers.provider.getBalance(addr2)
            const receipt2 = await (await lidoVault.connect(addr2).withdraw(SIDE.FIXED)).wait()
            const gasFees2 = calculateGasFees(receipt2)
            const balanceAfter2 = await ethers.provider.getBalance(addr2)

            // All staking earnings should have gone to addr2
            expect(balanceAfter2 - balanceBefore2).to.equal(fixedDeposit2 + stakingEarnings - gasFees2)
            expect(await lidoVault.fixedBearerToken(addr2.address)).to.equal(0)
          })

          it('Should get less than their initial deposit in the case of staking balance decreasing (vault-ended, multiple-participants, staking-loss)', async function () {
            const { lidoVault, lidoMock, addr1, addr2, addr3, fixedBearerToken } =
              await loadFixture(deployLidoVaultWithMockedLidoFixture)

            const lidoLastRequestId = BIG_INT_ZERO

            // start vault
            const variableDeposit = parseEther('20')

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
            await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })
            await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit })

            await lidoVault.connect(addr1).claimFixedPremium()
            await lidoVault.connect(addr2).claimFixedPremium()

            expect(await lidoVault.isStarted()).to.be.equal(true)

            const { endTime } = await getTimeState(lidoVault)
            await time.increaseTo(endTime + BIG_INT_ONE)

            // end vault
            expect(await lidoVault.isEnded()).to.be.equal(true)

            const shares1 = await lidoVault.fixedBearerToken(addr1.address)
            const shares2 = await lidoVault.fixedBearerToken(addr2.address)
            expect(shares1).to.equal(parseEther('600'))
            expect(shares2).to.equal(parseEther('400'))

            const stakingStETHBalanceBefore1 = await lidoMock.getPooledEthByShares(shares1)
            const stakingStETHBalanceBefore2 = await lidoMock.getPooledEthByShares(shares2)

            // mock staking loss - 100 ETH
            const stakingEarnings = parseEther('100')
            await lidoMock.subtractStakingEarnings(stakingEarnings)

            const stakingStETHBalanceAfter1 = await lidoMock.getPooledEthByShares(shares1)
            const stakingStETHBalanceAfter2 = await lidoMock.getPooledEthByShares(shares2)
            // eth = _sharesAmount.mulDiv(totalPooledEther, totalShares) = (600 * (2000 - 100)) / 2000 = 570 - 30 ETH loss from original deposit
            const stakingLoss1 = parseEther('30')
            expect(stakingStETHBalanceBefore1 - stakingStETHBalanceAfter1).to.equal(stakingLoss1)
            // eth = _sharesAmount.mulDiv(totalPooledEther, totalShares) = (400 * (2000 - 100)) / 2000 = 380 - 20 ETH loss from original deposit
            const stakingLoss2 = parseEther('20')
            expect(stakingStETHBalanceBefore2 - stakingStETHBalanceAfter2).to.equal(stakingLoss2)

            // addr1 withdraws
            const requestIds = [lidoLastRequestId + BIG_INT_ONE]
            await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)

            const balanceBefore = await ethers.provider.getBalance(addr1)
            const receipt = await (
              await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr1)

            expect(balanceAfter - balanceBefore).to.be.equal(fixedDeposit1 - stakingLoss1 - gasFees)
            await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, [])

            // no protocol fee should have been applied since the vault has not started
            const protocolFee = await lidoVault.appliedProtocolFee()
            expect(protocolFee).to.be.equal(0)

            // addr2 withdraws
            const balanceBefore2 = await ethers.provider.getBalance(addr2)
            const receipt2 = await (await lidoVault.connect(addr2).withdraw(SIDE.FIXED)).wait()
            const gasFees2 = calculateGasFees(receipt2)
            const balanceAfter2 = await ethers.provider.getBalance(addr2)

            expect(balanceAfter2 - balanceBefore2).to.be.equal(
              fixedDeposit2 - stakingLoss2 - gasFees2
            )
          })

          describe('Withdraw requested before vault start', () => {
            const fixedDeposit1 = parseEther('600')
            const fixedDeposit2 = parseEther('600')
            const fixedDeposit3 = parseEther('400')

            async function multipleParticipantsWithdrawBeforeVaultStart() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2, addr3, addr4 } = state

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              // deposit
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })

              // withdraw
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              await expectVaultNotStartedWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr1.address
              )
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              // vault started
              await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.FIXED, { value: fixedDeposit3 })
              await lidoVault.connect(addr4).deposit(SIDE.VARIABLE, { value: variableDeposit })
              await lidoVault.connect(addr2).claimFixedPremium()
              await lidoVault.connect(addr3).claimFixedPremium()

              // end vault
              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)
              expect(await lidoVault.isEnded()).to.equal(true)

              return state
            }

            it('Should be able to kickoff vault ended withdrawal but cannot finalize (vault-ended, withdraw-before-vault-started, multiple-participants)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                addr1,
                addr2,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(multipleParticipantsWithdrawBeforeVaultStart)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // can kick off vault end withdraw process
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )

              // cant call finalize due to vault not started withdrawal request
              await expect(
                lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
              ).to.be.revertedWith('WAR')
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              // can finalize vault not started withdrawal request
              const balanceBefore = await ethers.provider.getBalance(addr1)
              const receipt = await (
                await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)

              expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit1 - gasFees, 1) // no extra fees since they withdrew before the vault started
              expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

              // addr2 finalizes vault ended withdrawals
              const balanceBefore2 = await ethers.provider.getBalance(addr2)
              const receipt2 = await (
                await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.FIXED)
              ).wait()
              const gasFees2 = calculateGasFees(receipt2)
              const balanceAfter2 = await ethers.provider.getBalance(addr2)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              expect(balanceAfter2 - balanceBefore2).to.equalWithTolerance(
                fixedDeposit2 - gasFees2,
                1
              ) // no extra fees since they withdrew after vault end
              expect(await lidoVault.fixedBearerToken(addr2.address)).to.equal(0)
            })

            it('Should be able to withdraw after someone else finalizes (vault-ended, withdraw-before-vault-started, multiple-participants)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                addr1,
                addr2,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(multipleParticipantsWithdrawBeforeVaultStart)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              const requestIds = [lidoLastRequestId + BigInt(1)]
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )

              // addr2 finalizes vault ended withdrawals
              const balanceBefore2 = await ethers.provider.getBalance(addr2)
              const receipt2 = await (
                await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.FIXED)
              ).wait()
              const gasFees2 = calculateGasFees(receipt2)
              const balanceAfter2 = await ethers.provider.getBalance(addr2)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              expect(balanceAfter2 - balanceBefore2).to.equalWithTolerance(
                fixedDeposit2 - gasFees2,
                1
              ) // no extra fees since they withdrew after vault end
              expect(await lidoVault.fixedBearerToken(addr2.address)).to.equal(0)

              const balanceBefore = await ethers.provider.getBalance(addr1)
              const receipt = await (
                await lidoVault.connect(addr1).finalizeVaultNotStartedFixedWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)

              expect(balanceAfter - balanceBefore).to.equalWithTolerance(fixedDeposit1 - gasFees, 1) // no extra fees since they withdrew before the vault started
              expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)
            })
          })

          describe('Withdraw requested before vault end', () => {
            const fixedDeposit1 = parseEther('600')
            const fixedDeposit2 = parseEther('400')
            const upfrontPremium1 = (variableDeposit * BigInt(6)) / BigInt(10)

            async function multipleParticipantsWithdrawBeforeVaultEnd() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2, addr3 } = state

              // vault started
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit1 })
              await lidoVault.connect(addr2).deposit(SIDE.FIXED, { value: fixedDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit })
              await lidoVault.connect(addr1).claimFixedPremium()
              await lidoVault.connect(addr2).claimFixedPremium()

              // staking earnings
              await submitOracleReport()

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]

              // withdraw
              const { currentTimestamp } = await getTimeState(lidoVault)
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              await expectVaultOngoingFixedWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr1.address,
                currentTimestamp
              )
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

              // end vault
              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)
              expect(await lidoVault.isEnded()).to.equal(true)

              return state
            }

            it('Should be able to kickoff vault ended withdrawal and can finalize (vault-ended, withdraw-before-vault-ended, multiple-participants)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                addr1,
                addr2,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(multipleParticipantsWithdrawBeforeVaultEnd)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                lidoVault,
                upfrontPremium: upfrontPremium1,
                timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
              })

              // can kick off vault end withdraw process
              const requestIds = [lidoLastRequestId + BigInt(1)]
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )

              // can finalize ongoing withdrawal request and vault end
              const balanceBefore = await ethers.provider.getBalance(addr1)
              const receipt = await (
                await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              expect(balanceAfter - balanceBefore).to.equalWithTolerance(
                fixedDeposit1 - (gasFees + payBackAmount + scaledEarlyExitFee), 10
              )
              expect(await lidoVault.fixedToPendingWithdrawalAmount(addr1.address)).to.equal(0)
              await expectVaultOngoingFixedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)
              expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

              // addr2 can just withdraw
              const balanceBefore2 = await ethers.provider.getBalance(addr2)
              const receipt2 = await (await lidoVault.connect(addr2).withdraw(SIDE.FIXED)).wait()
              const gasFees2 = calculateGasFees(receipt2)
              const balanceAfter2 = await ethers.provider.getBalance(addr2)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              expect(balanceAfter2 - balanceBefore2).to.equal(fixedDeposit2 - gasFees2) // no extra fees since they withdrew after vault end
              expect(await lidoVault.fixedBearerToken(addr2.address)).to.equal(0)
            })

            it('Should be able to withdraw after someone else finalizes (vault-ended, withdraw-before-vault-ended, multiple-participants)', async function () {
              const {
                lidoVault,
                lidoWithdrawalQueueContract,
                addr1,
                addr2,
                fixedBearerToken,
                fixedETHDepositToken,
              } = await loadFixture(multipleParticipantsWithdrawBeforeVaultEnd)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                lidoVault,
                upfrontPremium: upfrontPremium1,
                timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
              })

              const requestIds = [lidoLastRequestId + BigInt(1)]
              await lidoVault.connect(addr2).withdraw(SIDE.FIXED)

              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )

              // addr2 finalizes vault ended withdrawals
              const balanceBefore2 = await ethers.provider.getBalance(addr2)
              const receipt2 = await (
                await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.FIXED)
              ).wait()
              const gasFees2 = calculateGasFees(receipt2)
              const balanceAfter2 = await ethers.provider.getBalance(addr2)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              expect(balanceAfter2 - balanceBefore2).to.equal(fixedDeposit2 - gasFees2) // no extra fees since they withdrew after vault end
              expect(await lidoVault.fixedBearerToken(addr2.address)).to.equal(0)
              expect(await lidoVault.fixedToPendingWithdrawalAmount(addr1.address)).to.equalWithTolerance(
                fixedDeposit1 - (payBackAmount + scaledEarlyExitFee), 10
              )
              await expectVaultOngoingFixedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)
              expect(await lidoVault.fixedBearerToken(addr1.address)).to.equal(0)

              // addr1 can just withdraw
              const balanceBefore = await ethers.provider.getBalance(addr1)
              const receipt = await (await lidoVault.connect(addr1).withdraw(SIDE.FIXED)).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr1)

              expect(balanceAfter - balanceBefore).to.equalWithTolerance(
                fixedDeposit1 - (gasFees + payBackAmount + scaledEarlyExitFee), 10
              )
              expect(await lidoVault.fixedToPendingWithdrawalAmount(addr1.address)).to.equal(0)
            })
          })
        })
      })

      describe('Variable Side', function () {
        it('Should emit the correct events if they are first to withdraw and finalize (vault-ended, events)', async function () {
          const { lidoVault, addr2, lidoWithdrawalQueueContract } = await loadFixture(
            endVaultFixture
          )

          const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
          const requestIds = [lidoLastRequestId + BIG_INT_ONE]

          const withdrawTrx = await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

          await expect(withdrawTrx)
            .to.emit(lidoVault, 'LidoWithdrawalRequested')
            .withArgs(addr2.address, anyValue, SIDE.VARIABLE, true, true)

          await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

          const finalizeTrx = await lidoVault
            .connect(addr2)
            .finalizeVaultEndedWithdrawals(SIDE.VARIABLE)

          await expect(finalizeTrx)
            .to.emit(lidoVault, 'LidoWithdrawalFinalized')
            .withArgs(addr2.address, anyValue, SIDE.VARIABLE, true, true)
          await expect(finalizeTrx)
            .to.emit(lidoVault, 'VariableFundsWithdrawn')
            .withArgs(anyUint, addr2.address, true, true)
        })

        it('Should emit the correct variable withdrawn event if they are not first to withdraw (vault-ended, events)', async function () {
          const { lidoVault, addr1, addr2, lidoWithdrawalQueueContract } = await loadFixture(
            endVaultFixture
          )

          const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
          const requestIds = [lidoLastRequestId + BIG_INT_ONE]
          await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
          await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
          await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)

          const withdrawTrx = await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

          await expect(withdrawTrx)
            .to.emit(lidoVault, 'VariableFundsWithdrawn')
            .withArgs(anyUint, addr2.address, true, true)
        })

        describe('No Earnings Generated', () => {
          describe('Single Variable Participant', function () {
            it('Should withdraw 0 ETH if they are first to withdraw and finalize (vault-ended)', async function () {
              const { lidoVault, lidoWithdrawalQueueContract, addr2, variableBearerToken } =
                await loadFixture(endVaultFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
              expect(variableBearerTokenTotalSupply).to.equal(variableDeposit)
              // should not have withdrawn funds yet
              expect(bearerTokenBalance).to.equal(variableDeposit)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(balanceAfter).to.equal(balanceBefore - gasFees)
              expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
            })

            it('Should withdraw 0 ETH if they withdraw after fixed', async function () {
              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2, variableBearerToken } =
                await loadFixture(endVaultFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              await lidoVault.connect(addr2).withdraw(SIDE.FIXED)

              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )
              await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
              // no fee should have been applied since there were no staking earnings
              expect(variableBearerTokenTotalSupply).to.equal(variableDeposit)
              // should not have withdrawn funds yet
              expect(bearerTokenBalance).to.equal(variableDeposit)

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(balanceAfter).to.equal(balanceBefore - gasFees)
              expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
            })

            it('Should withdraw 0 ETH if fixed withdraws first but variable calls finalization', async function () {
              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2, variableBearerToken } =
                await loadFixture(endVaultFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
              // no fee should have been applied since there were no staking earnings
              expect(variableBearerTokenTotalSupply).to.equal(variableDeposit)
              // should not have withdrawn funds yet
              expect(bearerTokenBalance).to.equal(variableDeposit)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(balanceAfter).to.equal(balanceBefore - gasFees)
              expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
            })
          })

          describe('Multiple Variable Participants', function () {
            const fixedDeposit = DEFAULTS.fixedSideCapacity

            const variableDeposit2 = parseEther('10')
            const variableDeposit3 = parseEther('20')

            async function multipleParticipantsEndVaultFixture() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, addr1, addr2, addr3 } = state

              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })

              await lidoVault.connect(addr1).claimFixedPremium()

              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)

              return state
            }

            it('Should withdraw 0 ETH for both participants (multiple-participants)', async function () {
              const { lidoVault, lidoWithdrawalQueueContract, addr2, addr3, variableBearerToken } =
                await loadFixture(multipleParticipantsEndVaultFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // withdraw #1
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
              expect(variableBearerTokenTotalSupply).to.equal(variableDeposit2 + variableDeposit3)
              // should not have withdrawn funds yet
              expect(bearerTokenBalance).to.equal(variableDeposit2)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[0]
              )

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(balanceAfter).to.equal(balanceBefore - gasFees)
              expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)

              // withdraw #2

              const bearerTokenBalance3 = await lidoVault.variableBearerToken(addr3.address)
              expect(bearerTokenBalance3).to.equal(variableDeposit3)

              const balanceBefore3 = await ethers.provider.getBalance(addr3)
              const receipt3 = await (await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)).wait()
              const gasFees3 = calculateGasFees(receipt3)
              const balanceAfter3 = await ethers.provider.getBalance(addr3)

              // should not have changed - should have executed a withdraw from the vaults balance
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              expect(balanceAfter3).to.equal(balanceBefore3 - gasFees3)
              expect(await lidoVault.variableBearerToken(addr3.address)).to.equal(0)
            })
          })
        })

        describe('Lido Negative Rebase', () => {
          it('Should withdraw 0 ETH if there is a lido negative rebase and they have previously withdrawn (vault-ended)', async function () {
            const { lidoVault, lidoMock, lidoVaultAddress, addr1, addr2, variableBearerToken } =
              await loadFixture(deployLidoVaultWithMockedLidoFixture)

            await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
            await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: parseEther('20') })
            expect(await lidoVault.isStarted()).to.be.true

            await lidoVault.connect(addr1).claimFixedPremium()

            // mock staking earnings - 100 ETH
            const stakingEarnings = parseEther('100')
            await lidoMock.addStakingEarningsForTargetETH(
              fixedDeposit + stakingEarnings,
              lidoVaultAddress
            )

            // first withdraw
            let balanceBefore = await ethers.provider.getBalance(addr2)
            await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
            await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
            expect(await ethers.provider.getBalance(addr2)).to.be.greaterThan(balanceBefore)

            // mock staking loss
            await lidoMock.subtractStakingEarnings(parseEther('200'))

            // end vault
            const { endTime } = await getTimeState(lidoVault)
            await time.increaseTo(endTime + BIG_INT_ONE)
            expect(await lidoVault.isEnded()).to.equal(true)

            await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

            balanceBefore = await ethers.provider.getBalance(addr2)
            const receipt = await (
              await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
            ).wait()
            const gasFees = calculateGasFees(receipt)
            const balanceAfter = await ethers.provider.getBalance(addr2)

            expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
            expect(balanceAfter).to.equal(balanceBefore - gasFees)
            expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
          })
        })

        describe('Only Fixed Early Exit Fees', () => {
          describe('Single Variable Participant', function () {
            let fixedSideTimestamp: bigint
            async function endVaultWithFixedEarlyExitFixture() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2 } = state

              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })
              expect(await lidoVault.isStarted()).to.be.equal(true)

              await lidoVault.connect(addr1).claimFixedPremium()

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // early fixed withdrawal
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              fixedSideTimestamp = await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1)
              const fixedRequestIds = [lidoLastRequestId + BIG_INT_ONE]
              await finalizeLidoWithdrawalRequests(fixedRequestIds, DEFAULTS.fixedSideCapacity)
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()

              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)
              expect(await lidoVault.isEnded()).to.be.equal(true)

              return state
            }

            it('Should have correct share of the fee earnings (vault-ended)', async function () {
              const { lidoVault, addr2, variableBearerToken } = await loadFixture(
                endVaultWithFixedEarlyExitFixture
              )

              // early exit fee + payback amount should have been applied
              const feeEarnings = await lidoVault.feeEarnings()
              const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                lidoVault,
                upfrontPremium: variableDeposit,
                timestamp: fixedSideTimestamp,
              })
              expect(feeEarnings).to.be.equal(scaledEarlyExitFee + payBackAmount)

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              expect(bearerTokenBalance).to.equal(variableDeposit)

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const amountWithdrawn = balanceAfter - balanceBefore

              // should not have any withdraw state
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                BIG_INT_ZERO
              )
              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                feeEarnings
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              // will be entire amount of feeEarnings since there is only 1 variable depositor
              expect(amountWithdrawn).to.be.equal(feeEarnings - gasFees)
            })

            it('Should fail to withdraw fee earnings again if they already withdrew', async function () {
              const { lidoVault, addr2 } = await loadFixture(endVaultWithFixedEarlyExitFixture)

              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              await expect(lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).to.be.revertedWith(
                'NBT'
              )
            })
          })

          describe('Multiple Variable Participants', function () {
            const variableDeposit2 = parseEther('20')
            const variableDeposit3 = parseEther('5')
            const variableDeposit4 = parseEther('5')
            let fixedSideTimestamp: bigint

            async function multipleParticipantsEndVaultWithFixedEarlyExitFixture() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, lidoWithdrawalQueueContract, addr1, addr2, addr3, addr4 } = state

              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })
              await lidoVault.connect(addr4).deposit(SIDE.VARIABLE, { value: variableDeposit4 })
              expect(await lidoVault.isStarted()).to.be.equal(true)

              await lidoVault.connect(addr1).claimFixedPremium()

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

              // early fixed withdrawal
              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)
              fixedSideTimestamp = await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1)
              const fixedRequestIds = [lidoLastRequestId + BIG_INT_ONE]
              await finalizeLidoWithdrawalRequests(fixedRequestIds, DEFAULTS.fixedSideCapacity)
              await lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()

              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)
              expect(await lidoVault.isEnded()).to.be.equal(true)

              return state
            }

            it('Should have correct share of the fee earnings (vault-ended, multiple-participants)', async function () {
              const { lidoVault, addr2, addr3, variableBearerToken } = await loadFixture(
                multipleParticipantsEndVaultWithFixedEarlyExitFixture
              )

              // early exit fee + payback amount should have been applied
              const feeEarnings = await lidoVault.feeEarnings()
              const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
                lidoVault,
                upfrontPremium: variableDeposit,
                timestamp: fixedSideTimestamp,
              })
              expect(feeEarnings).to.be.equal(scaledEarlyExitFee + payBackAmount)

              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()

              // addr2 withdraws
              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              expect(bearerTokenBalance).to.equal(variableDeposit2)

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const amountWithdrawn = balanceAfter - balanceBefore

              const feeEarningsShare =
                (bearerTokenBalance * feeEarnings) / variableBearerTokenTotalSupply

              // should not have any withdraw state
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                BIG_INT_ZERO
              )
              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                feeEarningsShare
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equal(feeEarningsShare - gasFees)

              // addr3 withdraws
              const bearerTokenBalance3 = await lidoVault.variableBearerToken(addr3.address)
              expect(bearerTokenBalance3).to.equal(variableDeposit3)

              const balanceBefore3 = await ethers.provider.getBalance(addr3)
              const receipt3 = await (await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)).wait()
              const gasFees3 = calculateGasFees(receipt3)
              const balanceAfter3 = await ethers.provider.getBalance(addr3)
              const amountWithdrawn3 = balanceAfter3 - balanceBefore3

              const feeEarningsShare3 =
                (bearerTokenBalance3 * feeEarnings) / variableBearerTokenTotalSupply

              // should not have any withdraw state
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                BIG_INT_ZERO
              )
              expect(
                await lidoVault.variableToWithdrawnFees(addr3.address)
              ).to.be.equalWithTolerance(feeEarningsShare3, 1)
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr3.address)
              expect(amountWithdrawn3).to.be.equalWithTolerance(feeEarningsShare3 - gasFees3, 1)
            })

            it('Should fail to withdraw fee earnings again if they already withdrew (multiple-participants)', async function () {
              const { lidoVault, addr2 } = await loadFixture(
                multipleParticipantsEndVaultWithFixedEarlyExitFixture
              )

              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              await expect(lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).to.be.revertedWith(
                'NBT'
              )
            })
          })
        })

        describe('Lido Staking Earnings Generated', () => {
          describe('Single Variable Participant', function () {
            async function generateStakingEarningsAndEndVaultFixture() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, addr1, addr2 } = state

              // fixedSideCapacity: parseEther('1000')
              // const fixedDeposit = DEFAULTS.fixedSideCapacity
              // const variableDeposit = parseEther('30')
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })

              await lidoVault.connect(addr1).claimFixedPremium()

              await submitOracleReport()

              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)

              return state
            }

            describe('Fee Receiver', function () {
              it('Should withdraw appliedProtocolFee if another user finalizes', async function () {
                const {
                  lidoVault,
                  lidoWithdrawalQueueContract,
                  lidoContract,
                  lidoVaultAddress,
                  deployer,
                  addr2,
                } = await loadFixture(generateStakingEarningsAndEndVaultFixture)

                const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
                const lidoETHAmount = await lidoContract.balanceOf(lidoVaultAddress)
                await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
                const earnings = lidoETHAmount - fixedDeposit
                // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
                const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
                await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
                await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)

                const appliedProtocolFee = await lidoVault.appliedProtocolFee()
                expect(appliedProtocolFee).to.be.equal(
                  (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
                )

                const balanceBefore = await ethers.provider.getBalance(deployer)
                const receipt = await (
                  await lidoVault.connect(deployer).withdraw(SIDE.VARIABLE)
                ).wait()
                const gasFees = calculateGasFees(receipt)
                const balanceAfter = await ethers.provider.getBalance(deployer)
                const amountWithdrawn = balanceAfter - balanceBefore

                expect(amountWithdrawn).to.be.equal(appliedProtocolFee - gasFees)
              })

              it('Should withdraw appliedProtocolFee if they finalize', async function () {
                const {
                  lidoVault,
                  lidoVaultAddress,
                  lidoWithdrawalQueueContract,
                  lidoContract,
                  deployer
                 } = await loadFixture(generateStakingEarningsAndEndVaultFixture)

                // check balance of contract
                const lidoStETHBalance = await lidoContract.balanceOf(lidoVaultAddress)
 
                const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
                console.log("lidoLastRequestId:",lidoLastRequestId);
                await lidoVault.connect(deployer).withdraw(SIDE.VARIABLE)
                // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
                const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
                await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

                // (lidoStETHBalance - fixedDeposit) is profit
                // protocolFeeBps / 10,000 is percent of profit for protocol fee
                const appliedProtocolFee =
                  (lidoStETHBalance - fixedDeposit) * BigInt(DEFAULTS.protocolFeeBps) / BIG_INT_10K

                const balanceBefore = await ethers.provider.getBalance(deployer)
                const receipt = await (
                  await lidoVault.connect(deployer).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
                ).wait()
                const gasFees = calculateGasFees(receipt)
                const balanceAfter = await ethers.provider.getBalance(deployer)
                const amountWithdrawn = balanceAfter - balanceBefore
                expect(amountWithdrawn).to.be.equal(appliedProtocolFee - gasFees)
                //expect(amountWithdrawn).to.be.greaterThan(0)
              })
            })

            it('Should withdraw their portion of the staking earnings if they are first to withdraw and finalize', async function () {
              const {
                lidoVault,
                lidoVaultAddress,
                lidoWithdrawalQueueContract,
                lidoContract,
                addr2,
                variableBearerToken,
              } = await loadFixture(generateStakingEarningsAndEndVaultFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const lidoStETHBalance = await lidoContract.balanceOf(lidoVaultAddress)
              const earnings = lidoStETHBalance - fixedDeposit

              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
              const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
              expect(variableBearerTokenTotalSupply).to.equal(variableDeposit)
              // should not have withdrawn funds yet
              expect(bearerTokenBalance).to.equal(variableDeposit)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[1]
              )

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)

              // since some earnings were made protocol fee should have been applied
              const protocolFee = await lidoVault.appliedProtocolFee()
              expect(protocolFee).to.be.equal(
                (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
              )
              const earningsShare =
                (bearerTokenBalance * (earnings - protocolFee)) / variableBearerTokenTotalSupply
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(balanceAfter - balanceBefore).to.equalWithTolerance(earningsShare - gasFees, 10)
              expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
              expect(vaultBalanceAfter - vaultBalanceBefore).to.be.equalWithTolerance(protocolFee + fixedDeposit, 10)
            })

            it('Should withdraw their portion of the staking earnings if they withdraw after fixed', async function () {
              const {
                lidoVault,
                lidoVaultAddress,
                lidoContract,
                lidoWithdrawalQueueContract,
                addr1,
                addr2,
                variableBearerToken,
              } = await loadFixture(generateStakingEarningsAndEndVaultFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const lidoStETHBalance = await lidoContract.balanceOf(lidoVaultAddress)
              const earnings = lidoStETHBalance - fixedDeposit

              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

              // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
              const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[1]
              )
              await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
              expect(variableBearerTokenTotalSupply).to.equal(variableDeposit)
              // should not have withdrawn funds yet
              expect(bearerTokenBalance).to.equal(variableDeposit)

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt = await (await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)

              // since some earnings were made protocol fee should have been applied
              const protocolFee = await lidoVault.appliedProtocolFee()
              expect(protocolFee).to.be.equal(
                (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
              )
              const earningsShare =
                (bearerTokenBalance * (earnings - protocolFee)) / variableBearerTokenTotalSupply

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(balanceAfter - balanceBefore).to.equalWithTolerance(earningsShare - gasFees, 10)
              expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
              expect(vaultBalanceBefore - vaultBalanceAfter).to.equalWithTolerance(earningsShare, 10)
            })

            it('Should fail to withdraw if they have already withdrawn', async function () {
              const { lidoVault, lidoWithdrawalQueueContract, addr2 } = await loadFixture(
                generateStakingEarningsAndEndVaultFixture
              )

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
              const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]

              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)

              await expect(lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).to.be.revertedWith(
                'NBT'
              )
            })

            describe('Withdraw requested before vault end', () => {
              let earnings: bigint

              async function withdrawBeforeVaultEndWithStakingEarnings() {
                const state = await loadFixture(deployLidoVaultFixture)

                const {
                  lidoVault,
                  lidoVaultAddress,
                  lidoWithdrawalQueueContract,
                  lidoContract,
                  addr1,
                  addr2,
                } = state

                // vault started
                await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
                await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit })
                await lidoVault.connect(addr1).claimFixedPremium()

                const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
                const requestIds = [lidoLastRequestId + BIG_INT_ONE]

                // staking earnings
                await submitOracleReport()

                const lidoStETHBalance = await lidoContract.balanceOf(lidoVaultAddress)
                earnings = lidoStETHBalance - fixedDeposit

                // withdraw
                await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
                await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                  lidoVault,
                  requestIds,
                  addr2.address
                )
                await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

                // end vault
                const { endTime } = await getTimeState(lidoVault)
                await time.increaseTo(endTime + BIG_INT_ONE)
                expect(await lidoVault.isEnded()).to.equal(true)

                return state
              }

              it('Should be able to kickoff vault ended withdrawal but cannot finalize', async function () {
                const {
                  lidoVault,
                  lidoWithdrawalQueueContract,
                  addr1,
                  addr2,
                  variableBearerToken,
                } = await loadFixture(withdrawBeforeVaultEndWithStakingEarnings)
                const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

                // can kick off vault end withdraw process
                await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

                const requestIds = [lidoLastRequestId + BigInt(1)]
                await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

                await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
                expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                  requestIds[0]
                )
                // cant call finalize due to ongoing withdrawal request
                await expect(
                  lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
                ).to.be.revertedWith('WAR')
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)
              
                // can finalize ongoing withdrawal request
                const balanceBefore = await ethers.provider.getBalance(addr2)
                const receipt = await (
                  await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
                ).wait()
                const gasFees = calculateGasFees(receipt)
                const balanceAfter = await ethers.provider.getBalance(addr2)
                console.log(await lidoVault.stakingBalance(), await lidoVault.fixedETHDepositTokenTotalSupply())
                // since some earnings were made protocol fee should have been applied
                const protocolFee = await lidoVault.appliedProtocolFee()
                expect(protocolFee).to.be.equal(
                  (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
                )
                expect(await lidoVault.vaultEndedStakingEarnings()).to.equal(0)
                expect(balanceAfter - balanceBefore).to.equalWithTolerance(earnings - protocolFee - gasFees, 10)
                // does not burn tokens yet
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equalWithTolerance(variableDeposit, 10)

                // vault ended
                await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)

                // withdraw again should burn bearer tokens and withdraw 0 ETH
                const balanceBefore2 = await ethers.provider.getBalance(addr2)
                const receipt2 = await (
                  await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
                ).wait()
                const gasFees2 = calculateGasFees(receipt2)
                const balanceAfter2 = await ethers.provider.getBalance(addr2)
                expect(balanceAfter2).to.equal(balanceBefore2 - gasFees2)
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
              })

              it('Should be able to withdraw after someone else finalizes', async function () {
                const {
                  lidoVault,

                  lidoWithdrawalQueueContract,
                  addr1,
                  addr2,
                  variableBearerToken,
                } = await loadFixture(withdrawBeforeVaultEndWithStakingEarnings)

                const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
                console.log(await lidoVault.stakingBalance(), await lidoVault.fixedETHDepositTokenTotalSupply())
                await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
                console.log(await lidoVault.stakingBalance(), await lidoVault.fixedETHDepositTokenTotalSupply())
                const requestIds = [lidoLastRequestId + BigInt(1)]
                await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)
                await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
                expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                  requestIds[0]
                )
                console.log(await lidoVault.stakingBalance(), await lidoVault.fixedETHDepositTokenTotalSupply(),
                await lidoVault.vaultEndedWithdrawalsFinalized(),(await lidoVault.vaultEndedWithdrawalRequestIds(0)))
                await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)

                // finalize ongoing withdrawal
                const balanceBefore = await ethers.provider.getBalance(addr2)
                const receipt = await (
                  await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
                ).wait()
                const gasFees = calculateGasFees(receipt)
                const balanceAfter = await ethers.provider.getBalance(addr2)

                // since some earnings were made protocol fee should have been applied
                const protocolFee = await lidoVault.appliedProtocolFee()
                expect(protocolFee).to.be.equal(
                  (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
                )
                expect(await lidoVault.vaultEndedStakingEarnings()).to.equal(0)
                expect(balanceAfter - balanceBefore).to.equalWithTolerance(earnings - protocolFee - gasFees, 10)
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(variableDeposit)

                // withdraw again should burn bearer tokens and withdraw 0 ETH
                const balanceBefore2 = await ethers.provider.getBalance(addr2)
                const receipt2 = await (
                  await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
                ).wait()
                const gasFees2 = calculateGasFees(receipt2)
                const balanceAfter2 = await ethers.provider.getBalance(addr2)
                expect(balanceAfter2).to.equal(balanceBefore2 - gasFees2)
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
              })
            })
          })

          describe('Multiple Variable Participants', function () {
            const fixedDeposit = DEFAULTS.fixedSideCapacity

            const variableDeposit2 = parseEther('10')
            const variableDeposit3 = parseEther('20')

            async function multipleParticipantsGenerateStakingEarningsAndEndVaultFixture() {
              const state = await loadFixture(deployLidoVaultFixture)

              const { lidoVault, addr1, addr2, addr3 } = state

              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })

              await lidoVault.connect(addr1).claimFixedPremium()

              await submitOracleReport()

              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)

              return state
            }

            it('Should withdraw their portion of the staking earnings over multiple withdrawals before and after vault end (multiple-participants)', async function () {
              const { lidoVault, lidoVaultAddress, addr1, addr2, addr3, addr4, lidoMock, lidoContract } =
                await loadFixture(deployLidoVaultWithMockedLidoFixture)

              const variableDeposit2 = parseEther('10')
              const variableDeposit3 = parseEther('5')
              const variableDeposit4 = parseEther('5')

              // vault started
              await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
              await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
              await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })
              await lidoVault.connect(addr4).deposit(SIDE.VARIABLE, { value: variableDeposit4 })
              await lidoVault.connect(addr1).claimFixedPremium()
              expect(await lidoVault.isStarted()).to.equal(true)

              // mock staking earnings - 100 ETH
              const lidoLastRequestId = BIG_INT_ZERO
              const requestIds = [lidoLastRequestId + BIG_INT_ONE]
              const stakingEarnings = parseEther('100')
              await lidoMock.addStakingEarningsForTargetETH(
                fixedDeposit + stakingEarnings,
                lidoVaultAddress
              )

              // addr2 withdraw #1
              const earningsShareAddr2 = parseEther('50') // 100 earnings / 2 since they own 50% of the bearer token supply
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds,
                addr2.address
              )
              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)).to.be.equal(
                earningsShareAddr2
              )

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const amountWithdrawn = balanceAfter - balanceBefore

              expect(await lidoVault.variableToWithdrawnFees(addr2.address)).to.be.equal(
                BIG_INT_ZERO
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr2.address)
              expect(amountWithdrawn).to.be.equalWithTolerance(earningsShareAddr2 - gasFees, 10)

              // addr3 withdraw #1
              const earningsShareAddr3 = parseEther('25') // 100 earnings / 4 since they own 25% of the bearer token supply

              const requestIds2 = [lidoLastRequestId + BigInt(2)]
              await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)

              await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                lidoVault,
                requestIds2,
                addr3.address
              )

              expect(await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)).to.be.equalWithTolerance(
                earningsShareAddr3,
                10
              )

              const balanceBefore3 = await ethers.provider.getBalance(addr3)
              const receipt3 = await (
                await lidoVault.connect(addr3).finalizeVaultOngoingVariableWithdrawals()
              ).wait()
              const gasFees3 = calculateGasFees(receipt3)
              const balanceAfter3 = await ethers.provider.getBalance(addr3)
              const amountWithdrawn3 = balanceAfter3 - balanceBefore3

              expect(await lidoVault.variableToWithdrawnFees(addr3.address)).to.be.equal(
                BIG_INT_ZERO
              )
              await expectVaultOngoingVariableWithdrawalArraysAreEqual(lidoVault, [], addr3.address)
              expect(amountWithdrawn3).to.be.equalWithTolerance(earningsShareAddr3 - gasFees3, 10)

              // mock more staking earnings - 50 ETH
              const stakingEarnings2 = parseEther('50')
              await lidoMock.addStakingEarningsForTargetETH(
                fixedDeposit +
                  stakingEarnings +
                  stakingEarnings2 -
                  (earningsShareAddr2 + earningsShareAddr3), // subtract already withdrawn funds
                lidoVaultAddress
              )

              // vault ended
              const { endTime } = await getTimeState(lidoVault)
              await time.increaseTo(endTime + BIG_INT_ONE)
              expect(await lidoVault.isEnded()).to.equal(true)

              // addr2 vault end withdraw
              const earningsShareAddr2Again = (await lidoContract.getPooledEthByShares(await lidoVault.stakingShares() + 
              await lidoVault.withdrawnStakingEarningsInStakes()) -  await lidoVault.fixedETHDepositTokenTotalSupply()) / 2n  - //  / 2 since they own 50% of the bearer token supply
              await lidoContract.getPooledEthByShares(await lidoVault.variableToWithdrawnStakingEarningsInShares(addr2.address))  
 
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              const balanceBefore2 = await ethers.provider.getBalance(addr2)
              const receipt2 = await (
                await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
              ).wait()
              const gasFees2 = calculateGasFees(receipt2)
              const balanceAfter2 = await ethers.provider.getBalance(addr2)
              const amountWithdrawn2 = balanceAfter2 - balanceBefore2

              console.log(amountWithdrawn2,earningsShareAddr2Again, gasFees2)
              expect(amountWithdrawn2).to.be.equal(
                earningsShareAddr2Again - gasFees2
              )

              // addr3 withdraw again
      
              const earningsShareAddr3Again = (await lidoContract.getPooledEthByShares(await lidoVault.withdrawnStakingEarningsInStakes()) + 
              await lidoVault.vaultEndedStakingEarnings())/4n  - //  / 4 since they own 25% of the bearer token supply
              await lidoContract.getPooledEthByShares(await lidoVault.variableToWithdrawnStakingEarningsInShares(addr3.address))  

              const balanceBefore3Again = await ethers.provider.getBalance(addr3)
              const receipt3Again = await (
                await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)
              ).wait()
              const gasFees3Again = calculateGasFees(receipt3Again)
              const balanceAfter3Again = await ethers.provider.getBalance(addr3)
              const amountWithdrawn3Again = balanceAfter3Again - balanceBefore3Again

              expect(amountWithdrawn3Again).to.be.equal(
                earningsShareAddr3Again - gasFees3Again
              )

              // addr4 withdraws
              const earningsShareAddr4 =  (await lidoContract.getPooledEthByShares(await lidoVault.withdrawnStakingEarningsInStakes()) + 
              await lidoVault.vaultEndedStakingEarnings())/4n  - //  / 4 since they own 25% of the bearer token supply
              await lidoContract.getPooledEthByShares(await lidoVault.variableToWithdrawnStakingEarningsInShares(addr4.address))  

              const balanceBefore4 = await ethers.provider.getBalance(addr4)
              const receipt4 = await (await lidoVault.connect(addr4).withdraw(SIDE.VARIABLE)).wait()
              const gasFees4 = calculateGasFees(receipt4)
              const balanceAfter4 = await ethers.provider.getBalance(addr4)
              const amountWithdrawn4 = balanceAfter4 - balanceBefore4

              expect(amountWithdrawn4).to.be.equal(earningsShareAddr4 - gasFees4)
            })

            it('Should withdraw their portion of the staking earnings if they are first to withdraw and finalize (multiple-participants)', async function () {
              const {
                lidoVault,
                lidoVaultAddress,
                lidoContract,
                lidoWithdrawalQueueContract,
                addr2,
                addr3,
                variableBearerToken,
              } = await loadFixture(multipleParticipantsGenerateStakingEarningsAndEndVaultFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const lidoStETHBalance = await lidoContract.balanceOf(lidoVaultAddress)
              const earnings = lidoStETHBalance - fixedDeposit

              // withdraw #1
              await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

              // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
              const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              const bearerTokenBalance3 = await lidoVault.variableBearerToken(addr3.address)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
              expect(variableBearerTokenTotalSupply).to.equal(variableDeposit2 + variableDeposit3)
              // should not have withdrawn funds yet
              expect(bearerTokenBalance).to.equal(variableDeposit2)
              expect(bearerTokenBalance3).to.equal(bearerTokenBalance3)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[1]
              )

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt = await (
                await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
              ).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)

              // since some earnings were made protocol fee should have been applied
              const protocolFee = await lidoVault.appliedProtocolFee()
              expect(protocolFee).to.be.equal(
                (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
              )
              // no early fixed withdrawals made
              expect(await lidoVault.withdrawnFeeEarnings()).to.be.equal(0)
              const earningsShare =
                (bearerTokenBalance * (earnings - protocolFee)) / variableBearerTokenTotalSupply
              const earningsShare3 =
                (bearerTokenBalance3 * (earnings - protocolFee)) / variableBearerTokenTotalSupply

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(balanceAfter - balanceBefore).to.equal(earningsShare - gasFees)
              expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
              expect(vaultBalanceAfter - vaultBalanceBefore).to.equalWithTolerance(
                protocolFee + fixedDeposit + earningsShare3,
                1
              )

              // withdraw #2
              const variableBearerTokenTotalSupply3 = await lidoVault.variableBearerTokenTotalSupply()
              expect(variableBearerTokenTotalSupply3).to.equal(variableDeposit3)

              const balanceBefore3 = await ethers.provider.getBalance(addr3)
              const vaultBalanceBefore3 = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt3 = await (await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)).wait()
              const gasFees3 = calculateGasFees(receipt3)
              const balanceAfter3 = await ethers.provider.getBalance(addr3)
              const vaultBalanceAfter3 = await ethers.provider.getBalance(lidoVaultAddress)

              expect(balanceAfter3 - balanceBefore3).to.equalWithTolerance(
                earningsShare3 - gasFees3,
                1
              )
              expect(await lidoVault.variableBearerToken(addr3.address)).to.equal(0)
              expect(vaultBalanceBefore3 - vaultBalanceAfter3).to.equalWithTolerance(
                earningsShare3,
                1
              )
            })

            it('Should withdraw their portion of the staking earnings if they withdraw after fixed (multiple-participants)', async function () {
              const {
                lidoVault,
                lidoVaultAddress,
                lidoContract,
                lidoWithdrawalQueueContract,
                addr1,
                addr2,
                addr3,
                variableBearerToken,
              } = await loadFixture(multipleParticipantsGenerateStakingEarningsAndEndVaultFixture)

              const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
              const lidoStETHBalance = await lidoContract.balanceOf(lidoVaultAddress)
              const earnings = lidoStETHBalance - fixedDeposit

              await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

              // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
              const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

              await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
              expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                requestIds[1]
              )
              await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)

              // withdraw #1
              const bearerTokenBalance = await lidoVault.variableBearerToken(addr2.address)
              const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
              expect(variableBearerTokenTotalSupply).to.equal(variableDeposit2 + variableDeposit3)
              // should not have withdrawn funds yet
              expect(bearerTokenBalance).to.equal(variableDeposit2)

              const balanceBefore = await ethers.provider.getBalance(addr2)
              const vaultBalanceBefore = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt = await (await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)).wait()
              const gasFees = calculateGasFees(receipt)
              const balanceAfter = await ethers.provider.getBalance(addr2)
              const vaultBalanceAfter = await ethers.provider.getBalance(lidoVaultAddress)

              // since some earnings were made protocol fee should have been applied
              const protocolFee = await lidoVault.appliedProtocolFee()
              expect(protocolFee).to.be.equal(
                (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
              )
              const earningsShare =
                (bearerTokenBalance * (earnings - protocolFee)) / variableBearerTokenTotalSupply

              expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
              await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
              expect(balanceAfter - balanceBefore).to.equal(earningsShare - gasFees)
              expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
              expect(vaultBalanceBefore - vaultBalanceAfter).to.equal(earningsShare)

              // withdraw #2
              const bearerTokenBalance3 = await lidoVault.variableBearerToken(addr3.address)
              const variableBearerTokenTotalSupply3 = await lidoVault.variableBearerTokenTotalSupply()
              expect(variableBearerTokenTotalSupply3).to.equal(variableDeposit3)
              // should not have withdrawn funds yet
              expect(bearerTokenBalance3).to.equal(variableDeposit3)

              const balanceBefore3 = await ethers.provider.getBalance(addr3)
              const vaultBalanceBefore3 = await ethers.provider.getBalance(lidoVaultAddress)
              const receipt3 = await (await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)).wait()
              const gasFees3 = calculateGasFees(receipt3)
              const balanceAfter3 = await ethers.provider.getBalance(addr3)
              const vaultBalanceAfter3 = await ethers.provider.getBalance(lidoVaultAddress)

              const earningsShare3 =
                (bearerTokenBalance3 * (earnings - protocolFee)) / variableBearerTokenTotalSupply

              expect(balanceAfter3 - balanceBefore3).to.equalWithTolerance(
                earningsShare3 - gasFees3,
                1
              )
              expect(await lidoVault.variableBearerToken(addr3.address)).to.equal(0)
              expect(vaultBalanceBefore3 - vaultBalanceAfter3).to.equalWithTolerance(
                earningsShare3,
                1
              )
            })

            describe('Withdraw requested before vault end', () => {
              const fixedDeposit = DEFAULTS.fixedSideCapacity

              const variableDeposit2 = parseEther('10')
              const variableDeposit3 = parseEther('20')
              let stakingEarnings: bigint

              async function multipleParticipantsWithdrawBeforeVaultEndNoStakingEarnings() {
                const state = await loadFixture(deployLidoVaultFixture)

                const {
                  lidoVault,
                  lidoVaultAddress,
                  lidoWithdrawalQueueContract,
                  lidoContract,
                  addr1,
                  addr2,
                  addr3,
                } = state

                // vault started
                await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
                await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
                await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })
                await lidoVault.connect(addr1).claimFixedPremium()
                expect(await lidoVault.isStarted()).to.equal(true)
                const stETHVaultBalanceBefore = await lidoContract.balanceOf(lidoVaultAddress)

                // staking earnings
                await submitOracleReport()
                const stETHVaultBalanceAfter = (await lidoContract.balanceOf(
                  lidoVaultAddress
                )) as bigint
                stakingEarnings = stETHVaultBalanceAfter - stETHVaultBalanceBefore

                // withdraw
                const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
                const requestIds = [lidoLastRequestId + BIG_INT_ONE]
                await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
                await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                  lidoVault,
                  requestIds,
                  addr2.address
                )
                await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

                // end vault
                const { endTime } = await getTimeState(lidoVault)
                await time.increaseTo(endTime + BIG_INT_ONE)
                expect(await lidoVault.isEnded()).to.equal(true)

                return state
              }

              it('Should have correct ETH earnings withdrawn (multiple-participants, variable-perpetual-withdrawal)', async function () {
                const {
                  lidoVault,
                  addr1,
                  addr2,
                  addr3,
                  addr4,
                  variableBearerToken,
                  lidoMock,
                  lidoVaultAddress,
                  lidoContract,
                } = await loadFixture(deployLidoVaultWithMockedLidoFixture)

                const variableDeposit2 = parseEther('10')
                const variableDeposit3 = parseEther('5')
                const variableDeposit4 = parseEther('5')

                // vault started
                await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
                await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
                await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })
                await lidoVault.connect(addr4).deposit(SIDE.VARIABLE, { value: variableDeposit4 })
                await lidoVault.connect(addr1).claimFixedPremium()
                expect(await lidoVault.isStarted()).to.equal(true)

                // mock staking earnings - 100 ETH
                const lidoLastRequestId = BIG_INT_ZERO
                const requestIds = [lidoLastRequestId + BIG_INT_ONE]
                const stakingEarnings = parseEther('100')
                await lidoMock.addStakingEarningsForTargetETH(
                  fixedDeposit + stakingEarnings,
                  lidoVaultAddress
                )

                // withdraw
                await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

                await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                  lidoVault,
                  requestIds,
                  addr2.address
                )
                expect(
                  await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)
                ).to.be.equal(
                  parseEther('50') // 100 earnings / 2 since they own 50% of the bearer token supply
                )

                // end vault
                const { endTime } = await getTimeState(lidoVault)
                await time.increaseTo(endTime + BIG_INT_ONE)
                expect(await lidoVault.isEnded()).to.equal(true)

                // kick off withdraw end process
                // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
                const requestIdsEnd = [lidoLastRequestId + BigInt(2), lidoLastRequestId + BigInt(3)]
                const stakingEarnings2 = parseEther('100')
                await lidoMock.addStakingEarningsForTargetETH(
                  fixedDeposit + stakingEarnings + stakingEarnings2 - parseEther('50'), // subtract what earnings already withdrawn
                  lidoVaultAddress
                )

                await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

                await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIdsEnd)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

                // finalize vault ended
                const vaultEndStakingEarnings = 150 // 200 total - 50 already withdrawn
                await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)

                const protocolFee = 0 // protocol fee set to 0 bps
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
                expect(await lidoVault.vaultEndedStakingEarnings()).to.be.equalWithTolerance(
                  parseEther(`${vaultEndStakingEarnings - protocolFee}`),
                  1
                )

                // other variable depositor should be able to withdraw their earnings
                const earningsShare3 = (await lidoContract.getPooledEthByShares(await lidoVault.withdrawnStakingEarningsInStakes()) + 
                await lidoVault.vaultEndedStakingEarnings())/4n  - // / 4 as addr2 have 25% of variable share
                await lidoContract.getPooledEthByShares(await lidoVault.variableToWithdrawnStakingEarningsInShares(addr3.address))
                const protocolFee3 = BIG_INT_ZERO // protocol fee set to 0 bps
                const balanceBefore3 = await ethers.provider.getBalance(addr3)
                const receipt3 = await (
                  await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)
                ).wait()
                const gasFees3 = calculateGasFees(receipt3)
                const balanceAfter3 = await ethers.provider.getBalance(addr3)
                console.log(balanceAfter3,balanceBefore3, earningsShare3, gasFees3, protocolFee3)
                expect(balanceAfter3 - balanceBefore3).to.equal(
                  earningsShare3 - gasFees3 - protocolFee3
                )
                expect(await lidoVault.variableBearerToken(addr3.address)).to.equal(0)

                // can finalize ongoing withdrawal request and get the 50 original requested to withdraw
                const earningsShare = parseEther('50')
                const balanceBefore = await ethers.provider.getBalance(addr2)
                const receipt = await (
                  await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
                ).wait()
                const gasFees = calculateGasFees(receipt)
                const balanceAfter = await ethers.provider.getBalance(addr2)
                expect(balanceAfter - balanceBefore).to.equalWithTolerance(earningsShare - gasFees, 10)

                // withdraw again should burn bearer tokens and withdraw remaining amount thats around 50 ETH (but less)
                const earningsShare2 = (await lidoContract.getPooledEthByShares(await lidoVault.withdrawnStakingEarningsInStakes()) + 
                await lidoVault.vaultEndedStakingEarnings())/2n   - // / 2 as addr2 have 50% of variable share
                await lidoContract.getPooledEthByShares(await lidoVault.variableToWithdrawnStakingEarningsInShares(addr2.address))  
                const balanceBefore2 = await ethers.provider.getBalance(addr2)
                const receipt2 = await (
                  await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
                ).wait()
                const gasFees2 = calculateGasFees(receipt2)
                const balanceAfter2 = await ethers.provider.getBalance(addr2)
                expect(balanceAfter2 - balanceBefore2).to.equal(
                  earningsShare2 - gasFees2
                )
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
              })

              it('Should have correct ETH earnings withdrawn if multiple ongoing withdrawals were made (multiple-participants, variable-perpetual-withdrawal)', async function () {
                const deployLidoVaultWithMockedLidoFixtureCustomFixedPremium = () =>
                  deployVaultWithMockedLido({ variableSideCapacity: parseEther('100'), protocolFeeBps: 0 })

                const {
                  lidoVault,
                  addr1,
                  addr2,
                  addr3,
                  addr4,
                  variableBearerToken,
                  lidoMock,
                  lidoVaultAddress,
                  lidoContract,
                } = await loadFixture(deployLidoVaultWithMockedLidoFixtureCustomFixedPremium)

                const variableDeposit2 = parseEther('50')
                const variableDeposit3 = parseEther('25')
                const variableDeposit4 = parseEther('25')

                // vault started
                await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
                await lidoVault.connect(addr2).deposit(SIDE.VARIABLE, { value: variableDeposit2 })
                await lidoVault.connect(addr3).deposit(SIDE.VARIABLE, { value: variableDeposit3 })
                await lidoVault.connect(addr4).deposit(SIDE.VARIABLE, { value: variableDeposit4 })
                await lidoVault.connect(addr1).claimFixedPremium()
                expect(await lidoVault.isStarted()).to.equal(true)

                // mock staking earnings - 100 ETH
                const stakingEarnings = parseEther('100')
                await lidoMock.addStakingEarningsForTargetETH(
                  fixedDeposit + stakingEarnings,
                  lidoVaultAddress
                )

                // withdraw #1
                const lidoLastRequestId = BIG_INT_ZERO
                const requestIds = [lidoLastRequestId + BIG_INT_ONE]
                await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

                await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                  lidoVault,
                  requestIds,
                  addr2.address
                )
                expect(
                  await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)
                ).to.be.equal(
                  parseEther('50') // 100 earnings / 2 since they own 50% of the bearer token supply
                )

                // withdraw #2
                const requestIds3 = [lidoLastRequestId + BigInt(2)]
                await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)

                await expectVaultOngoingVariableWithdrawalArraysAreEqual(
                  lidoVault,
                  requestIds3,
                  addr3.address
                )
                expect(
                  await lidoVault.variableToWithdrawnStakingEarnings(addr3.address)
                ).to.be.equalWithTolerance(
                  parseEther('25'), // 100 earnings / 4 since they own 25% of the bearer token supply
                  10
                )

                // add more staking earnings
                const stakingEarnings2 = parseEther('100')
                await lidoMock.addStakingEarningsForTargetETH(
                  fixedDeposit + stakingEarnings + stakingEarnings2 - parseEther('75'),
                  lidoVaultAddress
                )

                // end vault
                const { endTime } = await getTimeState(lidoVault)
                await time.increaseTo(endTime + BIG_INT_ONE)
                expect(await lidoVault.isEnded()).to.equal(true)

                // kick off withdraw end process
                // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
                const requestIdsEnd = [lidoLastRequestId + BigInt(3), lidoLastRequestId + BigInt(4)]
                await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

                await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIdsEnd)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

                // finalize vault ended
                const vaultEndStakingEarnings = 125 // 200 total - 75 already withdrawn
                await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)

                const protocolFee = 0 // protocol fee set to 0 bps
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
                expect(await lidoVault.vaultEndedStakingEarnings()).to.be.equalWithTolerance(
                  parseEther(`${vaultEndStakingEarnings - protocolFee}`),
                  1
                )

                // other variable depositor should be able to withdraw their earnings
                const earningsShare4 = (await lidoContract.getPooledEthByShares(await lidoVault.withdrawnStakingEarningsInStakes()) + 
                await lidoVault.vaultEndedStakingEarnings())/4n  - // / 4 as addr2 have 25% of variable share
                await lidoContract.getPooledEthByShares(await lidoVault.variableToWithdrawnStakingEarningsInShares(addr4.address))
                const protocolFee4 = BIG_INT_ZERO // protocol fee set to 0 bps
                const balanceBefore4 = await ethers.provider.getBalance(addr4)
                const receipt4 = await (
                  await lidoVault.connect(addr4).withdraw(SIDE.VARIABLE)
                ).wait()
                const gasFees4 = calculateGasFees(receipt4)
                const balanceAfter4 = await ethers.provider.getBalance(addr4)
                console.log(balanceAfter4,balanceBefore4, earningsShare4, gasFees4, protocolFee4)
                expect(balanceAfter4 - balanceBefore4).to.equal(
                  earningsShare4 - gasFees4 - protocolFee4
                )
                expect(await lidoVault.variableBearerToken(addr4.address)).to.equal(0)

                // can finalize ongoing withdrawal request and get the 50 original requested to withdraw
                const earningsShare = parseEther('50')
                const balanceBefore = await ethers.provider.getBalance(addr2)
                const receipt = await (
                  await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
                ).wait()
                const gasFees = calculateGasFees(receipt)
                const balanceAfter = await ethers.provider.getBalance(addr2)
                expect(balanceAfter - balanceBefore).to.equalWithTolerance(earningsShare - gasFees, 10)

                // withdraw again should burn bearer tokens and withdraw remaining amount that around 50 ETH (but less)
              
                const earningsShare2 = (await lidoContract.getPooledEthByShares(await lidoVault.withdrawnStakingEarningsInStakes()) + 
                await lidoVault.vaultEndedStakingEarnings())/2n  - // / 2 as addr2 have 50% of variable share
                await lidoContract.getPooledEthByShares(await lidoVault.variableToWithdrawnStakingEarningsInShares(addr2.address))  
                const balanceBefore2 = await ethers.provider.getBalance(addr2)
                const receipt2 = await (
                  await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
                ).wait()
                const gasFees2 = calculateGasFees(receipt2)
                const balanceAfter2 = await ethers.provider.getBalance(addr2)
                expect(balanceAfter2 - balanceBefore2).to.equal(
                  earningsShare2 - gasFees2
                )
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
              })

              it('Should be able to kickoff vault ended withdrawal but cannot finalize (multiple-participants, variable-perpetual-withdrawal)', async function () {
                const {
                  lidoVault,
                  lidoWithdrawalQueueContract,
                  addr1,
                  addr2,
                  addr3,
                  variableBearerToken,
                } = await loadFixture(multipleParticipantsWithdrawBeforeVaultEndNoStakingEarnings)

                const bearerTokenBalance = await lidoVault.variableBearerToken(addr2)
                const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()

                const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

                // can kick off vault end withdraw process
                // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
                const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
                await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)

                await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

                await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
                expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                  requestIds[1]
                )

                // cant call finalize due to ongoing withdrawal request
                await expect(
                  lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
                ).to.be.revertedWith('WAR')
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

                // can finalize ongoing withdrawal request
                const balanceBefore = await ethers.provider.getBalance(addr2)
                const receipt = await (
                  await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
                ).wait()
                const gasFees = calculateGasFees(receipt)
                const balanceAfter = await ethers.provider.getBalance(addr2)

                // vault ended finalize has not been called yet
                expect(await lidoVault.vaultEndedStakingEarnings()).to.equal(0)
                const earningsShare =
                  (bearerTokenBalance * stakingEarnings) / variableBearerTokenTotalSupply
                const protocolFeeShare =
                  (earningsShare * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K

                // since some earnings were made protocol fee should have been applied
                expect(await lidoVault.appliedProtocolFee()).to.be.equal(protocolFeeShare)
                expect(balanceAfter - balanceBefore).to.equalWithTolerance(
                  earningsShare - gasFees - protocolFeeShare,
                  10
                )
                expect(
                  await lidoVault.variableToWithdrawnStakingEarnings(addr2.address)
                ).to.equalWithTolerance(earningsShare - protocolFeeShare, 10)

                // does not burn tokens yet
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(
                  variableDeposit2
                )

                // finalize vault ended withdrawal
                await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)
                expect(await lidoVault.vaultEndedStakingEarnings()).to.be.greaterThan(0)

                // other variable depositor should be able to withdraw their earnings
                const bearerTokenBalance3 = await lidoVault.variableBearerToken(addr3)
                const earningsShare3 =
                  (bearerTokenBalance3 * stakingEarnings) / variableBearerTokenTotalSupply
                const protocolFee3 =
                  (earningsShare3 * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
                const balanceBefore3 = await ethers.provider.getBalance(addr3)
                const receipt3 = await (
                  await lidoVault.connect(addr3).withdraw(SIDE.VARIABLE)
                ).wait()
                const gasFees3 = calculateGasFees(receipt3)
                const balanceAfter3 = await ethers.provider.getBalance(addr3)
                expect(balanceAfter3 - balanceBefore3).to.equalWithTolerance(
                  earningsShare3 - gasFees3 - protocolFee3, 10
                )
                expect(await lidoVault.variableBearerToken(addr3.address)).to.equal(0)

                // withdraw again should burn bearer tokens and withdraw 0 ETH
                const balanceBefore2 = await ethers.provider.getBalance(addr2)
                const receipt2 = await (
                  await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
                ).wait()
                const gasFees2 = calculateGasFees(receipt2)
                const balanceAfter2 = await ethers.provider.getBalance(addr2)
                expect(balanceBefore2 - balanceAfter2).to.equalWithTolerance(gasFees2, 10)
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
              })

              it('Should be able to withdraw after someone else finalizes (multiple-participants, variable-perpetual-withdrawal)', async function () {
                const {
                  lidoVault,
                  lidoWithdrawalQueueContract,
                  addr1,
                  addr2,
                  variableBearerToken,
                } = await loadFixture(multipleParticipantsWithdrawBeforeVaultEndNoStakingEarnings)

                const bearerTokenBalance = await lidoVault.variableBearerToken(addr2)
                const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
                const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

                await lidoVault.connect(addr1).withdraw(SIDE.VARIABLE)

                // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
                const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
                await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

                await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
                expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                  requestIds[1]
                )

                await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)

                // finalize ongoing withdrawal
                const balanceBefore = await ethers.provider.getBalance(addr2)
                const receipt = await (
                  await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
                ).wait()
                const gasFees = calculateGasFees(receipt)
                const balanceAfter = await ethers.provider.getBalance(addr2)

                // since some earnings were made protocol fee should have been applied
                const protocolFee = await lidoVault.appliedProtocolFee()
                expect(protocolFee).to.be.equalWithTolerance(
                  stakingEarnings * BigInt(DEFAULTS.protocolFeeBps) / BIG_INT_10K,
                  1
                )                
                const earningsShare =
                  (bearerTokenBalance * (stakingEarnings - protocolFee)) /
                  variableBearerTokenTotalSupply

                expect(await lidoVault.vaultEndedStakingEarnings()).to.equalWithTolerance(
                  stakingEarnings - protocolFee - earningsShare,
                  10
                )
                expect(balanceAfter - balanceBefore).to.equalWithTolerance(
                  earningsShare - gasFees,
                  10
                )
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(
                  variableDeposit2
                )

                // withdraw again should burn bearer tokens and withdraw 0 ETH
                const balanceBefore2 = await ethers.provider.getBalance(addr2)
                const receipt2 = await (
                  await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
                ).wait()
                const gasFees2 = calculateGasFees(receipt2)
                const balanceAfter2 = await ethers.provider.getBalance(addr2)
                expect(balanceBefore2 - balanceAfter2).to.equalWithTolerance(gasFees2, 1)
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
              })
              it('Should be able to withdraw after someone else finalizes using finalizeVaultEndedWithdrawals (multiple-participants, variable-perpetual-withdrawal)', async function () {
                const {
                  lidoVault,
                  lidoWithdrawalQueueContract,
                  addr1,
                  addr2,
                  variableBearerToken,
                } = await loadFixture(multipleParticipantsWithdrawBeforeVaultEndNoStakingEarnings)

                const bearerTokenBalance = await lidoVault.variableBearerToken(addr2)
                const variableBearerTokenTotalSupply = await lidoVault.variableBearerTokenTotalSupply()
                const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

                await lidoVault.connect(addr1).withdraw(SIDE.VARIABLE)

                // 2 requests - fixed deposit (1000) + earnings > 1000 ETH max ETH withdrawal per lido request
                const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
                await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(false)

                await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)
                expect(await lidoWithdrawalQueueContract.getLastFinalizedRequestId()).to.equal(
                  requestIds[1]
                )

                await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
                expect(await lidoVault.vaultEndedWithdrawalsFinalized()).to.equal(true)

                // finalize ongoing withdrawal using finalizeVaultEndedWithdrawals
                const balanceBefore = await ethers.provider.getBalance(addr2)
                const receipt = await (
                  await lidoVault.connect(addr2).finalizeVaultOngoingVariableWithdrawals()
                ).wait()
                const gasFees = calculateGasFees(receipt)
                const balanceAfter = await ethers.provider.getBalance(addr2)
                // since some earnings were made protocol fee should have been applied
                const protocolFee = await lidoVault.appliedProtocolFee()
                expect(protocolFee).to.be.equalWithTolerance(
                  stakingEarnings * BigInt(DEFAULTS.protocolFeeBps) / BIG_INT_10K,
                  1
                )                
                const earningsShare =
                  (bearerTokenBalance * (stakingEarnings - protocolFee)) /
                  variableBearerTokenTotalSupply

                expect(await lidoVault.vaultEndedStakingEarnings()).to.equalWithTolerance(
                  stakingEarnings - protocolFee - earningsShare,
                  10
                )
                expect(balanceAfter - balanceBefore).to.equalWithTolerance(
                  earningsShare - gasFees,
                  10
                )
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(
                  variableDeposit2
                )

                // withdraw again should burn bearer tokens and withdraw 0 ETH
                const balanceBefore2 = await ethers.provider.getBalance(addr2)
                const receipt2 = await (
                  await lidoVault.connect(addr2).finalizeVaultEndedWithdrawals(SIDE.VARIABLE)
                ).wait()
                const gasFees2 = calculateGasFees(receipt2)
                const balanceAfter2 = await ethers.provider.getBalance(addr2)
                console.log(balanceBefore2, balanceAfter2, gasFees2)
                expect(balanceBefore2 - balanceAfter2).to.equalWithTolerance(gasFees2, 1)
                expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
              })
            })
          })
        })
      })

      describe('Fixed And Variable Side', function () {
        it('Should be able to withdraw as normal (vault-ended, fixed-and-variable-side)', async function () {
          const {
            lidoVault,
            lidoWithdrawalQueueContract,
            addr1,
            fixedETHDepositToken,
            fixedBearerToken,
            variableBearerToken,
            lidoContract,
            lidoVaultAddress,
          } = await loadFixture(deployLidoVaultFixture)

          await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
          await lidoVault.connect(addr1).deposit(SIDE.VARIABLE, { value: variableDeposit })

          await lidoVault.connect(addr1).claimFixedPremium()

          // generate earnings
          await submitOracleReport()

          const lidoETHAmount = await lidoContract.balanceOf(lidoVaultAddress)
          const earnings = lidoETHAmount - fixedDeposit
          const protocolFee = (earnings * BigInt(DEFAULTS.protocolFeeBps)) / BIG_INT_10K
          const earningsAndFees = earnings - protocolFee

          // end vault
          const { endTime } = await getTimeState(lidoVault)
          await time.increaseTo(endTime + BIG_INT_ONE)

          expect(await lidoVault.isEnded()).to.equal(true)

          const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()

          // fixed side withdraw
          await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

          const requestIds = [lidoLastRequestId + BIG_INT_ONE, lidoLastRequestId + BigInt(2)]
          await expectVaultEndedWithdrawalArraysAreEqual(lidoVault, requestIds)
          await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

          const balanceBefore = await ethers.provider.getBalance(addr1)

          const receipt = await (
            await lidoVault.connect(addr1).finalizeVaultEndedWithdrawals(SIDE.FIXED)
          ).wait()
          const gasFees = calculateGasFees(receipt)
          const balanceAfter = await ethers.provider.getBalance(addr1)
          const amountWithdrawn = balanceAfter - balanceBefore

          expect(amountWithdrawn).to.equalWithTolerance(fixedDeposit - gasFees, 1)
          expect(await lidoVault.fixedBearerToken(addr1.address)).to.be.equal(0)

          // variable side withdraw
          const balanceBeforeVariable = await ethers.provider.getBalance(addr1)
          const receiptVariable = await (
            await lidoVault.connect(addr1).withdraw(SIDE.VARIABLE)
          ).wait()
          const gasFeesVariable = calculateGasFees(receiptVariable)
          const balanceAfterVariable = await ethers.provider.getBalance(addr1)
          const amountWithdrawnVariable = balanceAfterVariable - balanceBeforeVariable

          expect(amountWithdrawnVariable).to.equalWithTolerance(earningsAndFees - gasFeesVariable, 10)
          expect(await lidoVault.variableBearerToken(addr1.address)).to.equal(0)
        })

        it('Should be able to withdraw as normal even if fixed side withdrew the entire balance before the vault ended (vault-ended, fixed-and-variable-side)', async function () {
          const {
            lidoVault,
            lidoWithdrawalQueueContract,
            addr1,
            addr2,
            fixedETHDepositToken,
            fixedBearerToken,
            variableBearerToken,
          } = await loadFixture(deployLidoVaultFixture)

          await lidoVault.connect(addr1).deposit(SIDE.FIXED, { value: fixedDeposit })
          await lidoVault
            .connect(addr1)
            .deposit(SIDE.VARIABLE, { value: (variableDeposit * BigInt(1)) / BigInt(10) })
          await lidoVault
            .connect(addr2)
            .deposit(SIDE.VARIABLE, { value: (variableDeposit * BigInt(9)) / BigInt(10) })

          await lidoVault.connect(addr1).claimFixedPremium()

          expect(await lidoVault.isStarted()).to.be.true

          // withdraw fixed
          const lidoLastRequestId = await lidoWithdrawalQueueContract.getLastRequestId()
          const requestIds = [lidoLastRequestId + BIG_INT_ONE]
          await lidoVault.connect(addr1).withdraw(SIDE.FIXED)

          const { payBackAmount, scaledEarlyExitFee } = await calculateFees({
            lidoVault,
            upfrontPremium: variableDeposit,
            timestamp: await lidoVault.getFixedOngoingWithdrawalRequestTimestamp(addr1),
          })

          // end vault
          const { endTime } = await getTimeState(lidoVault)
          await time.increaseTo(endTime + BIG_INT_ONE)

          expect(await lidoVault.isEnded()).to.equal(true)

          await finalizeLidoWithdrawalRequests(requestIds, DEFAULTS.fixedSideCapacity)

          // variable side withdraw
          const balanceBeforeVariable = await ethers.provider.getBalance(addr2)
          const receiptVariable = await (
            await lidoVault.connect(addr2).withdraw(SIDE.VARIABLE)
          ).wait()
          const gasFeesVariable = calculateGasFees(receiptVariable)
          const balanceAfterVariable = await ethers.provider.getBalance(addr2)
          const amountWithdrawnVariable = balanceAfterVariable - balanceBeforeVariable

          expect(amountWithdrawnVariable).to.equal(
            ((payBackAmount + scaledEarlyExitFee) * BigInt(9)) / BigInt(10) - gasFeesVariable
          )
          expect(await lidoVault.variableBearerToken(addr2.address)).to.equal(0)
          // should have claimed ongoing fixed withdrawal
          expectVaultOngoingFixedWithdrawalArraysAreEqual(lidoVault, [], addr1.address)
          // addr1 can no longer claim withdrawal too
          await expect(
            lidoVault.connect(addr1).finalizeVaultOngoingFixedWithdrawals()
          ).to.be.revertedWith('WNR')

          // addr1 fixed side withdraw
          const balanceBefore = await ethers.provider.getBalance(addr1)
          const receipt = await (await lidoVault.connect(addr1).withdraw(SIDE.FIXED)).wait()
          const gasFees = calculateGasFees(receipt)
          const balanceAfter = await ethers.provider.getBalance(addr1)
          const amountWithdrawn = balanceAfter - balanceBefore

          expect(amountWithdrawn).to.equalWithTolerance(
            fixedDeposit - (gasFees + scaledEarlyExitFee + payBackAmount),
            1
          )
          expect(await lidoVault.fixedBearerToken(addr1.address)).to.be.equal(0)

          // addr1 variable side withdraw
          const balanceBeforeVariable1 = await ethers.provider.getBalance(addr1)
          const receiptVariable1 = await (
            await lidoVault.connect(addr1).withdraw(SIDE.VARIABLE)
          ).wait()
          const gasFeesVariable1 = calculateGasFees(receiptVariable1)
          const balanceAfterVariable1 = await ethers.provider.getBalance(addr1)
          const amountWithdrawnVariable1 = balanceAfterVariable1 - balanceBeforeVariable1

          expect(amountWithdrawnVariable1).to.equalWithTolerance(
            ((payBackAmount + scaledEarlyExitFee) * BigInt(1)) / BigInt(10) - gasFeesVariable1,
            1
          )
          expect(await lidoVault.variableBearerToken(addr1.address)).to.equal(0)
        })
      })
    })
  })
  
   
})
