import chai, { expect } from 'chai'
import chaiAsPromised from 'chai-as-promised'
import { BigNumber, Signer, Wallet } from 'ethers'
import { hexlify } from 'ethers/lib/utils'
import hre, { getNamedSigner } from 'hardhat'
import { getMnemonic } from 'hardhat.config'
import { deploy } from 'helpers/deploy-helpers'
import moment from 'moment'
import { EIP712Utils } from 'test/helpers/EIP712Utils'
import {
  TellerAS,
  TellerASRegistry,
  TellerASEIP712Verifier,
  TestASAttestationResolver,
  TestASAttesterResolver,
  TestASDataResolver,
  TestASExpirationTimeResolver,
  TestASPayingResolver,
  TestASRecipientResolver,
  TestASTokenResolver,
  TestASValueResolver,
  TestERC20Token,
} from 'types/typechain'

chai.should()
chai.use(chaiAsPromised)

const { ethers, toBN, deployments } = hre

// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface SetupOptions {}

interface SetupReturn {
  tellerAS: TellerAS
  registry: TellerASRegistry
  tellerASEIP712Verifier: TellerASEIP712Verifier
  sender: Wallet
  sender2: Wallet
  recipient: Signer
  recipient2: Signer
}

const getASUUID = (schemaAsBytes: string, resolverAddress: string): string =>
  ethers.utils.solidityKeccak256(
    ['bytes', 'address'],
    [schemaAsBytes, resolverAddress]
  )
interface Options {
  from?: Wallet
  value?: BigNumber
}

const latest = async (): Promise<BigNumber> => {
  const block = await ethers.provider.getBlock('latest')
  return BigNumber.from(block.timestamp)
}

const setup = deployments.createFixture<SetupReturn, SetupOptions>(
  async (hre, _opts) => {
    await hre.deployments.fixture('teller-as', {
      keepExistingDeployments: false,
    })

    const tellerAS = await hre.contracts.get<TellerAS>('TellerAS')
    const tellerASEIP712Verifier =
      await hre.contracts.get<TellerASEIP712Verifier>('TellerASEIP712Verifier')
    const registry = await hre.contracts.get<TellerASRegistry>(
      'TellerASRegistry'
    )

    const mnemonicPhrase = getMnemonic()

    // eslint-disable-next-line @typescript-eslint/no-unused-expressions
    expect(mnemonicPhrase).to.exist

    const sender = hre.ethers.Wallet.fromMnemonic(
      mnemonicPhrase,
      `m/44'/60'/0'/0/0`
    ).connect(hre.ethers.provider)

    const sender2 = hre.ethers.Wallet.fromMnemonic(
      mnemonicPhrase,
      `m/44'/60'/0'/0/1`
    ).connect(hre.ethers.provider)

    const recipient = await getNamedSigner('borrower')
    const recipient2 = await getNamedSigner('borrower2')

    return {
      tellerAS,
      tellerASEIP712Verifier,
      registry,
      sender,
      sender2,
      recipient,
      recipient2,
    }
  }
)

describe('TellerAS', () => {
  let tellerAS: TellerAS
  let tellerASEIP712Verifier: TellerASEIP712Verifier
  let registry: TellerASRegistry
  let sender: Wallet
  let sender2: Wallet
  let recipient: Signer
  let recipient2: Signer
  let eip712Utils: EIP712Utils
  const data = '0x1234'
  const ZERO_BYTES = '0x'
  const ZERO_BYTES32 =
    '0x0000000000000000000000000000000000000000000000000000000000000000'
  let token: TestERC20Token
  const formatBytes32String = ethers.utils.formatBytes32String
  const AddressZero = ethers.constants.AddressZero

  beforeEach(async () => {
    const result = await setup()
    tellerAS = result.tellerAS
    tellerASEIP712Verifier = result.tellerASEIP712Verifier
    registry = result.registry
    sender = result.sender
    sender2 = result.sender2
    recipient = result.recipient
    recipient2 = result.recipient2
    eip712Utils = new EIP712Utils(tellerASEIP712Verifier.address)
  })

  describe('construction', () => {
    it('TellerAS should report a version', async () => {
      expect(await tellerAS.VERSION()).to.eq('0.8')
    })
    it('should initialize without any attestation categories or attestations', async () => {
      expect(await tellerAS.getAttestationsCount()).to.eq('0')
    })
  })
  describe('attesting', () => {
    let expirationTime: BigNumber

    beforeEach(async () => {
      expirationTime = toBN(moment.now()).add(
        moment.duration(30, 'days').milliseconds()
      )
    })

    for (const delegation of [false, true]) {
      context(
        `${delegation ? 'via an EIP712 delegation' : 'directly'}`,
        // eslint-disable-next-line @typescript-eslint/no-misused-promises
        async () => {
          // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
          const testAttestation = async (
            recipient: Signer | string,
            schema: string,
            expirationTime: BigNumber,
            refUUID: string,
            data: any,
            options?: Options
          ) => {
            const txSender = options?.from ?? sender
            const recipientAddress =
              typeof recipient !== 'string'
                ? await recipient.getAddress()
                : recipient

            const prevAttestationCount = await tellerAS.getAttestationsCount()
            const prevReceivedAttestationsUUIDsCount =
              await tellerAS.getReceivedAttestationUUIDsCount(
                recipientAddress,
                schema
              )
            const prevSentAttestationsUUIDsCount =
              await tellerAS.getSentAttestationUUIDsCount(
                await txSender.getAddress(),
                schema
              )
            const prevSchemaAttestationsUUIDsCount =
              await tellerAS.getSchemaAttestationUUIDsCount(schema)
            const prevRelatedAttestationsUUIDsCount =
              await tellerAS.getRelatedAttestationUUIDsCount(refUUID)

            let res
            if (!delegation) {
              res = await tellerAS
                .connect(txSender)
                .attest(
                  recipientAddress,
                  schema,
                  expirationTime,
                  refUUID,
                  data,
                  {
                    value: options?.value,
                  }
                )
            } else {
              const request = await eip712Utils.getAttestationRequest(
                recipientAddress,
                schema,
                expirationTime,
                refUUID,
                data,
                await tellerASEIP712Verifier.getNonce(
                  await txSender.getAddress()
                ),
                txSender.privateKey
              )
              res = await tellerAS
                .connect(txSender)
                .attestByDelegation(
                  recipientAddress,
                  schema,
                  expirationTime,
                  refUUID,
                  data,
                  await txSender.getAddress(),
                  request.v,
                  hexlify(request.r),
                  hexlify(request.s),
                  { value: options?.value }
                )

              const lastUUID = await tellerAS.getLastUUID()

              await expect(res)
                .to.emit(tellerAS, 'Attested')
                .withArgs(
                  recipientAddress,
                  await txSender.getAddress(),
                  lastUUID,
                  schema
                )

              const attestation = await tellerAS.getAttestation(
                lastUUID.toString()
              )
              expect(attestation.uuid).to.eq(
                lastUUID,
                'Attestation - incorrect UUID'
              )
              expect(attestation.schema).to.eq(
                schema,
                'Attestation - incorrect schema'
              )
              expect(attestation.recipient).to.eq(
                recipientAddress,
                'Attestation - incorrect recopient'
              )
              expect(attestation.attester).to.eq(
                await txSender.getAddress(),
                'Attestation - incorrect attester'
              )
              expect(attestation.expirationTime).to.eq(
                expirationTime,
                'Attestation - incorrect expiration time'
              )
              expect(attestation.revocationTime).to.eq(
                '0',
                'Attestation - incorrect revocation time'
              )
              expect(attestation.refUUID).to.eq(
                refUUID,
                'Attestation - incorrect refUUID'
              )
              expect(attestation.data).to.eq(
                data,
                'Attestation - incorrect data'
              )

              const receivedAttestationUUIDsCount =
                await tellerAS.getReceivedAttestationUUIDsCount(
                  recipientAddress,
                  schema
                )
              expect(receivedAttestationUUIDsCount).to.eq(
                prevReceivedAttestationsUUIDsCount.add('1'),
                'Incorrect received attestaion UUIDs count'
              )

              const receivedAttestationUUIDs =
                await tellerAS.getReceivedAttestationUUIDs(
                  recipientAddress,
                  schema,
                  0,
                  receivedAttestationUUIDsCount,
                  false
                )
              expect(receivedAttestationUUIDs).to.have.lengthOf(
                receivedAttestationUUIDsCount.toNumber(),
                'Incorrect received attestation UUIDs'
              )
              expect(
                receivedAttestationUUIDs[receivedAttestationUUIDs.length - 1]
              ).to.eq(attestation.uuid, 'Incorrect received attestation UUID')

              const sentAttestationUUIDsCount =
                await tellerAS.getSentAttestationUUIDsCount(
                  await txSender.getAddress(),
                  schema
                )
              expect(sentAttestationUUIDsCount).to.eq(
                prevSentAttestationsUUIDsCount.add('1'),
                'Incorrect sent attestation UUIDs count'
              )

              const sentAttestationUUIDs =
                await tellerAS.getSentAttestationUUIDs(
                  await txSender.getAddress(),
                  schema,
                  0,
                  sentAttestationUUIDsCount,
                  false
                )
              expect(sentAttestationUUIDs).to.have.lengthOf(
                sentAttestationUUIDsCount.toNumber(),
                'Incorrect sent attestation UUIDs'
              )
              expect(
                sentAttestationUUIDs[sentAttestationUUIDs.length - 1]
              ).to.eq(attestation.uuid, 'Incorrect sent attestation UUID')

              if (refUUID !== ZERO_BYTES32) {
                const relatedAttestationsUUIDsCount =
                  await tellerAS.getRelatedAttestationUUIDsCount(refUUID)
                expect(relatedAttestationsUUIDsCount).to.eq(
                  prevRelatedAttestationsUUIDsCount.add('1'),
                  'Incorrect related attesations UUIDS count'
                )

                const relatedAttestationsUUIDs =
                  await tellerAS.getRelatedAttestationUUIDs(
                    refUUID,
                    0,
                    receivedAttestationUUIDsCount,
                    false
                  )
                expect(relatedAttestationsUUIDs).to.have.lengthOf(
                  relatedAttestationsUUIDsCount.toNumber(),
                  'Incorrect related attesations UUIDS'
                )
                expect(
                  relatedAttestationsUUIDs[relatedAttestationsUUIDs.length - 1]
                ).to.eq(attestation.uuid, 'Incorrect related attestation UUID')
              }
            }
          }
          // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
          const testFailedAttestation = async (
            recipient: Signer,
            schema: string,
            expirationTime: BigNumber,
            refUUID: string,
            data: any,
            err: string,
            options?: Options
          ) => {
            const txSender = options?.from ?? sender
            const recipientAddress = await recipient.getAddress()

            if (!delegation) {
              await expect(
                tellerAS
                  .connect(txSender)
                  .attest(
                    recipientAddress,
                    schema,
                    expirationTime,
                    refUUID,
                    data,
                    { value: options?.value }
                  )
              ).to.be.revertedWith(err)
            } else {
              const request = await eip712Utils.getAttestationRequest(
                recipientAddress,
                schema,
                expirationTime,
                refUUID,
                data,
                await tellerASEIP712Verifier.getNonce(
                  await txSender.getAddress()
                ),
                txSender.privateKey
              )

              await expect(
                tellerAS
                  .connect(txSender)
                  .attestByDelegation(
                    recipientAddress,
                    schema,
                    expirationTime,
                    refUUID,
                    data,
                    await txSender.getAddress(),
                    request.v,
                    hexlify(request.r),
                    hexlify(request.s),
                    { value: options?.value }
                  )
              ).to.be.revertedWith(err)
            }
          }

          it('should be reverted when attesting to an unregistered schema', async () => {
            await testFailedAttestation(
              recipient,
              formatBytes32String('BAD'),
              expirationTime,
              ZERO_BYTES32,
              data,
              'InvalidSchema'
            )
          })
          // eslint-disable-next-line @typescript-eslint/no-misused-promises
          context('with registered schemas', async () => {
            const schema1 = formatBytes32String('AS1')
            const schema2 = formatBytes32String('AS2')
            const schema3 = formatBytes32String('AS3')
            const schema1Id = getASUUID(schema1, AddressZero)
            const schema2Id = getASUUID(schema2, AddressZero)
            const schema3Id = getASUUID(schema3, AddressZero)

            beforeEach(async () => {
              await registry.register(schema1, AddressZero)
              await registry.register(schema2, AddressZero)
              await registry.register(schema3, AddressZero)
            })

            it('should revert when attesting with passed expiration time', async () => {
              const expired = (await latest()).sub(
                toBN(moment.duration(1, 'days').asMilliseconds())
              )
              await testFailedAttestation(
                recipient,
                schema1Id,
                expired,
                ZERO_BYTES32,
                data,
                'InvalidExpirationTime'
              )
            })

            it('should allow attestation to an empty recipient', async () => {
              await testAttestation(
                AddressZero,
                schema1Id,
                expirationTime,
                ZERO_BYTES32,
                data
              )
            })

            it('should allow self attestations', async () => {
              await testAttestation(
                sender,
                schema2Id,
                expirationTime,
                ZERO_BYTES32,
                data,
                { from: sender }
              )
            })

            it('should allow multiple attestations', async () => {
              await testAttestation(
                recipient,
                schema1Id,
                expirationTime,
                ZERO_BYTES32,
                data
              )
              await testAttestation(
                sender,
                schema1Id,
                expirationTime,
                ZERO_BYTES32,
                data
              )
            })

            it('should allow multiple attestations to the same schema', async () => {
              await testAttestation(
                recipient,
                schema3Id,
                expirationTime,
                ZERO_BYTES32,
                data
              )
              await testAttestation(
                recipient,
                schema3Id,
                expirationTime,
                ZERO_BYTES32,
                data
              )
              await testAttestation(
                recipient,
                schema3Id,
                expirationTime,
                ZERO_BYTES32,
                data
              )
            })

            it('should allow attestation without expiration time', async () => {
              await testAttestation(
                recipient,
                schema1Id,
                ethers.constants.MaxUint256,
                ZERO_BYTES32,
                data
              )
            })

            it('should allow attestation without any data', async () => {
              await testAttestation(
                recipient,
                schema3Id,
                expirationTime,
                ZERO_BYTES32,
                ZERO_BYTES
              )
            })

            it('should store referenced attestation', async () => {
              await tellerAS.attest(
                await recipient.getAddress(),
                schema1Id,
                expirationTime,
                ZERO_BYTES32,
                data
              )
              const uuid = await tellerAS.getLastUUID()

              await testAttestation(
                recipient,
                schema3Id,
                expirationTime,
                uuid,
                data
              )
            })

            it('should revert when attesting to a non-existing attestation', async () => {
              await testFailedAttestation(
                recipient,
                schema3Id,
                expirationTime,
                formatBytes32String('INVALID'),
                data,
                'NotFound'
              )
            })

            it('should revert when sending ETH to a non-payable resolver', async () => {
              const schema4 = formatBytes32String('AS4')
              const targetRecipient = await getNamedSigner('borrower2')

              const resolver = await deploy<TestASRecipientResolver>({
                contract: 'TestASRecipientResolver',
                args: [await targetRecipient.getAddress()],
                hre,
              })
              expect(await resolver.isPayable()).to.eq(false)

              await expect(
                sender.sendTransaction({ to: resolver.address, value: toBN(1) })
              ).to.be.revertedWith('NotPayable')

              await registry.register(schema4, resolver.address)
              const schema4Id = getASUUID(schema4, resolver.address)

              await testFailedAttestation(
                recipient,
                schema4Id,
                expirationTime,
                ZERO_BYTES32,
                data,
                'NotPayable',
                { value: toBN(1) }
              )
            })
            context('with recipient resolver', () => {
              const schema4 = formatBytes32String('AS4')
              let schema4Id: string
              let targetRecipient: Signer

              beforeEach(async () => {
                targetRecipient = await getNamedSigner('borrower2')

                const resolver = await deploy<TestASRecipientResolver>({
                  contract: 'TestASRecipientResolver',
                  args: [await targetRecipient.getAddress()],
                  hre,
                })
                expect(await resolver.isPayable()).to.eq(false)

                await registry.register(schema4, resolver.address)
                schema4Id = getASUUID(schema4, resolver.address)
              })

              it('should revert when attesting to a wrong recipient', async () => {
                await testFailedAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data,
                  'InvalidAttestation'
                )
              })

              it('should allow attesting to the correct recipient', async () => {
                await testAttestation(
                  targetRecipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data
                )
              })
            })
            context('with data resolver', () => {
              const schema4 = formatBytes32String('schema4Id')
              let schema4Id: string

              beforeEach(async () => {
                const resolver = await deploy<TestASDataResolver>({
                  contract: 'TestASDataResolver',
                  hre,
                })
                expect(await resolver.isPayable()).to.eq(false)

                await registry.register(schema4, resolver.address)
                schema4Id = getASUUID(schema4, resolver.address)
              })

              it('should revert when attesting with wrong data', async () => {
                await testFailedAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  '0x1234',
                  'InvalidAttestation'
                )

                await testFailedAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  '0x02',
                  'InvalidAttestation'
                )
              })

              it('should allow attesting with correct data', async () => {
                await testAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  '0x00'
                )
                await testAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  '0x01'
                )
              })
            })
            context('with expiration time resolver', () => {
              const schema4 = formatBytes32String('schema4Id')
              let schema4Id: string
              let validAfter: BigNumber

              beforeEach(async () => {
                validAfter = toBN(moment.now()).add(
                  toBN(moment.duration(1, 'year').asMilliseconds())
                )
                const resolver = await deploy<TestASExpirationTimeResolver>({
                  contract: 'TestASExpirationTimeResolver',
                  args: [validAfter],
                  hre,
                })
                expect(await resolver.isPayable()).to.eq(false)

                await registry.register(schema4, resolver.address)
                schema4Id = getASUUID(schema4, resolver.address)
              })

              it('should revert when attesting with a wrong expiration time', async () => {
                await testFailedAttestation(
                  recipient,
                  schema4Id,
                  validAfter.sub(
                    toBN(moment.duration(1, 'day').asMilliseconds())
                  ),
                  ZERO_BYTES32,
                  data,
                  'InvalidAttestation'
                )
              })

              it('should allow attesting with the correct expiration time', async () => {
                await testAttestation(
                  recipient,
                  schema4Id,
                  validAfter.add(
                    toBN(moment.duration(1, 'seconds').asMilliseconds())
                  ),
                  ZERO_BYTES32,
                  data
                )
              })
            })
            context('with msg.sender resolver', () => {
              const schema4 = formatBytes32String('schema4Id')
              let schema4Id: string
              let targetSender: Wallet

              beforeEach(async () => {
                targetSender = hre.ethers.Wallet.createRandom().connect(
                  hre.ethers.provider
                )

                await hre.network.provider.send('hardhat_setBalance', [
                  await targetSender.getAddress(),
                  '0x3635C9ADC5DEA00000', // Fund created account ^
                ])

                const resolver = await deploy<TestASAttesterResolver>({
                  contract: 'TestASAttesterResolver',
                  args: [await targetSender.getAddress()],
                  hre,
                })
                expect(await resolver.isPayable()).to.eq(false)

                await registry.register(schema4, resolver.address)
                schema4Id = getASUUID(schema4, resolver.address)
              })

              it('should revert when attesting to the wrong msg.sender', async () => {
                await testFailedAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data,
                  'InvalidAttestation',
                  {
                    from: sender,
                  }
                )
              })

              it('should allow attesting to the correct msg.sender', async () => {
                await testAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data,
                  {
                    from: targetSender,
                  }
                )
              })
            })
            context('with msg.value resolver', () => {
              const schema4 = formatBytes32String('schema4Id')
              let schema4Id: string
              const targetValue = BigNumber.from(7862432)

              beforeEach(async () => {
                const resolver = await deploy<TestASValueResolver>({
                  contract: 'TestASValueResolver',
                  args: [targetValue],
                  hre,
                })
                expect(await resolver.isPayable()).to.eq(true)

                await registry.register(schema4, resolver.address)
                schema4Id = getASUUID(schema4, resolver.address)
              })

              it('should revert when attesting with wrong msg.value', async () => {
                await testFailedAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data,
                  'InvalidAttestation'
                )
                await testFailedAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data,
                  'InvalidAttestation',
                  {
                    value: targetValue.sub(BigNumber.from(1)),
                  }
                )
              })
              it('should allow attesting with correct msg.value', async () => {
                await testAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data,
                  {
                    value: targetValue,
                  }
                )
              })
            })
            context('with token resolver', () => {
              const schema4 = formatBytes32String('schema4Id')
              let schema4Id: string
              const targetAmount = BigNumber.from(22334)
              let resolver: TestASTokenResolver

              beforeEach(async () => {
                token = await deploy<TestERC20Token>({
                  contract: 'TestERC20Token',
                  args: ['TKN', 'TKN', BigNumber.from(9999999999)],
                  hre,
                })
                resolver = await deploy<TestASTokenResolver>({
                  contract: 'TestASTokenResolver',
                  args: [token.address, targetAmount],
                  hre,
                })
                expect(await resolver.isPayable()).to.eq(false)

                await registry.register(schema4, resolver.address)
                schema4Id = getASUUID(schema4, resolver.address)
              })

              it('should revert when attesting with wrong token amount', async () => {
                await testFailedAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data,
                  'ERC20: insufficient allowance'
                )

                await token.approve(
                  resolver.address,
                  targetAmount.sub(BigNumber.from(1))
                )
                await testFailedAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data,
                  'ERC20: insufficient allowance'
                )
              })

              it('should allow attesting with correct token amount', async () => {
                await token.approve(resolver.address, targetAmount)
                await testAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data
                )
              })
            })
            context('with attestation resolver', () => {
              const schema4 = formatBytes32String('schema4Id')
              let schema4Id: string
              let uuid: string

              beforeEach(async () => {
                await tellerAS.attest(
                  await recipient.getAddress(),
                  schema1Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data
                )
                uuid = await tellerAS.getLastUUID()

                const resolver = await deploy<TestASAttestationResolver>({
                  contract: 'TestASAttestationResolver',
                  args: [tellerAS.address],
                  hre,
                })
                expect(await resolver.isPayable()).to.eq(false)

                await registry.register(schema4, resolver.address)
                schema4Id = getASUUID(schema4, resolver.address)
              })

              it('should revert when attesting to a non-existing attestation', async () => {
                await testFailedAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  ZERO_BYTES32,
                  'InvalidAttestation'
                )
              })

              it('should allow attesting to an existing attestation', async () => {
                await testAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  uuid
                )
              })
            })
            context('with paying resolver', () => {
              const schema4 = formatBytes32String('schema4Id')
              let schema4Id: string
              const incentive = BigNumber.from(1000)
              let resolver: TestASPayingResolver

              beforeEach(async () => {
                resolver = await deploy<TestASPayingResolver>({
                  contract: 'TestASPayingResolver',
                  args: [incentive],
                  hre,
                })
                expect(await resolver.isPayable()).to.eq(true)

                await sender.sendTransaction({
                  to: resolver.address,
                  value: incentive.mul(BigNumber.from(2)),
                })

                await registry.register(schema4, resolver.address)
                schema4Id = getASUUID(schema4, resolver.address)
              })

              const getBalance = ethers.provider.getBalance

              it('should incentivize attesters', async () => {
                const prevResolverBalance = await getBalance(resolver.address)
                const prevRecipient2Balance = await getBalance(
                  await recipient.getAddress()
                )

                await testAttestation(
                  recipient,
                  schema4Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data
                )

                expect(await getBalance(resolver.address)).to.equal(
                  prevResolverBalance.sub(incentive)
                )
                expect(await getBalance(await recipient.getAddress())).to.equal(
                  prevRecipient2Balance.add(incentive)
                )
              })
            })
          })
          it('should revert when delegation attesting with a wrong signature', async () => {
            await expect(
              tellerAS.attestByDelegation(
                await recipient.getAddress(),
                formatBytes32String('BAD'),
                expirationTime,
                ZERO_BYTES32,
                ZERO_BYTES32,
                await sender.getAddress(),
                28,
                formatBytes32String('BAD'),
                formatBytes32String('BAD')
              )
            ).to.be.revertedWith('InvalidSignature')
          })
        }
      )
      describe('revocation', () => {
        const schema1 = formatBytes32String('AS1')
        const schema1Id = getASUUID(schema1, AddressZero)
        let uuid: string

        let expirationTime: BigNumber
        const data = '0x1234'

        beforeEach(async () => {
          await registry.register(schema1, AddressZero)

          const now = await latest()
          expirationTime = now.add(
            toBN(moment.duration(30, 'days').asMilliseconds())
          )
        })
        for (const delegation of [false, true]) {
          // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
          const testRevocation = async (uuid: string, options?: Options) => {
            const txSender = options?.from ?? sender
            let res

            if (!delegation) {
              res = await tellerAS.connect(txSender).revoke(uuid)
            } else {
              const request = await eip712Utils.getRevocationRequest(
                uuid,
                await tellerASEIP712Verifier.getNonce(
                  await txSender.getAddress()
                ),
                txSender,
                txSender.privateKey
              )

              res = await tellerAS
                .connect(txSender)
                .revokeByDelegation(
                  uuid,
                  await txSender.getAddress(),
                  request.v,
                  hexlify(request.r),
                  hexlify(request.s)
                )
            }

            await expect(res)
              .to.emit(tellerAS, 'Revoked')
              .withArgs(
                await recipient.getAddress(),
                await txSender.getAddress(),
                uuid,
                schema1Id
              )

            const attestation = await tellerAS.getAttestation(uuid)
            expect(attestation.revocationTime).to.equal(await latest())
          }

          // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
          const testFailedRevocation = async (
            uuid: string,
            err: string,
            options?: Options
          ) => {
            const txSender = options?.from ?? sender

            if (!delegation) {
              await expect(
                tellerAS.connect(txSender).revoke(uuid)
              ).to.be.revertedWith(err)
            } else {
              const request = await eip712Utils.getRevocationRequest(
                uuid,
                await tellerASEIP712Verifier.getNonce(
                  await txSender.getAddress()
                ),
                txSender,
                txSender.privateKey
              )

              await expect(
                tellerAS.revokeByDelegation(
                  uuid,
                  await txSender.getAddress(),
                  request.v,
                  hexlify(request.r),
                  hexlify(request.s)
                )
              ).to.be.revertedWith(err)
            }
          }

          context(
            `${delegation ? 'via an EIP712 delegation' : 'directly'}`,
            () => {
              beforeEach(async () => {
                await tellerAS.attest(
                  await recipient.getAddress(),
                  schema1Id,
                  expirationTime,
                  ZERO_BYTES32,
                  data
                )
                uuid = await tellerAS.getLastUUID()
              })

              it('should revert when revoking a non-existing attestation', async () => {
                await testFailedRevocation(
                  formatBytes32String('BAD'),
                  'NotFound'
                )
              })

              it("should revert when revoking a someone's else attestation", async () => {
                await testFailedRevocation(uuid, 'AccessDenied', {
                  from: sender2,
                })
              })

              it('should allow to revoke an existing attestation', async () => {
                await testRevocation(uuid)
              })

              it('should revert when revoking an already revoked attestation', async () => {
                await testRevocation(uuid)
                await testFailedRevocation(uuid, 'AlreadyRevoked')
              })
            }
          )
        }

        it('should revert when delegation revoking with a wrong signature', async () => {
          await expect(
            tellerAS.revokeByDelegation(
              ZERO_BYTES32,
              await sender.getAddress(),
              28,
              formatBytes32String('BAD'),
              formatBytes32String('BAD')
            )
          ).to.be.revertedWith('InvalidSignature')
        })
      })
      describe('pagination', () => {
        const attestationsCount = 100

        const data = '0x1234'
        let refUUID: string

        const sentAttestations: { [key: string]: string[] } = {}
        const receivedAttestations: { [key: string]: string[] } = {}
        const schemaAttestations: { [key: string]: string[] } = {}
        const relatedAttestations: { [key: string]: string[] } = {}

        const schema1 = formatBytes32String('AS1')
        const schema2 = formatBytes32String('AS2')
        const schema3 = formatBytes32String('AS3')
        const schema1Id = getASUUID(schema1, AddressZero)
        const schema2Id = getASUUID(schema2, AddressZero)
        const schema3Id = getASUUID(schema3, AddressZero)

        before(async () => {
          const result = await setup()
          registry = result.registry
          tellerASEIP712Verifier = result.tellerASEIP712Verifier
          tellerAS = result.tellerAS
          sender = result.sender
          sender2 = result.sender2
          recipient = result.recipient
          recipient2 = result.recipient2
          await registry.register(schema1, AddressZero)
          await registry.register(schema2, AddressZero)
          await registry.register(schema3, AddressZero)

          await tellerAS
            .connect(sender2)
            .attest(
              await recipient2.getAddress(),
              schema2Id,
              ethers.constants.MaxUint256,
              ZERO_BYTES32,
              data
            )
          refUUID = await tellerAS.getLastUUID()

          sentAttestations[await sender.getAddress()] = []
          receivedAttestations[await recipient.getAddress()] = []
          schemaAttestations[schema1Id] = []
          relatedAttestations[refUUID] = []

          for (let i = 0; i < attestationsCount; ++i) {
            await tellerAS
              .connect(sender)
              .attest(
                await recipient.getAddress(),
                schema1Id,
                ethers.constants.MaxUint256,
                refUUID,
                data
              )

            const uuid = await tellerAS.getLastUUID()
            sentAttestations[await sender.getAddress()].push(uuid)
            receivedAttestations[await recipient.getAddress()].push(uuid)
            schemaAttestations[schema1Id].push(uuid)
            relatedAttestations[refUUID].push(uuid)
          }
        })

        const scenarios1 = [
          [0, attestationsCount],
          [0, 1],
          [10, 1],
          [0, 50],
          [1, 90],
          [80, attestationsCount - 20],
          [95, attestationsCount - 5],
          [99, attestationsCount - 1],
        ]
        scenarios1.forEach((slice) => {
          describe(`slice [${slice}]`, () => {
            const [start, length] = slice

            it('should return an empty array of received attestations', async () => {
              expect(
                await tellerAS.getReceivedAttestationUUIDs(
                  await recipient.getAddress(),
                  schema2Id,
                  start,
                  length,
                  false
                )
              ).to.have.lengthOf(0)
              expect(
                await tellerAS.getReceivedAttestationUUIDs(
                  await recipient.getAddress(),
                  schema2Id,
                  start,
                  length,
                  true
                )
              ).to.have.lengthOf(0)
            })

            it('should return an empty array of sent attestations', async () => {
              expect(
                await tellerAS.getSentAttestationUUIDs(
                  await sender.getAddress(),
                  schema2Id,
                  start,
                  length,
                  false
                )
              ).to.have.lengthOf(0)
              expect(
                await tellerAS.getSentAttestationUUIDs(
                  await sender.getAddress(),
                  schema2Id,
                  start,
                  length,
                  true
                )
              ).to.have.lengthOf(0)
            })

            it('should return an empty array of schema attestations', async () => {
              expect(
                await tellerAS.getSchemaAttestationUUIDs(
                  schema3Id,
                  start,
                  length,
                  false
                )
              ).to.have.lengthOf(0)
              expect(
                await tellerAS.getSchemaAttestationUUIDs(
                  schema3Id,
                  start,
                  length,
                  true
                )
              ).to.have.lengthOf(0)
            })

            it('should return an empty array of related attestations', async () => {
              expect(
                await tellerAS.getRelatedAttestationUUIDs(
                  ZERO_BYTES32,
                  start,
                  length,
                  false
                )
              ).to.have.lengthOf(0)
              expect(
                await tellerAS.getRelatedAttestationUUIDs(
                  ZERO_BYTES32,
                  start,
                  length,
                  true
                )
              ).to.have.lengthOf(0)
            })
          })
        })
      })
    }
  })
})
