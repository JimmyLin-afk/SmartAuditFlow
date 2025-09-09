import chai, { expect } from 'chai'
import { solidity } from 'ethereum-waffle'
import { BigNumber, Signer, Wallet } from 'ethers'
import hre from 'hardhat'
import { getMnemonic } from 'hardhat.config'
import moment from 'moment'
import { EIP712Utils } from 'test/helpers/EIP712Utils'
import {
  MarketRegistry,
  TellerAS,
  TellerASEIP712Verifier,
} from 'types/typechain'

const { ethers, toBN, deployments } = hre

chai.should()
chai.use(solidity)

const paymentCycleDuration = moment.duration(30, 'days').asSeconds()
const loanDefaultDuration = moment.duration(180, 'days').asSeconds()
const loanExpirationDuration = moment.duration(1, 'days').asSeconds()
const feePercent = 0

const abiCoder = new ethers.utils.AbiCoder() // '0x0000'

// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface SetupOptions {}

interface SetupReturn {
  tellerAS: TellerAS
  marketRegistry: MarketRegistry
  tellerASEIP712Verifier: TellerASEIP712Verifier
  attestationWallet: Wallet
  eip712Utils: EIP712Utils
}

const setup = deployments.createFixture<SetupReturn, SetupOptions>(
  async (hre, _opts) => {
    await hre.deployments.fixture(['market-registry'], {
      keepExistingDeployments: false,
    })

    const tellerAS = await hre.contracts.get<TellerAS>('TellerAS')

    const marketRegistry = await hre.contracts.get<MarketRegistry>(
      'MarketRegistry'
    )

    const tellerASEIP712Verifier =
      await hre.contracts.get<TellerASEIP712Verifier>('TellerASEIP712Verifier')

    const mnemonicPhrase = getMnemonic()

    // eslint-disable-next-line @typescript-eslint/no-unused-expressions
    mnemonicPhrase.should.exist

    const attestationWallet = hre.ethers.Wallet.fromMnemonic(
      mnemonicPhrase,
      `m/44'/60'/0'/0/0`
    ).connect(hre.ethers.provider)

    const eip712Utils = new EIP712Utils(tellerASEIP712Verifier.address)

    return {
      tellerAS,
      marketRegistry,
      tellerASEIP712Verifier,
      attestationWallet,
      eip712Utils,
    }
  }
)

describe('MarketRegistry', () => {
  let tellerAS: TellerAS
  let marketRegistry: MarketRegistry
  let marketOwner: Signer
  let marketOwnerAddress: string
  let alternateOwner: Signer
  let alternateOwnerAddress: string
  let attestationWallet: Wallet
  let tellerASEIP712Verifier: TellerASEIP712Verifier
  let eip712Utils: EIP712Utils

  const uri = 'ipfs://QmMyDataHash'
  const uriTwo = 'ipfs://QmMyDataHashTwo'
  const uriThree = 'ipfs://QmMyDataHashThree'

  before(async () => {
    const result = await setup()
    tellerAS = result.tellerAS
    marketRegistry = result.marketRegistry
    marketOwner = await hre.getNamedSigner('lender')
    marketOwnerAddress = await marketOwner.getAddress()
    alternateOwner = await hre.getNamedSigner('borrower')
    alternateOwnerAddress = await alternateOwner.getAddress()
    attestationWallet = result.attestationWallet
    tellerASEIP712Verifier = result.tellerASEIP712Verifier
    eip712Utils = result.eip712Utils
  })
  const attestLender = async (): Promise<void> => {
    const lender = alternateOwner
    const lenderAddress = alternateOwnerAddress

    const attestationWalletAddress = await attestationWallet.getAddress()

    const marketId = 2 // requires attestation

    const expirationTime: BigNumber = toBN(moment.now()).add(
      moment.duration(30, 'days').seconds()
    )

    const schemaId = await marketRegistry.lenderAttestationSchemaId()

    const data = abiCoder.encode(
      ['uint256', 'address'],
      [marketId, lenderAddress]
    )

    const nonce = await tellerASEIP712Verifier.getNonce(
      attestationWalletAddress
    )

    const request = await eip712Utils.getAttestationRequest(
      lenderAddress,
      schemaId,
      expirationTime,
      ethers.utils.formatBytes32String(''),
      data,
      nonce,
      attestationWallet.privateKey
    )

    await marketRegistry
      .connect(lender)
      ['attestLender(uint256,address,uint256,uint8,bytes32,bytes32)'](
        marketId,
        lenderAddress,
        expirationTime,
        request.v,
        request.r,
        request.s
      )
      .should.emit(marketRegistry, 'LenderAttestation')
      .withArgs(marketId, lenderAddress)
  }
  const attestBorrower = async (): Promise<void> => {
    const borrower = alternateOwner
    const borrowerAddress = alternateOwnerAddress

    const attestationWalletAddress = await attestationWallet.getAddress()

    const marketId = 2 // requires attestation

    const expirationTime: BigNumber = toBN(moment.now()).add(
      moment.duration(30, 'days').seconds()
    )

    const schemaId = await marketRegistry.borrowerAttestationSchemaId()

    const data = abiCoder.encode(
      ['uint256', 'address'],
      [marketId, borrowerAddress]
    )

    const nonce = await tellerASEIP712Verifier.getNonce(
      attestationWalletAddress
    )

    const request = await eip712Utils.getAttestationRequest(
      borrowerAddress,
      schemaId,
      expirationTime,
      ethers.utils.formatBytes32String(''),
      data,
      nonce,
      attestationWallet.privateKey
    )

    await marketRegistry
      .connect(borrower)
      ['attestBorrower(uint256,address,uint256,uint8,bytes32,bytes32)'](
        marketId,
        borrowerAddress,
        expirationTime,
        request.v,
        request.r,
        request.s
      )
      .should.emit(marketRegistry, 'BorrowerAttestation')
      .withArgs(marketId, borrowerAddress)
  }

  describe('createMarket', () => {
    it('should create a new market', async () => {
      await marketRegistry
        .connect(marketOwner)
        [
          'createMarket(address,uint32,uint32,uint32,uint16,bool,bool,uint8,uint8,string)'
        ](
          marketOwnerAddress,
          paymentCycleDuration,
          loanDefaultDuration,
          loanExpirationDuration,
          feePercent,
          false,
          false,
          '0',
          0,
          uri
        )
        .should.emit(marketRegistry, 'MarketCreated')
        .withArgs(marketOwnerAddress, 1)
    })

    it('should fail to create the market with invalid market owner address', async () => {
      await marketRegistry
        .connect(marketOwner)
        [
          'createMarket(address,uint32,uint32,uint32,uint16,bool,bool,uint8,uint8,string)'
        ](
          ethers.constants.AddressZero,
          paymentCycleDuration,
          loanDefaultDuration,
          loanExpirationDuration,
          0,
          true,
          true,
          '0',
          0,
          uri
        )
        .should.be.revertedWith('Invalid owner address')
    })

    it('should create a second market', async () => {
      await marketRegistry
        .connect(marketOwner)
        [
          'createMarket(address,uint32,uint32,uint32,uint16,bool,bool,uint8,uint8,string)'
        ](
          marketOwnerAddress,
          paymentCycleDuration,
          loanDefaultDuration,
          loanExpirationDuration,
          0,
          true,
          true,
          '0',
          0,
          uriTwo
        )
        .should.emit(marketRegistry, 'MarketCreated')
        .withArgs(marketOwnerAddress, 2)
    })

    it('should create a third market', async () => {
      await marketRegistry
        .connect(marketOwner)
        [
          'createMarket(address,uint32,uint32,uint32,uint16,bool,bool,uint8,uint8,string)'
        ](
          marketOwnerAddress,
          paymentCycleDuration,
          loanDefaultDuration,
          loanExpirationDuration,
          0,
          true,
          true,
          '0',
          0,
          uriThree
        )
        .should.emit(marketRegistry, 'MarketCreated')
        .withArgs(marketOwnerAddress, 3)
    })

    it('should be able to create a market with same URI', async () => {
      await marketRegistry
        .connect(marketOwner)
        [
          'createMarket(address,uint32,uint32,uint32,uint16,bool,bool,uint8,uint8,string)'
        ](
          marketOwnerAddress,
          paymentCycleDuration,
          loanDefaultDuration,
          loanExpirationDuration,
          feePercent,
          true,
          true,
          '0',
          0,
          uri
        )
        .should.emit(marketRegistry, 'MarketCreated')
        .withArgs(marketOwnerAddress, 4)
    })
  })

  describe('setMarketMetadataURI', () => {
    it('should be able to update market metadata uri', async () => {
      const altURI = 'ipfs://QmMyDataHashAlt'
      // const altURIHex = `0x${Buffer.from(altURI).toString('hex')}`

      const setURI = await marketRegistry
        .connect(marketOwner)
        .setMarketURI(1, altURI)
        .should.emit(marketRegistry, 'SetMarketURI')
        .withArgs(1, altURI)

      expect(await marketRegistry.getMarketURI(1)).to.eql(altURI)
    })

    it('should not be able to update unowned market metadata uri', async () => {
      const altURI = 'ipfs://QmMyDataHashAlt'

      const setURI = await marketRegistry
        .connect(marketOwner)
        .setMarketURI(9, altURI)
        .should.be.revertedWith('Not the owner')
    })
  })

  describe('setPaymentCycle', () => {
    it('should be able to update market payment cycle duration', async () => {
      const setDuration = await marketRegistry
        .connect(marketOwner)
        .setPaymentCycle(1, 0, 60 * 60 * 60)
        .should.emit(marketRegistry, 'SetPaymentCycle')
        .withArgs(1, 0, 60 * 60 * 60)

      expect(await marketRegistry.getPaymentCycle(1)).to.eql([60 * 60 * 60, 0])
    })

    it('should not be able to update unowned market metadata uri', async () => {
      const setURI = await marketRegistry
        .connect(marketOwner)
        .setPaymentCycle(9, 0, 60 * 60 * 60)
        .should.be.revertedWith('Not the owner')
    })
  })

  describe('setBidExpirationTime', () => {
    it('should be able to update bid expiation time', async () => {
      await marketRegistry
        .connect(marketOwner)
        .setBidExpirationTime(1, 60)
        .should.emit(marketRegistry, 'SetBidExpirationTime')
        .withArgs(1, 60)

      expect(await marketRegistry.getBidExpirationTime(1)).to.eql(60)
    })
  })

  describe('transferMarketOwnership', () => {
    it('should be able to transfer market owner', async () => {
      await marketRegistry
        .connect(marketOwner)
        .transferMarketOwnership(1, alternateOwnerAddress)
        .should.emit(marketRegistry, 'SetMarketOwner')
        .withArgs(1, alternateOwnerAddress)

      expect(await marketRegistry.getMarketOwner(1)).to.eql(
        alternateOwnerAddress
      )

      await marketRegistry
        .connect(marketOwner)
        .transferMarketOwnership(1, marketOwnerAddress)
        .should.be.revertedWith('Not the owner')

      await marketRegistry
        .connect(alternateOwner)
        .transferMarketOwnership(1, marketOwnerAddress)

      expect(await marketRegistry.getMarketOwner(1)).to.eql(marketOwnerAddress)
    })
  })

  describe('closing a market', () => {
    it('should not be able to close a market if not owner', async () => {
      expect(await marketRegistry.isMarketClosed(3)).to.eql(false)

      await marketRegistry
        .connect(alternateOwner)
        .closeMarket(3)
        .should.be.revertedWith('Not the owner')
    })

    it('should be able to close a market', async () => {
      expect(await marketRegistry.isMarketClosed(3)).to.eql(false)

      await marketRegistry
        .connect(marketOwner)
        .closeMarket(3)
        .should.emit(marketRegistry, 'MarketClosed')
        .withArgs(3)

      expect(await marketRegistry.isMarketClosed(3)).to.eql(true)
    })
  })

  describe('attestLender', () => {
    it('should transfer market ownership to attestation wallet', async () => {
      const attestationWalletAddress = await attestationWallet.getAddress()

      const marketId = 2 // requires attestation

      await marketRegistry
        .connect(marketOwner)
        .transferMarketOwnership(marketId, attestationWalletAddress)

      await marketRegistry
        .getMarketOwner(marketId)
        .should.eventually.eql(attestationWalletAddress)
    })

    it('should fail if market ID is 0', async () => {
      const marketId = 0

      const lender = alternateOwner
      const lenderAddress = alternateOwnerAddress

      const attestationWalletAddress = await attestationWallet.getAddress()

      const expirationTime: BigNumber = toBN(moment.now()).add(
        moment.duration(30, 'days').seconds()
      )

      const schemaId = await marketRegistry.lenderAttestationSchemaId() // ethers.utils.formatBytes32String('schema4Id')

      const data = abiCoder.encode(
        ['uint256', 'address'],
        [marketId, lenderAddress]
      )

      const nonce = await tellerASEIP712Verifier.getNonce(
        attestationWalletAddress
      )

      const request = await eip712Utils.getAttestationRequest(
        lenderAddress,
        schemaId,
        expirationTime,
        ethers.utils.formatBytes32String(''),
        data,
        nonce,
        attestationWallet.privateKey
      )

      const attestResult = await marketRegistry
        .connect(lender)
        ['attestLender(uint256,address,uint256,uint8,bytes32,bytes32)'](
          marketId,
          lenderAddress,
          expirationTime,
          request.v,
          request.r,
          request.s
        )
        .should.be.revertedWith('InvalidSignature()')
    })

    it('should not fail if market does not require attestations', async () => {
      const marketId = 1

      const lender = alternateOwner
      const lenderAddress = alternateOwnerAddress

      const attestationWalletAddress = await attestationWallet.getAddress()

      await marketRegistry
        .connect(marketOwner)
        .transferMarketOwnership(marketId, attestationWalletAddress)

      const expirationTime: BigNumber = toBN(moment.now()).add(
        moment.duration(30, 'days').seconds()
      )

      const schemaId = await marketRegistry.lenderAttestationSchemaId() // ethers.utils.formatBytes32String('schema4Id')

      const data = abiCoder.encode(
        ['uint256', 'address'],
        [marketId, lenderAddress]
      )

      const nonce = await tellerASEIP712Verifier.getNonce(
        attestationWalletAddress
      )

      const request = await eip712Utils.getAttestationRequest(
        lenderAddress,
        schemaId,
        expirationTime,
        ethers.utils.formatBytes32String(''),
        data,
        nonce,
        attestationWallet.privateKey
      )

      const attestResult = await marketRegistry
        .connect(lender)
        ['attestLender(uint256,address,uint256,uint8,bytes32,bytes32)'](
          marketId,
          lenderAddress,
          expirationTime,
          request.v,
          request.r,
          request.s
        )
    })

    it('should fail if market does not have an owner set', async () => {
      const marketId = 999

      const lender = alternateOwner
      const lenderAddress = alternateOwnerAddress

      const attestationWalletAddress = await attestationWallet.getAddress()

      const expirationTime: BigNumber = toBN(moment.now()).add(
        moment.duration(30, 'days').seconds()
      )

      const schemaId = await marketRegistry.lenderAttestationSchemaId() // ethers.utils.formatBytes32String('schema4Id')

      const data = abiCoder.encode(
        ['uint256', 'address'],
        [marketId, lenderAddress]
      )

      const nonce = await tellerASEIP712Verifier.getNonce(
        attestationWalletAddress
      )

      const request = await eip712Utils.getAttestationRequest(
        lenderAddress,
        schemaId,
        expirationTime,
        ethers.utils.formatBytes32String(''),
        data,
        nonce,
        attestationWallet.privateKey
      )

      const attestResult = await marketRegistry
        .connect(lender)
        ['attestLender(uint256,address,uint256,uint8,bytes32,bytes32)'](
          marketId,
          lenderAddress,
          expirationTime,
          request.v,
          request.r,
          request.s
        )
        .should.be.revertedWith('InvalidSignature()')
    })

    it('should fail if the attestation signature is invalid', async () => {
      const lenderAddress = alternateOwnerAddress

      const attestationWalletAddress = await attestationWallet.getAddress()

      const marketId = 2 // requires attestation

      const expirationTime: BigNumber = toBN(moment.now()).add(
        moment.duration(30, 'days').seconds()
      )

      const schemaId = await marketRegistry.lenderAttestationSchemaId() // ethers.utils.formatBytes32String('schema4Id')

      const data = abiCoder.encode(
        ['uint256', 'address'],
        [marketId, lenderAddress]
      )

      const nonce = await tellerASEIP712Verifier.getNonce(
        attestationWalletAddress
      )

      const request = await eip712Utils.getAttestationRequest(
        lenderAddress,
        schemaId,
        expirationTime,
        ethers.utils.formatBytes32String(''),
        data,
        nonce,
        attestationWallet.privateKey
      )

      await marketRegistry
        .connect(marketOwner)
        ['attestLender(uint256,address,uint256,uint8,bytes32,bytes32)'](
          marketId,
          lenderAddress,
          expirationTime,
          request.v,
          request.s,
          request.r
        )
        .should.be.revertedWith('InvalidSignature()')
    })

    it('should be able to submit a valid signed attestation to MarketRegistry', async () => {
      await attestLender()
    })

    it('should be able to submit a non delegated attestation to MarketRegistry', async () => {
      const newLender = Wallet.createRandom()
      const marketId = 2 // requires attestation
      await marketRegistry
        .connect(attestationWallet)
        ['attestLender(uint256,address,uint256)'](
          marketId,
          newLender.address,
          moment.duration(moment.utc().add(1, 'year').unix()).asMilliseconds()
        )
        .should.emit(marketRegistry, 'LenderAttestation')
        .withArgs(marketId, newLender.address)
    })
  })

  describe('revokeLender', () => {
    it('should be able to revoke a lender', async () => {
      const lenderAddress = alternateOwnerAddress
      // Attest lender
      await attestLender()
      const marketId = 2 // requires attestation
      const nonce = await tellerASEIP712Verifier.getNonce(
        await attestationWallet.getAddress()
      )
      // Get uuid from market registry
      const uuid = (
        await marketRegistry.isVerifiedLender(marketId, lenderAddress)
      )[1]
      // Sign revocation request
      const request = await eip712Utils.getRevocationRequest(
        uuid,
        nonce,
        attestationWallet,
        attestationWallet.privateKey
      )
      // Revoke lender
      await marketRegistry
        .connect(marketOwner)
        ['revokeLender(uint256,address,uint8,bytes32,bytes32)'](
          marketId,
          lenderAddress,
          request.v,
          request.r,
          request.s
        )
        .should.emit(marketRegistry, 'LenderRevocation')
        .withArgs(marketId, lenderAddress)
    })

    it('should be able to revoke a lender without delegation to MarketRegistry', async () => {
      const lenderAddress = alternateOwnerAddress
      // Attest lender
      await attestLender()
      const marketId = 2 // requires attestation

      await marketRegistry
        .connect(attestationWallet)
        ['revokeLender(uint256,address)'](marketId, lenderAddress)
        .should.emit(marketRegistry, 'LenderRevocation')
        .withArgs(marketId, lenderAddress)
    })
  })

  describe('lenderExitMarket', () => {
    it('should not be able to exit a market as not the verified lender', async () => {
      const lenderAddress = alternateOwnerAddress
      const marketId = 2 // requires attestation
      // Attest lender by the market owner
      await attestLender()
      // Check verification status
      const isVerifiedBefore = (
        await marketRegistry.isVerifiedLender(marketId, lenderAddress)
      )[0]
      isVerifiedBefore.should.be.eq(true)
      // Try to exit market as not the lender
      await marketRegistry.connect(marketOwner).lenderExitMarket(marketId)
      // Check verification status
      const isVerifiedAfter = (
        await marketRegistry.isVerifiedLender(marketId, lenderAddress)
      )[0]
      isVerifiedAfter.should.be.eq(true)
    })
    it('should be able to exit a market as a verified lender', async () => {
      const lenderAddress = alternateOwnerAddress
      const lender = alternateOwner
      const marketId = 2 // requires attestation
      // Lender exit market
      await marketRegistry
        .connect(lender)
        .lenderExitMarket(marketId)
        .should.emit(marketRegistry, 'LenderExitMarket')
        .withArgs(marketId, lenderAddress)
      // Check verification status
      const isVerifiedAfter = (
        await marketRegistry.isVerifiedLender(marketId, lenderAddress)
      )[0]
      isVerifiedAfter.should.be.eq(false)
      // Try to exit the market again, no event should emit
      await marketRegistry
        .connect(lender)
        .lenderExitMarket(marketId)
        .should.not.emit(marketRegistry, 'LenderExitMarket')
        .withArgs(marketId, lenderAddress)
    })
  })

  describe('attestBorrower', () => {
    it('should fail if market ID is 0', async () => {
      const marketId = 0

      const borrower = alternateOwner
      const borrowerAddress = alternateOwnerAddress

      const attestationWalletAddress = await attestationWallet.getAddress()

      const expirationTime: BigNumber = toBN(moment.now()).add(
        moment.duration(30, 'days').seconds()
      )

      const schemaId = await marketRegistry.borrowerAttestationSchemaId() // ethers.utils.formatBytes32String('schema4Id')

      const data = abiCoder.encode(
        ['uint256', 'address'],
        [marketId, borrowerAddress]
      )

      const nonce = await tellerASEIP712Verifier.getNonce(
        attestationWalletAddress
      )

      const request = await eip712Utils.getAttestationRequest(
        borrowerAddress,
        schemaId,
        expirationTime,
        ethers.utils.formatBytes32String(''),
        data,
        nonce,
        attestationWallet.privateKey
      )

      const attestResult = await marketRegistry
        .connect(borrower)
        ['attestBorrower(uint256,address,uint256,uint8,bytes32,bytes32)'](
          marketId,
          borrowerAddress,
          expirationTime,
          request.v,
          request.r,
          request.s
        )
        .should.be.revertedWith('InvalidSignature()')
    })

    it('should fail if the borrower attestation signature is invalid', async () => {
      const borrowerAddress = alternateOwnerAddress

      const attestationWalletAddress = await attestationWallet.getAddress()

      const marketId = 2 // requires attestation

      const expirationTime: BigNumber = toBN(moment.now()).add(
        moment.duration(30, 'days').seconds()
      )

      const schemaId = await marketRegistry.borrowerAttestationSchemaId() // ethers.utils.formatBytes32String('schema4Id')

      const data = abiCoder.encode(
        ['uint256', 'address'],
        [marketId, borrowerAddress]
      )

      const nonce = await tellerASEIP712Verifier.getNonce(
        attestationWalletAddress
      )

      const request = await eip712Utils.getAttestationRequest(
        borrowerAddress,
        schemaId,
        expirationTime,
        ethers.utils.formatBytes32String(''),
        data,
        nonce,
        attestationWallet.privateKey
      )

      await marketRegistry
        .connect(marketOwner)
        ['attestBorrower(uint256,address,uint256,uint8,bytes32,bytes32)'](
          marketId,
          borrowerAddress,
          expirationTime,
          request.v,
          request.s,
          request.r
        )
        .should.be.revertedWith('InvalidSignature()')
    })

    it('should be able to submit a valid signed borrower attestation to MarketRegistry', async () => {
      await attestBorrower()
    })

    it('should fail when trying to attest a borrower with a lender attestation schema', async () => {
      const borrower = alternateOwner
      const borrowerAddress = alternateOwnerAddress

      const attestationWalletAddress = await attestationWallet.getAddress()

      const marketId = 2 // requires attestation

      const expirationTime: BigNumber = toBN(moment.now()).add(
        moment.duration(30, 'days').seconds()
      )

      const schemaId = await marketRegistry.lenderAttestationSchemaId()

      const data = abiCoder.encode(
        ['uint256', 'address'],
        [marketId, borrowerAddress]
      )

      const nonce = await tellerASEIP712Verifier.getNonce(
        attestationWalletAddress
      )

      const request = await eip712Utils.getAttestationRequest(
        borrowerAddress,
        schemaId,
        expirationTime,
        ethers.utils.formatBytes32String(''),
        data,
        nonce,
        attestationWallet.privateKey
      )

      await marketRegistry
        .connect(borrower)
        ['attestBorrower(uint256,address,uint256,uint8,bytes32,bytes32)'](
          marketId,
          borrowerAddress,
          expirationTime,
          request.v,
          request.r,
          request.s
        )
        .should.be.revertedWith('InvalidSignature()')
    })
  })

  describe('revokeBorrower', () => {
    it.skip('should fail if market does not have an owner set', async () => {
      const marketId = 1422
      const borrowerAddressToRevoke = alternateOwnerAddress
      const nonce = await tellerASEIP712Verifier.getNonce(
        await attestationWallet.getAddress()
      )
      const result = await marketRegistry.isVerifiedBorrower(
        marketId,
        borrowerAddressToRevoke
      )
      const request = await eip712Utils.getRevocationRequest(
        result[1],
        nonce,
        attestationWallet,
        attestationWallet.privateKey
      )
      await marketRegistry
        .connect(marketOwner)
        ['revokeBorrower(uint256,address,uint8,bytes32,bytes32)'](
          marketId,
          borrowerAddressToRevoke,
          request.v,
          request.r,
          request.s
        )
        .should.be.revertedWith('InvalidSignature()')
    })
    it('should be able to revoke a borrower', async () => {
      const borrrowerAddress = alternateOwnerAddress
      // Attest lender
      await attestBorrower()
      const marketId = 2 // requires attestation
      const nonce = await tellerASEIP712Verifier.getNonce(
        await attestationWallet.getAddress()
      )
      // Get uuid from market registry
      const uuid = (
        await marketRegistry.isVerifiedBorrower(marketId, borrrowerAddress)
      )[1]
      // Sign revocation request
      const request = await eip712Utils.getRevocationRequest(
        uuid,
        nonce,
        attestationWallet,
        attestationWallet.privateKey
      )
      // Revoke lender
      await marketRegistry
        .connect(marketOwner)
        ['revokeBorrower(uint256,address,uint8,bytes32,bytes32)'](
          marketId,
          borrrowerAddress,
          request.v,
          request.r,
          request.s
        )
        .should.emit(marketRegistry, 'BorrowerRevocation')
        .withArgs(marketId, borrrowerAddress)
    })
  })

  describe('borrowerExitMarket', () => {
    it('should not be able to exit a market as not the verified borrower', async () => {
      const borrowerAddress = alternateOwnerAddress
      const marketId = 2 // requires attestation
      // Attest lender by the market owner
      await attestBorrower()
      // Check verification status
      const isVerifiedBefore = (
        await marketRegistry.isVerifiedBorrower(marketId, borrowerAddress)
      )[0]
      isVerifiedBefore.should.be.eq(true)
      // Try to exit market as not the lender
      await marketRegistry.connect(marketOwner).borrowerExitMarket(marketId)
      // Check verification status
      const isVerifiedAfter = (
        await marketRegistry.isVerifiedBorrower(marketId, borrowerAddress)
      )[0]
      isVerifiedAfter.should.be.eq(true)
    })
    it('should be able to exit a market as a verified lender', async () => {
      const borrowerAddress = alternateOwnerAddress
      const borrower = alternateOwner
      const marketId = 2 // requires attestation
      // Borrower exit market
      await marketRegistry
        .connect(borrower)
        .borrowerExitMarket(marketId)
        .should.emit(marketRegistry, 'BorrowerExitMarket')
        .withArgs(marketId, borrowerAddress)
      // Check verification status
      const isVerifiedAfter = (
        await marketRegistry.isVerifiedBorrower(marketId, borrowerAddress)
      )[0]
      isVerifiedAfter.should.be.eq(false)
      // Try to exit the market again, no event should emit
      await marketRegistry
        .connect(borrower)
        .borrowerExitMarket(marketId)
        .should.not.emit(marketRegistry, 'BorrowerExitMarket')
        .withArgs(marketId, borrowerAddress)
    })
  })

  describe('isVerifiedLender', () => {
    it('should return false for non verified lender', async () => {
      const marketplaceId = 2

      const result = await marketRegistry.isVerifiedLender(
        marketplaceId,
        marketOwnerAddress
      )
      expect(result[0]).to.eql(false)
    })

    it('should return true for a verified lender', async () => {
      const marketplaceId = 2
      await attestLender()

      const result = await marketRegistry.isVerifiedLender(
        marketplaceId,
        alternateOwnerAddress
      )
      expect(result[0]).to.eql(true)
    })

    it('should return true if market does not require attestations', async () => {
      const marketplaceId = 1

      const result = await marketRegistry.isVerifiedLender(
        marketplaceId,
        marketOwnerAddress
      )
      expect(result[0]).to.eql(true)
    })
  })

  describe('can get market data', () => {
    it('should return market data', async () => {
      const marketplaceId = 2

      await marketRegistry
        .connect(attestationWallet)
        .transferMarketOwnership(marketplaceId, marketOwnerAddress)

      const fetchedData = await marketRegistry.getMarketData(marketplaceId)

      expect(fetchedData.owner).to.eql(marketOwnerAddress)
      expect(fetchedData.paymentCycleDuration).to.eql(paymentCycleDuration)
      expect(fetchedData.paymentDefaultDuration).to.eql(loanDefaultDuration)
      expect(fetchedData.loanExpirationTime).to.eql(loanExpirationDuration)
      expect(fetchedData.metadataURI).to.eql(uriTwo)
      expect(fetchedData.marketplaceFeePercent).to.eql(0)
      expect(fetchedData.lenderAttestationRequired).to.eql(true)
    })
  })

  describe('getAllVerifiedLendersForMarket', () => {
    it('should return proper array of verified lender addresses for a market', async () => {
      const list = await marketRegistry.getAllVerifiedLendersForMarket(2, 0, 10)

      expect(list.length).to.eql(2)

      expect(list[1]).to.eql(alternateOwnerAddress)
    })
  })

  describe('can set fee recipient', () => {
    it('should get fee recipient', async () => {
      const marketId = 2

      const feeRecipient = await marketRegistry.getMarketFeeRecipient(marketId)

      feeRecipient.should.eql(marketOwnerAddress)
    })

    it('should prevent setting fee recipient', async () => {
      const newFeeRecipient = Wallet.createRandom()

      await marketRegistry
        .connect(alternateOwner)
        .setMarketFeeRecipient(2, newFeeRecipient.address)
        .should.be.revertedWith('Not the owner')
    })

    it('should set fee recipient', async () => {
      const marketId = 2

      const newFeeRecipient = Wallet.createRandom()

      await marketRegistry
        .connect(marketOwner)
        .setMarketFeeRecipient(marketId, newFeeRecipient.address)
        .should.emit(marketRegistry, 'SetMarketFeeRecipient')
        .withArgs(marketId, newFeeRecipient.address)

      const feeRecipient = await marketRegistry.getMarketFeeRecipient(marketId)

      feeRecipient.should.eql(newFeeRecipient.address)
    })
  })
  describe('can update multiple settings for a market', () => {
    const marketId = 2
    const urifour = 'ipfs://QmMyDataHashFour'
    it('should prevent updating the settings for a market', async () => {
      await marketRegistry
        .connect(alternateOwner)
        .updateMarketSettings(
          marketId,
          paymentCycleDuration.toString(),
          0,
          0,
          0,
          0,
          0,
          false,
          false,
          urifour
        )
        .should.be.revertedWith('Not the owner')
    })

    it('should update the settings for a market', async () => {
      await marketRegistry
        .connect(marketOwner)
        .updateMarketSettings(
          marketId,
          paymentCycleDuration.toString(),
          0,
          0,
          0,
          0,
          0,
          false,
          false,
          urifour
        )

      const fetchedData = await marketRegistry.getMarketData(marketId)

      expect(fetchedData.owner).to.eql(marketOwnerAddress)
      expect(fetchedData.paymentCycleDuration).to.eql(paymentCycleDuration)
      expect(fetchedData.paymentDefaultDuration).to.eql(0)
      expect(fetchedData.metadataURI).to.eql(urifour)
    })
  })
})
