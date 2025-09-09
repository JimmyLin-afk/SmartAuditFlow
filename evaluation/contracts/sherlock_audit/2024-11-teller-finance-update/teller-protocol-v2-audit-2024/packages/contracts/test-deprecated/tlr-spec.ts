import chai, { expect } from 'chai'
import chaiAsPromised from 'chai-as-promised'
import { BigNumber as BN, Signer } from 'ethers'
import hre from 'hardhat'
import { deploy } from 'helpers/deploy-helpers'
import moment from 'moment'
import { TLR } from 'types/typechain'

chai.should()
chai.use(chaiAsPromised)

const { getNamedSigner, toBN, deployments } = hre

const initializedFixture = deployments.createFixture(async (_hre) => {
  const deployer = await getNamedSigner('deployer')
  const lender = await getNamedSigner('lender')
  const lender2 = await getNamedSigner('lender2')
  const dao = await getNamedSigner('dao')

  const totalSupply = toBN('100000000', '18')

  const tlr: TLR = await deploy({
    contract: 'TLR',
    args: [totalSupply, await dao.getAddress()],
    hre,
  })

  await tlr.connect(dao).mint(await dao.getAddress(), totalSupply)

  return {
    tlr,
    deployer,
    dao,
    lender,
    lender2,
  }
})

describe('TLR', () => {
  let tlr: TLR
  let cherise: Signer
  let jefe: Signer
  let dao: Signer

  beforeEach(async () => {
    const result = await initializedFixture()
    tlr = result.tlr
    cherise = result.lender
    jefe = result.lender2
    dao = result.dao
  })

  describe('metadata', () => {
    it('name', async () => {
      expect(await tlr.name()).to.eq('Teller')
    })
    it('symbol', async () => {
      expect(await tlr.symbol()).to.eq('TLR')
    })
    it('decimals', async () => {
      expect(await tlr.decimals()).to.eq(18)
    })
  })
  describe('Initial supply & transfers - sanity check', () => {
    it('total supply should have been minted to inital account', async () => {
      expect(await tlr.balanceOf(await dao.getAddress())).to.equal(
        await tlr.totalSupply()
      )
      expect(await tlr.balanceOf(await cherise.getAddress())).to.equal(0)
      expect(await tlr.balanceOf(await jefe.getAddress())).to.equal(0)
      expect(await tlr.getVotes(await cherise.getAddress())).to.equal(0)
      expect(await tlr.getVotes(await cherise.getAddress())).to.equal(0)
      expect(await tlr.getVotes(await jefe.getAddress())).to.equal(0)
    })
    const cheriseAmt = 100000
    const jefeAmt = 80000
    it('should transfer tokens', async () => {
      // Transfer to Cherise
      await tlr.connect(dao).transfer(await cherise.getAddress(), cheriseAmt)
      expect(await tlr.balanceOf(await cherise.getAddress())).to.equal(
        cheriseAmt
      )
      // Transfer to Jefe
      await tlr.connect(dao).transfer(await jefe.getAddress(), jefeAmt)
      expect(await tlr.balanceOf(await jefe.getAddress())).to.equal(jefeAmt)
    })
  })
  describe('delegateBySig', () => {
    let delegatee: string
    let nonce: number
    let expiry: number
    let typedSig: string
    let Domain: any
    let Types: any
    let v: number
    let r: string
    let s: string

    beforeEach(async () => {
      delegatee = await jefe.getAddress()
      nonce = 0

      Domain = {
        name: await tlr.name(),
        chainId: await hre.getChainId(),
        verifyingContract: tlr.address,
      }

      Types = {
        Delegation: [
          { name: 'delegatee', type: 'address' },
          { name: 'nonce', type: 'uint256' },
          { name: 'expiry', type: 'uint256' },
        ],
      }

      expiry =
        (
          await hre.ethers.provider.getBlock(
            await hre.ethers.provider.getBlockNumber()
          )
        ).timestamp + 100000000

      typedSig = await cherise.signMessage(
        hre.ethers.utils._TypedDataEncoder.encode(Domain, Types, {
          delegatee: await jefe.getAddress(),
          nonce: 0,
          expiry: expiry,
        })
      )

      const sig = hre.ethers.utils.splitSignature(typedSig)
      v = sig.v
      r = sig.r
      s = sig.s
    })

    it('reverts if the signature is bad', async () => {
      await expect(
        tlr.connect(cherise).delegateBySig(
          delegatee,
          nonce,
          expiry,
          12, // v,
          r,
          s
        )
      ).revertedWith("ECDSA: invalid signature 'v' value")
    })
    it('reverts if the nonce is bad', async () => {
      await expect(
        tlr.connect(cherise).delegateBySig(
          delegatee,
          11, // bad nonce
          expiry,
          v,
          r,
          s
        )
      ).revertedWith('ERC20Votes: invalid nonce')
    })
    it('reverts if the signaure has expirerd', async () => {
      await expect(
        tlr
          .connect(cherise)
          .delegateBySig(delegatee, nonce, expiry - 100000000, v, r, s)
      ).revertedWith('ERC20Votes: signature expired')
    })
    it('delegates successfully on behalf of the signatory', async () => {
      // Transfer
      await tlr.connect(dao).transfer(await cherise.getAddress(), toBN(10000))
      // Delegate
      await expect(
        tlr.connect(cherise).delegateBySig(delegatee, nonce, expiry, v, r, s)
      ).to.emit(tlr, 'DelegateChanged')
    })
  })
})
