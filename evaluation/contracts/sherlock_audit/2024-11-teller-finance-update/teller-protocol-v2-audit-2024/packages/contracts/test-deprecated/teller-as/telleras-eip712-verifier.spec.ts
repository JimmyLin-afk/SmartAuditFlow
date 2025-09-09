import {
  ATTEST_TYPED_SIGNATURE,
  Delegation,
  REVOKE_TYPED_SIGNATURE,
} from '@ethereum-attestation-service/eas-sdk'
import chai, { expect } from 'chai'
import chaiAsPromised from 'chai-as-promised'
import hre from 'hardhat'
import { TellerASEIP712Verifier } from 'types/typechain'

chai.should()
chai.use(chaiAsPromised)

const { ethers, toBN, deployments } = hre

// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface SetupOptions {}

interface SetupReturn {
  tellerASEIP712Verifier: TellerASEIP712Verifier
}

const setup = deployments.createFixture<SetupReturn, SetupOptions>(
  async (hre, _opts) => {
    await hre.deployments.fixture('teller-as', {
      keepExistingDeployments: false,
    })

    const tellerASEIP712Verifier =
      await hre.contracts.get<TellerASEIP712Verifier>('TellerASEIP712Verifier')

    return {
      tellerASEIP712Verifier,
    }
  }
)

describe('TellerASEIP712Verifier', () => {
  let tellerASEIP712Verifier: TellerASEIP712Verifier

  beforeEach(async () => {
    const result = await setup()
    tellerASEIP712Verifier = result.tellerASEIP712Verifier
  })
  it('should report a version', async () => {
    expect(await tellerASEIP712Verifier.VERSION()).to.eq('0.8')
  })
  it('should return the correct domain separator', async () => {
    const delegation = new Delegation({
      address: tellerASEIP712Verifier.address,
      version: await tellerASEIP712Verifier.VERSION(),
      chainId: (await ethers.provider.getNetwork()).chainId,
    })
    expect(await tellerASEIP712Verifier.DOMAIN_SEPARATOR()).to.eq(
      delegation.getDomainSeparator()
    )
  })
  const {
    utils: { keccak256, toUtf8Bytes },
  } = ethers
  it('should return the attest type hash', async () => {
    expect(await tellerASEIP712Verifier.ATTEST_TYPEHASH()).to.eq(
      keccak256(toUtf8Bytes(ATTEST_TYPED_SIGNATURE))
    )
  })
  it('should return the revoke type hash', async () => {
    expect(await tellerASEIP712Verifier.REVOKE_TYPEHASH()).to.eq(
      keccak256(toUtf8Bytes(REVOKE_TYPED_SIGNATURE))
    )
  })
})
