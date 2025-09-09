import { DeployFunction } from 'hardhat-deploy/dist/types'

const deployFn: DeployFunction = async (hre) => {
  hre.log('----------')
  hre.log('Proposing upgrade:  Lender Commitment Group')

  const TellerV2 = await hre.contracts.get('TellerV2')

  const tellerV2Address = await TellerV2.getAddress()

  const SmartCommitmentForwarder = await hre.contracts.get(
    'SmartCommitmentForwarder'
  )

  const smartCommitmentForwarderAddress =
    await SmartCommitmentForwarder.getAddress()

  //for sepolia
  const lenderCommitmentGroupProxyAddress =
    '0x3AF8DB041fcaFA539C2c78f73aa209383ba703ed'

  /*const LenderCommitmentGroup = await hre.ethers.getContractFactory(
    'LenderCommitmentGroup_Smart'
  )*/

  const LenderCommitmentGroup = await hre.contracts.get(
    'LenderCommitmentGroup_Smart'
  )

  //for sepolia
  const uniswapV3FactoryAddress = '0x0227628f3F023bb0B980b67D528571c95c6DaC1c'

  const networkName = hre.network.name

  //created pool https://sepolia.etherscan.io/tx/0x8ea20095c821f6066252457d7f0438030bc65bb441e1bea56c6ae0efd63016f0

  const principalTokenAddress = '0xfff9976782d46cc05630d1f6ebab18b2324d6b14' //weth
  const collateralTokenAddress = '0x72292c8464a33f6b5f6efcc0213a89a98c68668b' //0xbtc
  const uniswapPoolFee = 3000

  const marketId = 1
  const minInterestRate = 100
  const maxLoanDuration = 5000000
  const liquidityThresholdPercent = 10000
  const loanToValuePercent = 10000 //mzake sure this functions as normal.  If over 100%, getting much better loan terms and i wont repay.  If it is under 100%, it will likely repay.

  await hre.upgrades.proposeBatchTimelock({
    title: 'Lender Commitment Group Smart: Upgrade ',
    description: ` 
# Lender Commitment Group Smart Upgrade
* Upgrades Lender Commitment Group Smart.
`,
    _steps: [
      {
        proxy: LenderCommitmentGroup,
        implFactory: await hre.ethers.getContractFactory(
          'LenderCommitmentGroup_Smart'
        ),

        opts: {
          unsafeAllow: ['constructor', 'state-variable-immutable'],
          //  unsafeAllowRenames: true,
          constructorArgs: [
            tellerV2Address,
            smartCommitmentForwarderAddress,
            uniswapV3FactoryAddress,
          ],

          /* call: {
            fn: 'setCollateralManagerV2',
            args: [await collateralManagerV2.getAddress()]
          }
          */
        },
      },
    ],
  })

  /*
  const lenderCommitmentGroupSmart = await hre.upgrades.upgradeProxy(
    lenderCommitmentGroupProxyAddress,
    LenderCommitmentGroup,
    {
      unsafeAllow: ['constructor', 'state-variable-immutable'],
      constructorArgs: [
        smartCommitmentForwarderAddress,
        uniswapV3FactoryAddress
      ] 
    }
  )
*/

  return true
}

// tags and deployment
deployFn.id = 'lender-commitment-group-upgrade'
deployFn.tags = [
  'upgrade',
  'lender-commitment-group',
  'lender-commitment-group-upgrade',
]
deployFn.dependencies = ['lender-commitment-group-smart:deploy']
deployFn.skip = async (hre) => {
  return !hre.network.live || !['sepolia'].includes(hre.network.name)
}
export default deployFn
