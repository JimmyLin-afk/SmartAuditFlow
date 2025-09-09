import { task } from 'hardhat/config'

task('oz:force-import').setAction(async (args, hre) => {
  const fn = async (name: string): Promise<void> => {
    const { address: proxyAddress } = await hre.deployments.get(name)
    // const { args: implArgs } = await hre.deployments.get(
    //   `${name}_Implementation`
    // )

    const factory = await hre.ethers.getContractFactory(name)
    try {
      await hre.platform.forceImport(proxyAddress, factory, {
        // constructorArgs: implArgs,
      })
    } catch (e) {
      console.log(`Failed to force import for "${name}"`)
    }
  }

  await Promise.all(
    [
      'CollateralManager',
      'LenderCommitmentForwarder',
      'LenderManager',
      'MarketLiquidityRewards',
      'MarketRegistry',
      'MetaForwarder',
      'ReputationManager',
      'TellerV2',
    ].map(fn)
  )
})
