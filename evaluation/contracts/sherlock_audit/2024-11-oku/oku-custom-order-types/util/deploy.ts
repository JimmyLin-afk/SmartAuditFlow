import { Signer, ContractFactory } from "ethers";
export const DeployContract = async (factory: ContractFactory, deployer: Signer, ...args: any[]): Promise<any> => {
    const uContract
        = await factory.connect(deployer).deploy(...args)
    await uContract.waitForDeployment()
    return factory.attach(await uContract
        .getAddress())
}

