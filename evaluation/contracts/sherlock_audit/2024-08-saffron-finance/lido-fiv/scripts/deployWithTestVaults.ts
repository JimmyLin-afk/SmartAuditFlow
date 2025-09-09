import { ethers } from 'hardhat'
import { parseEther, formatEther } from 'ethers'
import hre from 'hardhat'
import LidoVaultABI from '../artifacts/contracts/LidoVault.sol/LidoVault.json';
import {
  DEFAULTS,
  SIDE
} from '../test/helpers'

import { VaultFactory } from '../typechain-types'

function getRandomNumber(min: number, max: number): number {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function getRandomBigNumber(min: bigint, max: bigint): bigint {
    const range = max - min + 1n; // Adding 1n to include the max value
    const randomNumber = BigInt(Math.floor(Number(Math.random() * Number(range)))); // Convert to BigInt
    return randomNumber + min;
}

async function main() {
  console.log(`Deploying to ${hre.network.name}`)
  const addrs = await hre.ethers.getSigners()

  const vaultFactoryContract = await ethers.getContractFactory('VaultFactory')
  
  const protocolFeeBps = 100;
  const earlyExitFeeBps = 1000;

  const vaultFactory: VaultFactory = (await vaultFactoryContract.deploy(
    protocolFeeBps,
    earlyExitFeeBps
  )) as any
  
  const vaultFactoryAddress = await vaultFactory.getAddress()
  console.log(`\x1b[32m${vaultFactoryAddress}\x1b[0m VaultFactory deployed`)
  
  // create vaults
  let vaultMax = 10;
  
  for(let i=0;i<vaultMax;i++) {
    const nextVaultId = await vaultFactory.nextVaultId();
    const fixedSideCapacity = parseEther(getRandomNumber( 10, 500 ).toString())
    const durationSeconds = getRandomNumber( 120, 31536000 )  // 2 minutes to a year
    const fixedPremiumBpsVault = getRandomNumber( 100, 500 )
    const variableSideCapacity = fixedSideCapacity.times(fixedPremiumBpsVault).div(10000);
  
    const lidoVault = await ethers.getContractFactory('LidoVault')
    const signer=getRandomNumber(0,3);
    //console.log(`create vault signer: \x1b[32m${addrs[signer].address}\x1b[0m`);
    let vaultTx = await vaultFactory.connect(addrs[signer]).createVault( fixedSideCapacity, durationSeconds, variableSideCapacity )
    await vaultTx.wait()
    const vaultCreator = (await vaultFactory.vaultInfo(nextVaultId))[0];
    const vaultAddress = (await vaultFactory.vaultInfo(nextVaultId))[1];
    
    const lidoVaultContract = new ethers.Contract(vaultAddress, LidoVaultABI.abi);
    console.log(`\nVault ${nextVaultId} created: \x1b[32m${vaultAddress}\x1b[0m \t\x1b[33mFIXED: ${formatEther(fixedSideCapacity)} ETH\x1b[0m \t\x1b[33mVARIABLE: ${formatEther(variableSideCapacity)} ETH\x1b[0m \t\t\x1b[33m${(durationSeconds/86400).toFixed(1)} days\x1b[0m \t\x1b[33m${(fixedPremiumBpsVault/100).toFixed(1)}% fixed premium\x1b[0m creator: \x1b[32m${addrs[signer].address}\x1b[0m`)
  
    //FIXED deposit nothing, partial, or full amount
    let howMuch = getRandomNumber(0,6);
    
    let depositSignerFixed = getRandomNumber(0,5);
    
    switch(howMuch) {
      case 0: case 1: case 2:
        //nothing
        break;
        
      case 3:
        //random partial from minimum for full
        let depositAmount=getRandomBigNumber(DEFAULTS.minimumDepositAmount, fixedSideCapacity)
        lidoVaultContract.connect(addrs[depositSignerFixed]).deposit(SIDE.FIXED, { value: depositAmount })
        console.log(`\tFIXED    deposit:\t\x1b[33m${formatEther(depositAmount)}\x1b[0m\t\x1b[32m${addrs[depositSignerFixed].address}\x1b[0m `)
        break;
        
      case 4: case 5: case 6:
        //full amount
        lidoVaultContract.connect(addrs[depositSignerFixed]).deposit(SIDE.FIXED, { value: fixedSideCapacity })
        console.log(`\tFIXED    deposit:\t\x1b[33m${formatEther(fixedSideCapacity)}\x1b[0m - FULL AMOUNT\t\x1b[32m${addrs[depositSignerFixed].address}\x1b[0m `)
        break
    }

    //VARIABLE deposit nothing, partial, or full amount
    howMuch = getRandomNumber(0,6);
    
    //get random variable signer, make sure is not same as fixed deposit signer
    let depositSignerVariable;
    do {
      depositSignerVariable = getRandomNumber(0,5);
    } while(depositSignerFixed == depositSignerVariable);
    
    //get variable side capacity
    let variableSideCapacity = await lidoVaultContract.connect(addrs[depositSignerVariable]).variableSideCapacity()
    //console.log("variableSideCapacity:",variableSideCapacity);
    switch(howMuch) {
      case 0: case 1: case 2:
        //nothing
        break;
        
      case 3:
        //random partial from minimum for full
        let depositAmount = getRandomBigNumber(DEFAULTS.minimumDepositAmount, variableSideCapacity)
        lidoVaultContract.connect(addrs[depositSignerVariable]).deposit(SIDE.VARIABLE, { value: depositAmount })
        console.log(`\tVARIABLE deposit:\t\x1b[33m${formatEther(depositAmount)}\x1b[0m\t\x1b[32m${addrs[depositSignerVariable].address}\x1b[0m `)
        break;
        
      case 4: case 5: case 6:
        //full amount
        lidoVaultContract.connect(addrs[depositSignerVariable]).deposit(SIDE.VARIABLE, { value: variableSideCapacity })
        console.log(`\tVARIABLE deposit:\t\x1b[33m${formatEther(variableSideCapacity)}\x1b[0m - FULL AMOUNT\t\x1b[32m${addrs[depositSignerVariable].address}\x1b[0m `)
        break
    }
    
        
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error)
    process.exit(1)
  })
