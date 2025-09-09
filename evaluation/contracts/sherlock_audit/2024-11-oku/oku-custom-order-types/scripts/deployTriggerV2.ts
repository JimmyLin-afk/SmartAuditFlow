import hre from "hardhat";
import { DeployContract } from "../util/deploy";
import { currentBlock, resetCurrentArb, resetCurrentBase, resetCurrentOP } from "../util/block";
import { AutomationMaster, AutomationMaster__factory, Bracket, Bracket__factory, IERC20__factory, IOracleRelay, OracleRelay__factory, StopLimit, StopLimit__factory, TokenEthRelay__factory, UniswapV3Pool__factory } from "../typechain-types";
import { Signer } from "ethers";
import { impersonateAccount } from "../util/impersonator";
import { setBalance } from "@nomicfoundation/hardhat-network-helpers";
import { decodeUpkeepData, generateUniTx } from "../util/msc";
import { a, b, o } from "../util/addresser";

const { ethers } = require("hardhat");

//"https://github.com/adrastia-oracle/oku-automation-config/blob/main/worker-config.ts"


const userAddr = "0x085909388fc0cE9E5761ac8608aF8f2F52cb8B89"


let masterAddr: string = a.Master
let stopLimitAddr: string = a.stopLimit
let bracketAddr: string = a.bracket
let permit2Addr: string = a.permit2


let wethOracleAddress: string
let usdcOracleAddress: string
let router02: string
let pool: string
let wethAddress: string
let usdcAddress: string
let wethFeedAddr: string
let usdcFeedAddr: string


let mainnet = true
let Master!: AutomationMaster
let StopLimit!: StopLimit
let Bracket!: Bracket

//SET THIS FOR TESTING
const testingNetwork = "op"

let masterKeeper: AutomationMaster
async function main() {
  console.log("STARTING")
  let networkName = hre.network.name
  console.log(networkName)

  if (networkName == "hardhat" || networkName == "localhost") {
    networkName = testingNetwork
    mainnet = false
    console.log("Testing on network : ", networkName)

  } else {
    console.log("Deploying for real to: ", networkName)
  }

  if (networkName == 'base') {
    if (!mainnet) {
      await resetCurrentBase()
      console.log("Testing on BASE @", (await currentBlock())?.number)
    }

    masterAddr = b.Master
    stopLimitAddr = b.stopLimit
    bracketAddr = b.bracket
    permit2Addr = b.permit2
    wethOracleAddress = b.wethOracleAddress
    usdcOracleAddress = b.usdcOracleAddress
    router02 = b.uniRouter
    wethAddress = b.wethAddress
    usdcAddress = b.nativeUsdcAddress
    wethFeedAddr = b.wethFeed
    usdcFeedAddr = b.usdcFeed
  }

  if (networkName == "op") {
    if (!mainnet) {
      await resetCurrentOP()
      console.log("RESET")
      console.log("Testing on OP @", (await currentBlock())?.number)

    }
    masterAddr = o.Master
    stopLimitAddr = o.stopLimit
    bracketAddr = o.bracket
    permit2Addr = o.permit2
    wethOracleAddress = o.wethOracleAddress
    usdcOracleAddress = o.usdcOracleAddress
    router02 = o.uniRouter
    pool: o.wethUsdcPoolAddress
    wethAddress = o.wethAddress
    usdcAddress = o.nativeUsdcAddress
    wethFeedAddr = o.wethFeed
    usdcFeedAddr = o.usdcFeed
  }

  if (networkName == "arbitrum") {

    if (!mainnet) {
      await resetCurrentArb()
      console.log("Testing on ARB @", (await currentBlock())?.number)

    }

    masterAddr = a.Master
    stopLimitAddr = a.stopLimit
    bracketAddr = a.bracket
    permit2Addr = a.permit2
    wethOracleAddress = a.wethOracleAddress
    usdcOracleAddress = a.usdcOracleAddress
    router02 = a.uniRouter
    pool: a.wethUsdcPoolAddress
    wethAddress = a.wethAddress
    usdcAddress = a.nativeUsdcAddress
    wethFeedAddr = a.wethFeed
    usdcFeedAddr = a.usdcFeed

  }
  let [signer] = await ethers.getSigners()

  Master = AutomationMaster__factory.connect(masterAddr, signer)
  StopLimit = StopLimit__factory.connect(stopLimitAddr, signer)
  Bracket = Bracket__factory.connect(bracketAddr, signer)

  if (!mainnet) {
    signer = await ethers.getSigner(userAddr)

    //testing does not scale tx cost correctly 
    await setBalance(await signer.getAddress(), ethers.parseEther("1"))

    await impersonateAccount(await signer.getAddress())

  }

  //do things
  //await deployOracles(signer)
  //await deployEverything(signer)
  //await updateSubKeeper(signer)
  //await wstethOracle(signer)
  //await moreOracles(signer)


  console.log("DONE")
}

const pythOracle = async (signer: Signer) => {

  //deploy pyth oracle

  //read price

  //register

}

const wstethOracle = async (signer: Signer) => {

  const oracle: IOracleRelay = await DeployContract(new TokenEthRelay__factory(signer), signer, o.wstethAddress, o.wstethEthFeed, o.wethOracleAddress)
  await oracle.deploymentTransaction()

  console.log("WSTETH: ", ethers.formatUnits(await oracle.currentValue(), 8))


}

const moreOracles = async (signer: Signer) => {

  const tokens = [
    o.wstethAddress,
    o.opAddress,
    o.wbtcAddress,
    o.uniAddress,
    o.aaveAddress,
    o.snxAddress
  ];

  const feeds = [
    o.wstethFeed,
    o.opFeed,
    o.wbtcFeed,
    o.uniFeed,
    o.aaveFeed,
    o.snxFeed
  ];

  const oracleAddresses: string[] = []

  for (let i = 0; i < tokens.length; i++) {

    const oracle: IOracleRelay = await DeployContract(new OracleRelay__factory(signer), signer, tokens[i], feeds[i])
    await oracle.deploymentTransaction()
    oracleAddresses.push(await oracle.getAddress())
    console.log("Deployed: ", oracleAddresses[i])
    //console.log("Price: ", ethers.formatUnits(await oracle.currentValue(), 8))

  }


  //register
  await Master.connect(signer).registerOracle(tokens, oracleAddresses)
  console.log("REGISTERED")

  if (mainnet) {
    console.log("Verifying...")
    try {
      await hre.run("verify:verify", {
        address: oracleAddresses[0],
        constructorArguments: [
          tokens[0],
          feeds[0]
        ]
      })
    } catch (error) {
      console.log("Error: ", error)
    }
    console.log("verified")
  }







}

const deployOracles = async (signer: Signer) => {

  const wethOracle: IOracleRelay = await DeployContract(new OracleRelay__factory(signer), signer, wethAddress, wethFeedAddr)
  wethOracleAddress = await wethOracle.getAddress()
  console.log("DEPLOYED ETH ORACLE: ", await wethOracle.getAddress())
  console.log("WETH: ", ethers.formatUnits((await wethOracle.currentValue()).toString(), 8))

  const usdcOracle: IOracleRelay = await DeployContract(new OracleRelay__factory(signer), signer, usdcAddress, usdcFeedAddr)
  usdcOracleAddress = await usdcOracle.getAddress()
  console.log("DEPLOYED USDC ORACLE: ", await usdcOracle.getAddress())
  console.log("USDC: ", ethers.formatUnits((await usdcOracle.currentValue()).toString(), 8))


  if (mainnet) {
    console.log("Verifying...")
    try {
      await hre.run("verify:verify", {
        address: await wethOracle.getAddress(),
        constructorArguments: [
          wethAddress,
          wethFeedAddr
        ]
      })
    } catch (error) {
      console.log("Error: ", error)
    }
    console.log("verified")
  }

}

const register = async (signer: Signer) => {

  const wbtcAddress = "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"
  //const wstethAddress = "0x5979D7b546E38E414F7E9822514be443A4800529"//todo only has cl price against eth
  const arbAddress = "0x912CE59144191C1204E64559FE8253a0e49E6548"
  const usdtAddress = "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9"

  //register sub keepers
  await Master.registerSubKeepers(
    stopLimitAddr,
    bracketAddr
  )


  const wbtcOracleAddress = "0x17B7bD832666Ac28A6Ad35a93d4efF4eB9A07a17"
  const arbOracleAddress = "0x47CBd328B185Ea8fC61Ead9a32d0edd79067b577"
  const usdtOracleAddress = "0x0E2a18163e6cB2eB11568Fad35E42dE4EE67EA9a"

  //register tokens => oracles
  const tokens = [wethAddress, usdcAddress, wbtcAddress, arbAddress, usdtAddress]
  const oracles = [wethOracleAddress, usdcOracleAddress, wbtcOracleAddress, arbOracleAddress, usdtOracleAddress]
  let tx = await Master.connect(signer).registerOracle(tokens, oracles)
  await tx.wait()
  console.log("REGISTERED ORACLES")

  tx = await Master.connect(signer).setMaxPendingOrders(25)
  await tx.wait()
  console.log("SET MAX PENDING ORDERS")

  tx = await Master.connect(signer).setMinOrderSize(ethers.parseUnits("0.25", 8))
  await tx.wait()
  console.log("SET MIN ORDER SIZE")
}

const deployEverything = async (signer: Signer) => {
  console.log("Deploying Everything")


  Master = await DeployContract(new AutomationMaster__factory(signer), signer)
  masterAddr = await Master.getAddress()
  console.log("DEPLOYED Master: ", await Master.getAddress())
  await Master.deploymentTransaction()
  //await new Promise(f => setTimeout(f, 5000));

  //stop loss limit
  Bracket = await DeployContract(new Bracket__factory(signer), signer, masterAddr, permit2Addr)
  bracketAddr = await Bracket.getAddress()
  console.log("DEPLOYED STOP_LOSS_LIMIT: ", await Bracket.getAddress())
  await Bracket.deploymentTransaction()
  //await new Promise(f => setTimeout(f, 5000));

  //stop limit
  StopLimit = await DeployContract(new StopLimit__factory(signer), signer, masterAddr, bracketAddr, permit2Addr)
  stopLimitAddr = await StopLimit.getAddress()
  console.log("DEPLOYED STOP_LIMIT: ", await StopLimit.getAddress())
  await StopLimit.deploymentTransaction()
  //await new Promise(f => setTimeout(f, 5000));

  //setup
  await register(signer)



  if (mainnet) {
    console.log("Starting verifications...");

    await Promise.all([
      (async () => {
        try {
          console.log("Verifying MASTER...");
          await hre.run("verify:verify", {
            address: await Master.getAddress(),
          });
          console.log("MASTER verified successfully");
        } catch (error) {
          console.error("Verification of MASTER failed:", error);
        }
      })(),
      (async () => {
        try {
          console.log("Verifying STOP_LIMIT...");
          await hre.run("verify:verify", {
            address: await StopLimit.getAddress(),
            constructorArguments: [
              masterAddr,
              bracketAddr,
              permit2Addr

            ]
          });
          console.log("STOP_LIMIT verified successfully");
        } catch (error) {
          console.error("Verification of STOP_LIMIT failed:", error);
        }
      })(),
      (async () => {
        try {
          console.log("Verifying STOP_LOSS_LIMIT...");
          await hre.run("verify:verify", {
            address: await Bracket.getAddress(),
            constructorArguments: [
              masterAddr,
              permit2Addr
            ]
          });
          console.log("STOP_LOSS_LIMIT verified successfully");
        } catch (error) {
          console.error("Verification of STOP_LOSS_LIMIT failed:", error);
        }
      })()
    ]);

    console.log("Verifications sent");

  }
}
const updateSubKeeper = async (signer: Signer) => {
  if (!mainnet) {
    signer = await ethers.getSigner("0x085909388fc0cE9E5761ac8608aF8f2F52cb8B89")

    //testing does not scale tx cost correctly 
    await setBalance(await signer.getAddress(), ethers.parseEther("1"))
    await impersonateAccount(await signer.getAddress())

  }

  const newBracket = await DeployContract(
    new Bracket__factory(signer),
    signer,
    a.Master,
    a.permit2,
  )
  await newBracket.deploymentTransaction()
  await new Promise(f => setTimeout(f, 5000));
  console.log("Bracket Deployed: ", await newBracket.getAddress())

  masterKeeper = AutomationMaster__factory.connect(a.Master, signer)

  await masterKeeper.registerSubKeepers(
    a.stopLimit,
    await newBracket.getAddress()
  )
  console.log("Registered Sub Keepers on Master: ", await masterKeeper.getAddress())

  if (mainnet) {
    console.log("Verifying...")
    try {
      await hre.run("verify:verify", {
        address: await newBracket.getAddress(),
        constructorArguments: [
          masterAddr,
          bracketAddr
        ]
      })
    } catch (error) {
      console.log(error)
    }
    console.log("verified")
  }


}


main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error)
    process.exit(1)
  })

/**
hh verify --network arbitrum 0x6D746d529F0D0C38A5abD561792917F8c4623E55 "0x6e521bcc7d159AE2168eb9990CEA09149c8F309C" "0x000000000022D473030F116dDEE9F6B43aC78BA3"
 */

