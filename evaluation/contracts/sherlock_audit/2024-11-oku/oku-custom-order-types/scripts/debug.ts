import hre, { network } from "hardhat";
import { currentBlock, hardhat_mine_timed, resetCurrentArb, resetCurrentArbBlock, resetCurrentBase, resetCurrentOP, resetCurrentOPblock } from "../util/block";
import { AutomationMaster, AutomationMaster__factory, Bracket, Bracket__factory, IERC20, IERC20__factory, IPermit2__factory, StopLimit, StopLimit__factory, UniswapV3Pool__factory } from "../typechain-types";
import { Signer } from "ethers";
import { impersonateAccount } from "../util/impersonator";
import { setBalance } from "@nomicfoundation/hardhat-network-helpers";
import { decodeUpkeepData, generateUniTx, generateUniTxData, MasterUpkeepData, permitSingle } from "../util/msc";
import { a, b, o } from "../util/addresser";
import { s, SwapParams } from "../test/triggerV2/scope";
const { ethers } = require("hardhat");


let Master: AutomationMaster
let StopLimit: StopLimit
let Bracket: Bracket
let mainnet = true
let masterAddr: string //"0x8327B0168858bd918A0177e89b2c172475F6B16f"//second deploy//0x4f38FA4F676a053ea497F295f855B2dC3580f517"//initial deploy
let bracketAddr: string
let stopLimitAddr: string
let permitAddr: string

//tokens
let WETH: IERC20
let USDC: IERC20


//SET THIS FOR TESTING
const testingNetwork = "arbitrum"
const userAddr = "0x085909388fc0cE9E5761ac8608aF8f2F52cb8B89"
const wethAmount = ethers.parseEther("0.0005")
const stopLimitDelta = ethers.parseUnits("1", 8)
const strikeDelta = ethers.parseUnits("5", 8)
const stopDelta = ethers.parseUnits("5", 8)
const testBips = 2000
let chainId = 42161

async function main() {
    console.log("STARTING")
    let networkName = hre.network.name
    console.log(networkName)

    let [signer] = await ethers.getSigners()

    const network = await ethers.provider.getNetwork();
    chainId = Number(network.chainId)
    console.log("GOT CHAINID: ", chainId)

    if (networkName == "hardhat" || networkName == "localhost") {
        networkName = testingNetwork
        mainnet = false
        console.log("Testing on network : ", networkName)

    } else {
        console.log("Sending for real to: ", networkName)
    }

    if (networkName == "arbitrum") {

        if (!mainnet) {
            await resetCurrentArbBlock(266974358)
            console.log("Testing on ARB @", (await currentBlock())?.number)

        }
        masterAddr = a.Master
        stopLimitAddr = a.stopLimit
        bracketAddr = a.bracket
        permitAddr = a.permit2

        WETH = IERC20__factory.connect(a.wethAddress, signer)
        USDC = IERC20__factory.connect(a.nativeUsdcAddress, signer)
    }

    if (networkName == "base") {

        if (!mainnet) {
            await resetCurrentBase()
            console.log("Testing on BASE @", (await currentBlock())?.number)

        }
        masterAddr = b.Master
        stopLimitAddr = b.stopLimit
        bracketAddr = b.bracket
        permitAddr = b.permit2

        WETH = IERC20__factory.connect(b.wethAddress, signer)
        USDC = IERC20__factory.connect(b.nativeUsdcAddress, signer)
    }

    if (networkName == "op") {

        if (!mainnet) {
            await resetCurrentOP
            console.log("Testing on OP @", (await currentBlock())?.number)

        }
        masterAddr = o.Master
        stopLimitAddr = o.stopLimit
        bracketAddr = o.bracket
        permitAddr = o.permit2

        WETH = IERC20__factory.connect(o.wethAddress, signer)
        USDC = IERC20__factory.connect(o.nativeUsdcAddress, signer)
    }


    Master = AutomationMaster__factory.connect(masterAddr, signer)
    StopLimit = StopLimit__factory.connect(stopLimitAddr, signer)
    Bracket = Bracket__factory.connect(bracketAddr, signer)

    if (!mainnet) {
        signer = await ethers.getSigner(userAddr)

        console.log("IMPERSONATING: ", await signer.getAddress())

        //testing does not scale tx cost correctly 
        await setBalance(await signer.getAddress(), ethers.parseEther("1"))
        await impersonateAccount(await signer.getAddress())

    }

    await checkOrder(signer)

}

const checkOrder = async (signer: Signer) => {

    //await resetCurrentArbBlock(266980619)
    await resetCurrentArb()

    const orderId = 4//266980619//current sl price, high str price low stop price
    //const orderId = 3//266977095//high sl price
    //const orderId = 2//266974358//low sl price

    //const result = await Master.checkUpkeep("0x")
    //console.log(result.upkeepNeeded)

    //get order
    const order = await StopLimit.orders(orderId)
    //console.log("order", order)

    const er = await Master.getExchangeRate(order.tokenIn, order.tokenOut)
    console.log("EXCHANGE RATE: ", er)


    await impersonateAccount(await signer.getAddress())
    await setBalance(await signer.getAddress(), ethers.parseEther("1"))

    let check = await Master.checkUpkeep("0x")
    console.log("UpkeepNeeded: ", check.upkeepNeeded)


    //create an order to compare
    await WETH.connect(signer).approve(await StopLimit.getAddress(), wethAmount)
    await StopLimit.connect(signer).createOrder(
        249399113866n,
        er,
        1n,
        wethAmount,
        await WETH.getAddress(),
        await USDC.getAddress(),
        await signer.getAddress(),
        500,
        500,
        500,
        false,
        false,
        "0x"
    )

    const filter = StopLimit.filters.OrderCreated
    const events = await StopLimit.queryFilter(filter, -1)
    const event = events[0].args
    const testOrderId = (event[0])

    const testOrder = await StopLimit.orders(testOrderId)
    console.log(testOrder)
    console.log("ORDER CREATED")

    await hardhat_mine_timed(10, 5)

    check = await Master.checkUpkeep("0x")
    console.log("UpkeepNeeded: ", check.upkeepNeeded)

    //252295854237n
    //252292592233n







    /**
  if (result.upkeepNeeded) {
 
 
         const data: MasterUpkeepData = await decodeUpkeepData(result.performData, signer)
         console.log("DECODED", data.txData)//todo
 
         const minAmountReceived = await Master.getMinAmountReceived(data.amountIn, data.tokenIn, data.tokenOut, data.bips)
         console.log("MAR: ", ethers.formatUnits(minAmountReceived, 6), minAmountReceived)
 
         const encodedTxData = await generateUniTx(
             a.uniRouter,
             UniswapV3Pool__factory.connect(a.bridgedUsdcPool, signer),
             bracketAddr,
             minAmountReceived,
             data
         )
 
         await Master.performUpkeep(encodedTxData)
         console.log("DONE")
 
 
 
 
 
 
     }
     */

}


main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error)
        process.exit(1)
    })
