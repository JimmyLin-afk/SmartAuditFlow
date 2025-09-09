import { AutomationMaster__factory, IERC20__factory, PlaceholderOracle__factory, StopLimit__factory, UniswapV3Pool__factory } from "../../typechain-types"
import { currentBlock, resetCurrentArbBlock } from "../../util/block"
import { expect } from "chai"
import { stealMoney } from "../../util/money"
import { decodeUpkeepData, generateUniTx, generateUniTxData, getGas, MasterUpkeepData } from "../../util/msc"
import { s, SwapParams } from "./scope"
import { DeployContract } from "../../util/deploy"
import { ethers } from "hardhat"


///All tests are performed as if on Arbitrum
///Testing is on the Arb WETH/USDC.e pool @ 500
describe("Test for failure - STOP LIMIT", () => {


    let currentPrice: bigint

    let steveOrder: number
    const steveStrikeDelta = BigInt(ethers.parseUnits("75", 8))
    const smallSlippage = 0
    const steveBips = 500

    const veryLargeWethAmount = s.steveWeth * 5n


    before(async () => {
        //reset price
        await s.wethOracle.setPrice(s.initialEthPrice)
        currentPrice = await s.Master.getExchangeRate(await s.WETH.getAddress(), s.USDC.getAddress())

        //fund steve
        await stealMoney(s.wethWhale, await s.Steve.getAddress(), await s.WETH.getAddress(), s.steveWeth)

    })

    beforeEach(async () => {
        //reset price
        await s.wethOracle.setPrice(s.initialEthPrice)
        currentPrice = await s.Master.getExchangeRate(await s.WETH.getAddress(), s.USDC.getAddress())

    })


    it("Order creation fails due to insufficient balance", async () => {
        await s.WETH.connect(s.Steve).approve(await s.StopLimit.getAddress(), veryLargeWethAmount)
        expect(s.StopLimit.connect(s.Steve).createOrder(
            currentPrice - steveStrikeDelta,
            currentPrice + steveStrikeDelta,
            0,
            veryLargeWethAmount,
            await s.WETH.getAddress(),
            await s.USDC.getAddress(),
            await s.Steve.getAddress(),
            smallSlippage,
            smallSlippage,
            5n,//fee
            0,
            false,
            false,//no permit
            "0x"
        )).to.be.revertedWith("ERC20: transfer amount exceeds balance")
    })

})


