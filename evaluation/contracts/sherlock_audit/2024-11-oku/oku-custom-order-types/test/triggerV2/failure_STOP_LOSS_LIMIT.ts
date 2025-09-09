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
describe("Test for failure - STOP LOSS LIMIT", () => {

    let currentPrice: bigint

    let steveOrder: bigint
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

    it("Swap fails due to slippage", async () => {
        await s.WETH.connect(s.Steve).approve(await s.Bracket.getAddress(), s.steveWeth)
        await s.Bracket.connect(s.Steve).createOrder(
            "0x",
            currentPrice + steveStrikeDelta,
            currentPrice - steveStrikeDelta,
            s.wethAmount,
            await s.WETH.getAddress(),
            await s.USDC.getAddress(),
            await s.Steve.getAddress(),
            5,//5 bips fee
            smallSlippage,
            smallSlippage,
            false,
            "0x"
        )

       
        const filter = s.Bracket.filters.OrderCreated
        const events = await s.Bracket.queryFilter(filter, -1)
        const event = events[0].args
        steveOrder = event[0]
        expect(Number(event.orderId)).to.not.eq(0, "First order")

        const order = await s.Bracket.orders(steveOrder)
        expect(order.recipient).to.eq(await s.Steve.getAddress(), "steve's order")

        //check upkeep
        let check = await s.Master.checkUpkeep("0x")
        expect(check.upkeepNeeded).to.eq(false, "no upkeep is needed yet")

        //adjust oracle
        await s.wethOracle.setPrice(currentPrice + steveStrikeDelta)

        //check upkeep
        check = await s.Master.checkUpkeep("0x")
        expect(check.upkeepNeeded).to.eq(true, "upkeep needed")

        //confirm pending order to be executed is steve's order
        const data: MasterUpkeepData = await decodeUpkeepData(check.performData, s.Steve)
        const pendingOrders = await s.Bracket.getPendingOrders()
        expect(pendingOrders[Number(data.pendingOrderIdx)]).to.eq(steveOrder, "steve's order is being filled")

        //try to fill, fail - slippage too low
        let minAmountReceived = await s.Master.getMinAmountReceived(data.amountIn, data.tokenIn, data.tokenOut, 500)//bips too high

        let encodedTxData = await generateUniTx(
            s.router02,
            s.UniPool,
            await s.Bracket.getAddress(),
            minAmountReceived,
            data
        )

        expect(s.Master.performUpkeep(encodedTxData)).to.be.revertedWith("Too Little Received")

        //try to cancel order that isn't yours
        expect(s.Bracket.connect(s.Bob).cancelOrder(steveOrder)).to.be.revertedWith("Only Order Owner")

        minAmountReceived = await s.Master.getMinAmountReceived(data.amountIn, data.tokenIn, data.tokenOut, data.bips)//actual bips are 0% slippage

        encodedTxData = await generateUniTx(
            s.router02,
            s.UniPool,
            await s.Bracket.getAddress(),
            minAmountReceived,
            data
        )

        //tx succeeds if we try to fill with 0% slippage, but swap fails
        expect(s.Master.performUpkeep(encodedTxData)).to.be.reverted


        //cancel order for future tests
        await s.Bracket.connect(s.Steve).cancelOrder(steveOrder)

    })


    it("Order creation fails due to insufficient balance", async () => {
        await s.WETH.connect(s.Steve).approve(await s.Bracket.getAddress(), veryLargeWethAmount)
        expect(s.Bracket.connect(s.Steve).createOrder(
            "0x",//no swap data
            currentPrice + steveStrikeDelta,
            currentPrice - steveStrikeDelta,
            veryLargeWethAmount,
            await s.WETH.getAddress(),
            await s.USDC.getAddress(),
            await s.Steve.getAddress(),
            5,//5 bips fee
            smallSlippage,
            smallSlippage,
            false,
            "0x"
        )).to.be.revertedWith("ERC20: transfer amount exceeds balance")

    })


    it("Spend pending balances", async () => {

        //create order
        await s.WETH.connect(s.Steve).approve(await s.Bracket.getAddress(), s.wethAmount)
        await s.Bracket.connect(s.Steve).createOrder(
            "0x",
            currentPrice + steveStrikeDelta,
            currentPrice - steveStrikeDelta,
            s.wethAmount,
            await s.WETH.getAddress(),
            await s.USDC.getAddress(),
            await s.Steve.getAddress(),
            5,//5 bips fee
            steveBips,
            steveBips,
            false,
            "0x"
        )

        //adjust oracle
        await s.wethOracle.setPrice(currentPrice + steveStrikeDelta)

        //check upkeep
        const check = await s.Master.checkUpkeep("0x")
        expect(check.upkeepNeeded).to.eq(true, "upkeep needed")


        //confirm pending order to be executed is steve's order

        const filter = s.Bracket.filters.OrderCreated
        const events = await s.Bracket.queryFilter(filter, -1)
        const event = events[0].args
        steveOrder = event[0]

        const order = await s.Bracket.orders(steveOrder)
        expect(order.recipient).to.eq(await s.Steve.getAddress(), "steve's order")
        const data: MasterUpkeepData = await decodeUpkeepData(check.performData, s.Steve)
        const pendingOrders = await s.Bracket.getPendingOrders()
        expect(pendingOrders[Number(data.pendingOrderIdx)]).to.eq(steveOrder, "steve's order is being filled")

        //now that we confirmed we are filling steve's order,
        //how much weth is on the contract, relative to how much we are supposed to be allowed to send (steve's order.amountIn)?
        const totalWeths = await s.WETH.balanceOf(await s.Bracket.getAddress())
        const expectedAmountIn = data.amountIn
        const expectedRemaining = totalWeths - expectedAmountIn

        //inject malicious amount into the tx data
        data.amountIn += ethers.parseEther("5")//increase the amount we pass in to encoded data

        //try to fill
        let minAmountReceived = await s.Master.getMinAmountReceived(data.amountIn, data.tokenIn, data.tokenOut, data.bips)
        let encodedTxData = await generateUniTx(
            s.router02,
            s.UniPool,
            await s.Bracket.getAddress(),
            minAmountReceived,
            data
        )
        expect(s.Master.performUpkeep(encodedTxData)).to.be.revertedWith("over spend")

        //now try to fill with mismatched amountIn, and receive refund
        const underFillAmount = ethers.parseEther("0.01")

        let initBalance = await s.WETH.balanceOf(await s.Steve.getAddress())

        data.amountIn = expectedAmountIn - underFillAmount
        minAmountReceived = await s.Master.getMinAmountReceived(data.amountIn, data.tokenIn, data.tokenOut, data.bips)
        encodedTxData = await generateUniTx(
            s.router02,
            s.UniPool,
            await s.Bracket.getAddress(),
            minAmountReceived,
            data
        )
        await s.Master.performUpkeep(encodedTxData)
        const delta = (await s.WETH.balanceOf(await s.Steve.getAddress())) - initBalance
        expect(delta).to.eq(underFillAmount, "Refund issued")


    })


})


