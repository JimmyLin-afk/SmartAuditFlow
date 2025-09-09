import { AbiCoder, AddressLike, BytesLike, Signer } from "ethers";
import {  AutomationMaster, Bracket, IERC20, OracleLess, PlaceholderOracle, StopLimit, UniswapV3Pool } from "../../typechain-types";
import { ethers } from "hardhat";

export type Order = {
    orderId: BigInt,
    strikePrice: BigInt,
    amountIn: BigInt,
    pairId: BigInt,
    recipient: AddressLike,
    slippageBips: BigInt,
    zeroForOne: Boolean,
    direction: Boolean
}

export type SwapParams = {
    swapTokenIn: AddressLike,
    swapAmountIn: bigint,
    swapTarget: AddressLike,
    swapBips: number,
    txData: BytesLike
}

export class TestScope {


    signers!: Signer[]

    Frank!: Signer
    Andy!: Signer //tests for failure on LIMIT
    Steve!: Signer //tests for failure on STOP_LOSS_LIMIT
    Bob!: Signer
    Charles!: Signer//test swap-on-fill
    Ian!: Signer //isolated testing
    Oscar!: Signer //Oracle-less testing
    Gary!: Signer //Oracle-less failure testing

    abi = new AbiCoder()

    LimitOrderRegistry = "0x54df9e11c7933a9ca3bd1e540b63da15edae40bf"//arbiscan
    pool = "0xc31e54c7a869b9fcbecc14363cf510d1c41fa443"//WETH/USDC.e pool @ 500
    router02 = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"

    wethWhale = "0xE4f718a0b06D91cF6ff436d4445315ABDF99247b"
    usdcWhale = "0x25681Ab599B4E2CEea31F8B498052c53FC2D74db"
    wethAmount = ethers.parseEther("1.65")
    usdcAmount = ethers.parseUnits("5000", 6)
    uniAmount = ethers.parseEther("665")
    arbAmount = ethers.parseEther("6580")

    andyWeth = this.wethAmount * 20n
    steveWeth = this.andyWeth

    //CL oracles are priced @ 1e8
    initialEthPrice = ethers.parseUnits("3391.95", 8)
    initialUsdcPrice = ethers.parseUnits("0.9998", 8)
    initialUniPrice = ethers.parseUnits("7.53", 8)
    initialArbPrice = ethers.parseUnits("0.7581", 8)


    Master!: AutomationMaster
    StopLimit!: StopLimit
    Bracket!: Bracket
    OracleLess!: OracleLess

    maxPendingOrders = 20
    minOrderSize = ethers.parseUnits("10", 8)



    wethOracle!: PlaceholderOracle
    usdcOracle!: PlaceholderOracle
    uniOracle!: PlaceholderOracle
    arbOracle!: PlaceholderOracle

    UniPool!: UniswapV3Pool
    WETH!: IERC20 //weth token0 0x82af49447d8a07e3bd95bd0d56f35241523fbab1
    USDC!: IERC20 //USDC.e token1 0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8
    ARB!: IERC20 //0x912CE59144191C1204E64559FE8253a0e49E6548
    UNI!: IERC20 //0xFa7F8980b0f1E64A2062791cc3b0871572f1F7f0

}

export const s = new TestScope()