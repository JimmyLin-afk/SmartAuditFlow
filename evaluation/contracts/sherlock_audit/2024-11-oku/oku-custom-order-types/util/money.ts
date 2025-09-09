import { ethers, network } from "hardhat";
import { IERC20__factory, IERC20 } from "../typechain-types";
import { BigFromN } from "./stringn";
import { setBalance } from "@nomicfoundation/hardhat-network-helpers";
import { BN } from "./number";
import { BigNumberish } from "ethers";

export const stealMoney = async (
    from: string,
    to: string,
    tokenAddr: string,
    amount: BigNumberish
) => {
    //fund with eth so we can steal from any addr that holds the token, including contracts
    await setBalance(from, ethers.parseEther("5"))
    
    await network.provider.request({
        method: "hardhat_impersonateAccount",
        params: [from],
    });
    const robberee = await ethers.provider.getSigner(from);
    const money = IERC20__factory.connect(tokenAddr, robberee);
    await money.connect(robberee).transfer(to, amount);
    await network.provider.request({
        method: "hardhat_stopImpersonatingAccount",
        params: [from],
    });
    return;
};

export const approveMoney = async (
    from: string,
    to: string,
    tokenAddr: string,
    amount: BigNumberish
) => {
    await network.provider.request({
        method: "hardhat_impersonateAccount",
        params: [from],
    });
    const robberee = await ethers.getSigner(from);
    const money = IERC20__factory.connect(tokenAddr, robberee);
    await money.approve(to, BigFromN(amount));
    await network.provider.request({
        method: "hardhat_stopImpersonatingAccount",
        params: [from],
    });
    return;
};

export const getMoney = async (
    from: string,
    tokenAddr: string
): Promise<BigNumber> => {
    const signer = await ethers.getSigner(from);
    const money = IERC20__factory.connect(tokenAddr, signer);
    return money.balanceOf(from);
};
