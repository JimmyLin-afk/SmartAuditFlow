import { BigNumber } from "ethers"
export const BN = (a: any): BigNumber => {
    if (typeof a === "string") {
        a = a.replaceAll(",", "",)
        let splt = a.split("e")
        if (splt.length == 1) {
            splt.push("0")
        }
        return BigNumber.from(splt[0]).mul(BigNumber.from(10).pow(splt[1]))
    }
    return BigNumber.from(a)
}