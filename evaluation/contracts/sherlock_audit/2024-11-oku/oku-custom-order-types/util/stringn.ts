import { BigNumber, BigNumberish } from "ethers";

export const BigFromN = (n: BigNumberish) => {
  if (BigNumber.isBigNumber(n)) {
    return n;
  }
  if (typeof n === "string") {
    if (n.includes("e")) {
      const spl = n.split("e");
      if (spl.length == 2) {
        const base = BigNumber.from(spl[0]);
        const mant = BigNumber.from(10).pow(spl[1]);
        const out = base.mul(mant);
        return out;
      }
    }
  }
  return BigNumber.from(n);
};
