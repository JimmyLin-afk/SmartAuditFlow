import { defineChain } from "viem";
import hre from "hardhat";

export const getBlast = async () => {
  const deployer = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const feeReceiver = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const wethAddress = "0x4300000000000000000000000000000000000004";
  const perpManager = "0xff2CDb9cdb79A60A31188FE37Bdc6774107cc268";
  const liquidationFeeReceiver = "0xF6336dd76300524Ef382FA9FC861305A37b929b6";

  const chain = defineChain({
    id: 81457,
    name: "blast",
    nativeCurrency: {
      name: "Ether",
      symbol: "ETH",
      decimals: 18
    },
    rpcUrls: {
      default: {
        http: ["https://holy-practical-owl.blast-mainnet.quiknode.pro/590d29b28faafbe6e3f80846cdc50e3f4c5356b0"]
      }
    }
  });
  const walletClient = await hre.viem.getWalletClient(deployer, { chain });
  const publicClient = await hre.viem.getPublicClient({ chain });
  const config = {
    client: {
      public: publicClient,
      wallet: walletClient
    }
  };
  return {
    config,
    deployer,
    feeReceiver,
    wethAddress,
    perpManager,
    liquidationFeeReceiver
  };
}

export const getBlastSepolia = async () => {
  const deployer = "0x129320410d1F827597Befcb01Dc7a037c7fbA6d5";
  const feeReceiver = "0x129320410d1F827597Befcb01Dc7a037c7fbA6d5";
  const wethAddress = "0x4200000000000000000000000000000000000023";
  const perpManager = "0xb52BAbD89eEDBeF6242784DC5c60C1E609538D06";

  const chain = defineChain({
    id: 168587773,
    name: "blast-sepolia",
    nativeCurrency: {
      name: "Ether",
      symbol: "ETH",
      decimals: 18
    },
    rpcUrls: {
      default: {
        http: ["https://yolo-restless-fire.blast-sepolia.quiknode.pro/6d8013584783e1f1ab32031f7091d7e21000c6af"]
      }
    }
  });
  const walletClient = await hre.viem.getWalletClient(deployer, { chain });
  const publicClient = await hre.viem.getPublicClient({ chain });
  const config = {
    client: {
      public: publicClient,
      wallet: walletClient
    }
  };
  return {
    config,
    deployer,
    feeReceiver,
    wethAddress,
    perpManager
  };
}