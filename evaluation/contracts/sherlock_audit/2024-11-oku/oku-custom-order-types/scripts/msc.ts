import hre, { ethers, network } from "hardhat";
import { DeployContract } from "../util/deploy";
import { currentBlock, resetCurrent, reset, resetCurrentOP, resetCurrentPoly } from "../util/block";
import { IOracleRelay__factory, PythOracle__factory } from "../typechain-types";
import { impersonateAccount } from "../util/impersonator";
import { Signer } from "ethers";
import { e, o } from "../util/addresser";
import axios from 'axios';
import { HermesClient } from "@pythnetwork/hermes-client";

const opLOR = "0x54dF9e11c7933a9cA3BD1E540B63dA15edAe40bf"
const mainnetLOR = "0x54dF9e11c7933a9cA3BD1E540B63dA15edAe40bf"
const baseLOR = "0xfF8b754c64e9a8473Bd6E1118d0eaC67F0A8Ae27"
const LOR = baseLOR

async function main() {

  const accounts = await ethers.getSigners();
  const user = accounts[0];


  const networkName = hre.network.name
  let mainnet = true

  if (networkName == "hardhat" || networkName == "localhost") {

    await reset(20730768)
    mainnet = false
    console.log("TEST @ ", await (await currentBlock())!.number)
  } else {
    console.log("Sending for real...")
  }

  await testPythOracle(user)
}

const testPythOracle = async (signer: Signer) => {
  await resetCurrentOP()
  const pythAddr = "0xff1a0f4744e8582DF1aE09D5611b887B6a12925C"
  const wethTokenId = "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace"//"0x9d4294bbcd1174d6f2003ec365831e64cc31d9f6f15a2b85399db8d5000960f6"

  const pythOracle = await DeployContract(new PythOracle__factory(signer), signer, pythAddr, wethTokenId, 50000, o.wethAddress)
  const refOracle = IOracleRelay__factory.connect(o.wethOracleAddress, signer)
  console.log("REFERENCE PRICE: ", await refOracle.currentValue())

  const connection = new HermesClient("https://hermes.pyth.network", {});
  const priceUpdates = await connection.getLatestPriceUpdates([wethTokenId]);
  const data = priceUpdates.binary.data[0]
  const formatted = data.startsWith("0x") ? data : "0x" + data;
  const testData = "0x504e41550100000003b801000000040d00d476f90ba2c371ed65d3af2611070885c4b1f7f96f1f4185a4df477370989e497e80c9b3fa17410fec4208c4d6fd9b7ad22454e9ed4f0e9283ee5acfcd4a02af010345b4159ecf79cb65d19433dce69b445f1bc324f139e96000bf294d3ef71c5b3737c00a670070ffcd3dac56f0d10430ec68d796323c161d3df1b0b6e336bed34b010462f1fe693b5cf9001237aa575e7552aade0614261c52ccf305a0ecf8574a2cf55afd93922b57c8b6796aceb7993319091116d85227edc185efd2b80c065b589e010854a66f5babd8bd91391a443ea9e09ad41e7564d29c2df78e0644b2739b4495550890dd76ccba45aa6174207695948f0fc35c7bc269ea581045f1418cefb7a39f010ad2eb9e96abd1718bbaa7d32ff2345baffa0b21457a63133f2e52d20e513e0d1b2e4b9d074a59e1c5ce435ffe18b86373257221f05882dceefdb9b7749e3ed9db010b99600fe20546c1d9b29c47eca96b917c609ae84b8273f13da891f0d8300ac3984c2c331e96c8a9decf6feff348b5f867f397ab419bff8ffea84fe79734299b30000c7b8e248ef7ad07d14b92331565cbbd0c7497b3cc4f5bb97dc7efe7a1e6aa32c91b8beec704ffd10ff99b654d3cda8c24dc2e4691db7c0c87028f9ac69cc4a4f2010d02953d9773fe85b3f4d0c07e0bdc990a80d496f165b7dd5804a74993032120211f125f3214fdaf3f3b5bd9fb974d02c2bc34519caae6ce6933457c83fdbe8199010e6ca03dfd9c22dfff0faef3fc9e61b24c2e95f3d3bfc596446f618bcaae475167143c51b63ac40d91058fb9b6c500b47cf3002e8fc06ab36a211bb5968be660e6010f41729f3b39efdce263ab7f81a21931b3b0547f0cfacf0a2bce8d0db37d06fc3a15cbc39776a3029dd281d12eeb779b99860a53ec4af28651b429eea992ab85ad00107ef1d7e24f02abdeada7928661f346500c432a37e2d8c1fe9aa3f5362561f3c31d634a43002ccdb8f53390c93164b865c2f150a5178671abcb98c25b05e469250111c54e805ef377e52884abb916bef00047cf7738d4df871ace4bdc4f101fd6971648254c9a622fa8f89916218e088a99036df82214aacff0cd9d98833b4037c79800129608cff8770a2307a135bb24a15413cac47dd82043dd8ddd618bd00875c729462b7d507e74e7c653ae343ddaa2af076670efc5f28c58b6365e1ddc8d8908411101673e6bb200000000001ae101faedac5851e32b9b23b5f9411a8c2bac4aae3ed4dd7b811dd1a72ea4aa710000000005a35e76014155575600000000000ab03275000027105edd8e80b7da3673380d9297d680b085f2f818f701005500ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace00000047812ecbe2000000000dcb1972fffffff800000000673e6bb200000000673e6bb200000047aa10dd70000000000c79d9b90bd2125c29e0a4b2ff4076d874b52bfd0b668751690c34674b2cf5a508bac9759af373290c98a1006ef78c19709aa6b7540e1ec10646001dd95f2ae6456c632a4c57b75a8eef0ae1a8d6855be16f631d4c4f0ae07b8a91e59a4c384fa2e04e989577ed142dae1656ee24698dd27616efc52c0be3b36a5b74636122567969f4320daea3ffce6b1a459c5e116a5039ca465272480178dc07861aeac711fcff33a2d5197f8b8eb4d1dc44d4482f7d08ffc74d5a7960a487b8d8ee60e2ece30b244e63e5eafff55f94c175ae8a7fd70d6c4389c61ebe0f390874fa2dc0a96e"

  await pythOracle.updatePrice([formatted], {value: ethers.parseEther("1")})


}

const addPoolsOp = async (signer: Signer) => {

  const opPools = [
    "0x68f5c0a2de713a54991e01858fd27a3832401849",
    "0x85149247691df622eaf1a8bd0cafd40bc45154a9",
    "0xfc1f3296458f9b2a27a0b91dd7681c4020e09d05",
    "0x1c3140ab59d6caf9fa7459c6f83d4b52ba881d36",
    "0x1fb3cf6e48f1e7b10213e7b6d87d4c073c7fdb7b",
    "0x85c31ffa3706d1cce9d525a00f1c7d4a2911754c",
    "0x2ab22ac86b25bd448a4d9dc041bd2384655299c4",
    "0x0392b358ce4547601befa962680bede836606ae2",
    "0x04f6c85a1b00f6d9b75f91fd23835974cc07e65c",
    "0xf1f199342687a7d78bcc16fce79fa2665ef870e1",
    "0x8323d063b1d12acce4742f1e3ed9bc46d71f4222",
    "0x73b14a78a0d396c521f954532d43fd5ffe385216",
    "0xb533c12fb4e7b53b5524eab9b47d93ff6c7a456f",
    "0x03af20bdaaffb4cc0a521796a223f7d85e2aac31",
    "0xa73c628eaf6e283e26a7b1f8001cf186aa4c0e8e",
    "0xbf16ef186e715668aa29cef57e2fd7f9d48adfe6",
    "0x252cbdff917169775be2b552ec9f6781af95e7f6",
    "0x535541f1aa08416e69dc4d610131099fa2ae7222",
    "0xb589969d38ce76d3d7aa319de7133bc9755fd840",
    "0xc858a329bf053be78d6239c4a4343b8fbd21472b",
    "0xd1f1bad4c9e6c44dec1e9bf3b94902205c5cd6c3",
    "0x95d9d28606ee55de7667f0f176ebfc3215cfd9c0",
    "0x766854992bd5363ebeeff0113f5a5795796befab",
    "0xd28f71e383e93c570d3edfe82ebbceb35ec6c412",
    "0xa8a5356ee5d02fe33d72355e4f698782f8f199e8",
    "0x4ce4a1a593ea9f2e6b2c05016a00a2d300c9ffd8",
    "0x9438a9d1bdeece02ed4431ac59613a128201e0b9",
    "0xa7bb0d95c6ba0ed0aca70c503b34bc7108589a47",
    "0xadb35413ec50e0afe41039eac8b930d313e94fa4",
    "0xb2ac2e5a3684411254d58b1c5a542212b782114d",
    "0x8eda97883a1bc02cf68c6b9fb996e06ed8fdb3e5",
    "0x19ea026886cbb7a900ecb2458636d72b5cae223b",
    "0x5adba6c5589c50791dd65131df29677595c7efa7",
    "0x730691cdac3cbd4d41fc5eb9d8abbb0cea795b94",
    "0x6168ec836d0b1f0c37381ec7ed1891a412872121",
    "0x2ae3d6096d8215ac2acddf30c60caa984ea5debe",
    "0x394a9fcbab8599437d9ec4e5a4a0eb7cb1fd2f69",
    "0xd52533a3309b393afebe3176620e8ccfb6159f8a",
    "0xbf595eb9a512b1c274125264aef84a2847158eb3",
    "0xf44acaa38be5e965c5ddf374e7a2ba270e580684",
    "0xac85eaf55e9c60ed40a683de7e549d23fdfbeb33",
    "0x250e21dddd306579458cf025c5e230665171fb31",
    "0xe9e3893921de87b1194a8108f9d70c24bde71c27",
    "0xc1738d90e2e26c35784a0d3e3d8a9f795074bca4",
    "0x37ffd11972128fd624337ebceb167c8c0a5115ff",
    "0x2e2d190ad4e0d7be9569baebd4d33298379b0502",
    "0xdc6cb16b8d635ef255cef370263c35ac0007186a",
    "0x8531e48a8611729185be9eaad945acbd6b32e256",
    "0x1d751bc1a723accf1942122ca9aa82d49d08d2ae",
    "0x0ca747e5c527e857d8a71b53b6efbad2866b9e04",
    "0x1d789e3b2146cfa96761996d14cd3cc54a8b133a",
  ]

  const master = MasterKeeper__factory.connect("0x6D746d529F0D0C38A5abD561792917F8c4623E55", signer)
  await master.connect(signer).addPools(opPools)


}

const checkAndExecute = async (signer: Signer, master: MasterKeeper) => {

  const result = await master.checkUpkeep("0x")
  console.log(result)

  if (result.upkeepNeeded == true) {
    console.log("Performing...")
    await master.performUpkeep(result.performData)
  }

}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error)
    process.exit(1)
  })


/**
hh verify --network op 0xfd41F406585eEE2A4BDF89b790172E1e5eE00036 "0x54dF9e11c7933a9cA3BD1E540B63dA15edAe40bf"
*/