import fs from 'fs'

import { Wallet } from 'ethers'

const wallet = Wallet.createRandom()

console.log(
  `🔐 Account Generated as ${wallet.address} and set as mnemonic in packages/hardhat`
)
console.log(
  "💬 Use 'yarn run account' to get more information about the deployment account."
)

fs.writeFileSync(`./${wallet.address}.secret`, wallet.mnemonic!.phrase)
fs.writeFileSync('./mnemonic.secret', wallet.mnemonic!.phrase)
