# Telcoin Contracts

![hardhat](https://img.shields.io/badge/hardhat-2.22.1-blue)
![node](https://img.shields.io/badge/node-v20.11.1-brightgreen.svg)
![solidity](https://img.shields.io/badge/solidity-0.8.24-red)

**Telcoin** is designed to complement telecom, mobile money, and e-wallet partners globally with both traditional fiat and blockchain transaction rails that underpin our fast and affordable digital financial service offerings. Telcoin combines the best parts of the burgeoning DeFi ecosystem with our compliance-first approach to each market, ensuring that the company takes on a fraction of traditional financial counter-party, execution, and custody risks.

Contract Overview

The AmirX suite of contracts offers a complex decentralized finance (DeFi) ecosystem based on stablecoin interactions, fee buybacks, and blacklist functionality for improved security. These contracts aim to support stablecoin handling, user interactions with DeFi swap mechanisms, and address restrictions through blacklisting. The setup builds on OpenZeppelin and other external libraries for safe and standardized functions.

Main Contracts

 1. AmirX:
 • Extends StablecoinHandler to implement a mechanism for DeFi token swaps and fee buybacks.
 • Core features include managing swaps across tokens, handling fees, executing buybacks, and facilitating referral payments.
 2. StablecoinHandler:
 • Manages the minting, burning, and swap mechanisms for stablecoins.
 • Supports DeFi interactions for designated stablecoins (XYZ tokens) and general ERC20 tokens.
 3. Blacklist:
 • Implements a blacklisting mechanism, restricting access for flagged addresses.
 • Provides roles for blacklisting/unblacklisting and allows internal transfer of funds from blacklisted accounts.
 4. Stablecoin:
 • Defines a stablecoin with minting and burning functionalities, pegged to a fiat currency.
 • Integrates blacklisting to prevent unauthorized interactions and allows role-based minting/burning.

Detailed Functionality

AmirX Contract

 • Swap Functionality: The swap function supports multi-token interactions by routing transactions through stablecoin and DeFi swap mechanisms.
 • Fee Dispersal: Handles fee buybacks, mainly in Telcoin (TEL), and allocates fees to referral addresses as needed.
 • Internal Helper Functions:
 • _feeDispersal: Manages fee distribution across referrers and safes.
 • _buyBack: Executes buybacks using a swap aggregator, supporting both ERC20 tokens and Polygon-based assets (POL).
 • Error Handling: Many functions use explicit error handling, which could benefit from comments explaining scenarios that lead to each error, especially for:
 • ZeroValueInput
 • Blacklisted

StablecoinHandler Contract

 • Stablecoin Swap: Provides core mechanisms to swap stablecoins, with modifiers to ensure non-zero inputs.
 • eXYZ Token Management:
 • Functions like UpdateXYZ allow role-based updates to XYZ token parameters, ensuring safe operations within supply limits.
 • getMinLimit and getMaxLimit are important to verify supply boundaries.
 • Internal _stablecoinSwap Function:
 • This function handles stablecoin transactions while applying necessary minting, burning, and transfer rules. It would benefit from comments describing each of the processing steps for new users unfamiliar with minting/burning logic.
 • Modifiers and Role-Based Access Control:
 • Several functions are controlled by roles like SWAPPER_ROLE and PAUSER_ROLE, restricting swap operations and pausing capabilities.

Blacklist Contract

 • Blacklist Management: addBlackList and removeBlackList are core functions allowing accounts to be flagged or cleared.
 • Error Handling:
 • Includes AlreadyBlacklisted and NotBlacklisted custom errors. A short comment explaining the use cases (e.g., to prevent redundant or incorrect blacklist operations) would be beneficial.
 • Event Emissions: Emits events when addresses are blacklisted/unblacklisted, which supports transparency in blacklist operations.

Stablecoin Contract

 • Minting and Burning: Supports minting and burning functionalities through mintTo, mint, burn, and burnFrom.
 • Blacklist Enforcement: Overrides _onceBlacklisted to capture funds of blacklisted accounts, effectively transferring them to the caller.
 • Decimals Management: Uses a storage slot to maintain decimals, ensuring consistency across deployments.
 • Rescue Operations: The erc20Rescue function allows the retrieval of mistakenly transferred funds, restricted by SUPPORT_ROLE.

## Running Tests

To get started, all you should need to install dependencies and run the unit tests are here.

```shell
npm install
npm test
```

Under the hood `npm test` is running `npx hardhat clean && npx hardhat coverage`

For a quicker unit test run:

```shell
npx hardhat test
```

### Notes

`Test` contracts are dummy contracts created for testing and are outside the scope of the audit. `Mock` contracts are created to be tested in place of the real contract. This is done for testing ease. In some cases using a slightly altered version is significantly simpler to test.

### Version

`nvm` use will switch to node `v20.11.1`

## References

```txt
                                     ttttttttttttttt,                           
                              *tttttttttttttttttttttttt,                        
                       *tttttttttttttttttttttttttttttttttt,                     
                ,tttttttttttttttttttttttttttttttttttttttttttt,                  
          .ttttttttttttttttttttttttttttttttttttttttttttttttttttt.               
        ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt.            
       ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt.         
      ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt       
     .ttttttttttttttttttttttttttttttttt    ttttttttttttttttttttttttttttttttt.   
     tttttttttttttttttttttttttttttttt     *ttttttttttttttttttttttttttttttttttt. 
     ttttttttttttttttttttttttttttt.       ttttttttttttttttttttttttttttttttttttt,
    *ttttttttttttttttttttttttt,          ************ttttttttttttttttttttttttttt
    tttttttttttttttttttttttt                        tttttttttttttttttttttttttttt
   *ttttttttttttttttttttttt*                        ttttttttttttttttttttttttttt,
   ttttttttttttttttttttttttttttt        *tttttttttttttttttttttttttttttttttttttt 
  ,tttttttttttttttttttttttttttt,       ,tttttttttttttttttttttttttttttttttttttt* 
  ttttttttttttttttttttttttttttt        ttttttttttttttttttttttttttttttttttttttt  
  tttttttttttttttttttttttttttt.       ,ttttttttttttttttttttttttttttttttttttttt  
 ttttttttttttttttttttttttttttt        ttttttttttttttttttttttttttttttttttttttt   
 ttttttttttttttttttttttttttttt        ttttttttttttttttttttttttttttttttttttttt   
 ttttttttttttttttttttttttttttt         *********tttttttttttttttttttttttttttt.   
 ttttttttttttttttttttttttttttt*                 tttttttttttttttttttttttttttt    
  *ttttttttttttttttttttttttttttt               tttttttttttttttttttttttttttt*    
    .tttttttttttttttttttttttttttttttttttttttttt*ttttttttttttttttttttttttttt     
       .ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt     
          .ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt      
             .tttttttttttttttttttttttttttttttttttttttttttttttttttttttttt,       
                .ttttttttttttttttttttttttttttttttttttttttttttttttttttt          
                   ,ttttttttttttttttttttttttttttttttttttttttttt*                
                      ,ttttttttttttttttttttttttttttttttt*                       
                         ,tttttttttttttttttttttttt.                             
                            ,*ttttttttttttt.                                    
```
