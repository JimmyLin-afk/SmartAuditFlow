# Pre-Optimizations
0ffc3e524c65a1d86db3aa35af8840a4db026a3a
Outputs of: `forge test --match-contract VVVETHStaking --gas-report`

| contracts/staking/VVVETHStaking.sol:VVVETHStaking contract       |                 |        |        |        |         |
|------------------------------------------------------------------|-----------------|--------|--------|--------|---------|
| Deployment Cost                                                  | Deployment Size |        |        |        |         |
| 1339975                                                          | 6457            |        |        |        |         |
| Function Name                                                    | min             | avg    | median | max    | # calls |
| DENOMINATOR                                                      | 273             | 273    | 273    | 273    | 6       |
| calculateAccruedVvvAmount((uint256,uint256,bool,uint8))(uint256) | 1746            | 3751   | 3751   | 5756   | 2       |
| calculateAccruedVvvAmount()(uint256)                             | 5060            | 5060   | 5060   | 5060   | 1       |
| calculateClaimableVvvAmount                                      | 3262            | 6173   | 7262   | 7272   | 11      |
| claimVvv                                                         | 308             | 37809  | 54649  | 59159  | 10      |
| durationToMultiplier                                             | 514             | 736    | 514    | 2514   | 9       |
| durationToSeconds                                                | 536             | 2411   | 2536   | 2536   | 32      |
| ethToVvvExchangeRate                                             | 204             | 204    | 204    | 204    | 6       |
| receive                                                          | 823             | 823    | 823    | 823    | 1       |
| restakeEth                                                       | 1685            | 82042  | 104575 | 114525 | 8       |
| setDurationMultipliers                                           | 10327           | 18258  | 18258  | 26189  | 2       |
| setNewStakesPermitted                                            | 2692            | 3139   | 2692   | 13992  | 46      |
| setVvvToken                                                      | 8936            | 23786  | 24570  | 24570  | 47      |
| stakeEth                                                         | 489             | 106769 | 118977 | 138877 | 43      |
| stakeId                                                          | 374             | 1374   | 1374   | 2374   | 4       |
| userStakeIds                                                     | 1156            | 2018   | 1626   | 3272   | 3       |
| userStakes                                                       | 1055            | 1055   | 1055   | 1055   | 20      |
| vvvToken                                                         | 404             | 404    | 404    | 404    | 1       |
| withdrawEth                                                      | 8853            | 24795  | 24795  | 40737  | 2       |
| withdrawStake                                                    | 1081            | 18238  | 11457  | 56357  | 22      |

# Post-Optimizations 1
fbb8b46461ae942c0cc8b659eedaebfd31a2eed1
Changes Made: in StakeData, stakedEthAmount was converted to uint224, and stakeStartTimestamp to uint32 

| contracts/staking/VVVETHStaking.sol:VVVETHStaking contract      |                 |       |        |        |         |    
|-----------------------------------------------------------------|-----------------|-------|--------|--------|---------|    
| Deployment Cost                                                 | Deployment Size |       |        |        |         |    
| 1376416                                                         | 6639            |       |        |        |         |    
| Function Name                                                   | min             | avg   | median | max    | # calls |    
| DENOMINATOR                                                     | 273             | 273   | 273    | 273    | 6       |    
| calculateAccruedVvvAmount((uint224,uint32,bool,uint8))(uint256) | 1439            | 3444  | 3444   | 5449   | 2       |    
| calculateAccruedVvvAmount()(uint256)                            | 4454            | 4454  | 4454   | 4454   | 1       |    
| calculateClaimableVvvAmount                                     | 2656            | 5567  | 6656   | 6666   | 11      |    
| claimVvv                                                        | 330             | 37286 | 54065  | 58575  | 10      |    
| durationToMultiplier                                            | 514             | 736   | 514    | 2514   | 9       |    
| durationToSeconds                                               | 558             | 2433  | 2558   | 2558   | 32      |    
| ethToVvvExchangeRate                                            | 204             | 204   | 204    | 204    | 6       |    
| receive                                                         | 823             | 823   | 823    | 823    | 1       |    
| restakeEth                                                      | 1632            | 65210 | 82484  | 92434  | 8       |    
| setDurationMultipliers                                          | 10349           | 18280 | 18280  | 26211  | 2       |    
| setNewStakesPermitted                                           | 2603            | 3050  | 2603   | 13903  | 46      |    
| setVvvToken                                                     | 8958            | 23808 | 24592  | 24592  | 47      |    
| stakeEth                                                        | 511             | 86268 | 96937  | 116837 | 43      |    
| stakeId                                                         | 396             | 1396  | 1396   | 2396   | 4       |    
| userStakeIds                                                    | 1156            | 2018  | 1626   | 3272   | 3       |    
| userStakes                                                      | 1008            | 1008  | 1008   | 1008   | 20      |    
| vvvToken                                                        | 404             | 404   | 404    | 404    | 1       |    
| withdrawEth                                                     | 8853            | 24795 | 24795  | 40737  | 2       |    
| withdrawStake                                                   | 1044            | 17209 | 11424  | 56324  | 22      | 

# Difference (Pre to 1)
`stakeEth` and `restakeEth` now save about 20k gas, other differences are negligible

| contracts/staking/VVVETHStaking.sol:VVVETHStaking contract      |           |       |       |       |         |
|-----------------------------------------------------------------|-----------|-------|-------|-------|---------|
| Deployment Cost                                                 | Deployment Size |       |        |        |         |
| 36441                                                           | 182       |       |       |       |         |
| Function Name                                                   | min       | avg   | median | max   | # calls |
| DENOMINATOR                                                     | 0         | 0     | 0     | 0     | 0       |
| calculateAccruedVvvAmount((uint224,uint32,bool,uint8))(uint256) | -307      | -307  | -307  | -307  | 0       |
| calculateAccruedVvvAmount()(uint256)                            | -606      | -606  | -606  | -606  | 0       |
| calculateClaimableVvvAmount                                     | -606      | -606  | -606  | -606  | 0       |
| claimVvv                                                        | 22        | -523  | -584  | -584  | 0       |
| durationToMultiplier                                            | 0         | 0     | 0     | 0     | 0       |
| durationToSeconds                                               | 22        | 22    | 22    | 22    | 0       |
| ethToVvvExchangeRate                                            | 0         | 0     | 0     | 0     | 0       |
| receive                                                         | 0         | 0     | 0     | 0     | 0       |
| restakeEth                                                      | -53       | -16832 | -22091 | -22091 | 0       |
| setDurationMultipliers                                          | 22        | 22    | 22    | 22    | 0       |
| setNewStakesPermitted                                           | -89       | -89   | -89   | -89   | 0       |
| setVvvToken                                                     | 22        | 22    | 22    | 22    | 0       |
| stakeEth                                                        | 22        | -20501 | -22040 | -22040 | 0       |
| stakeId                                                         | 22        | 22    | 22    | 22    | 0       |
| userStakeIds                                                    | 0         | 0     | 0     | 0     | 0       |
| userStakes                                                      | -47       | -47   | -47   | -47   | 0       |
| vvvToken                                                        | 0         | 0     | 0     | 0     | 0       |
| withdrawEth                                                     | 0         | 0     | 0     | 0     | 0       |
| withdrawStake                                                   | -37       | -1029 | -33   | -33   | 0       |
