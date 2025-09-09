# Pre-Optimizations 
0ffc3e524c65a1d86db3aa35af8840a4db026a3a
Outputs of: `forge test --match-contract VVVVesting --gas-report`

| contracts/vesting/VVVVesting.sol:VVVVesting contract |                 |        |        |         |         |
|------------------------------------------------------|-----------------|--------|--------|---------|---------|
| Deployment Cost                                      | Deployment Size |        |        |         |         |
| 1009680                                              | 5274            |        |        |         |         |
| Function Name                                        | min             | avg    | median | max     | # calls |
| VVVToken                                             | 349             | 349    | 349    | 349     | 1       |
| batchSetVestingSchedule                              | 8966            | 544967 | 286825 | 1597254 | 4       |
| calculateVestedAmountAtInterval                      | 553             | 1866   | 2178   | 3173    | 7       |
| getVestedAmount                                      | 1454            | 2383   | 2444   | 2540    | 10      |
| removeVestingSchedule                                | 5592            | 5592   | 5592   | 5592    | 2       |
| setVestedToken                                       | 8905            | 12016  | 12025  | 15110   | 4       |
| setVestingSchedule                                   | 8626            | 154976 | 173059 | 173059  | 18      |
| userVestingSchedules                                 | 1732            | 1732   | 1732   | 1732    | 11      |
| withdrawVestedTokens                                 | 1516            | 30233  | 3137   | 65844   | 9       |

# Post-Optimizations 1
4df51e47bb0d615589b4a0c3649ac7138fdba3a2
Changes Made: VestingSchedule data packing, without touching token amounts which can be affected by precision loss

| contracts/vesting/VVVVesting.sol:VVVVesting contract |                 |        |        |        |         |
|------------------------------------------------------|-----------------|--------|--------|--------|---------|
| Deployment Cost                                      | Deployment Size |        |        |        |         |
| 1091160                                              | 5693            |        |        |        |         |
| Function Name                                        | min             | avg    | median | max    | # calls |
| VVVToken                                             | 349             | 349    | 349    | 349    | 1       |
| batchSetVestingSchedule                              | 9032            | 290187 | 150973 | 849771 | 4       |
| calculateVestedAmountAtInterval                      | 553             | 1866   | 2178   | 3173   | 7       |
| getVestedAmount                                      | 1495            | 2417   | 2480   | 2599   | 10      |
| removeVestingSchedule                                | 5328            | 5328   | 5328   | 5328   | 2       |
| setVestedToken                                       | 8883            | 11994  | 12003  | 15088  | 4       |
| setVestingSchedule                                   | 8456            | 94193  | 104696 | 104696 | 18      |
| userVestingSchedules                                 | 1392            | 1392   | 1392   | 1392   | 11      |
| withdrawVestedTokens                                 | 1534            | 30164  | 3196   | 65658  | 9       |
 
# Difference (Pre to 1)

| contracts/vesting/VVVVesting.sol:VVVVesting contract |                 |        |        |        |         |
|-----------------------------------------------------|-----------------|--------|--------|--------|---------|
| Deployment Cost                                     | Deployment Size |        |        |        |         |
| 81480                                               | 419             |        |        |        |         |
| Function Name                                       | min             | avg    | median | max    | # calls |
| VVVToken                                            | 0               | 0      | 0      | 0      | 0       |
| batchSetVestingSchedule                             | 66              | -254780 | -135852 | -747483 | 0    |
| calculateVestedAmountAtInterval                     | 0               | 0      | 0      | 0      | 0       |
| getVestedAmount                                     | 41              | 34     | 36     | 59     | 0       |
| removeVestingSchedule                               | -264            | -264   | -264   | -264   | 0       |
| setVestedToken                                      | -22             | -22    | -22    | -22    | 0       |
| setVestingSchedule                                  | -170            | -60783 | -68363 | -68363 | 0       |
| userVestingSchedules                                | -340            | -340   | -340   | -340   | 0       |
| withdrawVestedTokens                                | 18              | -69    | 59     | -186   | 0       |

# Post-Optimizations 2
865d0f130b394ca27a49d174cb6d1bed7967409e
Changes Made: reduced precision of token amounts in VestingSchedule so entire struct fits in two words, reordered struct to optimal order for packing

One interesting note here is that the original ordering of the fields in VestingSchedule produced the most savings. The conventional wisdom is that ordering the struct fields such that consecutive fields add up to 32 bytes at a time will save the most gas, but this produced relatively little savings compared to leaving the token amounts as uint256. *Later found to be incorrect - see below, more savings were achieved*

| contracts/vesting/VVVVesting.sol:VVVVesting contract |                 |        |        |        |         |
|------------------------------------------------------|-----------------|--------|--------|--------|---------|
| Deployment Cost                                      | Deployment Size |        |        |        |         |
| 1174047                                              | 6107            |        |        |        |         |
| Function Name                                        | min             | avg    | median | max    | # calls |
| VVVToken                                             | 349             | 349    | 349    | 349    | 1       |
| batchSetVestingSchedule                              | 9010            | 228795 | 147407 | 611357 | 4       |
| calculateVestedAmountAtInterval                      | 553             | 1759   | 1992   | 2987   | 7       |
| getVestedAmount                                      | 1610            | 2416   | 2466   | 2585   | 10      |
| removeVestingSchedule                                | 5406            | 5406   | 5406   | 5406   | 2       |
| setVestedToken                                       | 8883            | 11994  | 12003  | 15088  | 4       |
| setVestingSchedule                                   | 8844            | 74823  | 82875  | 82875  | 18      |
| userVestingSchedules                                 | 1435            | 1435   | 1435   | 1435   | 11      |
| withdrawVestedTokens                                 | 1638            | 21337  | 3212   | 45759  | 9       |

# Difference (Pre to 2)

| contracts/vesting/VVVVesting.sol:VVVVesting contract |                 |        |        |        |         |
|-----------------------------------------------------|-----------------|--------|--------|--------|---------|
| Deployment Cost                                     | Deployment Size |        |        |        |         |
| 164367                                              | 833             |        |        |        |         |
| Function Name                                       | min             | avg    | median | max    | # calls |
| VVVToken                                            | 0               | 0      | 0      | 0      | 0       |
| batchSetVestingSchedule                             | 44              | -316172 | -139418 | -985897 | 0       |
| calculateVestedAmountAtInterval                     | 0               | -107   | -186   | -186   | 0       |
| getVestedAmount                                     | 156             | 33     | 22     | 45     | 0       |
| removeVestingSchedule                               | -186            | -186   | -186   | -186   | 0       |
| setVestedToken                                      | -22             | -22    | -22    | -22    | 0       |
| setVestingSchedule                                  | 218             | -80153 | -90184 | -90184 | 0       |
| userVestingSchedules                                | -297            | -297   | -297   | -297   | 0       |
| withdrawVestedTokens                                | 122             | -8896  | 75     | -20085 | 0       |

# Difference (1 to 2)
Additional gas savings compared to token amounts left as all uint256 in VestingSchedule

| contracts/vesting/VVVVesting.sol:VVVVesting contract |                 |        |        |        |         |
|-----------------------------------------------------|-----------------|--------|--------|--------|---------|
| Deployment Cost                                     | Deployment Size |        |        |        |         |
| 82887                                               | 414             |        |        |        |         |
| Function Name                                       | min             | avg    | median | max    | # calls |
| VVVToken                                            | 0               | 0      | 0      | 0      | 0       |
| batchSetVestingSchedule                             | -22             | -61392 | -3566  | -238414 | 0       |
| calculateVestedAmountAtInterval                     | 0               | -107   | -186   | -186   | 0       |
| getVestedAmount                                     | 115             | -1     | -14    | -14    | 0       |
| removeVestingSchedule                               | 78              | 78     | 78     | 78     | 0       |
| setVestedToken                                      | 0               | 0      | 0      | 0      | 0       |
| setVestingSchedule                                  | 388             | -19370 | -21821 | -21821 | 0       |
| userVestingSchedules                                | 43              | 43     | 43     | 43     | 0       |
| withdrawVestedTokens                                | 104             | -8827  | 16     | -19899 | 0       |

# Post-Optimizations 3
dc0946c396f3d46a2a13e5767bf8bd60913464bf
Changes Made: Corrected storage packing into two words (previous attempt was incorrect and therefore didn't save gas as expected)

| contracts/vesting/VVVVesting.sol:VVVVesting contract |                 |        |        |        |         |
|------------------------------------------------------|-----------------|--------|--------|--------|---------|
| Deployment Cost                                      | Deployment Size |        |        |        |         |
| 1163033                                              | 6052            |        |        |        |         |
| Function Name                                        | min             | avg    | median | max    | # calls |
| VVVToken                                             | 349             | 349    | 349    | 349    | 1       |
| batchSetVestingSchedule                              | 9010            | 220552 | 143011 | 587179 | 4       |
| calculateVestedAmountAtInterval                      | 553             | 1759   | 1992   | 2987   | 7       |
| getVestedAmount                                      | 1610            | 2533   | 2586   | 2699   | 10      |
| removeVestingSchedule                                | 5040            | 5040   | 5040   | 5040   | 2       |
| setVestedToken                                       | 8883            | 11994  | 12003  | 15088  | 4       |
| setVestingSchedule                                   | 8664            | 72859  | 80677  | 80677  | 18      |
| userVestingSchedules                                 | 1337            | 1337   | 1337   | 1337   | 11      |
| withdrawVestedTokens                                 | 1638            | 21387  | 3326   | 45781  | 9       |

# Difference (Pre to 3)

| contracts/vesting/VVVVesting.sol:VVVVesting contract |                 |        |        |         |         |
|------------------------------------------------------|-----------------|--------|--------|---------|---------|
| Deployment Cost                                     | Deployment Size |        |        |         |         |
| 153353                                              | 778             |        |        |         |         |
| Function Name                                       | min             | avg    | median | max     | # calls |
| VVVToken                                            | 0               | 0      | 0      | 0       | 0       |
| batchSetVestingSchedule                             | 44              | -324415| -143814| -1010075| 0       |
| calculateVestedAmountAtInterval                     | 0               | -107   | -186   | -186    | 0       |
| getVestedAmount                                     | 156             | 150    | 142    | 159     | 0       |
| removeVestingSchedule                               | -552            | -552   | -552   | -552    | 0       |
| setVestedToken                                      | -22             | -22    | -22    | -22     | 0       |
| setVestingSchedule                                  | 38              | -82117 | -92382 | -92382  | 0       |
| userVestingSchedules                                | -395            | -395   | -395   | -395    | 0       |
| withdrawVestedTokens                                | 122             | -8846  | 189    | -20063  | 0       |

# Difference (2 to 3)

| contracts/vesting/VVVVesting.sol:VVVVesting contract |                 |       |       |       |         |
|------------------------------------------------------|-----------------|-------|-------|-------|---------|
| Deployment Cost                                     | Deployment Size |       |       |       |         |
| -11014                                              | -55             |       |       |       |         |
| Function Name                                       | min             | avg   | median| max   | # calls |
| VVVToken                                            | 0               | 0     | 0     | 0     | 0       |
| batchSetVestingSchedule                             | 0               | -8243 | -4396 | -24178| 0       |
| calculateVestedAmountAtInterval                     | 0               | 0     | 0     | 0     | 0       |
| getVestedAmount                                     | 0               | 117   | 120   | 114   | 0       |
| removeVestingSchedule                               | -366            | -366  | -366  | -366  | 0       |
| setVestedToken                                      | 0               | 0     | 0     | 0     | 0       |
| setVestingSchedule                                  | -180            | -1964 | -2198 | -2198 | 0       |
| userVestingSchedules                                | -98             | -98   | -98   | -98   | 0       |
| withdrawVestedTokens                                | 0               | 50    | 114   | 22    | 0       |