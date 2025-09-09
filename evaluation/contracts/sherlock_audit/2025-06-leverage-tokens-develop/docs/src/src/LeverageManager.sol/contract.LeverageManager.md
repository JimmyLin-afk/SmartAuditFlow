# LeverageManager
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/LeverageManager.sol)

**Inherits:**
[ILeverageManager](/src/interfaces/ILeverageManager.sol/interface.ILeverageManager.md), AccessControlUpgradeable, ReentrancyGuardTransientUpgradeable, [FeeManager](/src/FeeManager.sol/abstract.FeeManager.md), UUPSUpgradeable

*The LeverageManager contract is an upgradeable core contract that is responsible for managing the creation of LeverageTokens.
It also acts as an entry point for users to mint and redeem LeverageTokens (shares), and for
rebalancers to rebalance LeverageTokens.
LeverageTokens are ERC20 tokens that are akin to shares in an ERC-4626 vault - they represent a claim on the equity held by
the LeverageToken. They can be created on this contract by calling `createNewLeverageToken`, and their configuration on the
LeverageManager is immutable.
Note: Although the LeverageToken configuration saved on the LeverageManager is immutable, the configured LendingAdapter and
RebalanceAdapter for the LeverageToken may be upgradeable contracts.
The LeverageManager also inherits the `FeeManager` contract, which is used to manage LeverageToken fees (which accrue to
the share value of the LeverageToken) and the treasury fees.
For mints of LeverageTokens (shares), the collateral and debt required is calculated by using the LeverageToken's
current collateral ratio. As such, the collateral ratio after a mint must be equal to the collateral ratio before a
mint, within some rounding error.
[CAUTION]
====
- LeverageTokens are susceptible to inflation attacks like ERC-4626 vaults:
"In empty (or nearly empty) ERC-4626 vaults, mints are at high risk of being stolen through frontrunning
with a "donation" to the vault that inflates the price of a share. This is variously known as a donation or inflation
attack and is essentially a problem of slippage. Vault deployers can protect against this attack by making an initial
mint of a non-trivial amount of the asset, such that price manipulation becomes infeasible. Redeems may
similarly be affected by slippage. Users can protect against this attack as well as unexpected slippage in general by
verifying the amount received is as expected, using a wrapper that performs these checks such as
https://github.com/fei-protocol/ERC4626#erc4626router-and-base[ERC4626Router]."
As such it is highly recommended that LeverageToken creators make an initial mint of a non-trivial amount of equity.
It is also recommended to use a router that performs slippage checks when minting and redeeming.
- LeverageToken creation is permissionless and can be configured with arbitrary lending adapters, rebalance adapters, and
underlying collateral and debt assets. As such, the adapters and tokens used by a LeverageToken are part of the risk
profile of the LeverageToken, and should be carefully considered by users before using a LeverageToken.
- LeverageTokens can be configured with arbitrary lending adapters, thus LeverageTokens are directly affected by the
specific mechanisms of the underlying lending market that their lending adapter integrates with. As mentioned above,
it is highly recommended that users research and understand the lending adapter used by the LeverageToken they are
considering using. Some examples:
- Morpho: Users should be aware that Morpho market creation is permissionless, and that the price oracle used by
by the market may be manipulatable.
- Aave v3: Allows rehypothecation of collateral, which may lead to reverts when trying to remove collateral from the
market during redeems and rebalances.*


## State Variables
### BASE_RATIO

```solidity
uint256 public constant BASE_RATIO = 1e18;
```


### UPGRADER_ROLE

```solidity
bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");
```


## Functions
### _getLeverageManagerStorage


```solidity
function _getLeverageManagerStorage() internal pure returns (LeverageManagerStorage storage $);
```

### initialize


```solidity
function initialize(address initialAdmin, address treasury, IBeaconProxyFactory leverageTokenFactory)
    external
    initializer;
```

### _authorizeUpgrade


```solidity
function _authorizeUpgrade(address newImplementation) internal override onlyRole(UPGRADER_ROLE);
```

### getLeverageTokenFactory

Returns the factory for creating new LeverageTokens


```solidity
function getLeverageTokenFactory() public view returns (IBeaconProxyFactory factory);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`factory`|`IBeaconProxyFactory`|Factory for creating new LeverageTokens|


### getLeverageTokenCollateralAsset

Returns the collateral asset for a LeverageToken


```solidity
function getLeverageTokenCollateralAsset(ILeverageToken token) public view returns (IERC20 collateralAsset);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to get collateral asset for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`collateralAsset`|`IERC20`|Collateral asset for the LeverageToken|


### getLeverageTokenDebtAsset

Returns the debt asset for a LeverageToken


```solidity
function getLeverageTokenDebtAsset(ILeverageToken token) public view returns (IERC20 debtAsset);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to get debt asset for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`debtAsset`|`IERC20`|Debt asset for the LeverageToken|


### getLeverageTokenRebalanceAdapter

Returns the rebalance adapter for a LeverageToken


```solidity
function getLeverageTokenRebalanceAdapter(ILeverageToken token) public view returns (IRebalanceAdapterBase module);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to get the rebalance adapter for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`module`|`IRebalanceAdapterBase`|adapter Rebalance adapter for the LeverageToken|


### getLeverageTokenConfig

Returns the entire configuration for a LeverageToken


```solidity
function getLeverageTokenConfig(ILeverageToken token) external view returns (LeverageTokenConfig memory config);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to get config for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`config`|`LeverageTokenConfig`|LeverageToken configuration|


### getLeverageTokenLendingAdapter

Returns the lending adapter for a LeverageToken


```solidity
function getLeverageTokenLendingAdapter(ILeverageToken token) public view returns (ILendingAdapter adapter);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to get lending adapter for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`adapter`|`ILendingAdapter`|Lending adapter for the LeverageToken|


### getLeverageTokenInitialCollateralRatio

Returns the initial collateral ratio for a LeverageToken

*Initial collateral ratio is followed when the LeverageToken has no shares and on mints when debt is 0.*


```solidity
function getLeverageTokenInitialCollateralRatio(ILeverageToken token) public view returns (uint256 ratio);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to get initial collateral ratio for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`ratio`|`uint256`|initialCollateralRatio Initial collateral ratio for the LeverageToken|


### getLeverageTokenState

Returns all data required to describe current LeverageToken state - collateral, debt, equity and collateral ratio


```solidity
function getLeverageTokenState(ILeverageToken token) public view returns (LeverageTokenState memory state);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to query state for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`state`|`LeverageTokenState`|LeverageToken state|


### createNewLeverageToken

Creates a new LeverageToken with the given config


```solidity
function createNewLeverageToken(LeverageTokenConfig calldata tokenConfig, string memory name, string memory symbol)
    external
    nonReentrant
    returns (ILeverageToken token);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`tokenConfig`|`LeverageTokenConfig`||
|`name`|`string`|Name of the LeverageToken|
|`symbol`|`string`|Symbol of the LeverageToken|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|Address of the new LeverageToken|


### previewMint

Previews mint function call and returns all required data

*Sender should approve leverage manager to spend collateralToAdd amount of collateral asset*


```solidity
function previewMint(ILeverageToken token, uint256 equityInCollateralAsset) public view returns (ActionData memory);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to preview mint for|
|`equityInCollateralAsset`|`uint256`|Equity to mint LeverageTokens (shares) for, denominated in the collateral asset|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ActionData`|previewData Preview data for mint - collateralToAdd Amount of collateral that sender needs to approve the LeverageManager to spend, this includes any fees - debtToBorrow Amount of debt that will be borrowed and sent to sender - equity Amount of equity that will be used for minting shares before fees, denominated in collateral asset - shares Amount of shares that will be minted to the sender - tokenFee Amount of shares that will be charged for the mint that are given to the LeverageToken - treasuryFee Amount of shares that will be charged for the mint that are given to the treasury|


### previewRedeem

Previews redeem function call and returns all required data

*Sender should approve leverage manager to spend debtToRepay amount of debt asset*


```solidity
function previewRedeem(ILeverageToken token, uint256 equityInCollateralAsset) public view returns (ActionData memory);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to preview redeem for|
|`equityInCollateralAsset`|`uint256`|Equity to receive by redeem denominated in collateral asset|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ActionData`|previewData Preview data for redeem - collateralToRemove Amount of collateral that will be removed from the LeverageToken and sent to the sender - debtToRepay Amount of debt that will be taken from sender and repaid to the LeverageToken - equity Amount of equity that will be received for the redeem before fees, denominated in collateral asset - shares Amount of shares that will be burned from sender - tokenFee Amount of shares that will be charged for the redeem that are given to the LeverageToken - treasuryFee Amount of shares that will be charged for the redeem that are given to the treasury|


### mint

Adds equity to a LeverageToken and mints shares of it to the sender


```solidity
function mint(ILeverageToken token, uint256 equityInCollateralAsset, uint256 minShares)
    external
    nonReentrant
    returns (ActionData memory actionData);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|The LeverageToken to mint shares of|
|`equityInCollateralAsset`|`uint256`|The amount of equity to mint shares for, denominated in the collateral asset of the LeverageToken|
|`minShares`|`uint256`|The minimum amount of shares to mint|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`actionData`|`ActionData`|Data about the mint - collateral Amount of collateral that was added, including any fees - debt Amount of debt that was added - equity Amount of equity that was added before fees, denominated in collateral asset - shares Amount of shares minted to the sender - tokenFee Amount of shares that was charged for the mint that are given to the LeverageToken - treasuryFee Amount of shares that was charged for the mint that are given to the treasury|


### redeem

Redeems equity from a LeverageToken and burns shares from sender


```solidity
function redeem(ILeverageToken token, uint256 equityInCollateralAsset, uint256 maxShares)
    external
    nonReentrant
    returns (ActionData memory actionData);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|The LeverageToken to redeem from|
|`equityInCollateralAsset`|`uint256`|The amount of equity to receive by redeeming denominated in the collateral asset of the LeverageToken|
|`maxShares`|`uint256`|The maximum amount of shares to burn|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`actionData`|`ActionData`|Data about the redeem - collateral Amount of collateral that was removed from LeverageToken and sent to sender - debt Amount of debt that was repaid to LeverageToken, taken from sender - equity Amount of equity that was received for redeem before fees, denominated in collateral asset - shares Amount of the sender's shares that were burned for the redeem - tokenFee Amount of shares that was charged for the redeem that are given to the LeverageToken - treasuryFee Amount of shares that was charged for the redeem that are given to the treasury|


### rebalance

Rebalances a LeverageToken based on provided actions

*Anyone can call this function. At the end function will just check if the affected LeverageToken is in a
better state than before rebalance. Caller needs to calculate and to provide tokens for rebalancing and he needs
to specify tokens that he wants to receive*


```solidity
function rebalance(
    ILeverageToken leverageToken,
    RebalanceAction[] calldata actions,
    IERC20 tokenIn,
    IERC20 tokenOut,
    uint256 amountIn,
    uint256 amountOut
) external nonReentrant;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`leverageToken`|`ILeverageToken`|LeverageToken to rebalance|
|`actions`|`RebalanceAction[]`|Rebalance actions to execute (add collateral, remove collateral, borrow or repay)|
|`tokenIn`|`IERC20`|Token to transfer in. Transfer from caller to the LeverageManager contract|
|`tokenOut`|`IERC20`|Token to transfer out. Transfer from the LeverageManager contract to caller|
|`amountIn`|`uint256`|Amount of tokenIn to transfer in|
|`amountOut`|`uint256`|Amount of tokenOut to transfer out|


### _convertToShares

Function that converts user's equity to shares

Function uses OZ formula for calculating shares

*Function should be used to calculate how much shares user should receive for their equity*


```solidity
function _convertToShares(ILeverageToken token, uint256 equityInCollateralAsset, ExternalAction action)
    internal
    view
    returns (uint256 shares);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to convert equity for|
|`equityInCollateralAsset`|`uint256`|Equity to convert to shares, denominated in collateral asset|
|`action`|`ExternalAction`|Action to convert equity for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`shares`|`uint256`|Shares|


### _previewAction

Previews parameters related to a mint action

*If the LeverageToken has zero total supply of shares (so the LeverageToken does not hold any collateral or debt,
or holds some leftover dust after all shares are redeemed), then the preview will use the target
collateral ratio for determining how much collateral and debt is required instead of the current collateral ratio.*

*If action is mint collateral will be rounded down and debt up, if action is redeem collateral will be rounded up and debt down*


```solidity
function _previewAction(ILeverageToken token, uint256 equityInCollateralAsset, ExternalAction action)
    internal
    view
    returns (ActionData memory data);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to preview mint for|
|`equityInCollateralAsset`|`uint256`|Amount of equity to give or receive, denominated in collateral asset|
|`action`|`ExternalAction`|Type of the action to preview, can be Mint or Redeem|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`data`|`ActionData`|Preview data for the action|


### _computeCollateralAndDebtForAction

Function that computes collateral and debt required by the position held by a LeverageToken for a given action and an amount of equity to add / remove


```solidity
function _computeCollateralAndDebtForAction(
    ILeverageToken token,
    uint256 equityInCollateralAsset,
    ExternalAction action
) internal view returns (uint256 collateral, uint256 debt);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to compute collateral and debt for|
|`equityInCollateralAsset`|`uint256`|Equity amount in collateral asset|
|`action`|`ExternalAction`|Action to compute collateral and debt for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`collateral`|`uint256`|Collateral to add / remove from the LeverageToken|
|`debt`|`uint256`|Debt to borrow / repay to the LeverageToken|


### _executeLendingAdapterAction

Executes actions on the LendingAdapter for a specific LeverageToken


```solidity
function _executeLendingAdapterAction(ILeverageToken token, ActionType actionType, uint256 amount) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to execute action for|
|`actionType`|`ActionType`|Type of the action to execute|
|`amount`|`uint256`|Amount to execute action with|


### _transferTokens

Helper function for transferring tokens, or no-op if token is 0 address

*If from address is this smart contract it will use the regular transfer function otherwise it will use transferFrom*


```solidity
function _transferTokens(IERC20 token, address from, address to, uint256 amount) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`IERC20`|Token to transfer|
|`from`|`address`|Address to transfer tokens from|
|`to`|`address`|Address to transfer tokens to|
|`amount`|`uint256`||


## Structs
### LeverageManagerStorage
*Struct containing all state for the LeverageManager contract*

**Note:**
storage-location: erc7201:seamless.contracts.storage.LeverageManager


```solidity
struct LeverageManagerStorage {
    IBeaconProxyFactory tokenFactory;
    mapping(ILeverageToken token => BaseLeverageTokenConfig) config;
}
```

