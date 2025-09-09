import { time } from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import type { Address } from 'abitype'
import hre from "hardhat";
import { expect } from "chai";
import { parseEther, zeroAddress, getAddress, maxUint256, encodeFunctionData, parseUnits, EncodeFunctionDataReturnType, Account, toFunctionSelector, GetContractReturnType } from "viem";
import { ClosePositionRequest, ClosePositionOrder, OrderType, FunctionCallData, OpenPositionRequest, Position, Vault, WithSignature, getEventPosition, getFee, getEmptyPosition } from "./utils/PerpStructUtils";
import { Signer, signClosePositionRequest, signClosePositionOrder, signOpenPositionRequest } from "./utils/SigningUtils";
import { getApproveAndSwapExactlyOutFunctionCallData, getApproveAndSwapFunctionCallData, getRouterSwapExactlyOutFunctionCallData, getRouterSwapFunctionCallData, getSwapExactlyOutFunctionCallData, getSwapFunctionCallData, getSweepTokenWithFeeCallData, getUnwrapWETH9WithFeeCallData } from "./utils/SwapUtils";
import { WETHAbi } from "./utils/WETHAbi";
import { LIQUIDATOR_ROLE, ORDER_SIGNER_ROLE, ORDER_EXECUTOR_ROLE, VAULT_ADMIN_ROLE } from "./utils/constants";
import { MockSwapRouterAbi } from "./utils/MockSwapRouterAbi";

const tradeFeeValue = 50n; // 0.5%
const feeDenominator = 10000n;

export type CreateClosePositionRequestParams = {
    position: Position,
    interest?: bigint,
    expiration?: number,
    amount?: bigint,
    referrer?: Address
}

export type CreateClosePositionOrderParams = {
    orderType: OrderType,
    traderSigner: Signer,
    positionId: bigint,
    makerAmount: bigint,
    takerAmount: bigint,
    createdAt?: number,
    expiration?: number
    executionFee?: bigint
}

export type CreateExactInSwapDataParams = {
    amount: bigint,
    tokenIn: Address,
    tokenOut: Address,
    swapRecipient: Address,
    swapFee?: bigint,
    unwrapEth?: boolean
}

export type CreateExactOutSwapDataParams = {
    amountInMax: bigint,
    amountOut: bigint,
    tokenIn: Address,
    tokenOut: Address,
    swapRecipient: Address,
    swapFee?: bigint,
    unwrapEth?: boolean
}

export async function deployPerpManager() {
    // Contracts are deployed using the first signer/account by default
    const [owner, user1, user2, liquidator, orderSigner, orderExecutor, vaultAdmin] = await hre.viem.getWalletClients();

    const contractName = "PerpManager";
    const PerpManager = await hre.ethers.getContractFactory(contractName);
    const address = 
        await hre.upgrades.deployProxy(
            PerpManager,
            [],
            { kind: 'uups'}
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const manager = await hre.viem.getContractAt(contractName, address);
    await manager.write.grantRole([LIQUIDATOR_ROLE, liquidator.account.address, 0]);
    await manager.write.grantRole([ORDER_SIGNER_ROLE, orderSigner.account.address, 0]);
    await manager.write.grantRole([ORDER_EXECUTOR_ROLE, orderExecutor.account.address, 0]);
    await manager.write.grantRole([VAULT_ADMIN_ROLE, vaultAdmin.account.address, 0]);
    return { manager, liquidator, orderSigner, user1, user2, owner, orderExecutor, vaultAdmin };
}

export async function deployWeth() {
    const weth = await hre.viem.deployContract("WETH9");
    await weth.write.deposit([], { value: parseEther("10") });
    return { weth, wethAddress: weth.address };
}

export async function deployVault(longPoolAddress: Address, shortPoolAddress: Address, addressProvider: Address, perpManager: Address, tokenAddress: Address, name: string, symbol: string) {
    const contractName = "WasabiVault";
    const WasabiVault = await hre.ethers.getContractFactory(contractName);
    const address = 
        await hre.upgrades.deployProxy(
            WasabiVault,
            [longPoolAddress, shortPoolAddress, addressProvider, perpManager, tokenAddress, name, symbol],
            { kind: 'uups'}
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const vault = await hre.viem.getContractAt(contractName, address);
    return { vault }
}

export async function deployMockV2VaultImpl() {
    const newVaultImpl = await hre.viem.deployContract("MockVaultV2");    
    return { newVaultImpl };
}

export async function deployAaveStrategy(vaultAddress: Address, perpManager: Address) {
    const mockAToken = await hre.viem.deployContract("MockAToken", ["MockAToken", "MAT"]);
    const mockAavePool = await hre.viem.deployContract("MockAavePool", [mockAToken.address]);
    const AaveStrategy = await hre.ethers.getContractFactory("AaveStrategy");
    const address = 
        await hre.upgrades.deployProxy(
            AaveStrategy,
            [vaultAddress, mockAavePool.address, perpManager],
            { kind: 'uups'}
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const strategy = await hre.viem.getContractAt("AaveStrategy", address);
    return { strategy, mockAToken, mockAavePool }
}

export async function deployDebtController() {
    const maxApy = 300n; // 300% APY
    const maxLeverage = 500n; // 5x Leverage

    // Contracts are deployed using the first signer/account by default
    const [owner, otherAccount] = await hre.viem.getWalletClients();
    const debtController = await hre.viem.deployContract("DebtController", [maxApy, maxLeverage]);
    const publicClient = await hre.viem.getPublicClient();

    return {
        debtController,
        maxApy,
        maxLeverage,
        owner,
        otherAccount,
        publicClient,
    };
}

export async function deployLongPoolMockEnvironment() {
    const wasabiLongPoolFixture = await deployWasabiLongPool();
    const {tradeFeeValue, contractName, wasabiLongPool, addressProvider, manager, user1, user2, partner, publicClient, feeDenominator, debtController, wethAddress, weth, orderSigner, vault, vaultAdmin, strategy, mockAToken, mockAavePool} = wasabiLongPoolFixture;
    const [owner] = await hre.viem.getWalletClients();

    const initialPrice = 10_000n;
    const priceDenominator = 10_000n;
    const leverage = 4n;

    const mockSwap = await hre.viem.deployContract("MockSwap", []);
    await weth.write.deposit([], { value: parseEther("500") });
    await weth.write.transfer([mockSwap.address, parseEther("500")]);

    const uPPG = await hre.viem.deployContract("MockERC20", ["μPudgyPenguins", 'μPPG']);
    await uPPG.write.mint([mockSwap.address, parseEther("500")]);
    await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice]);

    const usdc = await hre.viem.deployContract("USDC", []);
    await usdc.write.mint([mockSwap.address, parseUnits("10000", 6)]);
    await mockSwap.write.setPrice([usdc.address, wethAddress, 4n]);
    await mockSwap.write.setPrice([usdc.address, uPPG.address, 4n]);
    await wasabiLongPool.write.addQuoteToken([usdc.address]);

    const usdcVaultFixture = await deployVault(
        wasabiLongPool.address, zeroAddress, addressProvider.address, manager.address, usdc.address, "USDC Vault", "wUSDC");
    const usdcVault = usdcVaultFixture.vault;
    await wasabiLongPool.write.addVault([usdcVault.address], { account: vaultAdmin.account });

    const totalAmountIn = parseEther("1");
    const fee = getFee(totalAmountIn * leverage, tradeFeeValue);
    const downPayment = totalAmountIn - fee;
    const principal = downPayment * (leverage - 1n);
    const totalSize = principal + downPayment;

    const functionCallDataList: FunctionCallData[] =
        getApproveAndSwapFunctionCallData(mockSwap.address, wethAddress, uPPG.address, totalSize);

    await weth.write.deposit([], { value: parseEther("50"), account: user1.account });
    await weth.write.approve([wasabiLongPool.address, maxUint256], {account: user1.account});
    await weth.write.deposit([], { value: parseEther("50"), account: user2.account });
    await weth.write.approve([wasabiLongPool.address, maxUint256], {account: user2.account});

    const openPositionRequest: OpenPositionRequest = {
        id: 1n,
        currency: wethAddress,
        targetCurrency: uPPG.address,
        downPayment,
        principal,
        minTargetAmount: totalSize * initialPrice / priceDenominator,
        expiration: BigInt(await time.latest()) + 86400n,
        fee,
        functionCallDataList,
        existingPosition: getEmptyPosition(),
        referrer: zeroAddress
    };
    const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiLongPool.address, openPositionRequest);

    const getTradeAmounts = async (leverage: bigint, totalAmountIn: bigint, currency: Address) => {
        const fee = getFee(totalAmountIn * leverage, tradeFeeValue);
        const downPayment = totalAmountIn - fee;
        const principal = downPayment * (leverage - 1n);
        const totalSize = principal + downPayment;
        const minTargetAmount = totalSize * (currency == usdc.address ? 4n : initialPrice) / priceDenominator;
        return { fee, downPayment, principal, totalSize, minTargetAmount };
    }

    const getOpenPositionRequest = async ({
        id = 1n,
        currency = wethAddress,
        targetCurrency = uPPG.address,
        downPayment = openPositionRequest.downPayment,
        principal = openPositionRequest.principal,
        minTargetAmount = openPositionRequest.minTargetAmount,
        expiration = openPositionRequest.expiration,
        fee = openPositionRequest.fee,
        existingPosition = openPositionRequest.existingPosition,
        referrer = openPositionRequest.referrer
    }: {
        id?: bigint,
        currency?: Address,
        targetCurrency?: Address,
        downPayment?: bigint,
        principal?: bigint,
        minTargetAmount?: bigint,
        expiration?: bigint,
        fee?: bigint,
        existingPosition?: Position,
        referrer?: Address
    } = {}): Promise<OpenPositionRequest> => {
        const totalSize = principal + downPayment;
        const functionCallDataList = getApproveAndSwapFunctionCallData(mockSwap.address, currency, targetCurrency, totalSize);
        return {
            id,
            currency,
            targetCurrency,
            downPayment,
            principal,
            minTargetAmount,
            expiration,
            fee,
            functionCallDataList,
            existingPosition,
            referrer
        } as OpenPositionRequest;
    }

    const sendOpenPositionRequest = async (request: OpenPositionRequest) => {
        const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiLongPool.address, request);
        const hash = await wasabiLongPool.write.openPosition([request, signature], { account: user1.account });
        const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
        const event = (await wasabiLongPool.getEvents.PositionOpened())[0];
        const position: Position = await getEventPosition(event);

        return {
            position,
            hash,
            gasUsed,
            event
        }
    }

    const sendDefaultOpenPositionRequest = async (id?: bigint | undefined) => {
        const request = id ? {...openPositionRequest, id} : openPositionRequest;
        return sendOpenPositionRequest(request);
    }

    const sendReferredOpenPositionRequest = async (id?: bigint | undefined) => {
        const request = id ? {...openPositionRequest, id, referrer: partner.account.address} : {...openPositionRequest, referrer: partner.account.address};
        return sendOpenPositionRequest(request);
    }

    const createClosePositionRequest = async (params: CreateClosePositionRequestParams): Promise<ClosePositionRequest> => {
        let { position, interest, expiration, amount, referrer } = params;
        amount = amount || 0n;
        const functionCallDataList = getApproveAndSwapFunctionCallData(
            mockSwap.address,
            position.collateralCurrency, 
            position.currency, 
            amount == 0n ? position.collateralAmount : amount
        )
        const request: ClosePositionRequest = {
            expiration: expiration ? BigInt(expiration) : (BigInt(await time.latest()) + 300n),
            interest: interest || 0n,
            amount: amount || 0n,
            position,
            functionCallDataList,
            referrer: referrer || zeroAddress
        };
        return request;
    }

    const createSignedClosePositionRequest = async (params: CreateClosePositionRequestParams): Promise<WithSignature<ClosePositionRequest>> => {
        const request = await createClosePositionRequest(params);
        const signature = await signClosePositionRequest(orderSigner, contractName, wasabiLongPool.address, request);
        return { request, signature }
    }

    const createClosePositionOrder = async (params: CreateClosePositionOrderParams): Promise<ClosePositionOrder> => {
        const { orderType, positionId, makerAmount, takerAmount, createdAt, expiration, executionFee } = params;
        const order: ClosePositionOrder = {
            orderType,
            positionId,
            makerAmount,
            takerAmount,
            createdAt: createdAt ? BigInt(createdAt) : BigInt(await time.latest()),
            expiration: expiration ? BigInt(expiration) : (BigInt(await time.latest()) + 300n),
            executionFee: executionFee || 0n
        }
        return order;
    }

    const createSignedClosePositionOrder = async (params: CreateClosePositionOrderParams): Promise<WithSignature<ClosePositionOrder>> => {
        const {traderSigner} = params;
        const order = await createClosePositionOrder(params);
        const signature = await signClosePositionOrder(traderSigner, contractName, wasabiLongPool.address, order);
        return { request: order, signature }
    }

    const computeMaxInterest = async (position: Position): Promise<bigint> => {
        return await debtController.read.computeMaxInterest([position.collateralCurrency, position.principal, position.lastFundingTimestamp], { blockTag: 'pending' });
    }

    const computeLiquidationPrice = async (position: Position): Promise<bigint> => {
        const threshold = 500n; // 5 percent

        const currentInterest = await computeMaxInterest(position);
        const payoutLiquidationThreshold = position.principal * (threshold + tradeFeeValue) / (feeDenominator - tradeFeeValue);

        const liquidationAmount = payoutLiquidationThreshold + position.principal + currentInterest;
        return liquidationAmount * priceDenominator / position.collateralAmount;
    }

    const strategyDeposit = async (depositAmount: bigint) => {
        const ownerShares = await vault.read.balanceOf([owner.account.address]);
        const totalAssetsBefore = await vault.read.totalAssets();
        
        await vault.write.strategyDeposit([strategy.address, depositAmount], {account: owner.account});

        const totalAssetsAfterDeposit = await vault.read.totalAssets();
        const strategyDebtAfterDeposit = await vault.read.strategyDebt([strategy.address]);

        expect(await vault.read.balanceOf([owner.account.address])).to.equal(ownerShares, "Owner shares should be unchanged");
        expect(totalAssetsAfterDeposit).to.equal(totalAssetsBefore, "Total asset value should be unchanged");
        expect(strategyDebtAfterDeposit).to.equal(depositAmount, "Strategy debt should be recorded for strategy");

        const strategyDepositEvents = await vault.getEvents.StrategyDeposit();
        expect(strategyDepositEvents).to.have.lengthOf(1, "StrategyDeposit event not emitted");
        const strategyDepositEvent = strategyDepositEvents[0].args;
        expect(strategyDepositEvent.strategy).to.equal(getAddress(strategy.address));
        expect(strategyDepositEvent.collateral).to.equal(getAddress(mockAToken.address));
        expect(strategyDepositEvent.amountDeposited).to.equal(depositAmount);
        expect(strategyDepositEvent.collateralReceived).to.equal(depositAmount);
    }

    const strategyAddInterest = async (interest: bigint) => {
        await weth.write.deposit({ value: interest, account: owner.account });
        await weth.write.transfer([mockAavePool.address, interest], {account: owner.account});
        await mockAToken.write.mint([strategy.address, interest]);
    }

    const strategyClaim = async (interest: bigint) => {
        const strategyDebtBefore = await vault.read.strategyDebt([strategy.address]);
        const totalAssetsBefore = await vault.read.totalAssets();

        await vault.write.strategyClaim([strategy.address], { account: owner.account });

        const totalAssetsAfterClaim = await vault.read.totalAssets();
        const strategyDebtAfterClaim = await vault.read.strategyDebt([strategy.address]);

        expect(totalAssetsAfterClaim).to.equal(totalAssetsBefore + interest, "Total asset value should increase by interest");
        expect(strategyDebtAfterClaim).to.equal(strategyDebtBefore + interest, "Strategy debt should increase by interest");
        const strategyClaimEvents = await vault.getEvents.StrategyClaim();
        expect(strategyClaimEvents).to.have.lengthOf(1, "StrategyClaim event not emitted");
        const strategyClaimEvent = strategyClaimEvents[0].args;
        expect(strategyClaimEvent.strategy).to.equal(getAddress(strategy.address));
        expect(strategyClaimEvent.amount).to.equal(interest);
    }

    const strategyWithdraw = async (withdrawAmount: bigint) => {
        await vault.write.strategyWithdraw([strategy.address, withdrawAmount], {account: owner.account});

        const strategyWithdrawEvents = await vault.getEvents.StrategyWithdraw();
        expect(strategyWithdrawEvents).to.have.lengthOf(1, "AdminDebtRepaid event not emitted");
        const strategyWithdrawEvent = strategyWithdrawEvents[0].args;
        expect(strategyWithdrawEvent.strategy).to.equal(getAddress(strategy.address));
        expect(strategyWithdrawEvent.amountWithdraw).to.equal(withdrawAmount);
    }

    const upgradeVaultToTimelock = async (cooldownDuration: bigint = 864000n) => {
        const TimelockWasabiVault = await hre.ethers.getContractFactory("TimelockWasabiVault");
        await hre.upgrades.upgradeProxy(
            vault.address, 
            TimelockWasabiVault, 
            {
                kind: 'uups',
                call: {
                    fn: "setCooldownDuration",
                    args: [cooldownDuration]
                }
            }
        );
    }

    return {
        ...wasabiLongPoolFixture,
        mockSwap,
        uPPG,
        usdc,
        openPositionRequest,
        downPayment,
        totalAmountIn,
        totalSize,
        leverage,
        signature,
        initialPrice,
        priceDenominator,
        usdcVault,
        getOpenPositionRequest,
        getTradeAmounts,
        sendOpenPositionRequest,
        sendDefaultOpenPositionRequest,
        sendReferredOpenPositionRequest,
        createClosePositionRequest,
        createSignedClosePositionRequest,
        createClosePositionOrder,
        createSignedClosePositionOrder,
        computeLiquidationPrice,
        computeMaxInterest,
        strategyDeposit,
        strategyClaim,
        strategyWithdraw,
        strategyAddInterest,
        upgradeVaultToTimelock
    }
}

export async function deployAddressProvider() {
    const wethFixture = await deployWeth();
    const debtControllerFixture = await deployDebtController();
    const [owner, user1, user2, user3, user4] = await hre.viem.getWalletClients();
    const addressProvider = 
        await hre.viem.deployContract(
            "AddressProvider",
            [debtControllerFixture.debtController.address, zeroAddress, owner.account.address, wethFixture.wethAddress, user4.account.address, zeroAddress, zeroAddress]);
    return {
        ...wethFixture,
        ...debtControllerFixture,
        addressProvider,
        owner,
        user1,
        tradeFeeValue,
        feeDenominator,
        feeReceiver: owner.account.address,
        liquidationFeeReceiver: user4.account.address
    };
}

export async function deployAddressProvider2() {
    const debtControllerFixture = await deployDebtController();
    const [owner, user1] = await hre.viem.getWalletClients();
    const addressProvider = 
        await hre.viem.deployContract(
            "MockAddressProviderV2",
            [debtControllerFixture.debtController.address, zeroAddress, owner.account.address, zeroAddress, zeroAddress, zeroAddress, zeroAddress]);
    return {
        ...debtControllerFixture,
        addressProvider,
        owner,
        user1
    };
}

export async function deployWasabiLongPool() {
    const perpManager = await deployPerpManager();

    const addressProviderFixture = await deployAddressProvider();
    const {addressProvider, weth} = addressProviderFixture;

    // Setup
    const [owner, user1, user2, partner] = await hre.viem.getWalletClients();
    const publicClient = await hre.viem.getPublicClient();

    // Deploy WasabiLongPool
    const contractName = "WasabiLongPool";
    const WasabiLongPool = await hre.ethers.getContractFactory(contractName);
    const address = 
        await hre.upgrades.deployProxy(
            WasabiLongPool,
            [addressProviderFixture.addressProvider.address, perpManager.manager.address],
            { kind: 'uups'}
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const wasabiLongPool = await hre.viem.getContractAt(contractName, address);

    const implAddress = await hre.upgrades.erc1967.getImplementationAddress(address);

    const vaultFixture = await deployVault(
        wasabiLongPool.address, zeroAddress, addressProvider.address, perpManager.manager.address, weth.address, "WETH Vault", "wasabWETH");
    const vault = vaultFixture.vault;
    await wasabiLongPool.write.addVault([vault.address], {account: perpManager.vaultAdmin.account});
    await vault.write.depositEth([owner.account.address], { value: parseEther("20") });

    const aaveStrategyFixture = await deployAaveStrategy(vault.address, perpManager.manager.address);

    const CappedVaultCompetitionDepositor = await hre.ethers.getContractFactory("CappedVaultCompetitionDepositor");
    const competitionDepositorAddress = 
        await hre.upgrades.deployProxy(
            CappedVaultCompetitionDepositor,
            [vault.address, perpManager.manager.address],
            { kind: 'uups' }
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const competitionDepositor = await hre.viem.getContractAt("CappedVaultCompetitionDepositor", competitionDepositorAddress);

    const hasher = await hre.viem.deployContract("MockHasher", []);

    const feeShareBips = 5000n; // 50%
    const PartnerFeeManager = await hre.ethers.getContractFactory("PartnerFeeManager");
    const partnerFeeManagerAddress = 
        await hre.upgrades.deployProxy(
            PartnerFeeManager,
            [perpManager.manager.address, wasabiLongPool.address, zeroAddress],
            { kind: 'uups' }
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const partnerFeeManager = await hre.viem.getContractAt("PartnerFeeManager", partnerFeeManagerAddress);
    await partnerFeeManager.write.setFeeShareBips([partner.account.address, feeShareBips], {account: owner.account});

    await addressProvider.write.setPartnerFeeManager([partnerFeeManagerAddress], {account: owner.account});

    return {
        ...vaultFixture,
        ...addressProviderFixture,
        ...perpManager,
        ...aaveStrategyFixture,
        wasabiLongPool,
        owner,
        user1,
        user2,
        partner,
        publicClient,
        contractName,
        implAddress,
        competitionDepositor,
        hasher,
        partnerFeeManager,
        feeShareBips
    };
}

export async function deployWasabiShortPool() {
    const perpManager = await deployPerpManager();
    const addressProviderFixture = await deployAddressProvider();
    const {addressProvider, weth} = addressProviderFixture;

    // Setup
    const [owner, user1, user2, partner] = await hre.viem.getWalletClients();
    const publicClient = await hre.viem.getPublicClient();

    // Deploy WasabiShortPool
    const contractName = "WasabiShortPool";
    const WasabiShortPool = await hre.ethers.getContractFactory(contractName);
    const proxy = await hre.upgrades.deployProxy(
        WasabiShortPool,
        [addressProviderFixture.addressProvider.address, perpManager.manager.address],
        { kind: 'uups'}
    );
    await proxy.waitForDeployment();
    
    const address = getAddress(await proxy.getAddress());

    const wasabiShortPool = await hre.viem.getContractAt(contractName, address);

    // Deploy WasabiLongPool (for USDC deposits on close)
    const WasabiLongPool = await hre.ethers.getContractFactory("WasabiLongPool");
    const longPoolAddress = 
        await hre.upgrades.deployProxy(
            WasabiLongPool,
            [addressProviderFixture.addressProvider.address, perpManager.manager.address],
            { kind: 'uups'}
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const wasabiLongPool = await hre.viem.getContractAt("WasabiLongPool", longPoolAddress);

    const uPPG = await hre.viem.deployContract("MockERC20", ["μPudgyPenguins", 'μPPG']);
    const usdc = await hre.viem.deployContract("USDC", []);
    await wasabiShortPool.write.addQuoteToken([usdc.address], {account: owner.account});

    const vaultFixture = await deployVault(
        longPoolAddress, wasabiShortPool.address, addressProvider.address, perpManager.manager.address, uPPG.address, "PPG Vault", "wuPPG");
    const {vault} = vaultFixture;

    // Deploy WETH & USDC Vaults
    const usdcVaultFixture = await deployVault(
        wasabiLongPool.address, wasabiShortPool.address, addressProvider.address, perpManager.manager.address, usdc.address, "USDC Vault", "wUSDC");
    const usdcVault = usdcVaultFixture.vault;
    const wethVaultFixture = await deployVault(
        wasabiLongPool.address, wasabiShortPool.address, addressProvider.address, perpManager.manager.address, weth.address, "WETH Vault", "wWETH");
    const wethVault = wethVaultFixture.vault;

    const amount = parseEther("500");
    await uPPG.write.mint([amount]);
    await uPPG.write.approve([vault.address, amount]);
    await vault.write.deposit([amount, owner.account.address]);
    await wasabiShortPool.write.addVault([vault.address], {account: perpManager.vaultAdmin.account});
    await wasabiShortPool.write.addVault([wethVault.address], {account: perpManager.vaultAdmin.account});
    await wasabiShortPool.write.addVault([usdcVault.address], {account: perpManager.vaultAdmin.account});
    await wasabiLongPool.write.addVault([wethVault.address], {account: perpManager.vaultAdmin.account});
    await wasabiLongPool.write.addVault([usdcVault.address], {account: perpManager.vaultAdmin.account});

    const hasher = await hre.viem.deployContract("MockHasher", []);

    const feeShareBips = 5000n; // 50%
    const PartnerFeeManager = await hre.ethers.getContractFactory("PartnerFeeManager");
    const partnerFeeManagerAddress = 
        await hre.upgrades.deployProxy(
            PartnerFeeManager,
            [perpManager.manager.address, wasabiLongPool.address, wasabiShortPool.address],
            { kind: 'uups' }
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const partnerFeeManager = await hre.viem.getContractAt("PartnerFeeManager", partnerFeeManagerAddress);
    await partnerFeeManager.write.setFeeShareBips([partner.account.address, feeShareBips], {account: owner.account});

    await addressProvider.write.setPartnerFeeManager([partnerFeeManagerAddress], {account: owner.account});

    return {
        ...addressProviderFixture,
        ...perpManager,
        wasabiShortPool,
        wasabiLongPool,
        owner,
        user1,
        user2,
        partner,
        publicClient,
        contractName,
        uPPG,
        usdc,
        usdcVault,
        wethVault,
        vault,
        hasher,
        partnerFeeManager,
        feeShareBips
    };
}


export async function deployShortPoolMockEnvironment() {
    const wasabiShortPoolFixture = await deployWasabiShortPool();
    const {tradeFeeValue, contractName, wasabiShortPool, orderSigner, user1, partner, publicClient, feeDenominator, debtController, uPPG, wethAddress, weth, usdc} = wasabiShortPoolFixture;
    const [owner] = await hre.viem.getWalletClients();

    const initialPPGPrice = 10_000n;    // 1 PPG = 1 WETH
    const initialUSDCPrice = 4n;        // 1 USDC = 4/10000 WETH = 1/2500 WETH
    const priceDenominator = 10_000n;

    const mockSwap = await hre.viem.deployContract("MockSwap", []);
    await weth.write.deposit([], { value: parseEther("50") });
    await weth.write.transfer([mockSwap.address, parseEther("50")]);

    await uPPG.write.mint([mockSwap.address, parseEther("10")]);
    await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice]);

    await usdc.write.mint([mockSwap.address, parseUnits("10000", 6)]);
    await mockSwap.write.setPrice([usdc.address, wethAddress, initialUSDCPrice]);
    await mockSwap.write.setPrice([usdc.address, uPPG.address, initialUSDCPrice]);

    // Deploy some tokens to the short pool for collateral

    const leverage = 5n;
    const totalAmountIn = parseEther("1");
    const fee = getFee(totalAmountIn * (leverage + 1n), tradeFeeValue);
    const downPayment = totalAmountIn - fee;

    const swappedAmount = downPayment * initialPPGPrice / priceDenominator;
    const principal = swappedAmount * leverage;

    const functionCallDataList: FunctionCallData[] =
        getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, principal);

    await weth.write.deposit([], { value: parseEther("50"), account: user1.account });
    await weth.write.approve([wasabiShortPool.address, maxUint256], {account: user1.account});
    await usdc.write.mint([user1.account.address, parseUnits("10000", 6)]);
    await usdc.write.approve([wasabiShortPool.address, maxUint256], {account: user1.account});

    const openPositionRequest: OpenPositionRequest = {
        id: 1n,
        currency: uPPG.address,
        targetCurrency: wethAddress,
        downPayment,
        principal,
        minTargetAmount: principal * initialPPGPrice / priceDenominator,
        expiration: BigInt(await time.latest()) + 86400n,
        fee,
        functionCallDataList,
        existingPosition: getEmptyPosition(),
        referrer: zeroAddress
    };
    const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, openPositionRequest);

    const getTradeAmounts = async (leverage: bigint, totalAmountIn: bigint, currency: Address) => {
        const isUSDC = currency == usdc.address;
        const fee = getFee(totalAmountIn * (leverage + 1n), tradeFeeValue);
        const downPayment = totalAmountIn - fee;
        const swappedAmount = downPayment * (isUSDC ? ((10n ** (18n - 6n)) * initialUSDCPrice) : initialPPGPrice) / priceDenominator;
        const principal = swappedAmount * leverage;
        const totalSize = principal + downPayment;
        const minTargetAmount = isUSDC
            ? principal * initialPPGPrice / initialUSDCPrice / (10n ** (18n - 6n))
            : principal * initialPPGPrice / priceDenominator;
        return { fee, downPayment, principal, totalSize, minTargetAmount };
    }

    const getOpenPositionRequest = async ({
        id = 1n,
        currency = uPPG.address,
        targetCurrency = wethAddress,
        downPayment = openPositionRequest.downPayment,
        principal = openPositionRequest.principal,
        minTargetAmount = openPositionRequest.minTargetAmount,
        expiration = openPositionRequest.expiration,
        fee = openPositionRequest.fee,
        existingPosition = openPositionRequest.existingPosition,
        referrer = openPositionRequest.referrer
    }: {
        id?: bigint,
        currency?: Address,
        targetCurrency?: Address,
        downPayment?: bigint,
        principal?: bigint,
        minTargetAmount?: bigint,
        expiration?: bigint,
        fee?: bigint,
        existingPosition?: Position,
        referrer?: Address
    } = {}): Promise<OpenPositionRequest> => {
        const functionCallDataList: FunctionCallData[] =
            getApproveAndSwapFunctionCallData(mockSwap.address, currency, targetCurrency, principal);
        return {
            id,
            currency,
            targetCurrency,
            downPayment,
            principal,
            minTargetAmount,
            expiration,
            fee,
            functionCallDataList,
            existingPosition,
            referrer
        }
    }

    const sendOpenPositionRequest = async (request: OpenPositionRequest) => {
        const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
        const hash = await wasabiShortPool.write.openPosition([request, signature], { account: user1.account });
        const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
        const event = (await wasabiShortPool.getEvents.PositionOpened())[0];
        const position: Position = await getEventPosition(event);
        return {
            position,
            hash,
            gasUsed,
            event
        }
    }

    const sendDefaultOpenPositionRequest = async (id?: bigint | undefined) => {
        const request = id ? {...openPositionRequest, id} : openPositionRequest;
        return await sendOpenPositionRequest(request);
    }

    const sendReferredOpenPositionRequest = async (id?: bigint | undefined) => {
        const request = id ? {...openPositionRequest, id, referrer: partner.account.address} : {...openPositionRequest, referrer: partner.account.address};
        return await sendOpenPositionRequest(request);
    }

    const sendUSDCOpenPositionRequest = async (id?: bigint | undefined) => {
        const leverage = 5n;
        const totalAmountIn = parseUnits("500", 6);
        const fee = getFee(totalAmountIn * (leverage + 1n), tradeFeeValue);
        const downPayment = totalAmountIn - fee;
        const swappedAmount = downPayment * (10n ** (18n - 6n)) * initialUSDCPrice / priceDenominator;
        const principal = swappedAmount * leverage;
        const minTargetAmount = principal * initialPPGPrice / initialUSDCPrice / (10n ** (18n - 6n));

        const functionCallDataList: FunctionCallData[] = 
            getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, usdc.address, principal);
        const openPositionRequest: OpenPositionRequest = {
            id: id ?? 1n,
            currency: uPPG.address,
            targetCurrency: usdc.address,
            downPayment,
            principal,
            minTargetAmount,
            expiration: BigInt(await time.latest()) + 86400n,
            fee,
            functionCallDataList,
            existingPosition: getEmptyPosition(),
            referrer: zeroAddress
        };
        const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, openPositionRequest);

        const hash = await wasabiShortPool.write.openPosition([openPositionRequest, signature], { account: user1.account });

        const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed);
        const event = (await wasabiShortPool.getEvents.PositionOpened())[0];
        const position: Position = await getEventPosition(event);

        return {
            position,
            hash,
            gasUsed,
            event,
            downPayment,
        }
    }


    const createClosePositionRequest = async (params: CreateClosePositionRequestParams): Promise<ClosePositionRequest> => {
        let { position, interest, expiration, amount, referrer } = params;
        amount = amount || 0n;
        const amountOut = (amount > 0 ? amount : position.principal) + (interest || 0n);

        let functionCallDataList: FunctionCallData[] = [];

        const wethBalance = await weth.read.balanceOf([wasabiShortPool.address]);
        if (wethBalance < amountOut && position.currency === wethAddress) {
            const data = encodeFunctionData({
                abi: [WETHAbi.find(a => a.type === "function" && a.name === "deposit")!],
                functionName: "deposit"
            });
        
            const functionCallData: FunctionCallData = {
                to: wethAddress,
                value: amountOut - wethBalance,
                data
            }
            functionCallDataList.push(functionCallData);
        }

        functionCallDataList = [
            ...functionCallDataList,
            ...getApproveAndSwapExactlyOutFunctionCallData(
                mockSwap.address,
                position.collateralCurrency,
                position.currency,
                position.collateralAmount,
                amountOut
            )
        ]

        const request: ClosePositionRequest = {
            expiration: expiration ? BigInt(expiration) : (BigInt(await time.latest()) + 300n),
            interest: interest || 0n,
            amount: amount || 0n,
            position,
            functionCallDataList,
            referrer: referrer || zeroAddress
        };
        return request;
    }

    const createSignedClosePositionRequest = async (params: CreateClosePositionRequestParams): Promise<WithSignature<ClosePositionRequest>> => {
        const request = await createClosePositionRequest(params);
        const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
        return { request, signature }
    }

    const createClosePositionOrder = async (params: CreateClosePositionOrderParams): Promise<ClosePositionOrder> => {
        const { orderType, positionId, makerAmount, takerAmount, createdAt, expiration, executionFee } = params;
        const order: ClosePositionOrder = {
            orderType,
            positionId,
            makerAmount,
            takerAmount,
            createdAt: createdAt ? BigInt(createdAt) : BigInt(await time.latest()),
            expiration: expiration ? BigInt(expiration) : (BigInt(await time.latest()) + 300n),
            executionFee: executionFee || 0n
        }
        return order;
    }

    const createSignedClosePositionOrder = async (params: CreateClosePositionOrderParams): Promise<WithSignature<ClosePositionOrder>> => {
        const {traderSigner} = params;
        const order = await createClosePositionOrder(params);
        const signature = await signClosePositionOrder(traderSigner, contractName, wasabiShortPool.address, order);
        return { request: order, signature }
    }

    const computeMaxInterest = async (position: Position): Promise<bigint> => {
        return await debtController.read.computeMaxInterest([position.currency, position.principal, position.lastFundingTimestamp], { blockTag: 'pending' });
    }

    const computeLiquidationPrice = async (position: Position): Promise<bigint> => {
        const threshold = 500n; // 5 percent

        const currentInterest = await computeMaxInterest(position);
        const payoutLiquidationThreshold = position.collateralAmount * (threshold + tradeFeeValue) / (feeDenominator - tradeFeeValue);

        const liquidationAmount = position.collateralAmount - payoutLiquidationThreshold;
        return liquidationAmount * priceDenominator / (position.principal + currentInterest);
    }

    const getBalance = async (currency: string, address: string) => {
        if (currency === zeroAddress) {
            return await publicClient.getBalance({address: getAddress(address)});
        } else if (getAddress(currency) === getAddress(uPPG.address)) {
            return await uPPG.read.balanceOf([getAddress(address)]);
        } else {
            throw new Error(`Unknown currency ${currency}`);
        }
    }

    return {
        ...wasabiShortPoolFixture,
        mockSwap,
        uPPG,
        usdc,
        openPositionRequest,
        downPayment,
        principal,
        totalAmountIn,
        leverage,
        signature,
        initialPPGPrice,
        initialUSDCPrice,
        priceDenominator,
        getOpenPositionRequest,
        getTradeAmounts,
        sendOpenPositionRequest,
        sendDefaultOpenPositionRequest,
        sendReferredOpenPositionRequest,
        sendUSDCOpenPositionRequest,
        createClosePositionRequest,
        createSignedClosePositionRequest,
        createClosePositionOrder,
        createSignedClosePositionOrder,
        computeLiquidationPrice,
        computeMaxInterest,
        getBalance,
        signClosePositionRequest
    }
}

export async function deployWasabiPoolsAndRouter() {
    const perpManager = await deployPerpManager();

    const addressProviderFixture = await deployAddressProvider();
    const {addressProvider, weth, feeReceiver} = addressProviderFixture;

    // Setup
    const [owner, user1, user2, partner] = await hre.viem.getWalletClients();
    const publicClient = await hre.viem.getPublicClient();

    // Deploy WasabiLongPool
    const WasabiLongPool = await hre.ethers.getContractFactory("WasabiLongPool");
    const longPoolAddress = 
        await hre.upgrades.deployProxy(
            WasabiLongPool,
            [addressProviderFixture.addressProvider.address, perpManager.manager.address],
            { kind: 'uups'}
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const wasabiLongPool = await hre.viem.getContractAt("WasabiLongPool", longPoolAddress);

    // Deploy WasabiShortPool
    const WasabiShortPool = await hre.ethers.getContractFactory("WasabiShortPool");
    const shortPoolAddress =
        await hre.upgrades.deployProxy(
            WasabiShortPool,
            [addressProviderFixture.addressProvider.address, perpManager.manager.address],
            { kind: 'uups'}
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const wasabiShortPool = await hre.viem.getContractAt("WasabiShortPool", shortPoolAddress);

    const uPPG = await hre.viem.deployContract("MockERC20", ["μPudgyPenguins", 'μPPG']);
    const usdc = await hre.viem.deployContract("USDC", []);

    // Deploy WETH Vault
    const wethVaultFixture = await deployVault(
        longPoolAddress, shortPoolAddress, addressProvider.address, perpManager.manager.address, weth.address, "WETH Vault", "wasabWETH");
    const wethVault = wethVaultFixture.vault;
    await wasabiLongPool.write.addVault([wethVault.address], {account: perpManager.vaultAdmin.account});
    await wethVault.write.depositEth([owner.account.address], { value: parseEther("20") });

    // Deploy PPG Vault
    const ppgVaultFixture = await deployVault(
        longPoolAddress, shortPoolAddress, addressProvider.address, perpManager.manager.address, uPPG.address, "PPG Vault", "wuPPG");
    const ppgVault = ppgVaultFixture.vault;
    const amount = parseEther("50");
    await uPPG.write.mint([amount]);
    await uPPG.write.approve([ppgVault.address, amount]);
    await ppgVault.write.deposit([amount, owner.account.address]);
    await wasabiShortPool.write.addVault([ppgVault.address], {account: perpManager.vaultAdmin.account});
    await wasabiShortPool.write.addVault([wethVault.address], {account: perpManager.vaultAdmin.account});

    // Deploy MockSwap and MockSwapRouter
    const mockSwap = await hre.viem.deployContract("MockSwap", []);
    const mockSwapRouter = await hre.viem.deployContract("MockSwapRouter", [mockSwap.address, weth.address]);

    // Deploy WasabiRouter
    const swapFeeBips = 50n;    // 0.5% fee
    const WasabiRouter = await hre.ethers.getContractFactory("WasabiRouter");
    const routerAddress =
        await hre.upgrades.deployProxy(
            WasabiRouter,
            [longPoolAddress, shortPoolAddress, weth.address, perpManager.manager.address, mockSwapRouter.address, feeReceiver, swapFeeBips],
            { kind: 'uups'}
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const wasabiRouter = await hre.viem.getContractAt("WasabiRouter", routerAddress);
    await addressProvider.write.setWasabiRouter([routerAddress]);

    // Deploy ExactOutSwapper
    const ExactOutSwapper = await hre.ethers.getContractFactory("ExactOutSwapper");
    const exactOutSwapperAddress =
        await hre.upgrades.deployProxy(
            ExactOutSwapper,
            [perpManager.manager.address],
            { kind: 'uups'}
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const exactOutSwapper = await hre.viem.getContractAt("ExactOutSwapper", exactOutSwapperAddress);
    await exactOutSwapper.write.setWhitelistedAddress([mockSwapRouter.address, true]);
    const swapFunctionSelector = toFunctionSelector(mockSwapRouter.abi.find(a => a.type === "function" && a.name === "swap")!);
    await exactOutSwapper.write.setWhitelistedFunctionSelectors([[swapFunctionSelector], true]);

    const feeShareBips = 5000n; // 50%
    const PartnerFeeManager = await hre.ethers.getContractFactory("PartnerFeeManager");
    const partnerFeeManagerAddress = 
        await hre.upgrades.deployProxy(
            PartnerFeeManager,
            [perpManager.manager.address, wasabiLongPool.address, wasabiShortPool.address],
            { kind: 'uups' }
        )
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    const partnerFeeManager = await hre.viem.getContractAt("PartnerFeeManager", partnerFeeManagerAddress);
    await partnerFeeManager.write.setFeeShareBips([partner.account.address, feeShareBips], {account: owner.account});

    await addressProvider.write.setPartnerFeeManager([partnerFeeManagerAddress], {account: owner.account});

    return {
        ...addressProviderFixture,
        ...perpManager,
        wasabiLongPool,
        wasabiShortPool,
        wasabiRouter,
        wethVault,
        ppgVault,
        mockSwap,
        mockSwapRouter,
        exactOutSwapper,
        uPPG,
        usdc,
        owner,
        user1,
        user2,
        partner,
        publicClient,
        swapFeeBips,
        partnerFeeManager,
        feeShareBips
    };
}

export async function deployPoolsAndRouterMockEnvironment() {
    const wasabiPoolsAndRouterFixture = await deployWasabiPoolsAndRouter();
    const {wasabiRouter, wasabiLongPool, wasabiShortPool, mockSwap, mockSwapRouter, uPPG, usdc, weth, orderSigner, orderExecutor, feeReceiver, swapFeeBips, user1, user2, publicClient, wethAddress, debtController} = wasabiPoolsAndRouterFixture;

    const initialPPGPrice = 10_000n;    // 1 PPG = 1 WETH
    const initialUSDCPrice = 4n;        // 1 USDC = 4/10000 WETH = 1/2500 WETH
    const priceDenominator = 10_000n;

    await weth.write.deposit([], { value: parseEther("50") });
    await weth.write.transfer([mockSwap.address, parseEther("50")]);

    await uPPG.write.mint([mockSwap.address, parseEther("10")]);
    await uPPG.write.mint([user1.account.address, parseEther("10")]);
    await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice]);

    await usdc.write.mint([mockSwap.address, parseUnits("10000", 6)]);
    await mockSwap.write.setPrice([usdc.address, weth.address, initialUSDCPrice]);
    await mockSwap.write.setPrice([usdc.address, uPPG.address, initialUSDCPrice]);

    await weth.write.deposit([], { value: parseEther("50"), account: user1.account });
    await weth.write.approve([wasabiLongPool.address, maxUint256], {account: user1.account});
    await weth.write.approve([wasabiShortPool.address, maxUint256], {account: user1.account});
    await weth.write.deposit([], { value: parseEther("50"), account: user2.account });
    await weth.write.approve([wasabiLongPool.address, maxUint256], {account: user2.account});
    await weth.write.approve([wasabiShortPool.address, maxUint256], {account: user2.account});

    const leverage = 5n;
    const totalAmountIn = parseEther("1");
    const longFee = getFee(totalAmountIn * leverage, tradeFeeValue);
    const shortFee = getFee(totalAmountIn * (leverage + 1n), tradeFeeValue);
    const longDownPayment = totalAmountIn - longFee;
    const shortDownPayment = totalAmountIn - shortFee;
    const longPrincipal = longDownPayment * (leverage - 1n);
    const longTotalSize = longPrincipal + longDownPayment;
    const swappedAmount = shortDownPayment * initialPPGPrice / priceDenominator;
    const shortPrincipal = swappedAmount * leverage;
    const executionFee = parseEther("0.005");

    const longOpenPositionRequest: OpenPositionRequest = {
        id: 1n,
        currency: wethAddress,
        targetCurrency: uPPG.address,
        downPayment: longDownPayment,
        principal: longPrincipal,
        minTargetAmount: longTotalSize * initialPPGPrice / priceDenominator,
        expiration: BigInt(await time.latest()) + 86400n,
        fee: longFee,
        functionCallDataList: 
            getApproveAndSwapFunctionCallData(mockSwap.address, wethAddress, uPPG.address, longTotalSize),
        existingPosition: getEmptyPosition(),
        referrer: zeroAddress
    };
    const longOpenSignature = await signOpenPositionRequest(orderSigner, "WasabiLongPool", wasabiLongPool.address, longOpenPositionRequest);
    
    const shortOpenPositionRequest: OpenPositionRequest = {
        id: 1n,
        currency: uPPG.address,
        targetCurrency: wethAddress,
        downPayment: shortDownPayment,
        principal: shortPrincipal,
        minTargetAmount: shortPrincipal * initialPPGPrice / priceDenominator,
        expiration: BigInt(await time.latest()) + 86400n,
        fee: shortFee,
        functionCallDataList: 
            getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, shortPrincipal),
        existingPosition: getEmptyPosition(),
        referrer: zeroAddress
    };
    const shortOpenSignature = await signOpenPositionRequest(orderSigner, "WasabiShortPool", wasabiShortPool.address, shortOpenPositionRequest);

    const sendDefaultLongOpenPositionRequest = async (id?: bigint | undefined) => {
        const request = id ? {...longOpenPositionRequest, id} : longOpenPositionRequest;
        const signature = await signOpenPositionRequest(orderSigner, "WasabiLongPool", wasabiLongPool.address, request);
        const hash = await wasabiLongPool.write.openPosition([request, signature], { account: user1.account });
        const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
        const event = (await wasabiLongPool.getEvents.PositionOpened())[0];
        const position: Position = await getEventPosition(event);

        return {
            position,
            hash,
            gasUsed,
            event
        }
    }

    const sendRouterLongOpenPositionRequest = async (id?: bigint | undefined) => {
        const request = id ? {...longOpenPositionRequest, id} : longOpenPositionRequest;
        const routerRequest = {...request, functionCallDataList: [], interestToPay: 0n};
        const signature = await signOpenPositionRequest(orderSigner, "WasabiLongPool", wasabiLongPool.address, request);
        const traderSig = await signOpenPositionRequest(user1, "WasabiRouter", wasabiRouter.address, routerRequest);
        const hash = await wasabiRouter.write.openPosition(
            [wasabiLongPool.address, request, signature, traderSig, executionFee], 
            { account: orderExecutor.account }
        );
        const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
        const event = (await wasabiLongPool.getEvents.PositionOpened())[0];
        const position: Position = await getEventPosition(event);

        return {
            position,
            hash,
            gasUsed,
            event
        }
    }

    const sendDefaultShortOpenPositionRequest = async (id?: bigint | undefined) => {
        const request = id ? {...shortOpenPositionRequest, id} : shortOpenPositionRequest;
        const signature = await signOpenPositionRequest(orderSigner, "WasabiShortPool", wasabiShortPool.address, request);
        const hash = await wasabiShortPool.write.openPosition([request, signature], { account: user1.account });
        const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
        const event = (await wasabiShortPool.getEvents.PositionOpened())[0];
        const position: Position = await getEventPosition(event);

        return {
            position,
            hash,
            gasUsed,
            event
        }
    }

    const sendRouterShortOpenPositionRequest = async (id?: bigint | undefined) => {
        const request = id ? {...shortOpenPositionRequest, id} : shortOpenPositionRequest;
        const routerRequest = {...request, functionCallDataList: [], interestToPay: 0n};
        const signature = await signOpenPositionRequest(orderSigner, "WasabiShortPool", wasabiShortPool.address, request);
        const traderSig = await signOpenPositionRequest(user1, "WasabiRouter", wasabiRouter.address, routerRequest);
        const hash = await wasabiRouter.write.openPosition(
            [wasabiShortPool.address, request, signature, traderSig, executionFee], 
            { account: orderExecutor.account }
        );
        const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
        const event = (await wasabiShortPool.getEvents.PositionOpened())[0];
        const position: Position = await getEventPosition(event);

        return {
            position,
            hash,
            gasUsed,
            event
        }
    }

    const createCloseLongPositionRequest = async (params: CreateClosePositionRequestParams): Promise<ClosePositionRequest> => {
        const { position, interest, expiration, amount, referrer } = params;
        const request: ClosePositionRequest = {
            expiration: expiration ? BigInt(expiration) : (BigInt(await time.latest()) + 300n),
            interest: interest || 0n,
            amount: amount || 0n,
            position,
            functionCallDataList: getApproveAndSwapFunctionCallData(mockSwap.address, position.collateralCurrency, position.currency, position.collateralAmount),
            referrer: referrer || zeroAddress
        };
        return request;
    }

    const createCloseShortPositionRequest = async (params: CreateClosePositionRequestParams): Promise<ClosePositionRequest> => {
        const { position, interest, expiration, amount, referrer } = params;
        const amountOut = position.principal + (interest || 0n);

        let functionCallDataList: FunctionCallData[] = [];

        const wethBalance = await weth.read.balanceOf([wasabiShortPool.address]);
        if (wethBalance < amountOut) {
            const data = encodeFunctionData({
                abi: [WETHAbi.find(a => a.type === "function" && a.name === "deposit")!],
                functionName: "deposit"
            });
        
            const functionCallData: FunctionCallData = {
                to: wethAddress,
                value: amountOut - wethBalance,
                data
            }
            functionCallDataList.push(functionCallData);
        }

        functionCallDataList = [
            ...functionCallDataList,
            ...getApproveAndSwapExactlyOutFunctionCallData(
                mockSwap.address,
                position.collateralCurrency,
                position.currency,
                position.collateralAmount,
                amountOut
            )
        ]

        const request: ClosePositionRequest = {
            expiration: expiration ? BigInt(expiration) : (BigInt(await time.latest()) + 300n),
            interest: interest || 0n,
            amount: amount || 0n,
            position,
            functionCallDataList,
            referrer: referrer || zeroAddress
        };
        return request;
    }

    const createSignedCloseLongPositionRequest = async (params: CreateClosePositionRequestParams): Promise<WithSignature<ClosePositionRequest>> => {
        const request = await createCloseLongPositionRequest(params);
        const signature = await signClosePositionRequest(orderSigner, "WasabiLongPool", wasabiLongPool.address, request);
        return { request, signature }
    }

    const createSignedCloseShortPositionRequest = async (params: CreateClosePositionRequestParams): Promise<WithSignature<ClosePositionRequest>> => {
        const request = await createCloseShortPositionRequest(params);
        const signature = await signClosePositionRequest(orderSigner, "WasabiShortPool", wasabiShortPool.address, request);
        return { request, signature }
    }

    const createExactInRouterSwapData = async (params: CreateExactInSwapDataParams): Promise<EncodeFunctionDataReturnType> => {
        const hasFee = params.swapFee !== undefined;
        if (hasFee) {
            const callDatas: FunctionCallData[] = [];
            callDatas.push(getRouterSwapFunctionCallData(mockSwapRouter.address, params.tokenIn, params.tokenOut, params.amount, mockSwapRouter.address));
            if (params.tokenOut === wethAddress && params.unwrapEth) {
                callDatas.push(getUnwrapWETH9WithFeeCallData(mockSwapRouter.address, 0n, params.swapRecipient, swapFeeBips, feeReceiver));
            } else {
                callDatas.push(getSweepTokenWithFeeCallData(mockSwapRouter.address, params.tokenOut, 0n, params.swapRecipient, swapFeeBips, feeReceiver));
            }
            
            return encodeFunctionData({
                abi: [MockSwapRouterAbi.find(a => a.type === "function" && a.name === "multicall")],
                functionName: "multicall",
                args: [callDatas.map(f => f.data)]
            });
        } else {
            return getSwapFunctionCallData(mockSwap.address, params.tokenIn, params.tokenOut, params.amount).data;
        }
    }

    const createExactOutRouterSwapData = async (params: CreateExactOutSwapDataParams): Promise<EncodeFunctionDataReturnType> => {
        const hasFee = params.swapFee !== undefined;
        if (hasFee) {
            const callDatas: FunctionCallData[] = [];
            callDatas.push(getRouterSwapExactlyOutFunctionCallData(mockSwapRouter.address, params.tokenIn, params.tokenOut, params.amountInMax, params.amountOut, mockSwapRouter.address));
            if (params.tokenOut === wethAddress && params.unwrapEth) {
                callDatas.push(getUnwrapWETH9WithFeeCallData(mockSwapRouter.address, 0n, params.swapRecipient, swapFeeBips, feeReceiver));
            } else {
                callDatas.push(getSweepTokenWithFeeCallData(mockSwapRouter.address, params.tokenOut, 0n, params.swapRecipient, swapFeeBips, feeReceiver));
            }
            return encodeFunctionData({
                abi: [MockSwapRouterAbi.find(a => a.type === "function" && a.name === "multicall")],
                functionName: "multicall",
                args: [callDatas.map(f => f.data)]
            });
        } else {
            return getSwapExactlyOutFunctionCallData(mockSwap.address, params.tokenIn, params.tokenOut, params.amountInMax, params.amountOut).data;
        }
    }

    const computeLongMaxInterest = async (position: Position): Promise<bigint> => {
        return await debtController.read.computeMaxInterest([position.collateralCurrency, position.principal, position.lastFundingTimestamp], { blockTag: 'pending' });
    }

    const computeLongLiquidationPrice = async (position: Position): Promise<bigint> => {
        const threshold = 500n; // 5 percent

        const currentInterest = await computeLongMaxInterest(position);
        const payoutLiquidationThreshold = position.principal * (threshold + tradeFeeValue) / (feeDenominator - tradeFeeValue);

        const liquidationAmount = payoutLiquidationThreshold + position.principal + currentInterest;
        return liquidationAmount * priceDenominator / position.collateralAmount;
    }

    const computeShortMaxInterest = async (position: Position): Promise<bigint> => {
        return await debtController.read.computeMaxInterest([position.currency, position.principal, position.lastFundingTimestamp], { blockTag: 'pending' });
    }

    const computeShortLiquidationPrice = async (position: Position): Promise<bigint> => {
        const threshold = 500n; // 5 percent

        const currentInterest = await computeShortMaxInterest(position);
        const payoutLiquidationThreshold = position.collateralAmount * (threshold + tradeFeeValue) / (feeDenominator - tradeFeeValue);

        const liquidationAmount = position.collateralAmount - payoutLiquidationThreshold;
        return liquidationAmount * priceDenominator / (position.principal + currentInterest);
    }

    return {
        ...wasabiPoolsAndRouterFixture,
        mockSwap,
        leverage,
        totalAmountIn,
        longDownPayment,
        shortDownPayment,
        longPrincipal,
        shortPrincipal,
        longTotalSize,
        longOpenPositionRequest,
        longOpenSignature,
        shortOpenPositionRequest,
        shortOpenSignature,
        initialPPGPrice,
        initialUSDCPrice,
        priceDenominator,
        executionFee,
        swapFeeBips,
        sendDefaultLongOpenPositionRequest,
        sendDefaultShortOpenPositionRequest,
        sendRouterLongOpenPositionRequest,
        sendRouterShortOpenPositionRequest,
        createCloseLongPositionRequest,
        createSignedCloseLongPositionRequest,
        createCloseShortPositionRequest,
        createSignedCloseShortPositionRequest,
        createExactInRouterSwapData,
        createExactOutRouterSwapData,
        computeLongMaxInterest,
        computeLongLiquidationPrice,
        computeShortMaxInterest,
        computeShortLiquidationPrice
    }
}