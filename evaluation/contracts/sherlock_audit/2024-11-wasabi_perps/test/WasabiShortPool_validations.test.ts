import { time, loadFixture } from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import { expect } from "chai";
import { getAddress, zeroAddress } from "viem";
import { ClosePositionRequest, FunctionCallData, OpenPositionRequest, getFee, PayoutType, getEmptyPosition, AddCollateralRequest, RemoveCollateralRequest } from "./utils/PerpStructUtils";
import { signAddCollateralRequest, signClosePositionRequest, signOpenPositionRequest, signRemoveCollateralRequest } from "./utils/SigningUtils";
import { deployShortPoolMockEnvironment, deployWasabiShortPool } from "./fixtures";
import { getApproveAndSwapFunctionCallData, getApproveAndSwapFunctionCallDataExact } from "./utils/SwapUtils";

describe("WasabiShortPool - Validations Test", function () {
    describe("Deployment", function () {
        it("Should set the right address provider", async function () {
            const { wasabiShortPool, addressProvider } = await loadFixture(deployWasabiShortPool);
            expect(await wasabiShortPool.read.addressProvider()).to.equal(getAddress(addressProvider.address));
        });

        it("Should set the right owner", async function () {
            const { wasabiShortPool, owner, manager } = await loadFixture(deployWasabiShortPool);
            expect(await wasabiShortPool.read.owner()).to.equal(getAddress(manager.address));
        });
    });

    describe("Open Position Validations", function () {
        it("PrincipalTooHigh", async function () {
            const { wasabiShortPool, orderSigner, user1, totalAmountIn, maxLeverage, mockSwap, uPPG, wethAddress, tradeFeeValue, contractName, openPositionRequest, initialPPGPrice, priceDenominator } = await loadFixture(deployShortPoolMockEnvironment);
    
            const leverage = maxLeverage / 100n + 1n;
            const fee = getFee(totalAmountIn * (leverage + 2n), tradeFeeValue);
            const downPayment = totalAmountIn - fee;
        
            const swappedAmount = downPayment * initialPPGPrice / priceDenominator;
            const principal = swappedAmount * (leverage + 1n);

            const functionCallDataList: FunctionCallData[] =
                getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, principal);
            
            const request: OpenPositionRequest = {
                ...openPositionRequest,
                functionCallDataList,
                principal
            };
            const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
    
            await expect(wasabiShortPool.write.openPosition([request, signature], { value: totalAmountIn, account: user1.account }))
                .to.be.rejectedWith("PrincipalTooHigh", "Principal is too high");
        });

        it("InsufficientCollateralReceived", async function () {
            const { wasabiShortPool, orderSigner, user1, totalAmountIn, mockSwap, uPPG, wethAddress, contractName, openPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
    
            const principalThatWillBeUsed = openPositionRequest.principal * 97n / 100n;

            const functionCallDataList: FunctionCallData[] =
                getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, principalThatWillBeUsed);
            
            const request: OpenPositionRequest = {
                ...openPositionRequest,
                functionCallDataList
            };
            const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
    
            await expect(wasabiShortPool.write.openPosition([request, signature], { value: totalAmountIn, account: user1.account }))
                .to.be.rejectedWith("InsufficientCollateralReceived", "Not enough collateral received due to too little principal used");
        });

        it("InsufficientPrincipalUsed", async function () {
            const { wasabiShortPool, user1, openPositionRequest, totalAmountIn, orderSigner, contractName } = await loadFixture(deployShortPoolMockEnvironment);
            
            const request: OpenPositionRequest = { ...openPositionRequest, principal: 0n };
            const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

            await expect(wasabiShortPool.write.openPosition([request, signature], { value: totalAmountIn, account: user1.account }))
                .to.be.rejectedWith("InsufficientPrincipalUsed", "Cannot open positions with zero principal");
        })

        it("SenderNotTrader", async function () {
            const { wasabiShortPool, user1, user2, openPositionRequest, signature } = await loadFixture(deployShortPoolMockEnvironment);

            await expect(wasabiShortPool.write.openPositionFor([openPositionRequest, signature, user2.account.address], { account: user1.account }))
                .to.be.rejectedWith("SenderNotTrader", "Cannot open position for another user");
        });

        it("InvalidPosition", async function () {
            const { wasabiShortPool, user1, openPositionRequest, totalAmountIn, contractName, orderSigner } = await loadFixture(deployShortPoolMockEnvironment);

            const existingPosition = getEmptyPosition();
            let request: OpenPositionRequest = { ...openPositionRequest, existingPosition: { ...existingPosition, downPayment: 1n} };
            let signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

            await expect(wasabiShortPool.write.openPosition([request, signature], { value: totalAmountIn, account: user1.account }))
                .to.be.rejectedWith("InvalidPosition", "Existing position must be empty when opening a new position");

            request = { ...openPositionRequest, existingPosition: { ...existingPosition, principal: 1n} };
            signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

            await expect(wasabiShortPool.write.openPosition([request, signature], { value: totalAmountIn, account: user1.account }))
                .to.be.rejectedWith("InvalidPosition", "Existing position must be empty when opening a new position");
            
            request = { ...openPositionRequest, existingPosition: { ...existingPosition, collateralAmount: 1n} };
            signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

            await expect(wasabiShortPool.write.openPosition([request, signature], { value: totalAmountIn, account: user1.account }))
                .to.be.rejectedWith("InvalidPosition", "Existing position must be empty when opening a new position");
            
            request = { ...openPositionRequest, existingPosition: { ...existingPosition, feesToBePaid: 1n} };
            signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

            await expect(wasabiShortPool.write.openPosition([request, signature], { value: totalAmountIn, account: user1.account }))
                .to.be.rejectedWith("InvalidPosition", "Existing position must be empty when opening a new position");
        });
    });

    describe("Edit Position Validations", function () {
        describe("Increase Position", function () {
            it("InvalidPosition - request.id != request.existingPosition.id", async function () {
                const { wasabiShortPool, mockSwap, wethAddress, uPPG, user1, totalAmountIn, principal, initialPPGPrice, priceDenominator, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
                
                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const functionCallDataList: FunctionCallData[] =
                    getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, principal);
                const openPositionRequest: OpenPositionRequest = {
                    id: position.id + 1n, // Incorrect ID
                    currency: position.currency,
                    targetCurrency: position.collateralCurrency,
                    downPayment: position.downPayment,
                    principal: position.principal,
                    minTargetAmount: principal * initialPPGPrice / priceDenominator,
                    expiration: BigInt(await time.latest()) + 86400n,
                    fee: position.feesToBePaid,
                    functionCallDataList,
                    existingPosition: position,
                    referrer: zeroAddress
                };
                const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, openPositionRequest);

                await expect(wasabiShortPool.write.openPosition([openPositionRequest, signature], { value: totalAmountIn, account: user1.account }))
                    .to.be.rejectedWith("InvalidPosition", "Cannot increase position with different position ID");
            });

            it("InvalidPosition - stored hash != request.existingPosition.hash", async function () {
                const { wasabiShortPool, mockSwap, wethAddress, uPPG, user1, totalAmountIn, principal, initialPPGPrice, priceDenominator, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
                
                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const functionCallDataList: FunctionCallData[] =
                    getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, principal);
                const openPositionRequest: OpenPositionRequest = {
                    id: position.id,
                    currency: position.currency,
                    targetCurrency: position.collateralCurrency,
                    downPayment: position.downPayment,
                    principal: position.principal,
                    minTargetAmount: principal * initialPPGPrice / priceDenominator,
                    expiration: BigInt(await time.latest()) + 86400n,
                    fee: position.feesToBePaid,
                    functionCallDataList,
                    existingPosition: position,
                    referrer: zeroAddress
                };
                const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, openPositionRequest);

                // Increase position
                await wasabiShortPool.write.openPosition([openPositionRequest, signature], { value: totalAmountIn, account: user1.account });

                await time.increase(86400n); // 1 day later

                // Try to reuse the same request
                await expect(wasabiShortPool.write.openPosition([openPositionRequest, signature], { value: totalAmountIn, account: user1.account }))
                    .to.be.rejectedWith("InvalidPosition", "Cannot reuse the same request to increase position twice");
            });

            it("InvalidCurrency", async function () {
                const { wasabiShortPool, mockSwap, wethAddress, uPPG, user1, totalAmountIn, principal, initialPPGPrice, priceDenominator, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
                
                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const functionCallDataList: FunctionCallData[] =
                    getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, principal);
                const openPositionRequest: OpenPositionRequest = {
                    id: position.id,
                    currency: position.collateralCurrency, // Incorrect currency
                    targetCurrency: position.collateralCurrency,
                    downPayment: position.downPayment,
                    principal: position.principal,
                    minTargetAmount: principal * initialPPGPrice / priceDenominator,
                    expiration: BigInt(await time.latest()) + 86400n,
                    fee: position.feesToBePaid,
                    functionCallDataList,
                    existingPosition: position,
                    referrer: zeroAddress
                };
                const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, openPositionRequest);

                await expect(wasabiShortPool.write.openPosition([openPositionRequest, signature], { value: totalAmountIn, account: user1.account }))
                    .to.be.rejectedWith("InvalidCurrency", "request.currency must match request.existingPosition.currency");
            });

            it("InvalidTargetCurrency", async function () {
                const { wasabiShortPool, mockSwap, wethAddress, uPPG, user1, totalAmountIn, principal, initialPPGPrice, priceDenominator, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
                
                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const functionCallDataList: FunctionCallData[] =
                    getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, principal);
                const openPositionRequest: OpenPositionRequest = {
                    id: position.id,
                    currency: position.currency,
                    targetCurrency: position.currency, // Incorrect target currency
                    downPayment: position.downPayment,
                    principal: position.principal,
                    minTargetAmount: principal * initialPPGPrice / priceDenominator,
                    expiration: BigInt(await time.latest()) + 86400n,
                    fee: position.feesToBePaid,
                    functionCallDataList,
                    existingPosition: position,
                    referrer: zeroAddress
                };
                const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, openPositionRequest);

                await expect(wasabiShortPool.write.openPosition([openPositionRequest, signature], { value: totalAmountIn, account: user1.account }))
                    .to.be.rejectedWith("InvalidTargetCurrency", "request.targetCurrency must match request.existingPosition.collateralCurrency");
            });

            it("SenderNotTrader", async function () {
                const { wasabiShortPool, mockSwap, wethAddress, uPPG, user2, totalAmountIn, principal, initialPPGPrice, priceDenominator, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
                
                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const functionCallDataList: FunctionCallData[] =
                    getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, principal);
                const openPositionRequest: OpenPositionRequest = {
                    id: position.id,
                    currency: position.currency,
                    targetCurrency: position.collateralCurrency,
                    downPayment: position.downPayment,
                    principal: position.principal,
                    minTargetAmount: principal * initialPPGPrice / priceDenominator,
                    expiration: BigInt(await time.latest()) + 86400n,
                    fee: position.feesToBePaid,
                    functionCallDataList,
                    existingPosition: position,
                    referrer: zeroAddress
                };
                const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, openPositionRequest);

                await expect(wasabiShortPool.write.openPosition([openPositionRequest, signature], { value: totalAmountIn, account: user2.account }))
                    .to.be.rejectedWith("SenderNotTrader", "Cannot increase position on behalf of other traders");
            });
        });

        describe("Add Collateral", function () {
            it("InsufficientAmountProvided", async function () {
                const { wasabiShortPool, user1, totalAmountIn, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const request: AddCollateralRequest = {
                    amount: 0n,
                    interest: 0n,
                    expiration: BigInt(await time.latest()) + 86400n,
                    position
                }
                const signature = await signAddCollateralRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.addCollateral([request, signature], { value: totalAmountIn, account: user1.account }))
                    .to.be.rejectedWith("InsufficientAmountProvided", "Cannot add 0 collateral");
            });

            it("InvalidInterestAmount", async function () {
                const { wasabiShortPool, user1, totalAmountIn, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const request: AddCollateralRequest = {
                    amount: totalAmountIn,
                    interest: 1n,
                    expiration: BigInt(await time.latest()) + 86400n,
                    position
                }
                const signature = await signAddCollateralRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.addCollateral([request, signature], { value: totalAmountIn, account: user1.account }))
                    .to.be.rejectedWith("InvalidInterestAmount", "Cannot pay interest when adding collateral to short position");
            });

            it("InvalidPosition", async function () {
                const { wasabiShortPool, user1, totalAmountIn, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
                
                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const request: AddCollateralRequest = {
                    amount: totalAmountIn,
                    interest: 0n,
                    expiration: BigInt(await time.latest()) + 86400n,
                    position: {...position, id: position.id + 1n}
                }
                const signature = await signAddCollateralRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.addCollateral([request, signature], { value: totalAmountIn, account: user1.account }))
                    .to.be.rejectedWith("InvalidPosition", "Cannot add collateral to a position with a different ID");
            });

            it("SenderNotTrader", async function () {
                const { wasabiShortPool, user2, totalAmountIn, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
                
                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const request: AddCollateralRequest = {
                    amount: totalAmountIn,
                    interest: 0n,
                    expiration: BigInt(await time.latest()) + 86400n,
                    position
                }
                const signature = await signAddCollateralRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.addCollateral([request, signature], { value: totalAmountIn, account: user2.account }))
                    .to.be.rejectedWith("SenderNotTrader", "Cannot add collateral on behalf of other traders");
            })

            it("OrderExpired", async function () {
                const { wasabiShortPool, user1, totalAmountIn, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const request: AddCollateralRequest = {
                    amount: totalAmountIn,
                    interest: 0n,
                    expiration: BigInt(await time.latest()) - 1n,
                    position
                }
                const signature = await signAddCollateralRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.addCollateral([request, signature], { value: totalAmountIn, account: user1.account }))
                    .to.be.rejectedWith("OrderExpired", "Cannot add collateral to a position if the order is expired");
            })
        });

        describe("Remove Collateral", function () {
            it("InsufficientAmountProvided", async function () {
                const { wasabiShortPool, user1, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const request: RemoveCollateralRequest = {  
                    amount: 0n,
                    expiration: BigInt(await time.latest()) + 86400n,
                    position
                }
                const signature = await signRemoveCollateralRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.removeCollateral([request, signature], { account: user1.account }))
                    .to.be.rejectedWith("InsufficientAmountProvided", "Cannot remove 0 collateral");
            });

            it("OrderExpired", async function () {
                const { wasabiShortPool, user1, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const request: RemoveCollateralRequest = {
                    amount: position.collateralAmount,
                    expiration: BigInt(await time.latest()) - 1n,
                    position
                }
                const signature = await signRemoveCollateralRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.removeCollateral([request, signature], { account: user1.account }))
                    .to.be.rejectedWith("OrderExpired", "Cannot remove collateral if the order is expired");
            });

            it("InvalidPosition", async function () {
                const { wasabiShortPool, user1, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const request: RemoveCollateralRequest = {
                    amount: position.collateralAmount,
                    expiration: BigInt(await time.latest()) + 86400n,
                    position: {...position, id: position.id + 1n}
                }
                const signature = await signRemoveCollateralRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.removeCollateral([request, signature], { account: user1.account }))
                    .to.be.rejectedWith("InvalidPosition", "Cannot remove collateral to a position with a different ID");
            });

            it("SenderNotTrader", async function () {
                const { wasabiShortPool, user2, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const request: RemoveCollateralRequest = {
                    amount: 1n,
                    expiration: BigInt(await time.latest()) + 86400n,
                    position
                }
                const signature = await signRemoveCollateralRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.removeCollateral([request, signature], { account: user2.account }))
                    .to.be.rejectedWith("SenderNotTrader", "Cannot remove collateral to a position on behalf of other traders");
            });

            it("TooMuchCollateralSpent", async function () {
                const { wasabiShortPool, user1, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
                
                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later
                
                const request: RemoveCollateralRequest = {
                    amount: position.collateralAmount + 1n,
                    expiration: BigInt(await time.latest()) + 86400n,
                    position
                }
                const signature = await signRemoveCollateralRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.removeCollateral([request, signature], { account: user1.account }))
                    .to.be.rejectedWith("TooMuchCollateralSpent", "Too much collateral");
            });
        });

        describe("Decrease Position", function () {
            it("InsufficientPrincipalRepaid", async function () {
                const { createClosePositionRequest, signClosePositionRequest, wasabiShortPool, orderSigner, user1, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

                const { position } = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const request = await createClosePositionRequest({ position, amount: position.principal / 2n });
                // Increase amount to buy to be more than what will be bought using the generated swap function call
                request.amount = position.principal;
                const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account }))
                    .to.be.rejectedWith("InsufficientPrincipalRepaid", "Too much collateral");
            });

            it("ValueDeviatedTooMuch - Interest Paid", async function () {
                const { computeMaxInterest, createClosePositionRequest, signClosePositionRequest, wasabiShortPool, orderSigner, user1, mockSwap, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
    
                const { position } = await sendDefaultOpenPositionRequest();
    
                await time.increase(86400n); // 1 day later
    
                const interest = (await computeMaxInterest(position)) / 2n;
                const amount = position.principal / 2n;
                const request = await createClosePositionRequest({ position, interest });
                request.interest = interest * 105n / 100n; // Change interest to be invalid
                const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
    
                await expect(wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account }))
                    .to.be.rejectedWith("ValueDeviatedTooMuch", "Interest amount is invalid");
            });
        });
    });

    describe("Close Position Validations", function () {
        it("ValueDeviatedTooMuch - Interest Paid", async function () {
            const { computeMaxInterest, createClosePositionRequest, signClosePositionRequest, createSignedClosePositionRequest, wasabiShortPool, orderSigner, user1, totalAmountIn, maxLeverage, owner, tradeFeeValue, contractName, openPositionRequest, initialPPGPrice, priceDenominator, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

            const { position } = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            // Close Position
            const interest = (await computeMaxInterest(position)) / 2n;
            const request = await createClosePositionRequest({ position, interest });
            request.interest = interest * 105n / 100n; // Change interest to be invalid
            const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

            await expect(wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account }))
                .to.be.rejectedWith("ValueDeviatedTooMuch", "Interest amount is invalid");
        });

        it("TooMuchCollateralSpent", async function () {
            const { computeMaxInterest, createClosePositionRequest, signClosePositionRequest, createSignedClosePositionRequest, wasabiShortPool, orderSigner, user1, totalAmountIn, maxLeverage, owner, tradeFeeValue, contractName, openPositionRequest, initialPPGPrice, priceDenominator, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

            const { position } = await sendDefaultOpenPositionRequest();

            // Open extra positions to get more collateral in to the pool
            await sendDefaultOpenPositionRequest(2n);
            await sendDefaultOpenPositionRequest(3n);

            await time.increase(8640000n); // lots of time later, so that more collateral is spent to purchase debt (principal + interest) back

            // Close Position
            const interest = (await computeMaxInterest(position)) / 2n;
            const request = await createClosePositionRequest({ position, interest });
            const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

            await expect(wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account }))
                .to.be.rejectedWith("TooMuchCollateralSpent", "Too much collateral");
        });

        it("InsufficientPrincipalRepaid", async function () {
            const { sendDefaultOpenPositionRequest, orderSigner, user1, contractName, wasabiShortPool, mockSwap } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            // Craft a ClosePositionRequest with a malicious swap function call using MockSwap.swapExact
            const request: ClosePositionRequest = {
                expiration: BigInt(await time.latest()) + 300n,
                interest: 0n,
                amount: 0n,
                position,
                functionCallDataList: getApproveAndSwapFunctionCallDataExact(mockSwap.address, position.collateralCurrency, position.currency, position.collateralAmount, 1n), // bad amountOut
                referrer: zeroAddress
            };
            const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

            await expect(wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account }))
                .to.be.rejectedWith("InsufficientPrincipalRepaid");
        })
    });
});