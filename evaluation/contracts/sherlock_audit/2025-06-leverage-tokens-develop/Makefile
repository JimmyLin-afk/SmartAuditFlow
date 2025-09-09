# include .env file and export its env vars
# (-include to ignore error if it does not exist)
-include .env

.PHONY: test clean

# Build & test
build                   :; forge build
coverage                :; forge coverage
coverage-export         :; forge coverage --report lcov && genhtml lcov.info -o report --rc derive_function_end_line=0
gas                     :; forge test --gas-report
gas-check               :; forge snapshot --check --tolerance 1
snapshot                :; forge snapshot
clean                   :; forge clean
fmt                     :; forge fmt
test                    :; forge test -vvvv --gas-report
test-lite               :; FOUNDRY_PROFILE=lite forge test -vvvv

# Deploy
deploy-core-tenderly		:; forge script script/Core.s.sol:CoreDeploy --force --rpc-url tenderly --account ${DEPLOYER_ACCOUNT_NAME} --sender ${DEPLOYER_ACCOUNT_ADDRESS} --slow --broadcast -vvvv --verify --verifier-url ${TENDERLY_VERIFIER_URL} --etherscan-api-key ${TENDERLY_VERIFIER_KEY}
deploy-periphery-tenderly	:; forge script script/Periphery.s.sol:PeripheryDeploy --force --rpc-url tenderly --account ${DEPLOYER_ACCOUNT_NAME} --sender ${DEPLOYER_ACCOUNT_ADDRESS} --slow --broadcast -vvvv --verify --verifier-url ${TENDERLY_VERIFIER_URL} --etherscan-api-key ${TENDERLY_VERIFIER_KEY}
create-leverage-token-tenderly	:; forge script script/CreateLeverageToken.${symbol}.s.sol:CreateLeverageToken --force --rpc-url tenderly --account ${DEPLOYER_ACCOUNT_NAME} --sender ${DEPLOYER_ACCOUNT_ADDRESS} --slow --broadcast -vvvv --verify --verifier-url ${TENDERLY_VERIFIER_URL} --etherscan-api-key ${TENDERLY_VERIFIER_KEY}

deploy-core-base		:; forge script script/Core.s.sol:CoreDeploy --force --rpc-url base --account ${DEPLOYER_ACCOUNT_NAME} --sender ${DEPLOYER_ACCOUNT_ADDRESS} --slow --broadcast -vvvv --verify --verifier-url ${BASE_VERIFIER_URL} --etherscan-api-key ${BASE_VERIFIER_KEY}
deploy-periphery-base	:; forge script script/Periphery.s.sol:PeripheryDeploy --force --rpc-url base --account ${DEPLOYER_ACCOUNT_NAME} --sender ${DEPLOYER_ACCOUNT_ADDRESS} --slow --broadcast -vvvv  --verify --verifier-url ${BASE_VERIFIER_URL} --etherscan-api-key ${BASE_VERIFIER_KEY}
create-leverage-token-base	:; forge script script/CreateLeverageToken.${symbol}.s.sol:CreateLeverageToken --force --rpc-url base --account ${DEPLOYER_ACCOUNT_NAME} --sender ${DEPLOYER_ACCOUNT_ADDRESS} --slow --broadcast -vvvv --verify --verifier-url ${BASE_VERIFIER_URL} --etherscan-api-key ${BASE_VERIFIER_KEY}
deploy-pricing-adapter-base :; forge script script/DeployPricingAdapter.s.sol:DeployPricingAdapter --force --rpc-url base --account ${DEPLOYER_ACCOUNT_NAME} --sender ${DEPLOYER_ACCOUNT_ADDRESS} --slow --broadcast -vvvv --verify --verifier-url ${BASE_VERIFIER_URL} --etherscan-api-key ${BASE_VERIFIER_KEY}