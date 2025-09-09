.PHONY: abi deploy upgrade whitelist check slither mythril

ifdef ENV_FILE
include $(ENV_FILE)
endif

LATEST ?= V4
RPC_URL ?= http://127.0.0.1:8545
UPGRADE_TO ?= $(LATEST)

abi:
	cp out/Blueprint$(LATEST).sol/Blueprint$(LATEST).json artifacts/

deploy:
	forge script ./script/Deploy.s.sol --rpc-url $(RPC_URL) --broadcast --private-key $(PRIVATE_KEY)

upgrade:
	PROXY_ADDRESS=$(PROXY_ADDRESS) forge script ./script/Upgrade$(UPGRADE_TO).s.sol --rpc-url $(RPC_URL) --broadcast --private-key $(PRIVATE_KEY)

whitelist:
	PROXY_ADDRESS=$(PROXY_ADDRESS) forge script ./script/UpdateWhitelist.s.sol --rpc-url $(RPC_URL) --broadcast --private-key $(PRIVATE_KEY)

check:
	cast call --rpc-url $(RPC_URL) $(PROXY_ADDRESS) "VERSION()(string)"

slither:
	slither ./src/Blueprint$(LATEST).sol --checklist --filter-paths "openzeppelin"

mythril:
	myth analyze ./src/Blueprint$(LATEST).sol --solc-json solc_remappings.json
