#!/bin/bash
##############################################################################
# Sometimes when hitting AST compiler errors, the output doesn't tell you
# what file is causing the issue. This script helps you identify which
# contract is failing by compiling each contract individually.
##############################################################################

# Get all contract files
IFS=$'\n' read -r -d '' -a contract_files < <(find contracts test -name "*.sol" && printf '\0')

# Initialize an array to store contracts that need --via-ir
via_ir_contracts=()

# Function to check if a contract builds without --via-ir
check_contract() {
    local target_contract=$1
    local skip_contracts=()

    for contract in "${contract_files[@]}"; do
        if [ "$contract" != "$target_contract" ]; then
            skip_contracts+=("$contract")
        fi
    done

    forge clean
    if forge build --skip ${skip_contracts[*]}; then
        return 0
    else
        return 1
    fi
}

# Iterate through each contract
for contract in "${contract_files[@]}"; do
    echo "----------------------------------------"
    echo "Checking $contract..."
    if check_contract "$contract"; then
        echo "$contract does not require --via-ir"
    else
        echo "$contract requires --via-ir"
        via_ir_contracts+=("$contract")
    fi
    echo "----------------------------------------"
    echo
done

# Print the results
if [ ${#via_ir_contracts[@]} -eq 0 ]; then
    echo "All contracts can be built without --via-ir."
else
    echo "The following contracts require --via-ir:"
    printf '%s\n' "${via_ir_contracts[@]}"
    echo
    echo "Use the following command to build:"
    echo -n "forge build --via-ir --skip "
    
    contracts_to_skip=()
    for contract in "${contract_files[@]}"; do
        if [[ ! " ${via_ir_contracts[*]} " =~ " ${contract} " ]]; then
            contracts_to_skip+=("$contract")
        fi
    done
    
    echo "${contracts_to_skip[*]} test/**/*"
fi
