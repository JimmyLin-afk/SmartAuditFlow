#!/bin/bash

chains=(84532)

# Loop over each chainID
for chain in ${chains[@]}; do

  output_folder="abi/"
  input_file="broadcast/Deploy.s.sol/$chain/run-latest.json"

  # Create the output folder if it does not exist
  mkdir -p "$output_folder"

  # Extract contract names and addresses using jq
  contracts=$(jq -c '.transactions[] | {name: .contractName}' "$input_file")

  # Loop over each contract and create a JSON file with the contract address
  for contract in $contracts; do
    name=$(echo "$contract" | jq -r '.name')
    contract_json_file="out/$name.sol/$name.json"

    abi=$(jq -r '.abi' "$contract_json_file")

    echo "$abi" > "$output_folder/${name}.json"
  done

  echo "[DONE]: ABI extracted to 'packages/contracts/abi' directory."
done
