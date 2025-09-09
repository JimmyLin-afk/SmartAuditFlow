#!/usr/bin/env bash

yarn clean

## Compile Contracts ##
yarn compile

## Copy Contract Artifacts
mkdir -p build/contracts
find generated/artifacts/contracts/ \
  -type d \( -name "mock" -o -name "interfaces" \) -prune -o \
  -type f -name "*dbg.json" -prune -o \
  -type f -name "*.json" \
  -exec bash -c 'rsync --mkpath "$0" build/"$(dirname $(grep -o "contracts/.*" <<< "$0"))"/' {} \;

## Generate Contract Typings ##
cp -r generated/typechain build/typechain

## Export Contract Deployments ##
hardhat_dir=build/hardhat
contracts_export_file=$hardhat_dir/contracts.json
mkdir -p $hardhat_dir
echo '{}' > $contracts_export_file
yarn hardhat export --export-all $contracts_export_file
json=$(cat $contracts_export_file)
echo "$json" | jq '. |= del(."31337")' | jq '. |= with_entries({ key: .key, value: .value | to_entries | .[].value })' > $contracts_export_file

## Copy Math Library Helpers ##
yarn tsc -p teller-math-lib/tsconfig.json --outDir build/math
