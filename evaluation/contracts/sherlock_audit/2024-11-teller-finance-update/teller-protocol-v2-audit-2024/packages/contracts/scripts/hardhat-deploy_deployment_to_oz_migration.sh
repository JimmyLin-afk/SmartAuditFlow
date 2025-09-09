#!/bin/bash

prev_dir=$(pwd)
cd /home/passa/Documents/projects/Teller/teller-protocol-v2/packages/contracts

network="goerli"
oz_file=".openzeppelin/$network.json"

cat $oz_file | jq '.admin.address' | tr -d '"' | while read addr; do
  file=$(grep -rl "$addr" "deployments/$network" | grep 'DefaultProxyAdmin.json')
  basename $file

  cat $file | jq '.receipt.transactionHash' | tr -d '"' | while read txHash; do
    jq ".admin += { \"txHash\": \"$txHash\" }" "$oz_file" > tmp.$$.json && mv tmp.$$.json "$oz_file"
  done

  echo '--'
done

echo

i=0
cat $oz_file | jq '.proxies[].address' | tr -d '"' | while read addr; do
  file=$(grep -rl "$addr" "deployments/$network" | grep 'Proxy.json')
  echo "   Name: $(basename "$file")"

  cat $file | jq '.receipt.transactionHash' | tr -d '"' | while read txHash; do
    echo "Address: $addr"
    echo "Tx Hash: $txHash"
    jq ".proxies[$i] += { \"txHash\": \"$txHash\" }" "$oz_file" > tmp.$$.json && mv tmp.$$.json "$oz_file"
  done

  ((i++))
  echo '--'
done

echo

cat $oz_file | jq '.impls | to_entries | .[].key' | tr -d '"' | while read hash; do
  cat $oz_file | jq ".impls.\"$hash\".address" | tr -d '"' | while read addr; do
    file=$(grep -rl "$addr" "deployments/$network" | grep 'Implementation.json')
    echo "   Name: $(basename "$file")"

    cat $file | jq '.receipt.transactionHash' | tr -d '"' | while read txHash; do
      echo "Address: $addr"
      echo "Tx Hash: $txHash"
      jq ".impls.\"$hash\" += { \"txHash\": \"$txHash\" }" "$oz_file" > tmp.$$.json && mv tmp.$$.json "$oz_file"
    done

    echo '--'
  done
done

cd "$prev_dir"
