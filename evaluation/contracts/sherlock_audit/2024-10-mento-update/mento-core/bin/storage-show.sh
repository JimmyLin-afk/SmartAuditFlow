#!/bin/bash
##############################################################################
# Show the storage layout of a contract.
# Usage: yarn storage:show <contract-name> <version>
# Example: yarn storage:show StableToken
##############################################################################

CONTRACT=$1
env forge inspect $CONTRACT storage-layout \
  | jq ".storage | map(\"\(.slot):\(.offset):\(.label):\(.type)\") | .[]" -r \
  | column -s: -t
