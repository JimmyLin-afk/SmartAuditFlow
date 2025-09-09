#!/usr/bin/env bash

action=$1
description=$(sed 's/[[:upper:]]/\L&/g;s/ /_/g' <<< "$2")
contract_name=$3
contract_method=$4

migration_id="$(date +%s)__$description"
template_file="$(dirname "$0")/templates/$action.mustache"
migration_dirname="$(dirname "$0")/../../deploy/migrations"
migration_file_name="$migration_id.ts"
migration_file="$migration_dirname/$migration_file_name"

yarn mustache - "$template_file" "$migration_file" <<EOF
{
  "id": "$migration_id",
  "contractName": "$contract_name",
  "contractMethod": "$contract_method"
}
EOF

if [ $? -eq 0 ]; then
  echo "Created migration file: $(realpath "$migration_file")"
else
  echo "Failed to create migration file"
fi
