# VVV Smart Contracts
## Audit instructions
### Test & coverage
Run all commands in the root of the repository.

#### Initial setup
1. [Install npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
2. `$ npm install`
3. Install [Foundry](https://book.getfoundry.sh/getting-started/installation), or at least Forge

#### VC
To run tests and collect coverage for the VC Investment Ledger and Token Distributor contracts and their first party dependencies:
```
$ forge coverage --match-contract 'VVVVCInvestmentLedger*|VVVAuthorization*|VVVVCTokenDistributor*' | awk '!/^\| contracts|test|Total/ || /^\| contracts\/vc|auth\//'
```

## Contributing
Before contributing install the pre-commit hook by running the command below in the root of the repository.
```
npm run prepare
```
This will prevent you from getting blocked by the CI on your pull request.
