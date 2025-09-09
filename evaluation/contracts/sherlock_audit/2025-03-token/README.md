# Symm Token

# Overview

## Symmio
The **Symmio** token contract is an ERC20 token contract with additional capabilities:

### Roles and Permissions
- `DEFAULT_ADMIN_ROLE`: Full control, typically assigned to a timelock contract.
- `MINTER_ROLE`: Permission to mint tokens.

## SymmAllocationClaimer
The **SymmAllocationClaimer** contract manages user token allocations and facilitates claiming processes.

### Roles and Permissions
- `DEFAULT_ADMIN_ROLE`: Full control over the contract.
- `SETTER_ROLE`: Permission to set user allocations and related parameters.
- `PAUSER_ROLE`: Permission to pause the contract.
- `UNPAUSER_ROLE`: Permission to unpause the contract.
- `MINTER_ROLE`: Permission to claim admin-allocated tokens.

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- [Node.js](https://nodejs.org/) (LTS version recommended)
- [Hardhat](https://hardhat.org/)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

### Configuration
1. Create a `.env` file in the project root to store environment variables such as private keys. Example:
   ```
   PRIVATE_KEY=your-private-key
   ```

### Compilation
Compile the contracts using Hardhat:
```bash
npx hardhat compile
```

### Testing
Run the tests to ensure the contracts function as expected:
```bash
npx hardhat test
```
