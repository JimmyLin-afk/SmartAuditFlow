// SPDX-Licence-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

import { Collateral } from "./escrow/ICollateralEscrowV1.sol";

interface ITellerV2MarketForwarder {
    struct CreateLoanArgs {
        uint256 marketId;
        address lendingToken;
        uint256 principal;
        uint32 duration;
        uint16 interestRate;
        string metadataURI;
        address recipient;
        Collateral[] collateral;
    }
}
