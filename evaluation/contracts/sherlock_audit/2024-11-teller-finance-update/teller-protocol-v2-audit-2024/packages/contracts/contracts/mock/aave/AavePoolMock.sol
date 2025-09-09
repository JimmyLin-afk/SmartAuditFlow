pragma solidity ^0.8.0;

import { IFlashLoanSimpleReceiver } from "../../interfaces/aave/IFlashLoanSimpleReceiver.sol";

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract AavePoolMock {
    bool public flashLoanSimpleWasCalled;

    bool public shouldExecuteCallback = true;

    function setShouldExecuteCallback(bool shouldExecute) public {
        shouldExecuteCallback = shouldExecute;
    }

    function flashLoanSimple(
        address receiverAddress,
        address asset,
        uint256 amount,
        bytes calldata params,
        uint16 referralCode
    ) external returns (bool success) {
        uint256 balanceBefore = IERC20(asset).balanceOf(address(this));

        IERC20(asset).transfer(receiverAddress, amount);

        uint256 premium = amount / 100;
        address initiator = msg.sender;

        if (shouldExecuteCallback) {
            success = IFlashLoanSimpleReceiver(receiverAddress)
                .executeOperation(asset, amount, premium, initiator, params);

            require(success == true, "executeOperation failed");
        }

        IERC20(asset).transferFrom(
            receiverAddress,
            address(this),
            amount + premium
        );

        //require balance is what it was plus the fee..
        uint256 balanceAfter = IERC20(asset).balanceOf(address(this));

        require(
            balanceAfter >= balanceBefore + premium,
            "Must repay flash loan"
        );

        flashLoanSimpleWasCalled = true;
    }
}
