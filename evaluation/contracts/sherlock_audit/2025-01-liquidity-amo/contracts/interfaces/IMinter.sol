// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

interface IMinter {
    // State Variables
    function boostAddress() external view returns (address);

    function collateralAddress() external view returns (address);

    function treasury() external view returns (address);

    function boostDecimals() external view returns (uint8);

    function collateralDecimals() external view returns (uint8);

    // Events
    event TokenAddressesUpdated(address indexed boostAddress, address indexed collateralAddress);
    event TreasuryUpdated(address newTreasury);
    event TokenMinted(address indexed user, address indexed to, uint256 amount);
    event TokenProtocolMinted(address indexed user, address indexed to, uint256 amount);
    event TokenWithdrawn(address indexed tokenAddress, uint256 amount);

    // Function Signatures
    function pause() external;

    function unpause() external;

    function setTokens(address boost, address collateral) external;

    function setTreasury(address treasury) external;

    function mint(address to, uint256 amount) external;

    function protocolMint(address to, uint256 amount) external;

    function withdrawToken(address token, uint256 amount) external;
}
