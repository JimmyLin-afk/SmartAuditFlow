pragma solidity ^0.8.0;

// SPDX-License-Identifier: MIT

import "../interfaces/ILenderManager.sol";

import "@openzeppelin/contracts-upgradeable/token/ERC721/ERC721Upgradeable.sol";

contract LenderManagerMock is ILenderManager, ERC721Upgradeable {
    //bidId => lender
    mapping(uint256 => address) public registeredLoan;

    constructor() {}

    function registerLoan(uint256 _bidId, address _newLender)
        external
        override
    {
        registeredLoan[_bidId] = _newLender;
    }

    function ownerOf(uint256 _bidId)
        public
        view
        override(ERC721Upgradeable, IERC721Upgradeable)
        returns (address)
    {
        return registeredLoan[_bidId];
    }
}
