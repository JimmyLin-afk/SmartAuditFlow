pragma solidity >=0.8.0 <0.9.0;
// SPDX-License-Identifier: MIT

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";

contract TestERC721Token is ERC721 {
    uint256 public _totalSupply;

    constructor(string memory _name, string memory _symbol)
        ERC721(_name, _symbol)
    {}

    function mint(address recipient) public returns (uint256) {
        uint256 tokenId = _totalSupply++;
        _mint(recipient, tokenId);
        return tokenId;
    }
}
