pragma solidity >=0.8.0 <0.9.0;
// SPDX-License-Identifier: MIT

import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";

contract TestERC1155Token is ERC1155 {
    uint256 public _totalSupply;

    constructor(string memory uri) ERC1155(uri) {}

    function mint(address recipient) public returns (uint256) {
        return mint(recipient, 1);
    }

    function mint(address recipient, uint256 amount) public returns (uint256) {
        return mint(recipient, amount, "0x");
    }

    function mint(address recipient, uint256 amount, bytes memory data)
        public
        returns (uint256)
    {
        uint256 tokenId = _totalSupply++;
        _mint(recipient, tokenId, amount, data);
        return tokenId;
    }
}
