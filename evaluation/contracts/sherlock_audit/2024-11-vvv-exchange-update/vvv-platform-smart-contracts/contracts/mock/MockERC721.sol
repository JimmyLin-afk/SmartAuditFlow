// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";

contract MockERC721 is ERC721 {
    uint256 public tokenId;

    constructor() ERC721("MyToken", "MTK") {}

    function safeMint(address to) public {
        ++tokenId;
        _safeMint(to, tokenId);
    }
}
