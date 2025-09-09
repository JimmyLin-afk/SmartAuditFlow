pragma solidity ^0.8.0;

contract DynamicData {
    function getDynamicUintArray(
        uint256 x
    ) public pure returns (uint[] memory) {
        uint[] memory nftsID = new uint[](x);
        return nftsID;
    }

    function getDynamicAddressArray(
        uint256 x
    ) public pure returns (address[] memory) {
        address[] memory nftsID = new address[](x);
        return nftsID;
    }

    function getDynamicBoolArray(
        uint256 x
    ) public pure returns (bool[] memory) {
        bool[] memory nftsID = new bool[](x);
        return nftsID;
    }

    function getDynamicByteArray(uint256 x) public pure returns (bytes memory) {
        bytes memory nftsID = new bytes(x);
        return nftsID;
    }

    function getDynamicStringArray(
        uint256 x
    ) public pure returns (string[] memory) {
        string[] memory nftsID = new string[](x);
        return nftsID;
    }
}
