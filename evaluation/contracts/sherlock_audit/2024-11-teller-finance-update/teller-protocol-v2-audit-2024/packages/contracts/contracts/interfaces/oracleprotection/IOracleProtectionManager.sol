// SPDX-Licence-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

interface IOracleProtectionManager {
	function isOracleApproved(address _msgSender) external returns (bool) ;
		 
    function isOracleApprovedAllowEOA(address _msgSender) external returns (bool);

}