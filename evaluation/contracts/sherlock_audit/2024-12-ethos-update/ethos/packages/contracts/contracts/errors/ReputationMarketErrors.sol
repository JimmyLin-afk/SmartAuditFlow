// SPDX-License-Identifier: MIT
pragma solidity 0.8.26;
enum MarketCreationErrorCode {
  PROFILE_NOT_AUTHORIZED,
  PROFILE_MISMATCH
}

error InvalidProfileId();
error MarketAlreadyExists(uint256 profileId);
error MarketDoesNotExist(uint256 profileId);
error MarketCreationUnauthorized(
  MarketCreationErrorCode code,
  address addressStr,
  uint256 profileId
);
error InsufficientFunds();
error InsufficientLiquidity(uint256 creationCost);
error InsufficientVotesOwned(uint256 profileId, address addressStr);
error InsufficientVotesToSell(uint256 profileId);
error InvalidMarketConfigOption(string message);
error FeeTransferFailed(string message);
error InactiveMarket(uint256 profileId);
error UnauthorizedGraduation();
error UnauthorizedWithdrawal();
error MarketNotGraduated();
error NoFundsToWithdraw();
error ZeroAddressNotAllowed();
error SellSlippageLimitExceeded(uint256 minimumPricePerVote, uint256 actualPricePerVote);
