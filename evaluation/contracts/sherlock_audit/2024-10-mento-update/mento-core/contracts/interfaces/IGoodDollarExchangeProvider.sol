// SPDX-License-Identifier: MIT
pragma solidity >=0.5.17 <0.8.19;
pragma experimental ABIEncoderV2;

interface IGoodDollarExchangeProvider {
  /* ========================================== */
  /* ================= Events ================= */
  /* ========================================== */

  /**
   * @notice Emitted when the ExpansionController address is updated.
   * @param expansionController The address of the ExpansionController contract.
   */
  event ExpansionControllerUpdated(address indexed expansionController);

  /**
   * @notice Emitted when the GoodDollar DAO address is updated.
   * @param AVATAR The address of the GoodDollar DAO contract.
   */
  // solhint-disable-next-line var-name-mixedcase
  event AvatarUpdated(address indexed AVATAR);

  /**
   * @notice Emitted when the reserve ratio for a pool is updated.
   * @param exchangeId The id of the pool.
   * @param reserveRatio The new reserve ratio.
   */
  event ReserveRatioUpdated(bytes32 indexed exchangeId, uint32 reserveRatio);

  /* =========================================== */
  /* ================ Functions ================ */
  /* =========================================== */

  /**
   * @notice Initializes the contract with the given parameters.
   * @param _broker The address of the Broker contract.
   * @param _reserve The address of the Reserve contract.
   * @param _expansionController The address of the ExpansionController contract.
   * @param _avatar The address of the GoodDollar DAO contract.
   */
  function initialize(address _broker, address _reserve, address _expansionController, address _avatar) external;

  /**
   * @notice Sets the address of the GoodDollar DAO contract.
   * @param _avatar The address of the DAO contract.
   */
  function setAvatar(address _avatar) external;

  /**
   * @notice Sets the address of the Expansion Controller contract.
   * @param _expansionController The address of the Expansion Controller contract.
   */
  function setExpansionController(address _expansionController) external;

  /**
   * @notice Calculates the amount of G$ tokens to be minted as a result of the expansion.
   * @param exchangeId The ID of the pool to calculate the expansion for.
   * @param reserveRatioScalar Scaler for calculating the new reserve ratio.
   * @return amountToMint Amount of G$ tokens to be minted as a result of the expansion.
   */
  function mintFromExpansion(bytes32 exchangeId, uint256 reserveRatioScalar) external returns (uint256 amountToMint);

  /**
   * @notice Calculates the amount of G$ tokens to be minted as a result of the collected reserve interest.
   * @param exchangeId The ID of the pool the collected reserve interest is added to.
   * @param reserveInterest The amount of reserve asset tokens collected from interest.
   * @return amountToMint The amount of G$ tokens to be minted as a result of the collected reserve interest.
   */
  function mintFromInterest(bytes32 exchangeId, uint256 reserveInterest) external returns (uint256 amountToMint);

  /**
   * @notice Calculates the reserve ratio needed to mint the given G$ reward.
   * @param exchangeId The ID of the pool the G$ reward is minted from.
   * @param reward The amount of G$ tokens to be minted as a reward.
   */
  function updateRatioForReward(bytes32 exchangeId, uint256 reward) external;

  /**
   * @notice Pauses the Exchange, disabling minting.
   */
  function pause() external;

  /**
   * @notice Unpauses the Exchange, enabling minting again.
   */
  function unpause() external;
}
