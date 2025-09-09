// SPDX-License-Identifier: MIT
pragma solidity >=0.5.17 <0.8.19;
pragma experimental ABIEncoderV2;

interface IGoodDollarExpansionController {
  /**
   * @notice Struct holding the configuration for the expansion of an exchange.
   * @param expansionRate The rate of expansion in percentage with 1e18 being 100%.
   * @param expansionFrequency The frequency of expansion in seconds.
   * @param lastExpansion The timestamp of the last prior expansion.
   */
  struct ExchangeExpansionConfig {
    uint64 expansionRate;
    uint32 expansionFrequency;
    uint32 lastExpansion;
  }

  /* ------- Events ------- */

  /**
   * @notice Emitted when the GoodDollarExchangeProvider is updated.
   * @param exchangeProvider The address of the new GoodDollarExchangeProvider.
   */
  event GoodDollarExchangeProviderUpdated(address indexed exchangeProvider);

  /**
   * @notice Emitted when the distribution helper is updated.
   * @param distributionHelper The address of the new distribution helper.
   */
  event DistributionHelperUpdated(address indexed distributionHelper);

  /**
   * @notice Emitted when the Reserve address is updated.
   * @param reserve The address of the new Reserve.
   */
  event ReserveUpdated(address indexed reserve);

  /**
   * @notice Emitted when the GoodDollar DAO address is updated.
   * @param avatar The new address of the GoodDollar DAO.
   */
  event AvatarUpdated(address indexed avatar);

  /**
   * @notice Emitted when the expansion config is set for an pool.
   * @param exchangeId The ID of the pool.
   * @param expansionRate The rate of expansion.
   * @param expansionFrequency The frequency of expansion.
   */
  event ExpansionConfigSet(bytes32 indexed exchangeId, uint64 expansionRate, uint32 expansionFrequency);

  /**
   * @notice Emitted when a G$ reward is minted.
   * @param exchangeId The ID of the pool.
   * @param to The address of the recipient.
   * @param amount The amount of G$ tokens minted.
   */
  event RewardMinted(bytes32 indexed exchangeId, address indexed to, uint256 amount);

  /**
   * @notice Emitted when UBI is minted through collecting reserve interest.
   * @param exchangeId The ID of the pool.
   * @param amount The amount of G$ tokens minted.
   */
  event InterestUBIMinted(bytes32 indexed exchangeId, uint256 amount);

  /**
   * @notice Emitted when UBI is minted through expansion.
   * @param exchangeId The ID of the pool.
   * @param amount The amount of G$ tokens minted.
   */
  event ExpansionUBIMinted(bytes32 indexed exchangeId, uint256 amount);

  /* ------- Functions ------- */

  /**
   * @notice Initializes the contract with the given parameters.
   * @param _goodDollarExchangeProvider The address of the GoodDollarExchangeProvider contract.
   * @param _distributionHelper The address of the distribution helper contract.
   * @param _reserve The address of the Reserve contract.
   * @param _avatar The address of the GoodDollar DAO contract.
   */
  function initialize(
    address _goodDollarExchangeProvider,
    address _distributionHelper,
    address _reserve,
    address _avatar
  ) external;

  /**
   * @notice Returns the expansion config for the given exchange.
   * @param exchangeId The id of the exchange to get the expansion config for.
   * @return config The expansion config.
   */
  function getExpansionConfig(bytes32 exchangeId) external returns (ExchangeExpansionConfig memory);

  /**
   * @notice Sets the GoodDollarExchangeProvider address.
   * @param _goodDollarExchangeProvider The address of the GoodDollarExchangeProvider contract.
   */
  function setGoodDollarExchangeProvider(address _goodDollarExchangeProvider) external;

  /**
   * @notice Sets the distribution helper address.
   * @param _distributionHelper The address of the distribution helper contract.
   */
  function setDistributionHelper(address _distributionHelper) external;

  /**
   * @notice Sets the reserve address.
   * @param _reserve The address of the reserve contract.
   */
  function setReserve(address _reserve) external;

  /**
   * @notice Sets the AVATAR address.
   * @param _avatar The address of the AVATAR contract.
   */
  function setAvatar(address _avatar) external;

  /**
   * @notice Sets the expansion config for the given pool.
   * @param exchangeId The ID of the pool to set the expansion config for.
   * @param expansionRate The rate of expansion.
   * @param expansionFrequency The frequency of expansion.
   */
  function setExpansionConfig(bytes32 exchangeId, uint64 expansionRate, uint32 expansionFrequency) external;

  /**
   * @notice Mints UBI as G$ tokens for a given pool from collected reserve interest.
   * @param exchangeId The ID of the pool to mint UBI for.
   * @param reserveInterest The amount of reserve tokens collected from interest.
   */
  function mintUBIFromInterest(bytes32 exchangeId, uint256 reserveInterest) external;

  /**
   * @notice Mints UBI as G$ tokens for a given pool by comparing the contract's reserve balance to the virtual balance.
   * @param exchangeId The ID of the pool to mint UBI for.
   * @return amountMinted The amount of G$ tokens minted.
   */
  function mintUBIFromReserveBalance(bytes32 exchangeId) external returns (uint256 amountMinted);

  /**
   * @notice Mints UBI as G$ tokens for a given pool by calculating the expansion rate.
   * @param exchangeId The ID of the pool to mint UBI for.
   * @return amountMinted The amount of G$ tokens minted.
   */
  function mintUBIFromExpansion(bytes32 exchangeId) external returns (uint256 amountMinted);

  /**
   * @notice Mints a reward of G$ tokens for a given pool.
   * @param exchangeId The ID of the pool to mint a G$ reward for.
   * @param to The address of the recipient.
   * @param amount The amount of G$ tokens to mint.
   */
  function mintRewardFromReserveRatio(bytes32 exchangeId, address to, uint256 amount) external;
}
