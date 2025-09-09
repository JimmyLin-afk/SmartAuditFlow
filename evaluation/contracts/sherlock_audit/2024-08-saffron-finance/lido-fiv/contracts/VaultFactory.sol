// SPDX-License-Identifier: MIT

pragma solidity 0.8.18;

import "@openzeppelin/contracts/proxy/Clones.sol";
import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/utils/math/SafeCast.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "./LidoVault.sol";
import "./interfaces/ILidoVault.sol";

//import "./NonTransferrableVaultBearerToken.sol";

/// @title Saffron Fixed Income Vault Factory
/// @notice Configure and deploy vaults
contract VaultFactory is ILidoVaultInitializer, Ownable {
  /// @notice Master vault contract to clone from
  address public vaultContract;

  /// @notice Incrementing vault ID
  uint256 public nextVaultId = 1;

  /// @notice Protocol fee taken from earnings in basis points (one basis point = 1/100 of 1%)
  uint256 public protocolFeeBps;

  /// @notice Address that collects protocol fees
  address public protocolFeeReceiver;

  /// @notice Fixed income early exit fee in basis points (one basis point = 1/100 of 1%) that is quadratically scaled based off of early exit time
  uint256 public earlyExitFeeBps;

  struct VaultInfo {
    address creator;
    address addr;
  }

  /// @notice Info about vault, mapped by vault ID
  mapping(uint256 => VaultInfo) public vaultInfo;

  /// @notice ID of vault, mapped by vault address
  mapping(address => uint256) public vaultAddrToId;

  /// @notice Emitted when the vault bytecode is set
  /// @param creator Address of creator
  event VaultCodeSet(address indexed creator);

  /// @notice Emitted when a new vault is deployed
  /// @param vaultId ID of vault
  /// @param duration How long the vault will be locked once started, in seconds
  /// @param fixedSideCapacity Maximum capacity of fixed side
  /// @param variableSideCapacity Maximum capacity of variable side
  /// @param earlyExitFeeBps Fixed depositor early exit fee in basis points
  /// @param protocolFeeBps Protocol fee in basis points
  /// @param protocolFeeReceiver Address that collects protocol fee
  /// @param creator Address of vault creator
  /// @param vault Address of vault
  event VaultCreated(
    uint256 vaultId,
    uint256 duration,
    uint256 fixedSideCapacity,
    uint256 variableSideCapacity,
    uint256 earlyExitFeeBps,
    uint256 protocolFeeBps,
    address protocolFeeReceiver,
    address indexed creator,
    address indexed vault
  );

  /// @notice Emitted when the Factory owner changes earlyExitFeeBps
  /// @param earlyExitFeeBps Fixed income early exit fee in basis points (one basis point = 1/100 of 1%) that is quadratically scaled based off of early exit time
  event SetEarlyExitFeeBps(uint256 earlyExitFeeBps);

  /// @notice Emitted when the Factory owner changes protocolFeeReceiver
  /// @param protocolFeeReceiver Address that collects protocol fees
  event SetProtocolFeeReceiver(address protocolFeeReceiver);

  /// @notice Emitted when the Factory owner changes protocolFeeBps
  /// @param protocolFeeBps Protocol fee taken from earnings in basis points
  event SetProtocolFeeBps(uint256 protocolFeeBps);

  constructor(uint256 _protocolFeeBps, uint256 _earlyExitFeeBps) {
    require(_protocolFeeBps < 10_000, "IPB");
    protocolFeeReceiver = msg.sender;
    protocolFeeBps = _protocolFeeBps;
    earlyExitFeeBps = _earlyExitFeeBps;
    vaultContract = address(new LidoVault(true));
    emit VaultCodeSet(msg.sender);
  }

  function setProtocolFeeBps(uint256 _protocolFeeBps) external onlyOwner {
    require(_protocolFeeBps < 10_000, "IPB");
    protocolFeeBps = _protocolFeeBps;
    emit SetProtocolFeeBps(protocolFeeBps);
  }

  function setProtocolFeeReceiver(address _protocolFeeReceiver) external onlyOwner {
    protocolFeeReceiver = _protocolFeeReceiver;
    emit SetProtocolFeeReceiver(protocolFeeReceiver);
  }

  function setEarlyExitFeeBps(uint256 _earlyExitFeeBps) external onlyOwner {
    earlyExitFeeBps = _earlyExitFeeBps;
    emit SetEarlyExitFeeBps(earlyExitFeeBps);
  }

  /// @notice Deploys a new vault
  function createVault(
    uint256 _fixedSideCapacity,
    uint256 _duration,
    uint256 _variableSideCapacity
  ) public virtual {
    // Deploy vault (Note: this does not run constructor)
    address vaultAddress = Clones.clone(vaultContract);

    require(vaultAddress != address(0), "FTC");

    // Store vault info
    uint256 vaultId = nextVaultId++;
    vaultInfo[vaultId] = VaultInfo({creator: msg.sender, addr: vaultAddress});
    vaultAddrToId[vaultAddress] = vaultId;

    InitializationParams memory params = InitializationParams({
      vaultId: vaultId,
      duration: _duration,
      fixedSideCapacity: _fixedSideCapacity,
      variableSideCapacity: _variableSideCapacity,
      earlyExitFeeBps: earlyExitFeeBps,
      protocolFeeBps: protocolFeeBps,
      protocolFeeReceiver: protocolFeeReceiver
    });

    // Initialize vault
    ILidoVault(vaultAddress).initialize(params);

    emit VaultCreated(
      vaultId,
      _duration,
      _fixedSideCapacity,
      _variableSideCapacity,
      earlyExitFeeBps,
      protocolFeeBps,
      protocolFeeReceiver,
      msg.sender,
      vaultAddress
    );
  }

  /// @notice Check to see if a given vault was deployed by this factory
  /// @return True if address matches a vault deployed by this factory
  function wasDeployedByFactory(address addr) external view returns (bool) {
    return vaultAddrToId[addr] != 0;
  }
}
