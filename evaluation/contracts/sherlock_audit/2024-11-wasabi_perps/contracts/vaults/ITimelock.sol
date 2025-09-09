// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

interface ITimelock {
    error InsufficientCooldown(); // 0x1a731408
    error InvalidCooldownAmount(); // 0x8e99de94

    event CooldownStarted(address indexed account, uint256 amount);
    event CooldownDurationUpdated(uint256 oldDuration, uint256 newDuration);

    struct Cooldown {
        uint256 cooldownStart;
        uint256 amount;
    }

    struct TimelockStorage {
        uint256 cooldownDuration;
        mapping(address => Cooldown[]) cooldowns;
        mapping(address => uint256) nextCooldownIndex;
    }
    
    function getCooldowns(address account) external view returns (Cooldown[] memory);

    function getCooldownDuration() external view returns (uint256);

    function setCooldownDuration(uint256 duration) external;

    function startCooldown(uint256 amount) external;
}