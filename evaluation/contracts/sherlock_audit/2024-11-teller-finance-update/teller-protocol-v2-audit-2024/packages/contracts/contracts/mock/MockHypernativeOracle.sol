
import "../interfaces/oracleprotection/IHypernativeOracle.sol";

contract MockHypernativeOracle is IHypernativeOracle {
    mapping(address => bool) private blacklistedAccounts;
    mapping(address => bool) private blacklistedContexts;
    mapping(address => bool) private timeExceeded;

    function setBlacklistedAccount(address account, bool status) external {
        blacklistedAccounts[account] = status;
    }

    function setBlacklistedContext(address context, bool status) external {
        blacklistedContexts[context] = status;
    }

    function setTimeExceeded(address account, bool status) external {
        timeExceeded[account] = status;
    }

    function isBlacklistedAccount(address account) external view override returns (bool) {
        return blacklistedAccounts[account];
    }

    function isBlacklistedContext(address origin, address sender) external view override returns (bool) {
        return blacklistedContexts[origin];
    }

    function isTimeExceeded(address account) external view override returns (bool) {
        return timeExceeded[account];
    }

    function register(address account) external override {}

    function registerStrict(address account) external override {}
}
