pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract DebitaIncentives {
    event Incentivized(
        address indexed principle,
        address indexed incentivizeToken,
        uint amount,
        bool lendIncentivize,
        uint epoch
    );

    event ClaimedIncentives(
        address indexed user,
        address indexed principle,
        address indexed incentivizeToken,
        uint amount,
        uint epoch
    );

    event UpdatedFunds(
        address indexed lenders,
        address indexed principle,
        address indexed collateral,
        address borrower,
        uint epoch
    );

    event WhitelistedPair(
        address indexed principle,
        address indexed collateral,
        bool whitelisted
    );

    uint public blockDeployedContract; // timestamp of deployment
    uint public epochDuration = 14 days; // duration of an epoch
    address owner;
    address aggregatorContract;

    struct infoOfOffers {
        address principle; // address of the principle
        address lendOffer; // address of the lend offer
        uint principleAmount; // amount of principle
        uint lenderID; // ID of the lender
        uint apr; // APR of the offer
        uint ratio; // ratio of the offer
        uint collateralUsed; // collateral used
        uint maxDeadline; // max deadline
        bool paid; // has been paid
        bool collateralClaimed; // has collateral been claimed
        bool debtClaimed; // total debt claimed
        uint interestToClaim; // available interest to claim
        uint interestPaid; // interest already paid
    }

    struct InfoOfBribePerPrinciple {
        address principle; // address of the principle
        address[] bribeToken; // address of the bribe tokens
        uint[] amountPerLent;
        uint[] amountPerBorrow;
        uint epoch;
    }
    /* 
    -------
    Lend Incentives
    -------
    */
    // principle => (keccack256(bribe token, epoch)) => total incentives amount
    mapping(address => mapping(bytes32 => uint))
        public lentIncentivesPerTokenPerEpoch;

    // wallet address => keccack256(principle + epoch) => amount lent
    mapping(address => mapping(bytes32 => uint))
        public lentAmountPerUserPerEpoch;

    /* 
    --------
    Borrow Incentives
    --------
    */
    // principle => keccack(bribe token, epoch) => amount per Token
    mapping(address => mapping(bytes32 => uint))
        public borrowedIncentivesPerTokenPerEpoch;

    // wallet address => keccack256(principle + epoch) => amount
    mapping(address => mapping(bytes32 => uint)) public borrowAmountPerEpoch;

    // principle => epoch => total lent amount

    mapping(address => mapping(uint => uint)) public totalUsedTokenPerEpoch;

    // wallet => keccack256(principle + epoch + bribe token)  => amount claimed
    mapping(address => mapping(bytes32 => bool)) public claimedIncentives;

    /* 
    Security check
    */

    // principle => collateral => is whitelisted
    mapping(address => mapping(address => bool)) public isPairWhitelisted;
    mapping(address => bool) public isPrincipleWhitelisted;

    /* MAPPINGS FOR READ FUNCTIONS */

    // epoch uint => index  => principle address
    mapping(uint => mapping(uint => address)) public epochIndexToPrinciple;
    // epoch uint => amount of principles incentivized
    mapping(uint => uint) public principlesIncentivizedPerEpoch;

    // epoch uint => principle address => has been indexed
    mapping(uint => mapping(address => bool)) public hasBeenIndexed;

    // epoch => keccak(principle address, index) => bribeToken
    mapping(uint => mapping(bytes32 => address))
        public SpecificBribePerPrincipleOnEpoch;

    // epoch => principle => amount of bribe Tokens
    mapping(uint => mapping(address => uint))
        public bribeCountPerPrincipleOnEpoch;

    // epoch => incentive token => bool has been indexed
    mapping(uint => mapping(address => bool)) public hasBeenIndexedBribe;

    modifier onlyAggregator() {
        require(msg.sender == aggregatorContract, "Only aggregator");
        _;
    }

    constructor() {
        owner = msg.sender;
        blockDeployedContract = block.timestamp;
    }

    /**
     * @dev Claim the incentives for the user
     * @param principles array of principles used during the epoch
     * @param tokensIncentives array of tokens to claim per principle
     * @param epoch epoch to claim
     */

    function claimIncentives(
        address[] memory principles,
        address[][] memory tokensIncentives,
        uint epoch
    ) public {
        // get information
        require(epoch < currentEpoch(), "Epoch not finished");

        for (uint i; i < principles.length; i++) {
            address principle = principles[i];
            uint lentAmount = lentAmountPerUserPerEpoch[msg.sender][
                hashVariables(principle, epoch)
            ];
            // get the total lent amount for the epoch and principle
            uint totalLentAmount = totalUsedTokenPerEpoch[principle][epoch];

            uint porcentageLent;

            if (lentAmount > 0) {
                porcentageLent = (lentAmount * 10000) / totalLentAmount;
            }

            uint borrowAmount = borrowAmountPerEpoch[msg.sender][
                hashVariables(principle, epoch)
            ];
            uint totalBorrowAmount = totalUsedTokenPerEpoch[principle][epoch];
            uint porcentageBorrow;

            require(
                borrowAmount > 0 || lentAmount > 0,
                "No borrowed or lent amount"
            );

            porcentageBorrow = (borrowAmount * 10000) / totalBorrowAmount;

            for (uint j = 0; j < tokensIncentives[i].length; j++) {
                address token = tokensIncentives[i][j];
                uint lentIncentive = lentIncentivesPerTokenPerEpoch[principle][
                    hashVariables(token, epoch)
                ];
                uint borrowIncentive = borrowedIncentivesPerTokenPerEpoch[
                    principle
                ][hashVariables(token, epoch)];
                require(
                    !claimedIncentives[msg.sender][
                        hashVariablesT(principle, epoch, token)
                    ],
                    "Already claimed"
                );
                require(
                    (lentIncentive > 0 && lentAmount > 0) ||
                        (borrowIncentive > 0 && borrowAmount > 0),
                    "No incentives to claim"
                );
                claimedIncentives[msg.sender][
                    hashVariablesT(principle, epoch, token)
                ] = true;

                uint amountToClaim = (lentIncentive * porcentageLent) / 10000;
                amountToClaim += (borrowIncentive * porcentageBorrow) / 10000;

                IERC20(token).transfer(msg.sender, amountToClaim);

                emit ClaimedIncentives(
                    msg.sender,
                    principle,
                    token,
                    amountToClaim,
                    epoch
                );
            }
        }
    }

    /**
     * @dev Incentivize the pair --> anyone can incentivze the pair but it's mainly thought for chain incentives or points system
        * @param principles array of principles to incentivize
        * @param incentiveToken array of tokens you want to give as incentives
        * @param lendIncentivize array of bools to know if you want to incentivize the lend or the borrow
        * @param amounts array of amounts to incentivize
        * @param epochs array of epochs to incentivize

     */
    function incentivizePair(
        address[] memory principles,
        address[] memory incentiveToken,
        bool[] memory lendIncentivize,
        uint[] memory amounts,
        uint[] memory epochs
    ) public {
        require(
            principles.length == incentiveToken.length &&
                incentiveToken.length == lendIncentivize.length &&
                lendIncentivize.length == amounts.length &&
                amounts.length == epochs.length,
            "Invalid input"
        );

        for (uint i; i < principles.length; i++) {
            uint epoch = epochs[i];
            address principle = principles[i];
            address incentivizeToken = incentiveToken[i];
            uint amount = amounts[i];
            require(epoch > currentEpoch(), "Epoch already started");
            require(isPrincipleWhitelisted[principle], "Not whitelisted");

            // if principles has been indexed into array of the epoch
            if (!hasBeenIndexed[epochs[i]][principles[i]]) {
                uint lastAmount = principlesIncentivizedPerEpoch[epochs[i]];
                epochIndexToPrinciple[epochs[i]][lastAmount] = principles[i];
                principlesIncentivizedPerEpoch[epochs[i]]++;
                hasBeenIndexed[epochs[i]][principles[i]] = true;
            }

            // if bribe token has been indexed into array of the epoch
            if (!hasBeenIndexedBribe[epoch][incentivizeToken]) {
                uint lastAmount = bribeCountPerPrincipleOnEpoch[epoch][
                    principle
                ];
                SpecificBribePerPrincipleOnEpoch[epoch][
                    hashVariables(principle, lastAmount)
                ] = incentivizeToken;
                bribeCountPerPrincipleOnEpoch[epoch][incentivizeToken]++;
                hasBeenIndexedBribe[epoch][incentivizeToken] = true;
            }

            // transfer the tokens
            IERC20(incentivizeToken).transferFrom(
                msg.sender,
                address(this),
                amount
            );
            require(amount > 0, "Amount must be greater than 0");

            // add the amount to the total amount of incentives
            if (lendIncentivize[i]) {
                lentIncentivesPerTokenPerEpoch[principle][
                    hashVariables(incentivizeToken, epoch)
                ] += amount;
            } else {
                borrowedIncentivesPerTokenPerEpoch[principle][
                    hashVariables(incentivizeToken, epoch)
                ] += amount;
            }
            emit Incentivized(
                principles[i],
                incentiveToken[i],
                amounts[i],
                lendIncentivize[i],
                epochs[i]
            );
        }
    }

    // Update the funds of the user and the total amount of the principle
    // -- only aggregator whenever a loan is matched

    /**
     * @dev Update the funds of the user and the total amount of the principle
     * @param informationOffers array of information of the offers
     * @param collateral address of the collateral
     * @param lenders array of lenders
     * @param borrower address of the borrower
     */
    function updateFunds(
        infoOfOffers[] memory informationOffers,
        address collateral,
        address[] memory lenders,
        address borrower
    ) public onlyAggregator {
        for (uint i = 0; i < lenders.length; i++) {
            bool validPair = isPairWhitelisted[informationOffers[i].principle][
                collateral
            ];
            if (!validPair) {
                return;
            }
            address principle = informationOffers[i].principle;

            uint _currentEpoch = currentEpoch();

            lentAmountPerUserPerEpoch[lenders[i]][
                hashVariables(principle, _currentEpoch)
            ] += informationOffers[i].principleAmount;
            totalUsedTokenPerEpoch[principle][
                _currentEpoch
            ] += informationOffers[i].principleAmount;
            borrowAmountPerEpoch[borrower][
                hashVariables(principle, _currentEpoch)
            ] += informationOffers[i].principleAmount;

            emit UpdatedFunds(
                lenders[i],
                principle,
                collateral,
                borrower,
                _currentEpoch
            );
        }
    }

    // Get the amount of principles incentivized and the amount of bribes per principle
    function getBribesPerEpoch(
        uint epoch,
        uint offset,
        uint limit
    ) public view returns (InfoOfBribePerPrinciple[] memory) {
        // get the amount of principles incentivized
        uint totalPrinciples = principlesIncentivizedPerEpoch[epoch];
        if (totalPrinciples == 0) {
            return new InfoOfBribePerPrinciple[](0);
        }
        if (offset > totalPrinciples) {
            return new InfoOfBribePerPrinciple[](0);
        }
        if (limit > totalPrinciples) {
            limit = totalPrinciples;
        }
        uint length = limit - offset;
        InfoOfBribePerPrinciple[] memory bribes = new InfoOfBribePerPrinciple[](
            length
        );

        for (uint i = 0; i < length; i++) {
            address principle = epochIndexToPrinciple[epoch][i + offset];
            uint totalBribes = bribeCountPerPrincipleOnEpoch[epoch][principle];
            address[] memory bribeToken = new address[](totalBribes);
            uint[] memory amountPerLent = new uint[](totalBribes);
            uint[] memory amountPerBorrow = new uint[](totalBribes);

            for (uint j = 0; j < totalBribes; j++) {
                address token = SpecificBribePerPrincipleOnEpoch[epoch][
                    hashVariables(principle, j)
                ];
                uint lentIncentive = lentIncentivesPerTokenPerEpoch[principle][
                    hashVariables(token, epoch)
                ];
                uint borrowIncentive = borrowedIncentivesPerTokenPerEpoch[
                    principle
                ][hashVariables(token, epoch)];

                bribeToken[j] = token;
                amountPerLent[j] = lentIncentive;
                amountPerBorrow[j] = borrowIncentive;
            }

            bribes[i] = InfoOfBribePerPrinciple(
                principle,
                bribeToken,
                amountPerLent,
                amountPerBorrow,
                epoch
            );
        }
        return bribes;
    }

    function setAggregatorContract(address _aggregatorContract) public {
        require(msg.sender == owner, "Only owner");
        require(aggregatorContract == address(0), "Already set");
        aggregatorContract = _aggregatorContract;
    }

    function whitelListCollateral(
        address _principle,
        address _collateral,
        bool whitelist
    ) public {
        require(msg.sender == owner, "Only owner");
        if (isPrincipleWhitelisted[_principle] == false && whitelist) {
            isPrincipleWhitelisted[_principle] = whitelist;
        }
        isPairWhitelisted[_principle][_collateral] = whitelist;
        emit WhitelistedPair(_principle, _collateral, whitelist);
    }

    function deprecatePrinciple(address _principle) public {
        require(msg.sender == owner, "Only owner");
        isPrincipleWhitelisted[_principle] = false;
    }

    function hashVariables(
        address _principle,
        uint _epoch
    ) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(_principle, _epoch));
    }
    function hashVariablesT(
        address _principle,
        uint _epoch,
        address _tokenToClaim
    ) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(_principle, _epoch, _tokenToClaim));
    }
    function currentEpoch() public view returns (uint) {
        return ((block.timestamp - blockDeployedContract) / epochDuration) + 1;
    }
}
