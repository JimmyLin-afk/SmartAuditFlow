/*
 * @source: Source Code first verified at https://etherscan.io on Friday
 * @author: -
 * @vulnerable_at_lines: 59
 */

pragma solidity ^0.5.0;

// EthRoulette
//
// Guess the number secretly stored in the blockchain and win the whole contract balance!
// A new number is randomly chosen after each try.
//
// To play, call the play() method with the guessed number (1-20).  Bet price: 0.1 ether

contract EthRoulette {
    uint256 private secretNumber;
    uint256 public lastPlayed;
    uint256 public betPrice = 0.1 ether;
    address public ownerAddr;

    struct Game {
        address player;
        uint256 number;
    }
    Game[] public gamesPlayed;

    function EthRoulette() public {
        ownerAddr = msg.sender;
        shuffle();
    }

    function shuffle() internal {
        // randomly set secretNumber with a value between 1 and 20
        secretNumber =
            (uint8(sha3(now, block.blockhash(block.number - 1))) % 20) +
            1;
    }

    function play(uint256 number) public payable {
        require(msg.value >= betPrice && number <= 20);

        Game game;
        game.player = msg.sender;
        game.number = number;
        gamesPlayed.push(game);

        if (number == secretNumber) {
            // win!
            msg.sender.transfer(this.balance);
        }

        shuffle();
        lastPlayed = now;
    }

    function kill() public {
        if (msg.sender == ownerAddr && now > lastPlayed + 1 days) {
            // <yes> <report> unsafe_suicide
            suicide(msg.sender);
        }
    }

    function() public payable {}
}
