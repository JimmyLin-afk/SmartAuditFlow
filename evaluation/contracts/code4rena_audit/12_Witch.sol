// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.0;

import "./utils/access/AccessControl.sol";
import "./interfaces/vault/ILadle.sol";
import "./interfaces/vault/ICauldron.sol";
import "./interfaces/vault/DataTypes.sol";
import "./math/WMul.sol";
import "./math/WDiv.sol";
import "./math/WDivUp.sol";
import "./math/CastU256U128.sol";


contract Witch is AccessControl() {
    using WMul for uint256;
    using WDiv for uint256;
    using WDivUp for uint256;
    using CastU256U128 for uint256;

    event AuctionTimeSet(uint128 indexed auctionTime);
    event InitialProportionSet(uint128 indexed initialProportion);
    event Bought(bytes12 indexed vaultId, address indexed buyer, uint256 ink, uint256 art);
  
    uint128 public auctionTime = 4 * 60 * 60; 
    uint128 public initialProportion = 5e17; 

    ICauldron immutable public cauldron;
    ILadle immutable public ladle;
    mapping(bytes12 => address) public vaultOwners;

    constructor (ICauldron cauldron_, ILadle ladle_) {
        cauldron = cauldron_;
        ladle = ladle_;
    }

    /// @dev Set the auction time to calculate liquidation prices
    function setAuctionTime(uint128 auctionTime_) public auth {
        auctionTime = auctionTime_;
        emit AuctionTimeSet(auctionTime_);
    }

    /// @dev Set the proportion of the collateral that will be sold at auction start
    function setInitialProportion(uint128 initialProportion_) public auth {
        require (initialProportion_ <= 1e18, "Only at or under 100%");
        initialProportion = initialProportion_;
        emit InitialProportionSet(initialProportion_);
    }

    /// @dev Put an undercollateralized vault up for liquidation.
    function grab(bytes12 vaultId) public {
        DataTypes.Vault memory vault = cauldron.vaults(vaultId);
        vaultOwners[vaultId] = vault.owner;
        cauldron.grab(vaultId, address(this));
    }

    /// @dev Buy an amount of collateral off a vault in liquidation, paying at most `max` underlying.
    function buy(bytes12 vaultId, uint128 art, uint128 min) public {
        DataTypes.Balances memory balances_ = cauldron.balances(vaultId);

        require (balances_.art > 0, "Nothing to buy");                                    
        uint256 elapsed = uint32(block.timestamp) - cauldron.auctions(vaultId);           
        uint256 price;
        {
            // Price of a collateral unit, in underlying, at the present moment, for a given vault
            //
            //                ink                     min(auction, elapsed)
            // price = 1 / (------- * (p + (1 - p) * -----------------------))
            //                art                          auction
            (uint256 auctionTime_, uint256 initialProportion_) = (auctionTime, initialProportion);
            uint256 term1 = uint256(balances_.ink).wdiv(balances_.art);
            uint256 dividend2 = auctionTime_ < elapsed ? auctionTime_ : elapsed;
            uint256 divisor2 = auctionTime_;
            uint256 term2 = initialProportion_ + (1e18 - initialProportion_).wmul(dividend2.wdiv(divisor2));
            price = uint256(1e18).wdiv(term1.wmul(term2));
        }
        uint256 ink = uint256(art).wdivup(price);                                              
        require (ink >= min, "Not enough bought");

        ladle.settle(vaultId, msg.sender, ink.u128(), art);                                     
        if (balances_.art - art == 0) {
            cauldron.give(vaultId, vaultOwners[vaultId]);
            delete vaultOwners[vaultId];
        }

        emit Bought(vaultId, msg.sender, ink, art);
    }
}