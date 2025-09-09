// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility, private-vars-leading-underscore
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase, one-contract-per-file
pragma solidity ^0.8.18;

import { Ownable } from "openzeppelin-contracts/contracts/access/Ownable.sol";
import { Test } from "mento-std/Test.sol";

import { IChainlinkRelayerFactory } from "contracts/interfaces/IChainlinkRelayerFactory.sol";
import { IChainlinkRelayer } from "contracts/interfaces/IChainlinkRelayer.sol";
import { ChainlinkRelayerFactory } from "contracts/oracles/ChainlinkRelayerFactory.sol";

contract ChainlinkRelayerFactoryTest is Test {
  IChainlinkRelayerFactory relayerFactory;
  address owner = makeAddr("owner");
  address relayerDeployer = makeAddr("relayerDeployer");
  address nonOwner = makeAddr("nonOwner");
  address nonDeployer = makeAddr("nonDeployer");
  address mockSortedOracles = makeAddr("sortedOracles");
  address[4] mockAggregators = [
    makeAddr("aggregator1"),
    makeAddr("aggregator2"),
    makeAddr("aggregator3"),
    makeAddr("aggregator4")
  ];
  address[3] rateFeeds = [makeAddr("rateFeed1"), makeAddr("rateFeed2"), makeAddr("rateFeed3")];
  address aRateFeed = rateFeeds[0];
  string aRateFeedDescription = "CELO/USD";

  bytes constant NOT_ALLOWED_ERROR = abi.encodeWithSignature("NotAllowed()");

  event RelayerDeployed(
    address indexed relayerAddress,
    address indexed rateFeedId,
    string rateFeedDescription,
    IChainlinkRelayer.ChainlinkAggregator[] aggregators
  );
  event RelayerRemoved(address indexed relayerAddress, address indexed rateFeedId);
  event RelayerDeployerUpdated(address indexed newRelayerDeployer, address indexed oldRelayerDeployer);

  function oneAggregator(
    uint256 aggregatorIndex
  ) internal view returns (IChainlinkRelayer.ChainlinkAggregator[] memory aggregators) {
    aggregators = new IChainlinkRelayer.ChainlinkAggregator[](1);
    aggregators[0] = IChainlinkRelayer.ChainlinkAggregator(mockAggregators[aggregatorIndex], false);
  }

  function fourAggregators() internal view returns (IChainlinkRelayer.ChainlinkAggregator[] memory aggregators) {
    aggregators = new IChainlinkRelayer.ChainlinkAggregator[](4);
    aggregators[0] = IChainlinkRelayer.ChainlinkAggregator(mockAggregators[0], false);
    aggregators[1] = IChainlinkRelayer.ChainlinkAggregator(mockAggregators[1], false);
    aggregators[2] = IChainlinkRelayer.ChainlinkAggregator(mockAggregators[2], false);
    aggregators[3] = IChainlinkRelayer.ChainlinkAggregator(mockAggregators[3], false);
  }

  function setUp() public virtual {
    relayerFactory = IChainlinkRelayerFactory(new ChainlinkRelayerFactory(false));
    vm.prank(owner);
    relayerFactory.initialize(mockSortedOracles, relayerDeployer);
  }

  function expectedRelayerAddress(
    address rateFeedId,
    string memory rateFeedDescription,
    address sortedOracles,
    uint256 maxTimestampSpread,
    IChainlinkRelayer.ChainlinkAggregator[] memory aggregators,
    address relayerFactoryAddress
  ) internal view returns (address expectedAddress) {
    bytes32 salt = keccak256("mento.chainlinkRelayer");
    return
      address(
        uint160(
          uint256(
            keccak256(
              abi.encodePacked(
                bytes1(0xff),
                relayerFactoryAddress,
                salt,
                keccak256(
                  abi.encodePacked(
                    vm.getCode("ChainlinkRelayerV1"),
                    abi.encode(rateFeedId, rateFeedDescription, sortedOracles, maxTimestampSpread, aggregators)
                  )
                )
              )
            )
          )
        )
      );
  }

  function contractAlreadyExistsError(
    address relayerAddress,
    address rateFeedId
  ) public pure returns (bytes memory ContractAlreadyExistsError) {
    return abi.encodeWithSignature("ContractAlreadyExists(address,address)", relayerAddress, rateFeedId);
  }

  function relayerForFeedExistsError(address rateFeedId) public pure returns (bytes memory RelayerForFeedExistsError) {
    return abi.encodeWithSignature("RelayerForFeedExists(address)", rateFeedId);
  }

  function noRelayerForRateFeedId(address rateFeedId) public pure returns (bytes memory NoRelayerForRateFeedIdError) {
    return abi.encodeWithSignature("NoRelayerForRateFeedId(address)", rateFeedId);
  }
}

contract ChainlinkRelayerFactoryTest_initialize is ChainlinkRelayerFactoryTest {
  function test_setsSortedOracles() public {
    address realSortedOracles = relayerFactory.sortedOracles();
    assertEq(realSortedOracles, mockSortedOracles);
  }

  function test_setsOwner() public view {
    address realOwner = Ownable(address(relayerFactory)).owner();
    assertEq(realOwner, owner);
  }
}

contract ChainlinkRelayerFactoryTest_transferOwnership is ChainlinkRelayerFactoryTest {
  function test_setsNewOwner() public {
    vm.prank(owner);
    Ownable(address(relayerFactory)).transferOwnership(nonOwner);
    address realOwner = Ownable(address(relayerFactory)).owner();
    assertEq(realOwner, nonOwner);
  }

  function test_failsWhenCalledByNonOwner() public {
    vm.expectRevert("Ownable: caller is not the owner");
    vm.prank(nonOwner);
    Ownable(address(relayerFactory)).transferOwnership(nonOwner);
  }
}

contract ChainlinkRelayerFactoryTest_setRelayerDeployer is ChainlinkRelayerFactoryTest {
  function test_setRelayerDeployer() public {
    address newDeployer = makeAddr("newDeployer");
    vm.expectEmit(true, true, false, false, address(relayerFactory));
    emit RelayerDeployerUpdated({ newRelayerDeployer: newDeployer, oldRelayerDeployer: relayerDeployer });
    vm.prank(owner);
    relayerFactory.setRelayerDeployer(newDeployer);
    address realRelayerDeployer = relayerFactory.relayerDeployer();
    assertEq(realRelayerDeployer, newDeployer);
  }

  function test_failsWhenCalledByNonOwner() public {
    address newDeployer = makeAddr("newDeployer");
    vm.prank(nonOwner);
    vm.expectRevert("Ownable: caller is not the owner");
    relayerFactory.setRelayerDeployer(newDeployer);
  }
}

contract ChainlinkRelayerFactoryTest_renounceOwnership is ChainlinkRelayerFactoryTest {
  function test_setsOwnerToZeroAddress() public {
    vm.prank(owner);
    Ownable(address(relayerFactory)).renounceOwnership();
    address realOwner = Ownable(address(relayerFactory)).owner();
    assertEq(realOwner, address(0));
  }

  function test_failsWhenCalledByNonOwner() public {
    vm.expectRevert("Ownable: caller is not the owner");
    vm.prank(nonOwner);
    Ownable(address(relayerFactory)).renounceOwnership();
  }
}

contract ChainlinkRelayerFactoryTest_deployRelayer is ChainlinkRelayerFactoryTest {
  IChainlinkRelayer relayer;

  function test_setsRateFeed() public {
    vm.prank(relayerDeployer);
    relayer = IChainlinkRelayer(relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators()));
    address rateFeed = relayer.rateFeedId();
    assertEq(rateFeed, aRateFeed);
  }

  function test_setsRateFeedDescription() public {
    vm.prank(relayerDeployer);
    relayer = IChainlinkRelayer(relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators()));
    string memory rateFeedDescription = relayer.rateFeedDescription();
    assertEq(rateFeedDescription, aRateFeedDescription);
  }

  function test_setsMaxTimestampSpread() public {
    vm.prank(owner);
    relayer = IChainlinkRelayer(relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators()));
    uint256 maxTimestampSpread = relayer.maxTimestampSpread();
    assertEq(maxTimestampSpread, 300);
  }

  function test_setsAggregators() public {
    vm.prank(relayerDeployer);
    relayer = IChainlinkRelayer(relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators()));
    IChainlinkRelayer.ChainlinkAggregator[] memory expectedAggregators = fourAggregators();
    IChainlinkRelayer.ChainlinkAggregator[] memory actualAggregators = relayer.getAggregators();
    assertEq(expectedAggregators.length, actualAggregators.length);
    for (uint256 i = 0; i < expectedAggregators.length; i++) {
      assertEq(expectedAggregators[i].aggregator, actualAggregators[i].aggregator);
      assertEq(expectedAggregators[i].invert, actualAggregators[i].invert);
    }
  }

  function test_setsSortedOracles() public {
    vm.prank(relayerDeployer);
    relayer = IChainlinkRelayer(relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators()));
    address sortedOracles = relayer.sortedOracles();
    assertEq(sortedOracles, mockSortedOracles);
  }

  function test_deploysToTheCorrectAddress() public {
    vm.prank(relayerDeployer);
    relayer = IChainlinkRelayer(relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators()));
    address expectedAddress = expectedRelayerAddress({
      rateFeedId: aRateFeed,
      rateFeedDescription: aRateFeedDescription,
      sortedOracles: mockSortedOracles,
      maxTimestampSpread: 300,
      aggregators: fourAggregators(),
      relayerFactoryAddress: address(relayerFactory)
    });

    assertEq(address(relayer), expectedAddress);
  }

  function test_emitsRelayerDeployedEvent() public {
    address expectedAddress = expectedRelayerAddress({
      rateFeedId: aRateFeed,
      rateFeedDescription: aRateFeedDescription,
      sortedOracles: mockSortedOracles,
      maxTimestampSpread: 300,
      aggregators: fourAggregators(),
      relayerFactoryAddress: address(relayerFactory)
    });
    // solhint-disable-next-line func-named-parameters
    vm.expectEmit(true, true, true, true, address(relayerFactory));
    emit RelayerDeployed({
      relayerAddress: expectedAddress,
      rateFeedId: aRateFeed,
      rateFeedDescription: aRateFeedDescription,
      aggregators: fourAggregators()
    });
    vm.prank(relayerDeployer);
    relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators());
  }

  function test_remembersTheRelayerAddress() public {
    vm.prank(relayerDeployer);
    relayer = IChainlinkRelayer(relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators()));
    address storedAddress = relayerFactory.getRelayer(aRateFeed);
    assertEq(storedAddress, address(relayer));
  }

  function test_revertsWhenDeployingToAddressWithCode() public {
    address futureAddress = expectedRelayerAddress(
      aRateFeed,
      aRateFeedDescription,
      address(mockSortedOracles),
      300,
      fourAggregators(),
      address(relayerFactory)
    );
    vm.etch(futureAddress, abi.encode("This is a great contract's bytecode"));
    vm.expectRevert(contractAlreadyExistsError(address(futureAddress), aRateFeed));
    vm.prank(relayerDeployer);
    relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators());
  }

  function test_revertsWhenDeployingTheSameRelayer() public {
    vm.prank(relayerDeployer);
    relayer = IChainlinkRelayer(relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators()));
    vm.expectRevert(relayerForFeedExistsError(aRateFeed));
    vm.prank(relayerDeployer);
    relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators());
  }

  function test_revertsWhenDeployingForTheSameRateFeed() public {
    vm.prank(relayerDeployer);
    relayer = IChainlinkRelayer(relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators()));
    vm.expectRevert(relayerForFeedExistsError(aRateFeed));
    vm.prank(relayerDeployer);
    relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, oneAggregator(0));
  }

  function test_revertsWhenCalledByNonDeployer() public {
    vm.expectRevert(NOT_ALLOWED_ERROR);
    vm.prank(nonDeployer);
    relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators());
  }

  function test_worksWhenCalledByOwner() public {
    vm.prank(owner);
    relayerFactory.deployRelayer(aRateFeed, aRateFeedDescription, 300, fourAggregators());
  }
}

contract ChainlinkRelayerFactoryTest_getRelayers is ChainlinkRelayerFactoryTest {
  function test_emptyWhenNoRelayers() public view {
    address[] memory relayers = relayerFactory.getRelayers();
    assertEq(relayers.length, 0);
  }

  function test_returnsRelayerWhenThereIsOne() public {
    vm.prank(relayerDeployer);
    address relayerAddress = relayerFactory.deployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(0));
    address[] memory relayers = relayerFactory.getRelayers();
    assertEq(relayers.length, 1);
    assertEq(relayers[0], relayerAddress);
  }

  function test_returnsMultipleRelayersWhenThereAreMore() public {
    vm.prank(relayerDeployer);
    address relayerAddress1 = relayerFactory.deployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(0));
    vm.prank(relayerDeployer);
    address relayerAddress2 = relayerFactory.deployRelayer(rateFeeds[1], aRateFeedDescription, 0, oneAggregator(1));
    vm.prank(relayerDeployer);
    address relayerAddress3 = relayerFactory.deployRelayer(rateFeeds[2], aRateFeedDescription, 0, oneAggregator(2));
    address[] memory relayers = relayerFactory.getRelayers();
    assertEq(relayers.length, 3);
    assertEq(relayers[0], relayerAddress1);
    assertEq(relayers[1], relayerAddress2);
    assertEq(relayers[2], relayerAddress3);
  }

  function test_returnsADifferentRelayerAfterRedeployment() public {
    vm.prank(relayerDeployer);
    address relayerAddress1 = relayerFactory.deployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(0));
    vm.prank(relayerDeployer);
    relayerFactory.deployRelayer(rateFeeds[1], aRateFeedDescription, 0, oneAggregator(1));
    vm.prank(relayerDeployer);
    address relayerAddress2 = relayerFactory.redeployRelayer(rateFeeds[1], aRateFeedDescription, 0, oneAggregator(2));
    address[] memory relayers = relayerFactory.getRelayers();
    assertEq(relayers.length, 2);
    assertEq(relayers[0], relayerAddress1);
    assertEq(relayers[1], relayerAddress2);
  }

  function test_doesntReturnARemovedRelayer() public {
    vm.prank(relayerDeployer);
    address relayerAddress1 = relayerFactory.deployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(0));
    vm.prank(relayerDeployer);
    address relayerAddress2 = relayerFactory.deployRelayer(rateFeeds[1], aRateFeedDescription, 0, oneAggregator(1));
    vm.prank(relayerDeployer);
    relayerFactory.deployRelayer(rateFeeds[2], aRateFeedDescription, 0, oneAggregator(2));
    vm.prank(relayerDeployer);
    relayerFactory.removeRelayer(rateFeeds[2]);
    address[] memory relayers = relayerFactory.getRelayers();
    assertEq(relayers.length, 2);
    assertEq(relayers[0], relayerAddress1);
    assertEq(relayers[1], relayerAddress2);
  }
}

contract ChainlinkRelayerFactoryTest_removeRelayer is ChainlinkRelayerFactoryTest {
  address relayerAddress;

  function setUp() public override {
    super.setUp();

    vm.prank(relayerDeployer);
    relayerAddress = relayerFactory.deployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(0));
  }

  function test_removesTheRelayer() public {
    vm.prank(relayerDeployer);
    relayerFactory.removeRelayer(aRateFeed);
    vm.prank(relayerDeployer);
    address relayer = relayerFactory.getRelayer(aRateFeed);
    assertEq(relayer, address(0));
  }

  function test_emitsRelayerRemovedEvent() public {
    // solhint-disable-next-line func-named-parameters
    vm.expectEmit(true, true, true, false, address(relayerFactory));
    emit RelayerRemoved({ relayerAddress: relayerAddress, rateFeedId: aRateFeed });
    vm.prank(relayerDeployer);
    relayerFactory.removeRelayer(aRateFeed);
  }

  function test_doesntRemoveOtherRelayers() public {
    vm.prank(relayerDeployer);
    address newRelayerAddress = relayerFactory.deployRelayer(rateFeeds[1], aRateFeedDescription, 0, oneAggregator(1));
    vm.prank(relayerDeployer);
    relayerFactory.removeRelayer(aRateFeed);
    address[] memory relayers = relayerFactory.getRelayers();

    assertEq(relayers.length, 1);
    assertEq(relayers[0], newRelayerAddress);
  }

  function test_revertsOnNonexistentRelayer() public {
    vm.expectRevert(noRelayerForRateFeedId(rateFeeds[1]));
    vm.prank(relayerDeployer);
    relayerFactory.removeRelayer(rateFeeds[1]);
  }

  function test_revertsWhenCalledByNonDeployer() public {
    vm.expectRevert(NOT_ALLOWED_ERROR);
    vm.prank(nonDeployer);
    relayerFactory.removeRelayer(aRateFeed);
  }

  function test_worksWhenCalledByOwner() public {
    vm.prank(owner);
    relayerFactory.removeRelayer(aRateFeed);
  }
}

contract ChainlinkRelayerFactoryTest_redeployRelayer is ChainlinkRelayerFactoryTest {
  address oldAddress;

  function setUp() public override {
    super.setUp();
    vm.prank(relayerDeployer);
    oldAddress = relayerFactory.deployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(0));
  }

  function test_setsRateFeedOnNewRelayer() public {
    vm.prank(relayerDeployer);
    IChainlinkRelayer relayer = IChainlinkRelayer(
      relayerFactory.redeployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(1))
    );

    address rateFeed = relayer.rateFeedId();
    assertEq(rateFeed, rateFeeds[0]);
  }

  function test_setsAggregatorOnNewRelayer() public {
    vm.prank(relayerDeployer);
    IChainlinkRelayer relayer = IChainlinkRelayer(
      relayerFactory.redeployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(1))
    );

    IChainlinkRelayer.ChainlinkAggregator[] memory expectedAggregators = oneAggregator(1);
    IChainlinkRelayer.ChainlinkAggregator[] memory actualAggregators = relayer.getAggregators();
    assertEq(expectedAggregators.length, actualAggregators.length);
    for (uint256 i = 0; i < expectedAggregators.length; i++) {
      assertEq(expectedAggregators[i].aggregator, actualAggregators[i].aggregator);
      assertEq(expectedAggregators[i].invert, actualAggregators[i].invert);
    }
  }

  function test_setsSortedOraclesOnNewRelayer() public {
    vm.prank(relayerDeployer);
    IChainlinkRelayer relayer = IChainlinkRelayer(
      relayerFactory.redeployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(1))
    );

    address sortedOracles = relayer.sortedOracles();
    assertEq(sortedOracles, mockSortedOracles);
  }

  function test_deploysToTheCorrectNewAddress() public {
    vm.prank(relayerDeployer);
    IChainlinkRelayer relayer = IChainlinkRelayer(
      relayerFactory.redeployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(1))
    );

    address expectedAddress = expectedRelayerAddress(
      aRateFeed,
      aRateFeedDescription,
      mockSortedOracles,
      0,
      oneAggregator(1),
      address(relayerFactory)
    );

    assertEq(address(relayer), expectedAddress);
  }

  function test_emitsRelayerRemovedAndDeployedEvents() public {
    address expectedAddress = expectedRelayerAddress({
      rateFeedId: aRateFeed,
      rateFeedDescription: aRateFeedDescription,
      sortedOracles: mockSortedOracles,
      maxTimestampSpread: 0,
      aggregators: oneAggregator(1),
      relayerFactoryAddress: address(relayerFactory)
    });
    // solhint-disable-next-line func-named-parameters
    vm.expectEmit(true, true, true, false, address(relayerFactory));
    emit RelayerRemoved({ relayerAddress: oldAddress, rateFeedId: aRateFeed });
    // solhint-disable-next-line func-named-parameters
    vm.expectEmit(true, true, true, false, address(relayerFactory));
    emit RelayerDeployed({
      relayerAddress: expectedAddress,
      rateFeedId: aRateFeed,
      rateFeedDescription: aRateFeedDescription,
      aggregators: oneAggregator(1)
    });
    vm.prank(relayerDeployer);
    relayerFactory.redeployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(1));
  }

  function test_remembersTheNewRelayerAddress() public {
    vm.prank(relayerDeployer);
    address relayer = relayerFactory.redeployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(1));
    address storedAddress = relayerFactory.getRelayer(aRateFeed);
    assertEq(storedAddress, relayer);
  }

  function test_revertsWhenDeployingTheSameExactRelayer() public {
    vm.expectRevert(contractAlreadyExistsError(oldAddress, aRateFeed));
    vm.prank(relayerDeployer);
    relayerFactory.redeployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(0));
  }

  function test_revertsWhenCalledByNonDeployer() public {
    vm.expectRevert(NOT_ALLOWED_ERROR);
    vm.prank(nonDeployer);
    relayerFactory.redeployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(1));
  }

  function test_worksWhenCalledByOwner() public {
    vm.prank(owner);
    relayerFactory.redeployRelayer(rateFeeds[0], aRateFeedDescription, 0, oneAggregator(1));
  }
}
