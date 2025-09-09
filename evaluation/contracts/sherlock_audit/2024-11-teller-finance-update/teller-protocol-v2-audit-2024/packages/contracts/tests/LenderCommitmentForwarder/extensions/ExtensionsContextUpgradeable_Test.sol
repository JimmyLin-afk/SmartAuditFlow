import { Testable } from "../../Testable.sol";

import { ExtensionsContextUpgradeable } from "../../../contracts/LenderCommitmentForwarder/extensions/ExtensionsContextUpgradeable.sol";

contract ExtensionsContextMock is ExtensionsContextUpgradeable {}

contract ExtensionsContext_Test is Testable {
    constructor() {}

    User private extensionContract;

    User private borrower;
    User private lender;

    ExtensionsContextMock extensionsContext;

    function setUp() public {
        borrower = new User();
        lender = new User();

        extensionsContext = new ExtensionsContextMock();
    }

    function test_addingExtension() public {
        bool isTrustedBefore = extensionsContext.hasExtension(
            address(borrower),
            address(extensionContract)
        );

        //the user will approve
        vm.prank(address(borrower));
        extensionsContext.addExtension(address(extensionContract));

        vm.prank(address(borrower));
        bool isTrustedAfter = extensionsContext.hasExtension(
            address(borrower),
            address(extensionContract)
        );

        assertFalse(isTrustedBefore, "Should not be trusted forwarder before");
        assertTrue(isTrustedAfter, "Should be trusted forwarder after");
    }
}

contract User {}
