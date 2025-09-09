import { loadFixture } from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import { expect } from "chai";
import { getAddress } from "viem";
import { deployLongPoolMockEnvironment, deployPerpManager } from "./fixtures";
import hre from "hardhat";
import { ADMIN_ROLE, LIQUIDATOR_ROLE, ORDER_SIGNER_ROLE } from "./utils/constants";

describe("PerpManager", function () {
    describe("Basic Tests", function () {
        it("Only manager can assign roles", async function () {
            const { manager, user1, owner, orderSigner, liquidator } = await loadFixture(deployPerpManager);

            let address = getAddress(user1.account.address);
            await expect(manager.write.grantRole([ADMIN_ROLE, address, 0], { account: user1.account }))
                .to.be.rejectedWith(`AccessManagerUnauthorizedAccount("${address}", ${ADMIN_ROLE})`, "Only admin can assign roles");

            address = getAddress(liquidator.account.address);
            await expect(manager.write.grantRole([ADMIN_ROLE, user1.account.address, 0], { account: liquidator.account }))
                .to.be.rejectedWith(`AccessManagerUnauthorizedAccount("${address}", ${ADMIN_ROLE})`, "Only admin can assign roles");
            await expect(manager.write.grantRole([LIQUIDATOR_ROLE, user1.account.address, 0], { account: liquidator.account }))
                .to.be.rejectedWith(`AccessManagerUnauthorizedAccount("${address}", ${ADMIN_ROLE})`, "Only admin can assign roles");

            address = getAddress(orderSigner.account.address);
            await expect(manager.write.grantRole([ADMIN_ROLE, user1.account.address, 0], { account: orderSigner.account }))
                .to.be.rejectedWith(`AccessManagerUnauthorizedAccount("${address}", ${ADMIN_ROLE})`, "Only admin can assign roles");
            await expect(manager.write.grantRole([ORDER_SIGNER_ROLE, user1.account.address, 0], { account: orderSigner.account }))
                .to.be.rejectedWith(`AccessManagerUnauthorizedAccount("${address}", ${ADMIN_ROLE})`, "Only admin can assign roles");

            await manager.write.grantRole([ADMIN_ROLE, address, 0], { account: owner.account });
            
            const event = (await manager.getEvents.RoleGranted())[0].args!;
            
            expect(event.account).to.equal(address);
            expect(event.roleId).to.equal(ADMIN_ROLE);
        });
    });
});