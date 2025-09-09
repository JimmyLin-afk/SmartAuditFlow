import { loadFixture } from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import { expect } from "chai";
import { getAddress } from "viem";
import { deployLongPoolMockEnvironment } from "./fixtures";
import hre from "hardhat";
import { ADMIN_ROLE } from "./utils/constants";

describe("WasabiLongPool - Proxy Validations", function () {
    describe("Upgrade Contract", function () {
        it("Upgrade and call new function", async function () {
            const { wasabiLongPool, user1, owner, liquidator, manager} = await loadFixture(deployLongPoolMockEnvironment);
            await hre.upgrades.silenceWarnings();

            console.log('owner', owner.account.address);

            const contractName = "MockWasabiLongPoolV2";
            const MockWasabiLongPoolV2 = await hre.ethers.getContractFactory(contractName);
            const address = 
                await hre.upgrades.upgradeProxy(
                    wasabiLongPool.address,
                    MockWasabiLongPoolV2,
                    { call: { fn: "setSomeNewValue", args: [1234n] }, unsafeAllow: ['missing-initializer'] }
                )
                .then(c => c.waitForDeployment())
                .then(c => c.getAddress()).then(getAddress);
            const wasabiLongPool2 = await hre.viem.getContractAt(contractName, address);

            expect(await wasabiLongPool2.read.owner()).to.equal(manager.address);

            await wasabiLongPool2.write.setSomeNewValue([42n], {account: owner.account.address});
            expect(await wasabiLongPool2.read.someNewValue()).to.equal(42);

            await expect(wasabiLongPool2.write.setSomeNewValue([43n], {account: user1.account.address}))
                .to.be.rejectedWith(`AccessManagerUnauthorizedAccount("${getAddress(user1.account.address)}", ${ADMIN_ROLE})`, "Only owner can set address provider");

            await wasabiLongPool2.write.setSomeNewValue([43n], {account: owner.account.address});
            expect(await wasabiLongPool2.read.someNewValue()).to.equal(43);
        });

        it("Can't upgrade implementation contract", async function () {
            const {implAddress, addressProvider, owner , wasabiLongPool, manager} = await loadFixture(deployLongPoolMockEnvironment);

            const wasabiLongPoolImpl = await hre.viem.getContractAt("WasabiLongPool", implAddress);

            expect(wasabiLongPool.address).to.not.equal(wasabiLongPoolImpl.address);

            await expect(wasabiLongPoolImpl.write.initialize([addressProvider.address, manager.address], {account: owner.account.address}))
                .to.be.rejectedWith("InvalidInitialization");
        });
    });
});