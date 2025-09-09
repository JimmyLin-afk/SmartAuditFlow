import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";
import { expect } from "chai";
import { ethers } from "hardhat";
import { Stablecoin } from "../typechain-types";

describe("Blacklist", () => {
    const BLACKLISTER_ROLE = ethers.keccak256(ethers.toUtf8Bytes('BLACKLISTER_ROLE'));

    let deployer: SignerWithAddress;
    let blacklisted: SignerWithAddress;
    let blacklist: Stablecoin;

    beforeEach("setup", async () => {
        [deployer, blacklisted] = await ethers.getSigners();

        const Blacklist_Factory = await ethers.getContractFactory("Stablecoin", deployer);
        blacklist = await Blacklist_Factory.deploy();

        await blacklist.initialize("Blacklist", "BL", 18);
    });

    describe("Static Values", () => {
        it("BLACKLISTER_ROLE", async () => {
            expect(await blacklist.BLACKLISTER_ROLE()).to.equal(BLACKLISTER_ROLE);
        });
    });

    describe("Blacklisting", () => {
        beforeEach("setup", async () => {
            await blacklist.grantRole(BLACKLISTER_ROLE, deployer);
            await blacklist.addBlackList(blacklisted);
            expect(await blacklist.blacklisted(blacklisted)).to.equal(true);
        });

        it("blacklisted", async () => {
            expect(await blacklist.blacklisted(blacklisted)).to.equal(true);
        });

        it("AlreadyBlacklisted", async () => {
            await expect(blacklist.addBlackList(blacklisted)).to.be.revertedWithCustomError(blacklist, 'AlreadyBlacklisted').withArgs(blacklisted);
            expect(await blacklist.blacklisted(blacklisted)).to.equal(true);
        });

        it("removeBlackList", async () => {
            await expect(blacklist.removeBlackList(blacklisted)).to.emit(blacklist, 'RemovedBlacklist').withArgs(blacklisted);
            expect(await blacklist.blacklisted(blacklisted)).to.equal(false);
        });
    });

    describe("Removing from Blacklist", () => {
        beforeEach("setup", async () => {
            await blacklist.grantRole(BLACKLISTER_ROLE, deployer);
            expect(await blacklist.blacklisted(blacklisted)).to.equal(false);
        });

        it("addBlackList", async () => {
            await expect(blacklist.addBlackList(blacklisted)).to.emit(blacklist, 'AddedBlacklist').withArgs(blacklisted);
            expect(await blacklist.blacklisted(blacklisted)).to.equal(true);
        });

        it("NotBlacklisted", async () => {
            await expect(blacklist.removeBlackList(blacklisted)).to.be.revertedWithCustomError(blacklist, 'NotBlacklisted').withArgs(blacklisted);
            expect(await blacklist.blacklisted(blacklisted)).to.equal(false);
        });
    });
});