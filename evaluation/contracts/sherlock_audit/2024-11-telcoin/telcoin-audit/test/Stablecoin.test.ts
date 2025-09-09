import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";
import { expect } from "chai";
import { ethers } from "hardhat";
import { Stablecoin } from "../../typechain-types";

describe("Stablecoin", () => {
    const MINTER_ROLE = ethers.keccak256(ethers.toUtf8Bytes('MINTER_ROLE'));
    const BURNER_ROLE = ethers.keccak256(ethers.toUtf8Bytes('BURNER_ROLE'));
    const SUPPORT_ROLE = ethers.keccak256(ethers.toUtf8Bytes('SUPPORT_ROLE'));
    const BLACKLISTER_ROLE = ethers.keccak256(ethers.toUtf8Bytes('BLACKLISTER_ROLE'));

    let deployer: SignerWithAddress;
    let holder: SignerWithAddress;
    let stablecoin: Stablecoin;

    beforeEach("setup", async () => {
        [deployer, holder] = await ethers.getSigners();

        const Stablecoin_Factory = await ethers.getContractFactory("Stablecoin", deployer);
        stablecoin = await Stablecoin_Factory.deploy();

        await stablecoin.initialize("US Dollar", "eXYZ", 6);
    });

    describe("Static Values", () => {
        it("MINTER_ROLE", async () => {
            expect(await stablecoin.MINTER_ROLE()).to.equal(MINTER_ROLE);
        });

        it("BURNER_ROLE", async () => {
            expect(await stablecoin.BURNER_ROLE()).to.equal(BURNER_ROLE);
        });

        it("SUPPORT_ROLE", async () => {
            expect(await stablecoin.SUPPORT_ROLE()).to.equal(SUPPORT_ROLE);
        });

        it("decimals", async () => {
            expect(await stablecoin.decimals()).to.equal(6);
        });
    });

    describe("Alter supply", () => {
        beforeEach("setup", async () => {
            await stablecoin.grantRole(MINTER_ROLE, deployer);
        });

        describe("Mint", () => {
            it("mint", async () => {
                await expect(stablecoin.mint(100)).to.be.not.reverted;
                expect(await stablecoin.balanceOf(deployer)).to.equal(100);
            });

            it("mintTo", async () => {
                await expect(stablecoin.mintTo(holder, 100)).to.be.not.reverted;
                expect(await stablecoin.balanceOf(holder)).to.equal(100);
            });
        });

        describe("burn", () => {
            beforeEach("setup", async () => {
                await stablecoin.grantRole(BURNER_ROLE, deployer);
            });

            it("burn", async () => {
                await stablecoin.mint(100);
                await expect(stablecoin.burn(100)).to.be.not.reverted;
                expect(await stablecoin.balanceOf(deployer)).to.equal(0);
            });

            it("burnFrom", async () => {
                await expect(stablecoin.mintTo(holder, 100)).to.be.not.reverted;
                await stablecoin.connect(holder).approve(deployer, 100);
                await expect(stablecoin.connect(deployer).burnFrom(holder, 100)).to.be.not.reverted;
                expect(await stablecoin.balanceOf(holder)).to.equal(0);
            });
        });
    });

    describe("blacklist", () => {
        beforeEach("setup", async () => {
            await stablecoin.grantRole(BLACKLISTER_ROLE, deployer);
            await stablecoin.grantRole(MINTER_ROLE, deployer);
        });

        it("burn", async () => {
            await stablecoin.mintTo(holder, 100);
            await expect(stablecoin.addBlackList(holder)).to.be.not.reverted;
            expect(await stablecoin.balanceOf(deployer)).to.equal(100);
            expect(await stablecoin.balanceOf(holder)).to.equal(0);
        });
    });

    describe("Auxiliary", () => {
        beforeEach("setup", async () => {
            await stablecoin.grantRole(MINTER_ROLE, deployer);
            await stablecoin.grantRole(SUPPORT_ROLE, deployer);
            await stablecoin.mintTo(stablecoin, 1);
        });

        it("erc20Rescue", async () => {
            await expect(stablecoin.erc20Rescue(stablecoin, holder, 1)).to.not.be.reverted;
            expect(await stablecoin.balanceOf(holder)).to.equal(1);
            expect(await stablecoin.balanceOf(stablecoin)).to.equal(0);
        });
    });
});