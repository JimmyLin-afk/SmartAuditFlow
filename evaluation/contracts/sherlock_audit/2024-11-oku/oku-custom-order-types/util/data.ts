import { AdrastiaConfig, BatchConfig } from "../../src/config/adrastia-config";

const ONE_GWEI = BigInt("1000000000");

const STD_WRITE_DELAY = 30_000; // 30 seconds for chains with <= 15 second blocks
const EXT_WRITE_DELAY = 60_000; // 60 seconds for chains with 30 second blocks

const MULTICALL3_ADDRESS = "0xcA11bde05977b3631167028862bE2a173976CA11";

const workerIndex = parseInt(process.env.ADRASTIA_WORKER_INDEX ?? "1");

const POLYGON_UPTIME_WEBHOOK_URL = process.env.POLYGON_UPTIME_WEBHOOK_URL;
const POLYGONZKEVM_UPTIME_WEBHOOK_URL = process.env.POLYGONZKEVM_UPTIME_WEBHOOK_URL;
const ZKSYNCERA_UPTIME_WEBHOOK_URL = process.env.ZKSYNCERA_UPTIME_WEBHOOK_URL;
const MOONBEAM_UPTIME_WEBHOOK_URL = process.env.MOONBEAM_UPTIME_WEBHOOK_URL;
const BOBAETHEREUM_UPTIME_WEBHOOK_URL = process.env.BOBAETHEREUM_UPTIME_WEBHOOK_URL;
const ETHEREUM_UPTIME_WEBHOOK_URL = process.env.ETHEREUM_UPTIME_WEBHOOK_URL;
const FILECOIN_UPTIME_WEBHOOK_URL = process.env.FILECOIN_UPTIME_WEBHOOK_URL;
const ROOTSTOCK_UPTIME_WEBHOOK_URL = process.env.ROOTSTOCK_UPTIME_WEBHOOK_URL;
const SCROLL_UPTIME_WEBHOOK_URL = process.env.SCROLL_UPTIME_WEBHOOK_URL;
const ARBITRUMONE_UPTIME_WEBHOOK_URL = process.env.ARBITRUMONE_UPTIME_WEBHOOK_URL;
const OPTIMISM_UPTIME_WEBHOOK_URL = process.env.OPTIMISM_UPTIME_WEBHOOK_URL;
const BSC_UPTIME_WEBHOOK_URL = process.env.BSC_UPTIME_WEBHOOK_URL;
const BASE_UPTIME_WEBHOOK_URL = process.env.BASE_UPTIME_WEBHOOK_URL;

const STANDARD_BATCH_CONFIG: BatchConfig = {
    customerId: "gfx-labs-oku",
    batchId: "oku-0",
    logging: process.env.DATADOG_API_KEY && {
        type: "datadog",
        sourceToken: process.env.DATADOG_API_KEY,
        region: "us5",
        level: "notice",
    },
};

const config: AdrastiaConfig = {
    httpCacheSeconds: 30,
    chains: {
        polygon: {
            txConfig: {
                // Historical max gas price: 2000 Gwei => 10 MATIC min balance (worst case)
                // 1y avg gas price: 225 Gwei => 1.125 MATIC min balance (average case)
                gasLimit: 5_000_000n, // 5M
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: "0x275617327c958bD06b5D6b871E7f491D76113dd8",
            uptimeWebhookUrl: POLYGON_UPTIME_WEBHOOK_URL,
            batches: {
                19: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0x54dF9e11c7933a9cA3BD1E540B63dA15edAe40bf", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 200n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                        maxGasPrice: 2000n * ONE_GWEI,
                    },
                    tokens: [
                        {
                            address: [
                                "0x847b64f9d3a95e977d157866447a5c0a5dfa0ee5",
                                "0x0e44ceb592acfc5d3f09d996302eb4c499ff8c10",
                                "0x167384319b41f7094e62f7506409eb38079abff8",
                                "0x94ab9e4553ffb839431e37cc79ba8905f45bfbea",
                                "0x88f3c15523544835ff6c738ddb30995339ad57d6",
                                "0x45dda9cb7c25131df268515131f647d726f50608",
                                "0xa374094527e1673a86de625aa59517c5de346d32",
                                "0x9b08288c3be4f62bbf8d1c20ac9c5e6f9467d8b7",
                                "0x50eaedb835021e4a108b7290636d62e9765cc6d7",
                                "0x86f1d8390222a3691c28938ec7404a1661e618e0",
                                "0xeef1a9507b3d505f0062f2be9453981255b503c8",
                                "0x1f6082db7c8f4b199e17090cd5c8831a1dad1997",
                                "0xdac8a8e6dbf8c690ec6815e0ff03491b2770255d",
                                "0x3a5329ee48a06671ad1bf295b8a233ee9b9b975e",
                                "0x0a63d3910ffc1529190e80e10855c4216407cc45",
                                "0x5645dcb64c059aa11212707fbf4e7f984440a8cf",
                                "0x7de263d0ad6e5d208844e65118c3a02a9a5d56b6",
                                "0x2aceda63b5e958c45bd27d916ba701bc1dc08f7a",
                                "0x4d05f2a005e6f36633778416764e82d1d12e7fbb",
                                "0x3d0acd52ee4a9271a0ffe75f9b91049152bac64b",
                                "0x3e31ab7f37c048fc6574189135d108df80f0ea26",
                                "0xd866fac7db79994d08c0ca2221fee08935595b4b",
                                "0x98b9162161164de1ed182a0dfa08f5fbf0f733ca",
                                "0x4ccd010148379ea531d6c587cfdd60180196f9b1",
                                "0xfe343675878100b344802a6763fd373fdeed07a4",
                                "0x357faf5843c7fd7fb4e34fbeabdac16eabe8a5bc",
                                "0xb6e57ed85c4c9dbfef2a68711e9d6f36c56e0fcb",
                                "0x0a28c2f5e0e8463e047c203f00f649812ae67e4f",
                                "0xd36ec33c8bed5a9f7b6630855f1533455b98a418",
                                "0xa4d8c89f0c20efbe54cba9e7e7a7e509056228d9",
                                "0x36165b14423425228d7ef62b3ffa799d446347c1",
                                "0x31083a78e11b18e450fd139f9abea98cd53181b7",
                                "0x79e4240e33c121402dfc9009de266356c91f241d",
                                "0x254aa3a898071d6a2da0db11da73b02b4646078f",
                                "0x7e02ae3f794ebade542c92973eb1c46d7e2e935d",
                                "0x3d86a4b8c1b55509792d57e0c038128cc9c14fe7",
                                "0x3f5228d0e7d75467366be7de2c31d0d098ba2c23",
                                "0x3fa147d6309abeb5c1316f7d8a7d8bd023e0cd80",
                                "0x1e5bd2ab4c308396c06c182e1b7e7ba8b2935b83",
                                "0xcf0bb95967cd006f5eaa1463c9d710d1e1550a96",
                                "0xa9077cdb3d13f45b8b9d87c43e11bce0e73d8631",
                                "0xc4c06c9a239f94fc0a1d3e04d23c159ebe8316f1",
                                "0x647fb01a63de9a551b39c7915693b25e6bcec502",
                                "0xb2131540048397d2c958bb5cc38b6db0e3c8fe88",
                                "0x6b75f2189f0e11c52e814e09e280eb1a9a8a094a",
                                "0x41e64a5bc929fa8e6a9c8d7e3b81a13b21ff3045",
                                "0x19c5505638383337d2972ce68b493ad78e315147",
                                "0x8837a61644d523cbe5216dde226f8f85e3aa9be3",
                                "0xefa98fdf168f372e5e9e9b910fcdfd65856f3986",
                                "0x32fae204835e08b9374493d6b4628fd1f87dd045",
                                "0x941061770214613ba0ca3db9a700c39587bb89b6",
                                "0xe6ba22265aefe9dc392f544437acce2aedf8ef36",
                                "0xda908c0bf14ad0b61ea5ebe671ac59b2ce091cbf",
                                "0xa236278bec0e0677a48527340cfb567b4e6e9adc",
                                "0x781067ef296e5c4a4203f81c593274824b7c185d",
                                "0xf369277650ad6654f25412ea8bfbd5942733babc",
                                "0x60ea32a35f64628328f578c872dd6c6d81626aba",
                                "0xc42bf5cd16d9eb1e892b66bb32a3892dcb7bb75c",
                                "0x7f9121b4f4e040fd066e9dc5c250cf9b4338d5bc",
                                "0x1cedfb6819e3ce98ea0e7ea253e6866d7fcccc16",
                                "0x2a08c38c7e1fa969325e2b64047abb085dec3756",
                                "0xa830ff28bb7a46570a7e43dc24a35a663b9cfc2e",
                                "0xfa22d298e3b0bc1752e5ef2849cec1149d596674",
                                "0x849ec65748107aedc518dbc42961f358ea1361a7",
                                "0xbb98b3d2b18aef63a3178023a920971cf5f29be4",
                                "0xb035b6593fcf5ebff11fb16730c6bc023a61f9d3",
                                "0x1de01edb36d0f51762dac6645f0639462cf73933",
                            ],
                            batch: 19,
                        },
                    ],
                },
            ],
        },
        polygonZkEVM: {
            txConfig: {
                // Historical max gas price: 100 Gwei => 0.5 ETH min balance (worst case)
                // 1y avg gas price: 3.05 Gwei => 0.01525 ETH min balance (average case)
                gasLimit: 5_000_000n, // 5M
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: "0x4dd2886836eB5966dd2F5a223182E7889CD7F8a6",
            uptimeWebhookUrl: POLYGONZKEVM_UPTIME_WEBHOOK_URL,
            batches: {
                2: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0x5f1ef1d278013567c3c67e18d2d35bfa9954f723", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: "0xd2c7e323c6f5415dc0ee4e3811901b3380d316fd",
                            batch: 2,
                        },
                    ],
                },
            ],
        },
        zkSyncEra: {
            txConfig: {
                // Gas price estimate: 0.13 GWEI => 0.026 ETH min balance (estimate)
                gasLimit: 200_000_000n, // 200M - gas usage fluctuates
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: "0x77Fce0B11B6F342B5f7a68114A03cAbb808a77e5",
            uptimeWebhookUrl: ZKSYNCERA_UPTIME_WEBHOOK_URL,
            batches: {
                0: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0x0FD66bD1e0974e2535CB424E6675D60aC52a84Fa", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: "0xff577f0e828a878743ecc5e2632cbf65cecf17cf",
                            batch: 0,
                        },
                    ],
                },
            ],
        },
        moonbeam: {
            txConfig: {
                // Historical max gas price: 1405 Gwei => 8.43 GLMR min balance (worst case)
                // 1y avg gas price: 382 Gwei => 2.3 GLMR min balance (average case)
                gasLimit: 6_000_000n, // 6M
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: "0x4dd2886836eB5966dd2F5a223182E7889CD7F8a6",
            uptimeWebhookUrl: MOONBEAM_UPTIME_WEBHOOK_URL,
            batches: {
                0: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0x54dF9e11c7933a9cA3BD1E540B63dA15edAe40bf", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: "0xB64fD2Cf30588e4ACbb92e98b28d976a61914D29",
                            batch: 0,
                        },
                    ],
                },
            ],
        },
        bobaEthereum: {
            txConfig: {
                // Gas price estimate: 1.25 Gwei => 0.0075 BOBA min balance (estimate)
                gasLimit: 6_000_000n, // 6M
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: "0x4dd2886836eB5966dd2F5a223182E7889CD7F8a6",
            uptimeWebhookUrl: BOBAETHEREUM_UPTIME_WEBHOOK_URL,
            batches: {
                0: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0xfEFb60591cffc694C0137983a9091D64Af8Ecbac", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: "0xdF37543dae7986E48E3ce83F390A828A9F3D23BA",
                            batch: 0,
                        },
                    ],
                },
            ],
        },
        ethereum: {
            txConfig: {
                // 1y avg gas price: 34 Gwei => 0.034 ETH min balance (average case)
                gasLimit: 1_000_000n, // 1M
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: "0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696",
            uptimeWebhookUrl: ETHEREUM_UPTIME_WEBHOOK_URL,
            batches: {
                0: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0x54dF9e11c7933a9cA3BD1E540B63dA15edAe40bf", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                        maxGasPrice: 700n * ONE_GWEI,
                    },
                    tokens: [
                        {
                            address: [
                                "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
                                "0x11b815efb8f581194ae79006d24e0d814b7697f6",
                                "0x3416cf6c708da44db2624d63ea0aaef7113527c6",
                                "0x4585fe77225b41b697c938b018e2ac67ac5a20c0",
                                "0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa",
                                "0x9a772018fbd77fcd2d25657e5c547baff3fd7d16",
                                "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
                                "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36",
                                "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
                                "0xc7bbec68d12a0d1830360f8ec58fa599ba1b0e9b",
                                "0x4622df6fb2d9bee0dcdacf545acdb6a2b2f4f863",
                                "0x9db9e0e53058c89e5b94e29621a205198648425b",
                                "0xe8c6c9227491c0a8156a0106a0204d881bb7e531",
                                "0x19fe9857bb3652e9007f2347a1f877ffa9215f7f",
                                "0x1c98562a2fab5af19d8fb3291a36ac3c618835d9",
                                "0x5777d92f208679db4b9778590fa3cab3ac9e2168",
                                "0x60594a405d53811d3bc4766596efd80fd545a270",
                                "0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8",
                                "0xa6cc3c2531fdaa6ae1a3ca84c2855806728693e8",
                                "0x6c063a6e8cd45869b5eb75291e65a3de298f3aa8",
                                "0x92560c178ce069cc014138ed3c2f5221ba71f58a",
                                "0x844eb5c280f38c7462316aad3f338ef9bda62668",
                                "0xc64350c0eab6faed8f17cc2cdff0761c53fb4152",
                                "0x48da0965ab2d2cbf1c17c09cfb5cbe67ad5b1406",
                                "0xa3f558aebaecaf0e11ca4b2199cc5ed341edfd74",
                                "0x4e0924d3a751be199c426d52fb1f2337fa96f736",
                                "0x11950d141ecb863f01007add7d1a342041227b58",
                                "0xefd784093ddd12e24231fa6b792c09d03a4f7b7e",
                                "0x1d42064fc4beb5f8aaf85f4617ae8b3b5b8bd801",
                                "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35",
                                "0x824a30f2984f9013f2c8d0a29c0a3cc5fd5c0673",
                                "0x3afdc5e6dfc0b0a507a8e023c9dce2cafc310316",
                                "0x127452f3f9cdc0389b0bf59ce6131aa3bd763598",
                                "0x510100d5143e011db24e2aa38abe85d73d5b2177",
                                "0xfb76cd5e9fb9351137c3ce0ac1c23212c46995a7",
                                "0xe42318ea3b998e8355a3da364eb9d48ec725eb45",
                                "0x59354356ec5d56306791873f567d61ebf11dfbd5",
                                "0x735a26a57a0a0069dfabd41595a970faf5e1ee8b",
                                "0xdbac78be00503d10ae0074e5e5873a61fc56647c",
                                "0x9902affdd3b8ef60304958c60377110c6d6ab1df",
                                "0xe21876afd4c632b22870df250e5df1754c1875e8",
                                "0xf3a150716cb43142d51d1940f336cf8e01ab83a0",
                                "0xf4c5e0f4590b6679b3030d29a84857f226087fef",
                                "0x553e9c493678d8606d6a5ba284643db2110df823",
                                "0xa344855388c9f2760e998eb2207b58de6e7d0360",
                                "0xd1d5a4c0ea98971894772dcd6d2f1dc71083c44e",
                                "0x290a6a7460b308ee3f19023d2d00de604bcf5b42",
                                "0xebce363564fa8b55d85aaf681156087116b148db",
                                "0x198063c23ac1317ff3cc57d9f54faac6b675d89f",
                                "0x30ea22c879628514f1494d4bbfef79d21a6b49a2",
                                "0xac4b3dacb91461209ae9d41ec517c2b9cb1b7daf",
                                "0x840deeef2f115cf50da625f7368c24af6fe74410",
                                "0xc5af84701f98fa483ece78af83f11b6c38aca71d",
                                "0x7bea39867e4169dbe237d55c8242a8f2fcdcc387",
                                "0x0c30062368eEfB96bF3AdE1218E685306b8E89Fa",
                            ],
                            batch: 0,
                        },
                    ],
                },
            ],
        },
        filecoin: {
            txConfig: {
                // Estimate gas price: 1.25 nanoFIL => 1.875 nanoFIL min balance (estimate)
                gasLimit: 1_500_000_000n, // 1.5B
                transactionTimeout: EXT_WRITE_DELAY * 2,
                txType: 2,
                waitForConfirmations: 1,
            },
            multicall2Address: "0x732f5baac411e427d130fed03c2e82a3e0d64d35",
            uptimeWebhookUrl: FILECOIN_UPTIME_WEBHOOK_URL,
            batches: {
                0: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : EXT_WRITE_DELAY * workerIndex, // Note: blocks are 30 seconds apart
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0xded8791056AA39581460a005A3c400A281E24bd7", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: [
                                "0x9d8eA62e1264ab667d234b5610774A08e608E3b8",
                                "0x14D44c7Ef81F6c18f5D22e0962f0279D83E80b05",
                            ],
                            batch: 0,
                        },
                    ],
                },
            ],
        },
        rootstock: {
            txConfig: {
                // 1y avg gas price: 0.041 RBTC => 0.00021 RBTC min balance (average case)
                gasLimit: 5_000_000n, // 5M
                transactionTimeout: EXT_WRITE_DELAY * 2,
                waitForConfirmations: 1,
            },
            multicall2Address: "0x996a9858cDfa45Ad68E47c9A30a7201E29c6a386",
            uptimeWebhookUrl: ROOTSTOCK_UPTIME_WEBHOOK_URL,
            batches: {
                0: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : EXT_WRITE_DELAY * workerIndex, // Note: blocks are 30 seconds apart
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0x83B1cF411f57F7373bBFF81dCE81437e768F4252", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: [
                                "0xdC72fed793F1e660ba6096948eA27b0490218cA3",
                                "0x022650756421f2e636d4138054331cbfafb55d9e",
                                "0xd2ffe51ab4e622a411abbe634832a19d919e9c55",
                                "0x549a5d92412161a1a2828549a657a49dd9fa046c",
                                "0xcba7abe98fd6a65259837d76a3409841c1dd4288",
                                "0x069081b7de3edbdfcee339e01c72e4802b259278",
                                "0x3151d3093797412642685ad16e74a83859f2011c",
                                "0xd7a68c0a51c753e1c2931d46c14fadf6ab14ad5e",
                                "0xbe092d38045ef7f9cdc5014278a4239e896bf5ca",
                            ],
                            batch: 0,
                        },
                    ],
                },
            ],
        },
        scroll: {
            txConfig: {
                // Historical max gas price: 300 Gwei => 1.5 ETH min balance (worst case)
                // 1y avg gas price: 0.62 Gwei => 0.0032 ETH min balance (average case)
                gasLimit: 5_000_000n, // 5M
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: "0x3b615B1AC55bc34e51a81D3dea67467F32bcb8C2",
            uptimeWebhookUrl: SCROLL_UPTIME_WEBHOOK_URL,
            batches: {
                0: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0xeC3E5eeC51D8C3D4f03DABB84B4Db313a739f377", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 150n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: [
                                "0x813df550a32d4a9d42010d057386429ad2328ed9",
                                "0x7211c32bfc1841cab1158d18fee62c9b8905ddfe",
                                "0x79525529166d9b9fb3fa05ab2e1a11c8d758809f",
                                "0xbaf7b19598e10eeed86289cd8d7d4ee5438715f9",
                                "0x138182900d26adf6272b357faef77757349165a1",
                                "0xca800f2bf3718fd86ad780d846dde346dfce05d6",
                                "0x3cc5375f08d5df15611c3a446d31fa99a08bd182",
                                "0x622cf39de8d94c81b154f27dd6edde4e48d5c473",
                                "0x74df2424b359de3e4299566bf0db93380b8da877",
                                "0x69cd5ed53a3662b54bf8e8a2fa6c48d4c1c08ed3",
                                "0x02ed4bd217c5c27f66f2a461a83dbe0316593729",
                                "0xa2a37b632fe6a1cce7a7874a65f9e4ce62c7f05a",
                                "0xba63cb90f7b46e2b163e8c8adf57cf8b2a73500e",
                                "0xf1783f3377b3a70465c193ef33942c0803121ba0",
                            ],
                            batch: 0,
                        },
                    ],
                },
            ],
        },
        arbitrumOne: {
            txConfig: {
                // Estimated gas price: 0.125 Gwei => 0.0025 ETH min balance (estimate)
                gasLimit: 20_000_000n, // 20M - gas usage fluctuates
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: "0x842eC2c7D803033Edf55E478F461FC547Bc54EB2",
            uptimeWebhookUrl: ARBITRUMONE_UPTIME_WEBHOOK_URL,
            batches: {
                4: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0x54df9e11c7933a9ca3bd1e540b63da15edae40bf", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: [
                                "0xc31e54c7a869b9fcbecc14363cf510d1c41fa443",
                                "0xc6f780497a95e246eb9449f5e4770916dcd6396a",
                                "0xc6962004f452be9203591991d15f6b388e09e8d0",
                                "0xcda53b1f66614552f834ceef361a8d12a0b8dad8",
                                "0x641c00a822e8b671738d32a431a4fb6074e5c79d",
                                "0x2f5e87c9312fa29aed5c179e456625d79015299c",
                                "0xb0f6ca40411360c03d41c5ffc5f179b8403cdcf8",
                                "0x92c63d0e701caae670c9415d91c474f686298f00",
                                "0x8e295789c9465487074a65b1ae9ce0351172393f",
                                "0xdbaeb7f0dfe3a0aafd798ccecb5b22e708f7852c",
                                "0xb791ad21ba45c76629003b4a2f04c0d544406e37",
                                "0x17c14d2c404d167802b16c450d3c99f88f2c4f4d",
                                "0x59d72ddb29da32847a4665d08ffc8464a7185fae",
                                "0xbe3ad6a5669dc0b8b12febc03608860c31e2eef6",
                                "0x8c9d230d45d6cfee39a6680fb7cb7e8de7ea8e71",
                                "0x1aeedd3727a6431b8f070c0afaa81cc74f273882",
                                "0x468b88941e7cc0b88c1869d68ab6b570bcef62ff",
                                "0x0e4831319a50228b9e450861297ab92dee15b44f",
                                "0x446bf9748b4ea044dd759d9b9311c70491df8f29",
                                "0xc7341e85996eeb05897d3dec79448b6e4ccc09cf",
                                "0xaa9e653252ed9e87a9bd545b974efbfb2f389f3f",
                                "0x81c48d31365e6b526f6bbadc5c9aafd822134863",
                                "0x149e36e72726e0bcea5c59d40df2c43f60f5a22d",
                                "0x97bca422ec0ee4851f2110ea743c1cd0a14835a1",
                                "0x7cf803e8d82a50504180f417b8bc7a493c0a0503",
                                "0x35218a1cbac5bbc3e57fd9bd38219d37571b3537",
                                "0xaeeceec4b31d3c1057210115bf176ebb05b9805d",
                                "0xc91b7b39bbb2c733f0e7459348fd0c80259c8471",
                                "0x1f497a4dd4bfc984296a72358363db405be6259b",
                                "0xa67f72f21bd9f91db2da2d260590da5e6c437009",
                                "0xac70bd92f89e6739b3a08db9b6081a923912f73d",
                                "0xc473e2aee3441bf9240be85eb122abb059a3b57c",
                                "0x53c6ca2597711ca7a73b6921faf4031eedf71339",
                                "0xc82819f72a9e77e2c0c3a69b3196478f44303cf4",
                                "0x91308bc9ce8ca2db82aa30c65619856cc939d907",
                                "0xd37af656abf91c7f548fffc0133175b5e4d3d5e6",
                                "0x03a33aee5f4947b9c1cde042c5f037d1be304be7",
                                "0xd02a4969dc12bb889754361f8bcf3385ac1b2077",
                                "0x52f9d14bed8ce6536da063aaf274ae2747ef4853",
                                "0x135e49cc315fed87f989e072ee11132686cf84f3",
                                "0x92fd143a8fa0c84e016c2765648b9733b0aa519e",
                                "0xaebdca1bc8d89177ebe2308d62af5e74885dccc3",
                                "0x655c1607f8c2e73d5b4ddabce9ba8792b87592b6",
                                "0xcf2f020eeee1c1774c86ea70691913ae5e8affcb",
                                "0x7f580f8a02b759c350e6b8340e7c2d4b8162b6a9",
                                "0x80a9ae39310abf666a87c743d6ebbd0e8c42158e",
                                "0xa961f0473da4864c5ed28e00fcc53a3aab056c1b",
                                "0xe3466d556a89adc56d6db0b9f0610ed747226ddd",
                                "0x30ab5db560b5f85cabbafc2882c98a4e72903bec",
                                "0x681c2a4d924223563dbe35da64e9a0f6a4967fae",
                                "0x14353445c8329df76e6f15e9ead18fa2d45a8bb6",
                                "0x48b0ab72c2591849e678e7d6f272b75ef9b863f7",
                                "0x6f38e884725a116c9c7fbf208e79fe8828a2595f",
                                "0x47b6de853d181626918eecb1e0ed946a5cf96449",
                                "0x4fd47e5102dfbf95541f64ed6fe13d4ed26d2546",
                                "0x09ba302a3f5ad2bf8853266e271b005a5b3716fe",
                                "0x8f5ccd50f076665c67bb3f21eb3eff3bbc8bf078",
                                "0x0491830925ae61db25c01538874c8af9e53352de",
                                "0xa8bd646f72ea828ccbc40fa2976866884f883409",
                                "0x74d0ae8b8e1fca6039707564704a25ad2ee036b0",
                                "0x2f020e708811c054f146eebcc4d5a215fd4eec26",
                                "0x68c685fd52a56f04665b491d491355a624540e85",
                                "0x536f4b1cbeda8387a57b45a549ca6b250f7f1d90",
                                "0x4992de010f1f1b69f7c650b13fb279aa5ee7fef3",
                                "0xbaaf1fc002e31cb12b99e4119e5e350911ec575b",
                                "0x5969efdde3cf5c0d9a88ae51e47d721096a97203",
                                "0x6cf5ed8ac4b99d1bb55e5b6a6644550daacbd527",
                                "0xa169d1ab5c948555954d38700a6cdaa7a4e0c3a0",
                                "0xf782cd5748bfc21d30b03174bb3e30fdd111a897",
                                "0x021a4e8adce1dd0cfac1d8561005c1e1ffc55608",
                                "0xe754841b77c874135caca3386676e886459c2d61",
                                "0xc725f30f8e692387577622edbcd5216b0f5cf524",
                                "0xc24f7d8e51a64dc1238880bd00bb961d54cbeb29",
                                "0xdd092f5dce127961af6ebe975978c084c935bcc8",
                                "0x05bbaaa020ff6bea107a9a1e06d2feb7bfd79ed2",
                                "0x1dfc1054e0e2a10e33c9ca21aad5aa8a1cce91e3",
                                "0xe8629b6a488f366d27dad801d1b5b445199e2ada",
                                "0x4762a162bb535b736b83d49a87b3e1ae3267c80c",
                                "0x42161084d0672e1d3f26a9b53e653be2084ff19c",
                                "0x7c06736e41236fecd681dd3353aa77ecd19ea565",
                                "0x56ebd63a756b94d3de9cea194896b4920b64fb01",
                                "0x0ce541eac2adf14c8eeeb36d588a5db21df9e6c6",
                                "0x3d18c836be1674e8ecc6906224c3e871a1b3a13f",
                                "0x037c7a3dc49da6b20f0a964e7b3a8fd37bcea39a",
                                "0xc6af8e73e2261264ef95466b97b13e03bd88165e",
                                "0x833dd4616efbb74ab8dc3a639e4e449137fb534e",
                                "0x84436a2af97f37018db116ae8e1b691666db3d00",
                                "0x0ad1e922e764df5ab6d636f5d21ecc2e41e827f0",
                                "0x37af2c75a71f90e4e289a5ee56c13111bf4ddff4",
                                "0xee9bf1d1e23784067bd7b0b3496f865038b766eb",
                                "0x0dcf98667c5400b7bc8de4ec2e4d03c5cd11fa85",
                                "0x16c1344e8537aadbbf0ffb166b46acf9acf2fedd",
                                "0xc0cf0f380ddb44dbcaf19a86d094c8bba3efa04a",
                                "0x82f223c5f64d892785e7a5d11470ffa81ebce614",
                                "0xbbe36e6f0331c6a36ab44bc8421e28e1a1871c1e",
                                "0xe4d9faddd9bca5d8393bee915dc56e916ab94d27",
                                "0x14af1804dbbf7d621ecc2901eef292a24a0260ea",
                                "0xe9e6b9aaafaf6816c3364345f6ef745ccfc8660a",
                                "0x0fd451d06e0fa0ad2bae12e8190c6001a4531ef2",
                                "0x4b5b629ddab83fae1edff103ab4cc905cbec70c4",
                                "0x5ee828101b4b6092be915f4cfefe9a5d178e5454",
                                "0x4cd207d3bb951dffee3fae27a0be70019bb72666",
                                "0xa59b4cfaa5ea1506178e82d53a768fcf69945d65",
                            ],
                            batch: 4,
                        },
                    ],
                },
            ],
        },
        optimism: {
            txConfig: {
                // Estimated gas price: 0.01 Gwei => 0.00005 ETH min balance (estimate)
                gasLimit: 5_000_000n, // 5M
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: "0xFbdd194376de19a88118e84E279b977f165d01b8",
            uptimeWebhookUrl: OPTIMISM_UPTIME_WEBHOOK_URL,
            batches: {
                3: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0x54df9e11c7933a9ca3bd1e540b63da15edae40bf", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: [
                                "0x68f5c0a2de713a54991e01858fd27a3832401849",
                                "0x85149247691df622eaf1a8bd0cafd40bc45154a9",
                                "0xfc1f3296458f9b2a27a0b91dd7681c4020e09d05",
                                "0x1c3140ab59d6caf9fa7459c6f83d4b52ba881d36",
                                "0x1fb3cf6e48f1e7b10213e7b6d87d4c073c7fdb7b",
                                "0x85c31ffa3706d1cce9d525a00f1c7d4a2911754c",
                                "0x2ab22ac86b25bd448a4d9dc041bd2384655299c4",
                                "0x0392b358ce4547601befa962680bede836606ae2",
                                "0x04f6c85a1b00f6d9b75f91fd23835974cc07e65c",
                                "0xf1f199342687a7d78bcc16fce79fa2665ef870e1",
                                "0x8323d063b1d12acce4742f1e3ed9bc46d71f4222",
                                "0x73b14a78a0d396c521f954532d43fd5ffe385216",
                                "0xb533c12fb4e7b53b5524eab9b47d93ff6c7a456f",
                                "0x03af20bdaaffb4cc0a521796a223f7d85e2aac31",
                                "0xa73c628eaf6e283e26a7b1f8001cf186aa4c0e8e",
                                "0xbf16ef186e715668aa29cef57e2fd7f9d48adfe6",
                                "0x252cbdff917169775be2b552ec9f6781af95e7f6",
                                "0x535541f1aa08416e69dc4d610131099fa2ae7222",
                                "0xb589969d38ce76d3d7aa319de7133bc9755fd840",
                                "0xc858a329bf053be78d6239c4a4343b8fbd21472b",
                                "0xd1f1bad4c9e6c44dec1e9bf3b94902205c5cd6c3",
                                "0x95d9d28606ee55de7667f0f176ebfc3215cfd9c0",
                                "0x766854992bd5363ebeeff0113f5a5795796befab",
                                "0xd28f71e383e93c570d3edfe82ebbceb35ec6c412",
                                "0xa8a5356ee5d02fe33d72355e4f698782f8f199e8",
                                "0x4ce4a1a593ea9f2e6b2c05016a00a2d300c9ffd8",
                                "0x9438a9d1bdeece02ed4431ac59613a128201e0b9",
                                "0xa7bb0d95c6ba0ed0aca70c503b34bc7108589a47",
                                "0xadb35413ec50e0afe41039eac8b930d313e94fa4",
                                "0xb2ac2e5a3684411254d58b1c5a542212b782114d",
                                "0x8eda97883a1bc02cf68c6b9fb996e06ed8fdb3e5",
                                "0x19ea026886cbb7a900ecb2458636d72b5cae223b",
                                "0x5adba6c5589c50791dd65131df29677595c7efa7",
                                "0x730691cdac3cbd4d41fc5eb9d8abbb0cea795b94",
                                "0x6168ec836d0b1f0c37381ec7ed1891a412872121",
                                "0x2ae3d6096d8215ac2acddf30c60caa984ea5debe",
                                "0x394a9fcbab8599437d9ec4e5a4a0eb7cb1fd2f69",
                                "0xd52533a3309b393afebe3176620e8ccfb6159f8a",
                                "0xbf595eb9a512b1c274125264aef84a2847158eb3",
                                "0xf44acaa38be5e965c5ddf374e7a2ba270e580684",
                                "0xac85eaf55e9c60ed40a683de7e549d23fdfbeb33",
                                "0x250e21dddd306579458cf025c5e230665171fb31",
                                "0xe9e3893921de87b1194a8108f9d70c24bde71c27",
                                "0xc1738d90e2e26c35784a0d3e3d8a9f795074bca4",
                                "0x37ffd11972128fd624337ebceb167c8c0a5115ff",
                                "0x2e2d190ad4e0d7be9569baebd4d33298379b0502",
                                "0xdc6cb16b8d635ef255cef370263c35ac0007186a",
                                "0x8531e48a8611729185be9eaad945acbd6b32e256",
                                "0x1d751bc1a723accf1942122ca9aa82d49d08d2ae",
                                "0x0ca747e5c527e857d8a71b53b6efbad2866b9e04",
                                "0x1d789e3b2146cfa96761996d14cd3cc54a8b133a",
                            ],
                            batch: 3,
                        },
                    ],
                },
            ],
        },
        bsc: {
            txConfig: {
                // 1y avg gas price: 5.8 Gwei => 0.029 BNB min balance (average case)
                gasLimit: 5_000_000n, // 5M
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: MULTICALL3_ADDRESS,
            uptimeWebhookUrl: BSC_UPTIME_WEBHOOK_URL,
            batches: {
                0: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0x19b9bD76028caB6F414ed1Fc57400b75B5cA0627", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: [
                                "0x47a90a2d92a8367a91efa1906bfc8c1e05bf10c4",
                                "0x28df0835942396b7a1b7ae1cd068728e6ddbbafd",
                                "0xb125aa15ad943d96e813e4a06d0c34716f897e26",
                                "0x0f338ec12d3f7c3d77a4b9fcc1f95f3fb6ad0ea6",
                                "0xf9878a5dd55edc120fde01893ea713a4f032229c",
                                "0x2c3c320d49019d4f9a92352e947c7e5acfe47d68",
                                "0xc98f01bf2141e1140ef8f8cad99d4b021d10718f",
                                "0x3fb2623567e21f8c50f0ae86f54ef4849b4eb47b",
                                "0x5289a8dbf7029ee0b0498a84777ed3941d9acfec",
                                "0xf2c9339945bff71dd0bffd3c142164112cd05dc6",
                                "0xf5c616e7b58226b8081dcc7e4a7123a63734eef6",
                                "0x6fe9e9de56356f7edbfcbb29fab7cd69471a4869",
                                "0x0525190f3532b99b32fb42b2697807bd80268565",
                                "0xcb99fe720124129520f7a09ca3cbef78d58ed934",
                                "0x06396509195eb9e07c38a016694dc9ff535b128a",
                            ],
                            batch: 0,
                        },
                    ],
                },
            ],
        },
        base: {
            txConfig: {
                // Estimated gas price: 1.15 Gwei => 0.00575 ETH min balance (estimate)
                gasLimit: 5_000_000n, // 5M
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: MULTICALL3_ADDRESS,
            uptimeWebhookUrl: BASE_UPTIME_WEBHOOK_URL,
            batches: {
                0: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0xfF8b754c64e9a8473Bd6E1118d0eaC67F0A8Ae27", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: [
                                "0x4c36388be6f416a29c8d8eee81c771ce6be14b18",
                                "0x06959273e9a65433de71f5a452d529544e07ddd0",
                                "0xc9034c3e7f58003e6ae0c8438e7c8f4598d5acaa",
                                "0xd0b53d9277642d899df5c87a3966a349a798f224",
                                "0x48413707b70355597404018e7c603b261fcadf3f",
                                "0x4b0aaf3ebb163dd45f663b38b6d93f6093ebc2d3",
                                "0x0d5959a52e7004b601f0be70618d01ac3cdce976",
                                "0x10648ba41b8565907cfa1496765fa4d95390aa0d",
                                "0x22f9623817f152148b4e080e98af66fbe9c5adf8",
                                "0xae2ce200bdb67c472030b31f602f0756c9aeb61c",
                                "0xd5638bf58e2762fa40bd753490f693cbb1986709",
                                "0x018046b1d182f7c0978c07610e1173c8e11913fd",
                                "0x24e1cbd6fed006ceed9af0dce688acc7951d57a9",
                                "0x97a25cc2793f0ffa90e1667cf7b3c1f130737189",
                                "0xfcc89a1f250d76de198767d33e1ca9138a7fb54b",
                                "0x3bc5180d5439b500f381f9a46f15dd6608101671",
                                "0xa555149210075702a734968f338d5e1cbd509354",
                                "0x5197195ac878741b192f84ff6d7da5a85b9e634b",
                                "0xe745a591970e0fa981204cf525e170a2b9e4fb93",
                                "0x7e904aaf3439402eb21958fe090bd852d5e882cf",
                            ],
                            batch: 0,
                        },
                    ],
                },
            ],
        },
        mantaPacific: {
            txConfig: {
                // Estimated gas price: 0.125 Gwei => 0.00065 ETH min balance (estimate)
                gasLimit: 5_000_000n, // 5M
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: MULTICALL3_ADDRESS,
            batches: {
                0: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0xFE83E1DDa189D71093f2a716A4D01d591d6Ca66C", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: [],
                            batch: 0,
                        },
                    ],
                },
            ],
        },
        blast: {
            txConfig: {
                gasLimit: 5_000_000n, // 5M
                transactionTimeout: STD_WRITE_DELAY * 2,
            },
            multicall2Address: MULTICALL3_ADDRESS,
            batches: {
                0: {
                    ...STANDARD_BATCH_CONFIG,
                    pollingInterval: workerIndex == 1 ? 1_000 : 30_000,
                    writeDelay: workerIndex == 1 ? 1_000 : STD_WRITE_DELAY * workerIndex,
                },
            },
            oracles: [
                {
                    // type: aci-address
                    address: "0x0337d36A3dF76d882369E3cBF984a2EA40f6636F", // LimitOrderRegistry
                    txConfig: {
                        gasPriceMultiplierDividend: 125n + BigInt(workerIndex - 1) * 25n,
                        gasPriceMultiplierDivisor: 100n,
                    },
                    tokens: [
                        {
                            address: ["0xf52b4b69123cbcf07798ae8265642793b2e8990c"],
                            batch: 0,
                        },
                    ],
                },
            ],
        },
    },
};

export default config;