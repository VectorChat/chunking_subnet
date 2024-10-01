import { ApiPromise, WsProvider, Keyring } from '@polkadot/api';
import { KeyringPair } from '@polkadot/keyring/types';
import { BN, hexToU8a, u8aToHex } from '@polkadot/util';
import { base58Decode, base58Encode } from '@polkadot/util-crypto';
import fs from 'fs';
import path from 'path';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import "../__generated__/interfaces/augment-api.ts"
import { IpfsCommitment, IpfsInscription } from '../types.ts';
import { MILLISECONDS_PER_BLOCK } from './constants.ts';
import { logger } from '../logger.ts';

const WS_URL = 'ws://127.0.0.1:9946';
const COLDKEY_NAME = 'owner-localnet';

const TEST_IPFS_CLUSTER_IDS = [
    '12D3KooWMmZqkk1Ek8vonm3FE3rNMLqqxwspRiKwx5ZD5yn4tPLG',
    '12D3KooWS92ucchSQ4djSrhJdjVAnd2TVPvMAhSeveRvdDa1bHG5',
    '12D3KooWAxXgjXS2vBj5wbW8rm1uFD9s1VKbt5wqWvaHv3g9NpGy',
    '12D3KooWEUgzttDhsHR5KtQcmCUVEZJWbAiQyWjmboMSfDTegqGk',
];

const BT_HOTKEY_NAMES = [
    'validator1',
    'validator2',
    'validator3',
    'validator4',
];

export function loadHotkey(coldkeyName: string, hotkeyName: string): KeyringPair {
    const keyring = new Keyring({ type: 'sr25519' });
    const filePath = path.join(process.env.HOME!, '.bittensor', 'wallets', coldkeyName, 'hotkeys', hotkeyName);
    const keyData = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const mnemonic = keyData.secretPhrase;
    return keyring.addFromUri(mnemonic);
}

export function decodeIpfsId(ipfsId: string): Uint8Array {
    return base58Decode(ipfsId);
}

export async function inscribeIpfsClusterId(api: ApiPromise, netuid: number, hotkey: KeyringPair, ipfsId: Uint8Array) {
    const rawVariant = `raw${ipfsId.length}`;

    const commitmentInfo = {
        fields: [{
            [rawVariant]: u8aToHex(ipfsId)
        }]
    };

    logger.verbose('Commitment Info:', JSON.stringify(commitmentInfo, null, 2));

    const tx = api.tx.commitments.setCommitment(netuid, commitmentInfo);

    logger.info("Submitting transaction... (note that the tx is signed right before sending, which is why it's not shown as signed here)\n", tx.toHuman())
    return new Promise<void>((resolve, reject) => {
        tx.signAndSend(hotkey, ({ status, events }) => {
            logger.info("Transaction status:", status.toHuman())
            if (status.isFinalized) {
                let foundCommitmentEvent = false;
                events.forEach(({ event }) => {
                    if (api.events.commitments.Commitment.is(event)) {
                        logger.info(`Commitment set for ${hotkey.address}. Block hash: ${status.asFinalized}`);
                        foundCommitmentEvent = true;
                    }
                });
                if (!foundCommitmentEvent) {
                    logger.error('No commitment event found in extrinsic events');
                    reject(new Error('No commitment event found in extrinsic events'));
                } else {
                    resolve();
                }
            }
        }).catch(reject);
    });
}



export function parseIpfsClusterIdFromExtrinsic(commitmentInfo: any): string | null {
    const fields = commitmentInfo?.get('fields');
    const firstField = fields?.[0];
    const rawBytes = firstField?.__internal__raw as Uint8Array;
    if (rawBytes === undefined) {
        logger.error('rawBytes is undefined for commitment info:', commitmentInfo.toHuman());
        return null;
    }
    const ipfsId = base58Encode(rawBytes);
    return ipfsId;
}

export function parseIpfsClusterIdFromStorage(commitmentInfo: any): IpfsCommitment | null {
    const fields = commitmentInfo?.info?.get('fields');

    const firstField = fields?.[0];
    const rawBytes = firstField?.__internal__raw as Uint8Array;

    if (rawBytes === undefined) {
        logger.error('rawBytes is undefined for commitment info:', commitmentInfo.toHuman());
        return null;
    }
    if (rawBytes.length !== 38) {
        logger.error('Invalid commitment info length:', rawBytes.length, "expected 38");
        return null
    }
    const ipfsId = base58Encode(rawBytes);
    return {
        ipfsClusterId: ipfsId,
        inscribedAt: (commitmentInfo.block as BN).toNumber()
    };
}

export async function getInscription(api: ApiPromise, netuid: number, hotkey: string) {
    const commitment = await api.query.commitments.commitmentOf(netuid, hotkey);
    if (commitment.isNone) {
        return null;
    }
    return parseIpfsClusterIdFromStorage(commitment.unwrap());
}

export async function queryCommitmentsForIpfsClusterIds(api: ApiPromise, netuid: number) {
    const commitments = await api.query.commitments.commitmentOf.entries(netuid);

    const ipfsClusterIdCommitments: IpfsInscription[] = [];
    commitments.forEach(([key, commitment]) => {
        const [_, accountId] = key.args;
        logger.verbose(`Querying commitment for account: ${accountId.toString()}`)
        if (commitment.isSome) {
            const commitmentInfo = commitment.unwrap();
            const parsedCommitment = parseIpfsClusterIdFromStorage(commitmentInfo);
            if (parsedCommitment !== null) {
                ipfsClusterIdCommitments.push({
                    ...parsedCommitment,
                    hotkey: accountId.toString()
                });
            }
        } else {
            logger.verbose(`No commitment found for ${accountId.toString()}`);
        }
    });

    return ipfsClusterIdCommitments;
}

export async function doInscribe(api: ApiPromise, netuid: number, ipfsId: string, bittensorColdkeyName: string, bittensorHotkeyName: string) {
    console.log("Doing inscribe", {
        netuid,
        ipfsId,
        bittensorHotkeyName
    });
    const hotkey = loadHotkey(bittensorColdkeyName, bittensorHotkeyName);
    logger.verbose(`Loaded hotkey: ${hotkey.address}`);
    const ipfsBytes = decodeIpfsId(ipfsId);
    logger.verbose(`IPFS bytes (hex): ${u8aToHex(ipfsBytes)}`);
    logger.verbose(`IPFS bytes length: ${ipfsBytes.length}`);
    await inscribeIpfsClusterId(api, netuid, hotkey, ipfsBytes);
}

/**
 * Checks if the given ipfs cluster id is already inscribed for the given netuid and hotkey. Returns the estimated sleep time in seconds until the rate limit completes,
 * if the ipfs cluster id is already inscribed. Else, returns null meaning that the ipfs cluster id should be inscribed.
 *  
 * @param api the polkadot api instance
 * @param netuid the subnet idenitifer
 * @param ipfsClusterId the ipfs cluster id to check for
 * @param bittensorColdkeyName the coldkey name for the bittensor coldkey that owns the hotkey
 * @param bittensorHotkeyName the hotkey name for the bittensor hotkey that will eventually sign the inscription 
 * @param inscribeRateLimit the rate limit for commitments/inscribes in blocks 
 * 
 * @returns the estimated sleep time in milliseconds if the ipfs cluster id is already inscribed
 */
export async function shouldInscribe(api: ApiPromise, netuid: number, ipfsClusterId: string, bittensorColdkeyName: string, bittensorHotkeyName: string, inscribeRateLimit: number): Promise<number | null> {
    const hotkey = loadHotkey(bittensorColdkeyName, bittensorHotkeyName);
    const curInscription = await getInscription(api, netuid, hotkey.address);
    if (curInscription === null) {
        // if the hotkey has no inscription, we should inscribe
        logger.verbose(`No inscription found for ${hotkey.address}`)
        return null;
    }

    const inscribedAt = curInscription.inscribedAt

    logger.verbose(`Inscription found for ${hotkey.address} at block ${inscribedAt} with rate limit ${inscribeRateLimit}`)

    const nextAllowedBlock = inscribedAt + inscribeRateLimit;

    logger.verbose(`Next allowed block: ${nextAllowedBlock}`)

    const currentBlockNumber = await api.query.system.number()

    logger.verbose(`Current block number: ${currentBlockNumber}`)

    if (currentBlockNumber.toNumber() >= nextAllowedBlock) {
        // if the current block number surpasses the rate limit, we should inscribe
        return null;
    }

    const blocksUntilAllowed = nextAllowedBlock - currentBlockNumber.toNumber();

    logger.verbose(`Blocks until allowed: ${blocksUntilAllowed}`)

    const millisecondsUntilAllowed = blocksUntilAllowed * MILLISECONDS_PER_BLOCK;

    logger.verbose(`Milliseconds until allowed: ${millisecondsUntilAllowed}`)

    return millisecondsUntilAllowed;
}

/**
 * Main function, used for testing
 */
async function main() {

    const argv = yargs(hideBin(process.argv))
        .option('netuid', {
            alias: 'n',
            type: 'number',
            description: 'Network UID',
            default: 1
        })
        .command('set', 'Set commitments', {}, async (argv) => {
            const provider = new WsProvider(WS_URL);
            const api = await ApiPromise.create({ provider });

            for (let i = 0; i < TEST_IPFS_CLUSTER_IDS.length; i++) {
                const ipfsId = TEST_IPFS_CLUSTER_IDS[i];
                const hotkeyName = BT_HOTKEY_NAMES[i];
                const coldkeyName = COLDKEY_NAME;
                console.log(`Processing IPFS ID: ${ipfsId}`);
                try {
                    await doInscribe(api, argv.netuid as number, ipfsId, coldkeyName, hotkeyName);
                } catch (e) {
                    console.error(`Error inscribing ${ipfsId}:`, e)
                }
            }

            await api.disconnect();
        })
        .command('query', 'Query commitments', {}, async (argv) => {
            const provider = new WsProvider(WS_URL);
            const api = await ApiPromise.create({ provider });

            const ipfsClusterIdCommitments = await queryCommitmentsForIpfsClusterIds(api, argv.netuid as number);
            console.log('IPFS Cluster ID Commitments:', ipfsClusterIdCommitments);

            await api.disconnect();
        })
        .demandCommand(1, 'You need to specify a command')
        .help()
        .parse();
}

if (require.main === module) {
    main().catch(console.error);
}