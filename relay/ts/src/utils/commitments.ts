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

    console.log('Commitment Info:', JSON.stringify(commitmentInfo, null, 2));

    const tx = api.tx.commitments.setCommitment(netuid, commitmentInfo);

    console.log("Submitting transaction... (note that the tx is signed right before sending, which is why it's not shown as signed here)\n", tx.toHuman())
    return new Promise<void>((resolve, reject) => {
        tx.signAndSend(hotkey, ({ status, events }) => {
            console.log("Transaction status:", status.toHuman())
            if (status.isFinalized) {
                let foundCommitmentEvent = false;
                events.forEach(({ event }) => {
                    if (api.events.commitments.Commitment.is(event)) {
                        console.log(`Commitment set for ${hotkey.address}. Block hash: ${status.asFinalized}`);
                        foundCommitmentEvent = true;
                    }
                });
                if (!foundCommitmentEvent) {
                    console.error('No commitment event found in extrinsic events');
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
        console.error('rawBytes is undefined for commitment info:', commitmentInfo.toHuman());
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
        console.error('rawBytes is undefined for commitment info:', commitmentInfo.toHuman());
        return null;
    }
    if (rawBytes.length !== 38) {
        console.error('Invalid commitment info length:', rawBytes.length, "expected 38");
        return null
    }
    const ipfsId = base58Encode(rawBytes);
    return {
        ipfsClusterId: ipfsId,
        inscribedAt: (commitmentInfo.block as BN).toNumber()
    };
}

export async function getInscription(api: ApiPromise, netuid: number, bittensorHotkeyName: string) {
    const hotkey = loadHotkey(COLDKEY_NAME, bittensorHotkeyName);
    const inscriptions = await api.query.inscriptions.inscriptionOf.entries(hotkey.address);
    return inscriptions;
}

export async function queryCommitmentsForIpfsClusterIds(api: ApiPromise, netuid: number) {
    const commitments = await api.query.commitments.commitmentOf.entries(netuid);

    const ipfsClusterIdCommitments: IpfsInscription[] = [];
    commitments.forEach(([key, commitment]) => {
        const [_, accountId] = key.args;
        console.log('Account:', accountId.toString());
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
            console.log(`No commitment found for ${accountId.toString()}`);
        }
        console.log('---');
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
    console.log(`Loaded hotkey: ${hotkey.address}`);
    const ipfsBytes = decodeIpfsId(ipfsId);
    console.log(`IPFS bytes (hex): ${u8aToHex(ipfsBytes)}`);
    console.log(`IPFS bytes length: ${ipfsBytes.length}`);
    await inscribeIpfsClusterId(api, netuid, hotkey, ipfsBytes);
}

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