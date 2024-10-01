import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { doInscribe, shouldInscribe } from './utils/commitments';
import { ApiPromise, WsProvider } from '@polkadot/api';
import { sleep } from './utils/misc';
import axios from 'axios';
import path from 'path';
import { MILLISECONDS_PER_BLOCK } from './utils/constants';
import { logger } from './logger';

async function getIpfsIdFromCluster(ipfsClusterRestUrl: string) {
    try {
        const endpoint = path.join(ipfsClusterRestUrl, 'id');
        const response = await axios.get(endpoint);
        return String(response.data.id);
    } catch (error) {
        console.error('Error fetching IPFS ID from cluster:', error);
        throw error;
    }
}

async function main() {
    const argv = await yargs(hideBin(process.argv))
        .option('netuid', {
            alias: 'n',
            type: 'number',
            description: 'Subnet Identifier',
        })
        .option('ipfs-cluster-id', {
            alias: 'i',
            type: 'string',
            description: 'IPFS Cluster peer ID',
        })
        .option('bittensor-hotkey-name', {
            alias: 'h',
            type: 'string',
            description: 'Name of the bittensor hotkey used to sign the inscribe transaction',
        })
        .option('bittensor-coldkey-name', {
            alias: 'c',
            type: 'string',
            description: 'Name of the bittensor coldkey/wallet that owns the hotkey used to sign the inscribe transaction',
        })
        .option('inscribe-rate-limit', {
            alias: 'r',
            type: 'number',
            description: 'Inscribe rate limit in Bittensor blocks (1 block = 12 seconds)',
            default: 100   // default rate limit on chain    
        })
        .option('ws-url', {
            alias: 'w',
            type: 'string',
            description: 'Websocket URL of the Bittensor node to connect to',
            default: 'ws://127.0.0.1:9946'
        })
        .option('inscribe-fail-sleep-ms', {
            alias: 'f',
            type: 'number',
            description: 'Sleep time in milliseconds after a failed inscribe attempt',
            default: 10_000
        })
        .option('ipfs-cluster-rest-url', {
            alias: 'u',
            type: 'string',
            description: 'REST URL of the IPFS Cluster to fetch the IPFS ID from',
            default: 'http://cluster:9094'
        })
        .option('log-level', {
            alias: 'l',
            type: 'string',
            description: 'Log level',
            default: 'info'
        })
        .demandOption(['netuid', 'bittensor-hotkey-name', 'bittensor-coldkey-name',])
        .help()
        .parse();

    const parsedInputArgs = argv

    logger.info(`Setting log level to: ${parsedInputArgs.logLevel}`)
    logger.level = parsedInputArgs.logLevel

    logger.info(`Parsed input args:\n${JSON.stringify(parsedInputArgs, null, 2)}`)

    const ipfsClusterId = parsedInputArgs.ipfsClusterId ?? await getIpfsIdFromCluster(parsedInputArgs.ipfsClusterRestUrl)

    logger.info(`Using IPFS Cluster ID: ${ipfsClusterId}`)

    while (true) {
        const provider = new WsProvider(argv.wsUrl);

        const api = await ApiPromise.create({ provider });

        try {
            const sleepMs = await shouldInscribe(
                api,
                parsedInputArgs.netuid,
                ipfsClusterId,
                parsedInputArgs.bittensorColdkeyName,
                parsedInputArgs.bittensorHotkeyName,
                parsedInputArgs.inscribeRateLimit,
            )


            if (sleepMs === null) {

                logger.info(`Inscribing ${ipfsClusterId} at ${new Date().toLocaleTimeString()}`)
                await doInscribe(
                    api,
                    parsedInputArgs.netuid,
                    ipfsClusterId,
                    parsedInputArgs.bittensorColdkeyName,
                    parsedInputArgs.bittensorHotkeyName,
                )

                logger.info(`Inscribed ${ipfsClusterId} at ${new Date().toLocaleTimeString()}`);
            } else {
                const nextAllowedTime = new Date(Date.now() + sleepMs)
                const blocksLeft = sleepMs / MILLISECONDS_PER_BLOCK
                logger.info(`Sleeping for ${sleepMs}ms (${sleepMs / 1000 / 60} minutes)..., next inscribe allowed at ${nextAllowedTime.toLocaleTimeString()} in ~${blocksLeft} blocks`)
                await sleep(sleepMs)
            }
        } catch (e) {
            logger.error(e)

            logger.info(`Unable to inscribe ${ipfsClusterId}. Sleeping for ${parsedInputArgs.inscribeFailSleepMs}ms before retrying...`)
            await sleep(parsedInputArgs.inscribeFailSleepMs)
        }
    }
}


if (require.main === module) {
    main().catch(console.error);
}