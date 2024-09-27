import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { doInscribe } from './utils/commitments';
import { ApiPromise, WsProvider } from '@polkadot/api';
import z from 'zod';
import { sleep } from './utils/misc';

async function main() {
    const argv = await yargs(hideBin(process.argv))
        .option('netuid', {
            alias: 'n',
            type: 'number',
            description: 'Subnet Identifier',
        })
        .option('ipfs-id', {
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
            default: 5_000
        })
        .demandOption(['netuid', 'ipfs-id', 'bittensor-hotkey-name'])
        .help()
        .parse();

    const inputSchema = z.object({
        netuid: z.number().gte(0),
        ipfsId: z.string(),
        bittensorHotkeyName: z.string(),
        bittensorColdkeyName: z.string(),
        inscribeRateLimit: z.number().gt(0),
        wsUrl: z.string(),
        inscribeFailSleepMs: z.number().gte(0),
    });

    const parsedInputArgs = inputSchema.parse(argv);

    console.log("Parsed input args:", parsedInputArgs);

    while (true) {
        const provider = new WsProvider(argv.wsUrl);

        const api = await ApiPromise.create({ provider });

        const sleepMs = parsedInputArgs.inscribeRateLimit * 12 * 1000

        try {


            await doInscribe(
                api,
                parsedInputArgs.netuid,
                parsedInputArgs.ipfsId,
                parsedInputArgs.bittensorColdkeyName,
                parsedInputArgs.bittensorHotkeyName,
            )

            console.log(`Inscribed ${parsedInputArgs.ipfsId} at ${new Date().toISOString()}`);

            console.log(`Sleeping for ${sleepMs}ms...`)
            await sleep(sleepMs)
        } catch (e) {
            console.error(e)

            console.log(`Unable to inscribe ${parsedInputArgs.ipfsId}. Sleeping for ${parsedInputArgs.inscribeFailSleepMs}ms before retrying...`)
            await sleep(parsedInputArgs.inscribeFailSleepMs)
        }
    }
}


if (require.main === module) {
    main().catch(console.error);
}