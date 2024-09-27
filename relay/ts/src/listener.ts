import { ApiPromise, WsProvider } from "@polkadot/api";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import { queryCommitmentsForIpfsClusterIds } from "./utils/commitments";
import { fromRao } from "./utils/rao";
import { IpfsInscription } from "./types";
import axios from "axios";


const latestInscriptionMap: Record<string, IpfsInscription> = {};

/**
 * Updates the latest inscription map for the given netuid. Fetches the latest commitments via a websocket call. 
 *  
 * @param api 
 * @param netuid 
 */
async function updateLatestInscriptionMap(api: ApiPromise, netuid: number) {
    const ipfsClusterIdCommitments = await queryCommitmentsForIpfsClusterIds(api, netuid)

    for (const inscription of ipfsClusterIdCommitments) {
        latestInscriptionMap[inscription.hotkey] = inscription
    }
    console.log("updated latest inscription map with", ipfsClusterIdCommitments.length, "commitments")
}

/**
 * The set of IPFS Cluster IDs that are currently trusted.
 */
const trustedIpfsClusterIds: Set<string> = new Set()

/**
 * Checks if the given IPFS Cluster ID can be trusted based on the given criteria.
 * 
 * @param api - The Polkadot API instance.
 * @param hotkey - The hotkey associated with the IPFS Cluster ID. This is the account that should have set the commitment on chain.
 * @param minStake - The minimum stake required for a validator to be considered trusted.
 * @param netuid - The subnet unique identifier.
 * @param inscription - The IPFS inscription data.
 * @param currentBlockNumber - The current block number at the time of checking trustworthiness.
 * @param timeWindow - The time window in blocks to consider the inscription valid.
 * @returns - True if the IPFS Cluster ID can be trusted, false otherwise.
 */
async function canBeTrusted(api: ApiPromise, hotkey: string, minStake: number, netuid: number, inscription: IpfsInscription, currentBlockNumber: number, timeWindow: number) {
    console.log("Checking if", inscription.ipfsClusterId, "can be trusted")
    console.log("getting uid for", {
        netuid,
        hotkey,
    })
    const uid = await api.query.subtensorModule.uids(netuid, hotkey)
    if (uid.isNone) {
        console.log("Hotkey", hotkey, "not found in metagraph")
        return false;
    }
    console.log("uid", uid.unwrap().toNumber())
    const validatorPermitArray = await api.query.subtensorModule.validatorPermit(netuid)
    const hasValidatorPermit = validatorPermitArray[uid.unwrap().toNumber()].toPrimitive()
    if (!hasValidatorPermit) {
        console.log("Hotkey", hotkey, "does not have a validator permit")
        return false;
    }
    const stakeRao = await api.query.subtensorModule.totalHotkeyStake(hotkey)
    const stakeTao = fromRao(stakeRao)
    if (stakeTao.lt(minStake)) {
        console.log("Hotkey", hotkey, "has stake", stakeTao, "which is less than", minStake)
        return false;
    }

    if (inscription.inscribedAt < currentBlockNumber - timeWindow) {
        console.log("Hotkey", hotkey, "inscribed at", inscription.inscribedAt, "which is older than", timeWindow, "blocks", {
            inscribedAt: inscription.inscribedAt,
            timeWindow,
            currentBlockNumber,
        })
        return false;
    }
    console.log(`Inscription will expire in ${inscription.inscribedAt + timeWindow - currentBlockNumber} blocks`, {
        inscribedAt: inscription.inscribedAt,
        timeWindow,
        currentBlockNumber,
    })

    return true;
}

/**
 * Updates the set of trusted IPFS Cluster IDs based on the latest inscription map and the given criteria that determines trustworthiness.
 * 
 * @param api - The Polkadot API instance.
 * @param netuid - The subnet unique identifier.
 * @param minStake - The minimum stake required for a validator to be considered trusted.
 * @param timeWindow - The time window in blocks to consider the inscription valid.
 * @param currentBlockNumber - The current block number at the time of checking trustworthiness.
 * @returns - True if the set of trusted IPFS Cluster IDs changed, false otherwise.
 */
async function updateTrustedIpfsClusterIds(api: ApiPromise, netuid: number, minStake: number, timeWindow: number, currentBlockNumber: number) {
    let wasChange = false;
    for (const [hotkey, ipfsInscription] of Object.entries(latestInscriptionMap)) {
        console.log("handling", ipfsInscription)
        const wasTrusted = trustedIpfsClusterIds.has(ipfsInscription.ipfsClusterId)

        const currentlyTrusted = await canBeTrusted(api, hotkey, minStake, netuid, ipfsInscription, currentBlockNumber, timeWindow)

        console.log({
            wasTrusted,
            currentlyTrusted,
        })

        if (wasTrusted && !currentlyTrusted) {
            console.log("Removing trusted IPFS Cluster ID", ipfsInscription.ipfsClusterId)
            trustedIpfsClusterIds.delete(ipfsInscription.ipfsClusterId)
            wasChange = true;
        } else if (!wasTrusted && currentlyTrusted) {
            console.log("Adding trusted IPFS Cluster ID", ipfsInscription.ipfsClusterId)
            trustedIpfsClusterIds.add(ipfsInscription.ipfsClusterId)
            wasChange = true;
        }
    }

    return wasChange;
}

/**
 * Checks if two sets are equal.
 * 
 * @param a - The first set.
 * @param b - The second set.
 * @returns - True if the sets are equal, false otherwise.
 */
function setIsEqual(a: Set<string>, b: Set<string>) {
    return a.size === b.size && Array.from(a).every((value) => b.has(value));
}

// /**
//  * Updates the service.json file with the trusted peers. It only updates the file if the current set of trusted peers is different from the stored set of trusted peers.
//  * 
//  * @param serviceJsonFilePath - The path to the service.json file.
//  * @param alwaysUpdate - Whether to always update the service.json file.
//  * @returns - True if the service.json file was updated, false otherwise.
//  */
// function updateTrustedPeersInServiceJsonFile(serviceJsonFilePath: string, alwaysUpdate = false) {
//     const serviceJson = JSON.parse(fs.readFileSync(serviceJsonFilePath, 'utf8'));

//     const serviceJsonTrustedPeers = serviceJson.consensus.crdt.trusted_peers || []

//     const trustedPeers = Array.from(trustedIpfsClusterIds)

//     const currentEqualsStored = setIsEqual(new Set(serviceJsonTrustedPeers), trustedIpfsClusterIds)

//     const doUpdate = alwaysUpdate || !currentEqualsStored

//     if (!doUpdate) {
//         console.log("service.json already up to date with trusted peers", trustedPeers)
//         return false;
//     }

//     console.log("updating service json file because alwaysUpdate is true or current does not match stored", {
//         alwaysUpdate,
//         currentEqualsStored,
//     })

//     serviceJson.consensus.crdt.trusted_peers = trustedPeers

//     fs.writeFileSync(serviceJsonFilePath, JSON.stringify(serviceJson, null, 2));
//     console.log("updated service.json with trusted peers", trustedPeers, "at", serviceJsonFilePath)
//     return true;
// }

/**
 * Updates the trusted peers in the IPFS Cluster service and restarts the service via the IPFS Manager API.
 * 
 * @param trustedPeers - The set of trusted IPFS Cluster IDs.
 * @returns - True if the trusted peers were updated and the service was restarted, false otherwise.
 */
async function updateTrustedPeersViaApi(trustedPeers: string[]) {
    try {
        const response = await axios.put('http://ipfs-manager:3000/update-trusted-peers', {
            trustedPeers: Array.from(trustedPeers)
        });
        console.log('Updated trusted peers:', response.data);
        return true;
    } catch (error) {
        console.error('Failed to update trusted peers:', error);
        return false;
    }
}

/**
 * Helper function to update the trusted peers in the IPFS Cluster service and restart the service.
 * 
 * @param trustedPeers - The set of trusted IPFS Cluster IDs.
 * @returns - True if the trusted peers were updated and the service was restarted, false otherwise.
 */
async function updateTrustedPeersAndRestartService(trustedPeers: Set<string>) {
    const updatedPeers = Array.from(trustedPeers);
    const success = await updateTrustedPeersViaApi(updatedPeers);
    if (success) {
        console.log('Trusted peers updated and IPFS Cluster service restarted');
    } else {
        console.log('Failed to update trusted peers or restart IPFS Cluster service');
    }
    return success;
}


/**
 * Main function to start the listener.
 */
async function main() {
    const argv = await yargs(hideBin(process.argv))
        .option('ws-url', {
            type: 'string',
            default: 'ws://127.0.0.1:9946',
            description: 'The URL of the Polkadot node to connect to'
        })
        .option('netuid', {
            type: 'number',
            description: 'The ID of the subnet to listen for',
        })
        .option('min-stake', {
            type: 'number',
            description: 'The minimum stake in TAO for a validator that advertises its IPFS cluster ID to be considered trusted.',
        })
        .option('time-window', {
            type: 'number',
            description: 'The number of blocks to look back to consider an IPFS cluster ID trusted.',
            default: 300 // 1 hour 
        })
        .option('allow-unsuccessful-commitments', {
            type: 'boolean',
            description: 'Whether to allow unsuccessful commitments to be considered handled, should only be used when debugging'
        })
        .demandOption(['netuid', 'min-stake'])
        .help()
        .parse();

    // print the argv
    console.log(argv);

    const provider = new WsProvider(argv.wsUrl);

    const api = await ApiPromise.create({ provider });

    // subscribe to finalized heads:
    //  - update latest inscription map
    //  - update trusted ipfs cluster ids based off of latest inscriptions
    //  - update service.json if there were any changes
    //  - restart ipfs cluster service if there were any changes
    const unsub = await api.rpc.chain.subscribeFinalizedHeads(async (header) => {

        const block = await api.rpc.chain.getBlock(header.hash)
        const blockNumber = block.block.header.number.toNumber()
        console.log(`Processing block #${blockNumber}`)

        await updateLatestInscriptionMap(api, argv.netuid)
        console.log("updated latest inscription map", latestInscriptionMap)

        const wasChange = await updateTrustedIpfsClusterIds(api, argv.netuid, argv.minStake, argv.timeWindow, blockNumber)
        console.log("updated trusted ipfs cluster ids", { wasChange })

        console.log("Trusted IPFS Cluster IDs:", trustedIpfsClusterIds)

        if (wasChange) {
            const didUpdate = await updateTrustedPeersAndRestartService(trustedIpfsClusterIds)
            if (didUpdate) {
                console.log('Successfully updated trusted peers and restarted IPFS Cluster service');
            } else {
                console.log('Failed to update trusted peers or restart IPFS Cluster service');
            }
        }
    })
}

/**
 * Entry point for the listener.
 */
if (require.main === module) {
    main()
}