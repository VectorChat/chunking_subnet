import { ApiPromise, WsProvider } from "@polkadot/api";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import { queryCommitmentsForIpfsClusterIds } from "./utils/commitments";
import { fromRao } from "./utils/rao";
import { IpfsInscription } from "./types";
import fs from "fs";
import express from "express";
import z from "zod";

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
    console.log("update latest inscription map with", ipfsClusterIdCommitments.length, "commitments")
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
    console.log("getting uid for\n", {
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
        console.log("-".repeat(50))
        console.log("handling\n", ipfsInscription)
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

/**
 * Updates the service.json file with the trusted peers. It only updates the file if the current set of trusted peers is different from the stored set of trusted peers.
 * 
 * @param serviceJsonFilePath - The path to the service.json file.
 * @param alwaysUpdate - Whether to always update the service.json file.
 * @returns - True if the service.json file was updated, false otherwise.
 */
function updateTrustedPeersInServiceJsonFile(serviceJsonFilePath: string, alwaysUpdate = false) {
    try {
        const serviceJson = JSON.parse(fs.readFileSync(serviceJsonFilePath, 'utf8'));

        const serviceJsonTrustedPeers = serviceJson.consensus.crdt.trusted_peers || []

        const trustedPeers = Array.from(trustedIpfsClusterIds)

        const currentEqualsStored = setIsEqual(new Set(serviceJsonTrustedPeers), trustedIpfsClusterIds)

        const doUpdate = alwaysUpdate || !currentEqualsStored

        console.log({
            serviceJsonTrustedPeers,
            trustedPeers,
            currentEqualsStored,
            doUpdate,
        })

        if (!doUpdate) {
            console.log("service.json already up to date with trusted peers", trustedPeers)
            return false;
        }

        console.log("updating service json file because alwaysUpdate is true or current does not match stored", {
            alwaysUpdate,
            currentEqualsStored,
        })

        serviceJson.consensus.crdt.trusted_peers = trustedPeers

        fs.writeFileSync(serviceJsonFilePath, JSON.stringify(serviceJson, null, 2));
        console.log("updated service.json with trusted peers", trustedPeers, "at", serviceJsonFilePath)
        return true;
    } catch (e) {
        console.error("error updating service.json file", e)
        return false;
    }
}
/**
 * Restarts the IPFS Cluster service via a file. The extended ipfs cluster image will watch for file existence and restart the service.
 * 
 * @param restartFilePath - The path to the file that will be created to restart the service.
 */
function restartIpfsClusterServiceViaFile(restartFilePath: string) {
    fs.writeFileSync(restartFilePath, "restart")
    console.log("restart file written, ipfs cluster service will restart soon")
}

/**
 * Updates the peer addresses in the service.json file. These contain the full peer multiaddrs that should be connected to on boot. 
 *  
 * @param serviceJsonFilePath - The path to the service.json file.
 * @param peerAddresses - The list of peer addresses to update.
 * @returns - True if the service.json file was updated, false otherwise.
 */
function updatePeerAddressesInServiceJsonFile(serviceJsonFilePath: string, peerMultiaddrs: string[]) {
    const serviceJson = JSON.parse(fs.readFileSync(serviceJsonFilePath, 'utf8'));

    const currentPeerAddresses: string[] = serviceJson.cluster.peer_addresses || []

    const currentPeerAddressesSet = new Set(currentPeerAddresses)

    const newPeerAddressesSet = new Set(peerMultiaddrs)

    const didChange = !setIsEqual(currentPeerAddressesSet, newPeerAddressesSet)

    if (!didChange) {
        console.log("service.json already up to date with peer addresses", peerMultiaddrs)
        return false;
    }

    serviceJson.cluster.peer_addresses = Array.from(newPeerAddressesSet)
    fs.writeFileSync(serviceJsonFilePath, JSON.stringify(serviceJson, null, 2));
    console.log("updated service.json with peer addresses", peerMultiaddrs, "at", serviceJsonFilePath)
    return true;
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
            default: false
        })
        .demandOption(['netuid', 'min-stake'])
        .help()
        .parse();

    // print the argv
    console.log(argv);

    const envSchema = z.object({
        LEADER_IPFS_CLUSTER_ID: z.string(),
        // LEADER_IPFS_CLUSTER_MULTIADDR: z.string(),
    })

    const env = envSchema.parse(process.env)

    const provider = new WsProvider(argv.wsUrl);

    const api = await ApiPromise.create({ provider });

    // add the leader ipfs cluster id to the trusted list
    trustedIpfsClusterIds.add(env.LEADER_IPFS_CLUSTER_ID)

    console.log("Automatically added leader ipfs cluster id to trusted list", env.LEADER_IPFS_CLUSTER_ID)

    // // update the service.json file with the leader ipfs cluster multiaddr for proper bootstrapping 
    // const didUpdate = updatePeerAddressesInServiceJsonFile(argv.serviceJsonFilePath, [env.LEADER_IPFS_CLUSTER_MULTIADDR])
    // if (didUpdate) {
    //     console.log("Updated service.json with leader ipfs cluster multiaddr", env.LEADER_IPFS_CLUSTER_MULTIADDR, "restarting ipfs cluster service")
    //     restartIpfsClusterServiceViaFile(argv.restartFilePath)
    // } else {
    //     console.log("Not restarting IPFS Cluster service because service.json was not updated")
    // }

    // subscribe to finalized heads:
    //  - update latest inscription map
    //  - update trusted ipfs cluster ids based off of latest inscriptions
    //  - update service.json if there were any changes
    //  - restart ipfs cluster service if there were any changes
    const unsub = await api.rpc.chain.subscribeFinalizedHeads(async (header) => {

        const block = await api.rpc.chain.getBlock(header.hash)
        const blockNumber = block.block.header.number.toNumber()

        console.log("-".repeat(100))
        console.log(`Processing block #${blockNumber}`)
        console.log("-".repeat(100))

        await updateLatestInscriptionMap(api, argv.netuid)
        console.log("updated latest inscription map\n", latestInscriptionMap)

        const wasChange = await updateTrustedIpfsClusterIds(api, argv.netuid, argv.minStake, argv.timeWindow, blockNumber)
        console.log("updated trusted ipfs cluster ids\n", { wasChange })

        console.log("Trusted IPFS Cluster IDs:\n", trustedIpfsClusterIds)

        if (wasChange) {
            const didUpdate = updateTrustedPeersInServiceJsonFile(argv.serviceJsonFilePath, argv.alwaysUpdateServiceJson)
            if (didUpdate) {
                restartIpfsClusterServiceViaFile(argv.restartFilePath)
            } else {
                console.log("Not restarting IPFS Cluster service because service.json was not updated")
            }
        }
    })

    const app = express()

    app.get('/trusted-peers', (req, res) => {
        res.json(Array.from(trustedIpfsClusterIds))
    })

    app.get('/health', (req, res) => {
        console.log("health check")
        res.send('OK')
    })

    app.listen(argv.port, () => {
        console.log('Server is running on port', argv.port)
    })
}

/**
 * Entry point for the listener.
 */
if (require.main === module) {
    main()
}