import { ApiPromise, WsProvider } from "@polkadot/api";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import type { Event, SignedBlock } from "@polkadot/types/interfaces"
import { Option } from "@polkadot/types"
import { parseIpfsClusterIdFromExtrinsic } from "./utils/commitments";
import { getExtrinsicErrorString } from "./utils/extrinsic";
import { fromRao } from "./utils/rao";

// Represents an inscription on the chain for a specific hotkey
export type IpfsInscription = {
    ipfsClusterId: string;
    inscribedAt: number;
    hotkey: string;
}

const latestInscriptionMap: Record<string, IpfsInscription> = {};


async function updateLatestInscriptionMap(api: ApiPromise, block: SignedBlock, netuid: number, allowUnsuccessfulCommitments: boolean) {
    const extrinsics = block.block.extrinsics;

    const apiAt = await api.at(block.block.header.hash);
    const allRecords = await apiAt.query.system.events()

    const blockNumber = block.block.header.number.toNumber();

    console.log(`Processing block #${blockNumber} with ${extrinsics.length} extrinsics`)

    for (let index = 0; index < extrinsics.length; index++) {
        const extrinsic = extrinsics[index];
        if (!api.tx.commitments.setCommitment.is(extrinsic)) {
            continue;
        }
        console.log("Found commitment extrinsic")

        const commitmentNetuid = extrinsic.args[0]

        if (commitmentNetuid.toNumber() !== netuid) {
            console.log("Skipping commitment extrinsic for netuid", commitmentNetuid.toNumber())
            continue;
        }
        console.log("Found commitment extrinsic for netuid", netuid)

        const extrinsicEvents = allRecords
            // filter the specific events based on the phase and then the
            // index of our extrinsic in the block
            .filter(({ phase }) =>
                phase.isApplyExtrinsic &&
                phase.asApplyExtrinsic.eq(index)
            )
            .map(({ event }) => (event as unknown as Event))

        const extrinsicIsSuccess = extrinsicEvents.some((e) => api.events.system.ExtrinsicSuccess.is(e));
        if (!extrinsicIsSuccess && !allowUnsuccessfulCommitments) {
            console.log("unable to find success event for extrinsic", extrinsic.toHuman())
            const errorString = getExtrinsicErrorString(extrinsicEvents, api)
            console.log("Parsed extrinsic error:", errorString)
            continue;
        }


        console.log(`Found successful commitment at block ${blockNumber} for netuid ${netuid}`)
        const commitmentInfo = extrinsic.args[1]
        console.log("commitmentInfo", commitmentInfo.toHuman())

        const ipfsClusterId = parseIpfsClusterIdFromExtrinsic(commitmentInfo)

        const parsedCommitment: IpfsInscription | null = ipfsClusterId ? {
            ipfsClusterId,
            hotkey: extrinsic.signer.toString(),
            inscribedAt: blockNumber
        } : null

        const signer = extrinsic.signer.toString()
        if (parsedCommitment === null) {
            console.log(`Failed to parse commitment for signer ${signer}, skipping...`)
            continue
        }


        console.log(`IPFS Cluster ID: ${ipfsClusterId}`)
        console.log(`Hotkey: ${parsedCommitment.hotkey}`)

        latestInscriptionMap[parsedCommitment.hotkey] = parsedCommitment
        console.log("Updated inscription map for signer", signer, "with", latestInscriptionMap[signer])
    }
}

const trustedIpfsClusterIds: string[] = [];

async function canBeTrusted(api: ApiPromise, hotkey: string, minStake: number, netuid: number, inscription: IpfsInscription, currentBlockNumber: number, timeWindow: number) {
    console.log("Checking if", inscription.ipfsClusterId, "can be trusted")
    const uid = await api.query.subtensorModule.uids(netuid, hotkey)
    if (uid.isNone) {
        console.log("Hotkey", hotkey, "not found in uids")
        return false;
    }
    const validatorPermitArray = await api.query.subtensorModule.validatorPermit(netuid)
    const hasValidatorPermit = validatorPermitArray[uid.unwrap().toNumber()].toPrimitive()
    if (!hasValidatorPermit) {
        console.log("Hotkey", hotkey, "not found in validatorPermit")
        return false;
    }
    const stakeRao = await api.query.subtensorModule.totalHotkeyStake(hotkey)
    const stakeTao = fromRao(stakeRao)
    if (stakeTao.lt(minStake)) {
        console.log("Hotkey", hotkey, "has stake", stakeTao, "which is less than", minStake)
        return false;
    }

    if (inscription.inscribedAt < currentBlockNumber - timeWindow) {
        console.log("Hotkey", hotkey, "inscribed at", inscription.inscribedAt, "which is older than", timeWindow, "blocks")
        return false;
    }

    return true;
}


async function updateTrustedIpfsClusterIds(api: ApiPromise, netuid: number, minStake: number, timeWindow: number, currentBlockNumber: number) {
    for (const [hotkey, ipfsInscription] of Object.entries(latestInscriptionMap)) {
        if (await canBeTrusted(api, hotkey, minStake, netuid, ipfsInscription, currentBlockNumber, timeWindow)) {
            console.log("Adding trusted IPFS Cluster ID", ipfsInscription.ipfsClusterId)
            trustedIpfsClusterIds.push(ipfsInscription.ipfsClusterId)
        }
    }
}

async function main() {
    const argv = yargs(hideBin(process.argv))
        .option('ws-url', {
            type: 'string',
            default: 'ws://127.0.0.1:9946',
            description: 'The URL of the Polkadot node to connect to'
        })
        .option('netuid', {
            type: 'number',
            description: 'The ID of the subnet to listen for'
        })
        .option('min-stake', {
            type: 'number',
            description: 'The minimum stake in TAO for a validator that advertises its IPFS cluster ID to be considered trusted.'
        })
        .option('time-window', {
            type: 'number',
            description: 'The number of blocks to look back to consider an IPFS cluster ID trusted.'
        })
        .option('allow-unsuccessful-commitments', {
            type: 'boolean',
            description: 'Whether to allow unsuccessful commitments to be considered handled, useful for debugging'
        })
        .help()
        .parse();

    // print the argv
    console.log(argv);

    const provider = new WsProvider(argv.wsUrl);

    const api = await ApiPromise.create({ provider });

    // for (let blockNumber = 1; blockNumber <= 10; blockNumber++) {
    //     const blockHash = await api.rpc.chain.getBlockHash(blockNumber)
    //     const block = await api.rpc.chain.getBlock(blockHash)
    //     await updateLatestInscriptionMap(api, block, argv.netuid, argv.allowUnsuccessfulCommitments)
    //     console.log("updated latest inscription map")

    //     await updateTrustedIpfsClusterIds(api, argv.netuid, argv.minStake, argv.timeWindow, block.block.header.number.toNumber())
    //     console.log("updated trusted ipfs cluster ids")
    // }
    // console.log("latest inscription map", latestInscriptionMap)
    // console.log("Trusted IPFS Cluster IDs:", trustedIpfsClusterIds)

    // listen to commitments for the target netuid
    // const unsub = await api.query.commitments.commitmentOf.entries(argv.netuid, (data) => {
    //     console.log("Got data", data);
    // });
    // const unsub = await api.rpc.chain.subscribeFinalizedHeads(async (header) => {

    //     const block = await api.rpc.chain.getBlock(header.hash)

    //     await updateLatestInscriptionMap(api, block, argv.netuid, argv.allowUnsuccessfulCommitments)
    //     console.log("updated latest inscription map")

    //     await updateTrustedIpfsClusterIds(api, argv.netuid, argv.minStake, argv.timeWindow, block.block.header.number.toNumber())
    //     console.log("updated trusted ipfs cluster ids")


    //     console.log("Trusted IPFS Cluster IDs:", trustedIpfsClusterIds)
    // })

}

if (require.main === module) {
    main()
}