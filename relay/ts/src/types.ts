
export type IpfsCommitment = {
    ipfsClusterId: string;
    inscribedAt: number;

}

// Represents an commitment on the chain for a specific hotkey
export type IpfsInscription = IpfsCommitment & {
    hotkey: string;
}

