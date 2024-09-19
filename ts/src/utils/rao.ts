import { BN } from "@polkadot/util";
import Decimal from "decimal.js";

export const RAO_DECIMALS = 9;

export function fromRao(rao: string | number | BN): Decimal {
    return new Decimal(rao.toString()).div(new Decimal(10).pow(RAO_DECIMALS));
}

export function toRao(decimal: Decimal): BN {
    return new BN(decimal.mul(new Decimal(10).pow(RAO_DECIMALS)).toString());
}