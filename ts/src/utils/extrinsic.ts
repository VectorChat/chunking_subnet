import { ApiPromise } from "@polkadot/api";
import { Event } from "@polkadot/types/interfaces";

export function getExtrinsicErrorString(events: Event[], api: ApiPromise) {
    let errorInfo: string | null = null;
    for (const event of events) {
        if (api.events.system.ExtrinsicFailed.is(event)) {
            const [dispatchError, _dispatchInfo] = event.data

            if (dispatchError.isModule) {
                try {
                    const decoded = api.registry.findMetaError(dispatchError.asModule);
                    errorInfo = `${decoded.section}.${decoded.name}`;
                } catch (e) {
                    errorInfo = dispatchError.toString();
                }
            } else {
                errorInfo = dispatchError.toString();
            }

        }
    }
    return errorInfo;
}