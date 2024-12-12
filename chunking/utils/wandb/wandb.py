from datetime import datetime, timedelta
import os
import time
import chunking
import wandb
from wandb.apis.public.runs import Runs, Run
import bittensor as bt

# from gql import gql

RUN_FRAGMENT = """fragment RunFragment on Run {
    id
    tags
    name
    displayName
    sweepName
    state
    config
    group
    jobType
    commit
    readOnly
    createdAt
    heartbeatAt
    description
    notes
    systemMetrics
    summaryMetrics
    historyLineCount
    user {
        name
        username
    }
    historyKeys
}"""

QUERY_TEMPLATE = """
query Run($project: String!, $entity: String!, $name: String!) {
    project(name: $project, entityName: $entity) {
        run(name: $name) {
            ...RunFragment
        }
    }
}
%s
"""


class WandbLogger:
    def __init__(self, validator):
        self.validator = validator

        if not self.is_off():
            self.run = self._setup_wandb()
            self.start_time = time.time()

        self.api = wandb.Api()

    def is_off(self):
        return self.validator.config.wandb.wandb_off

    def finish(self):
        if not self.is_off():
            wandb.finish()

    def log(self, *args, **kwargs):
        if not self.is_off():
            wandb.log(*args, **kwargs)
        else:
            bt.logging.info(f"Wandb is off, not logging {args} {kwargs}")

    def restart_if_past_time_delta(self, time_delta: timedelta):
        # query = gql(QUERY_TEMPLATE % RUN_FRAGMENT)
        #         query = gql("""
        #         query Run($project: String!, $entity: String!, $name: String!) {
        #             project(name: $project, entityName: $entity) {
        #                 run(name: $name) {
        #                     id
        #                     tags
        #                     name
        #                     displayName
        #                     sweepName
        #                     state
        #                     config
        #                     group
        #                     jobType
        #                     commit
        #                     readOnly
        #                     createdAt
        #                     heartbeatAt
        #                     description
        #                     notes
        #                     systemMetrics
        #                     summaryMetrics
        #                     historyLineCount
        #                     user {
        #                         name
        #                         username
        #                     }
        #                     historyKeys
        #                 }
        #             }
        #         }
        # """)

        # variables = {
        #     "project": self.validator.config.wandb.project_name,
        #     "entity": chunking.ENTITY,
        #     "name": self.validator.config.run_name,
        # }

        # bt.logging.info(f"query: {query}")
        # bt.logging.info(f"variables: {variables}")

        # result = self.api.client.execute(query, variable_values=variables)

        cur_datetime = datetime.now()
        start_datetime = datetime.fromtimestamp(self.start_time)

        if start_datetime + time_delta < cur_datetime:
            bt.logging.info(
                f"Restarting wandb run. Start time: {start_datetime}. Time delta arg: {time_delta}. Current time: {cur_datetime}"
            )
            self.run = self._setup_wandb(force_restart=True)
            self.start_time = time.time()

    def _find_valid_wandb_run(self, runs: Runs) -> Run | None:
        """
        Find a valid wandb run for this validator given a list of wandb runs. The run must be signed by the validator's hotkey.
        """
        for run in runs:
            sig = run.config.get("signature")
            if not sig:
                continue

            verified = self.validator.wallet.hotkey.verify(
                run.id.encode(), bytes.fromhex(sig)
            )

            if verified:
                bt.logging.info(f"Found valid run: {run.id}")
                return run
            else:
                bt.logging.warning(
                    f"Found invalid run: {run.id}, looking for another run"
                )

        return None

    def _get_latest_wandb_run(self, project_name: str) -> Run | None:
        """
        Get the latest valid wandb run for this validator.
        """
        api = wandb.Api()
        latest_runs: Runs = api.runs(
            f"{chunking.ENTITY}/{project_name}",
            {
                "config.hotkey": self.validator.wallet.hotkey.ss58_address,
                "config.type": "validator",
            },
            order="-created_at",
        )
        return self._find_valid_wandb_run(latest_runs)

    def _start_new_wandb_run(
        self, project_name: str, run_name: str, reinit: bool = False
    ) -> Run:
        """
        Start a new wandb run for this validator if no valid wandb run exists for this validator and version.
        """
        run = wandb.init(
            name=run_name,
            project=project_name,
            entity=chunking.ENTITY,
            config=self.validator.config,
            dir=self.validator.config.full_path,
            reinit=reinit,
        )
        signature = self.validator.wallet.hotkey.sign(run.id.encode()).hex()
        self.validator.config.signature = signature
        bt.logging.success(
            f"Started wandb run for project '{project_name}', name: '{run_name}', id: '{run.id}'"
        )
        return run

    def _find_existing_wandb_run(self, version: str, uid: int) -> Run | None:
        """
        Find a valid wandb run for this validator given a list of wandb runs. The run must be signed by the validator's hotkey and match the validator's uid and version.
        """
        api = wandb.Api()
        latest_runs: Runs = api.runs(
            f"{chunking.ENTITY}/{self.validator.config.wandb.project_name}",
            {
                "config.hotkey": self.validator.wallet.hotkey.ss58_address,
                "config.type": "validator",
                "config.version": version,
                "config.uid": uid,
            },
            order="-created_at",
        )
        bt.logging.debug(
            f"Found {len(latest_runs)} runs with version {version} and uid {uid}"
        )
        return self._find_valid_wandb_run(latest_runs)

    def _resume_wandb_run(self, run: Run, project_name: str):
        """
        Resume a wandb run for this validator.
        """
        run = wandb.init(
            entity=chunking.ENTITY, project=project_name, id=run.id, resume="must"
        )
        bt.logging.success(
            f"Resumed wandb run '{run.name}' for project '{project_name}'"
        )
        return run

    def _get_wandb_project_name(self):
        if (
            self.validator.config.subtensor.chain_endpoint == "test"
            or self.validator.config.subtensor.network == "test"
        ):
            return "chunking-testnet"
        return self.validator.config.wandb.project_name

    def _setup_wandb(self, force_restart: bool = False):
        """
        Setup wandb for this validator.

        This function will start a new wandb run if no valid wandb run exists for this validator and version.
        If a valid wandb run exists, it will resume the wandb run.
        """
        if (
            os.environ.get("WANDB_API_KEY") is None
            or os.environ.get("WANDB_API_KEY") == ""
        ):
            raise Exception("WANDB_API_KEY environment variable must be set")

        else:
            try:
                project_name = self._get_wandb_project_name()

                latest_run = self._get_latest_wandb_run(project_name)

                run_name = f"validator-{self.validator.uid}-{chunking.__version__}"

                self.validator.config.uid = self.validator.uid
                self.validator.config.hotkey = self.validator.wallet.hotkey.ss58_address
                self.validator.config.run_name = run_name
                self.validator.config.version = chunking.__version__
                self.validator.config.type = "validator"

                run: Run | None = None

                if not latest_run or force_restart:
                    run = self._start_new_wandb_run(
                        project_name, run_name, reinit=force_restart
                    )
                else:
                    # check if uid or version has changed
                    if (
                        latest_run.config.get("version") != chunking.__version__
                        or latest_run.config.get("uid") != self.validator.uid
                    ):
                        bt.logging.info(
                            f"Found run with different version or uid ({latest_run.name})"
                        )

                        existing_run = self._find_existing_wandb_run(
                            chunking.__version__, self.validator.uid
                        )

                        if not existing_run:
                            bt.logging.info(
                                f"Could not find existing run with version {chunking.__version__} and uid {self.validator.uid}, starting new run"
                            )
                            run = self._start_new_wandb_run(project_name, run_name)
                        else:
                            bt.logging.info(
                                f"Found existing run with version {chunking.__version__} and uid {self.uid}, resuming run"
                            )
                            run = self._resume_wandb_run(existing_run, project_name)
                    else:
                        run = self._resume_wandb_run(latest_run, project_name)

                # always update config
                wandb.config.update(self.validator.config, allow_val_change=True)

                return run

            except Exception as e:
                raise Exception(f"Error in init_wandb: {e}")
