__version__ = "2.0.1"

version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

from .forward import forward, get_miner_groups, create_groups
from .reward import reward
from .task_api import Task
