__version__ = "2.1.0"

version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

from .forward import forward, get_miner_groups
from .reward import reward
from .task_api import Task
from .types import TaskType
from .integrated_api import setup_routes
