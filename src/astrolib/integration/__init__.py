from typing import List
from typing import NamedTuple
from typing import Optional

from astrolib import Matrix


class IntegratorResults(NamedTuple):
    epoch: float
    state: float | Matrix
    total_step_seconds: float
    intermediate_step_seconds: List[float]
    projected_step_seconds: Optional[float] = None
