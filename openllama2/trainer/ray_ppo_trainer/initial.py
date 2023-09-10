import abc

import ray


@ray.remote
class Initial(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    def is_ready(self) -> bool:
        return True
