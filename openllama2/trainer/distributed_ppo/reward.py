import ray
import abc


@ray.remote
class Reward(abc.ABC):
    def __init__(self) -> None:
        super().__init__()


    def is_ready(self) -> bool:
        return True
