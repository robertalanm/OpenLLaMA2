import ray
import abc


@ray.remote
class Initial(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
