import abc
import ray
import actor
import critic
import initial
import reward

import openllama2.utils.ray as ray_utils
from openllama2.trainer.ppo_utils import (
    Experience,
    RayExperienceMaker,
    FixedKLController,
)


@ray.remote
class RayPPOTrainer(abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self._remote_actor = actor.Actor.remote(actor_config)
        self._remote_critic = critic.Critic.remote(critic_config)
        self._remote_initial = initial.Initial.remote(initial_config)
        self._remote_reward = reward.Reward.remote(reward_config)

        self._experience_maker = RayExperienceMaker(
            self._remote_actor,
            self._remote_critic,
            self._remote_initial,
            self._remote_reward,
        )

        def _remote_is_ready(ray_actor):
            return ray_actor.is_ready.remote()

        def _condition_predecitor():
            li = [
                self._remote_actor,
                self._remote_critic,
                self._remote_initial,
                self._remote_reward,
            ]
            results = [_remote_is_ready(ray_actor) for ray_actor in li]
            ray.get(results)

        # As we should start these ray actors on GPUs that may take long time, we give 1 minute.
        ray_utils.wait_for_condition(_condition_predecitor, timeout=60)



    def fit(self) -> None:
        

