from abc import ABC
import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openllama2.models.utils import compute_reward, masked_mean
from .experience_maker import Experience, get_advantages_and_returns


class RayExperienceMaker(ABC):
    def __init__(
        self,
        remote_actor,
        remote_critic,
        remote_initial,
        remote_reward,
        kl_controller,
    ) -> None:
        super.__init__()
        self._remote_actor = remote_actor
        self._remote_critic = remote_critic
        self._remote_initial = remote_initial
        self._remote_reward = remote_reward
        self._kl_controller = kl_controller

    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs):
        # TODO(qwang): These should be changed in no duplicate lines.
        o1 = self._remote_actor.eval.remote()
        o2 = self._remote_critic.eval.remote()
        o3 = self._remote_initial.eval.remote()
        o4 = self._remote_reward.eval.remote()
        ray.get([o1, o2, o3, o4])

        for i in tqdm(range(1), desc=f"Generate sequence"):
            sequences_obj, attention_mask_obj, action_mask_obj = self._remote_actor.generate.remote(
                input_ids, **generate_kwargs
            )
            sequences, attention_mask, action_mask = ray.get([sequences_obj, attention_mask_obj, action_mask_obj])

        num_actions = action_mask.size(1)
        for i in tqdm(range(1), desc=f"Actor forward"):
            action_log_probs_obj = self._remote_actor.forward(sequences, num_actions, attention_mask)
            action_log_probs = ray.get(action_log_probs_obj)

        for i in tqdm(range(1), desc=f"Init model forward"):
            base_action_log_probs_obj = self._remote_initial.remote(sequences, num_actions, attention_mask)
            base_action_log_probs = ray.get(base_action_log_probs_obj)

        for i in tqdm(range(1), desc=f"Value model forward"):
            value_obj = self._remote_critic.remote(sequences, action_mask, attention_mask)
            value = ray.get(value_obj)

        for i in tqdm(range(1), desc=f"Reward model forward"):
            reward_value_obj = self._remote_reward.forward.remote(sequences, attention_mask)
            r = ray.get(reward_value_obj)

        # Compute these stuff on local.
        reward, kl = compute_reward(
            r, self.kl_ctl.value, action_log_probs, base_action_log_probs, action_mask=action_mask
        )
        advantage, returns = get_advantages_and_returns(
            value, reward, action_mask, generate_kwargs["gamma"], generate_kwargs["lambd"]
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }

        return Experience(sequences, action_log_probs, value, returns, advantage, attention_mask, action_mask, info)
