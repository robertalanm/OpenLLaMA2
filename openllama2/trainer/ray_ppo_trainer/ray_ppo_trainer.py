import abc
import math
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

import ray
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import openllama2.utils.ray as ray_utils
from openllama2.models.utils import masked_mean
from openllama2.trainer.ppo_utils import (
    AdaptiveKLController,
    Experience,
    FixedKLController,
    NaiveExperienceMaker,
    NaiveReplayBuffer,
    RayExperienceMaker,
)

from .actor import Actor
from .critic import Critic
from .initial import Initial
from .reward import Reward


@ray.remote
class RayPPOTrainer(abc.ABC):
    def __init__(self, actor_config, critic_config, initial_config, reward_config) -> None:
        super().__init__()
        self._remote_actor = Actor.remote(actor_config)
        self._remote_critic = Critic.remote(critic_config)
        self._remote_initial = Initial.remote(initial_config)
        self._remote_reward = Reward.remote(reward_config)

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

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

    def fit(
        self,
        prompts_dataloader,
        pretrain_dataloader,
        num_episodes: 1,
        rollout_batch_size: 1024,
    ) -> None:
        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # tokenizer
        def tokenize_fn(texts):
            batch = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=self.prompt_max_len,
                padding=True,
                truncation=True,
            )
            return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

        update_timesteps = rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)
        global_step = 0

        torch.cuda.empty_cache()
        for episode in range(num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
            pbar = tqdm(
                self.prompts_dataloader,
                desc=f"Episode [{episode+1}/{num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in pbar:
                inputs = tokenize_fn(rand_prompts)
                experience = self.experience_maker.make_experience(**inputs, **self.generate_kwargs)
                self.replay_buffer.append(experience)

                global_step = global_step + 1
                if global_step % update_timesteps == 0:
                    torch.cuda.empty_cache()
                    self.replay_buffer.normalize("advantages", self.strategy)
                    status = self.ppo_train()
                    self.replay_buffer.clear()
                    self.kl_ctl.update(status["kl"], rollout_batch_size)

                    self.strategy.print(status)
                    if self._wandb is not None and self.strategy.is_rank_0():
                        logs = {
                            "train/%s" % k: v
                            for k, v in {
                                **status,
                                "global_step": global_step // update_timesteps,
                            }.items()
                        }
                        self._wandb.log(logs)
                    torch.cuda.empty_cache()

    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch+1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience)

                # for DP
                # weighted mean for kl
                status["kl"] *= status["response_length"]
                status = self.strategy.all_reduce(status)
                status["kl"] /= status["response_length"]

                status_list.append(status)
                short_status = {
                    "pg": status["policy_loss"],
                    "cri": status["critic_loss"],
                    "vals": status["values"],
                    "rm": status["reward"],
                    "ret": status["return"],
                    "glen": status["response_length"],
                    "tlen": status["total_length"],
                    "kl": status["kl"],
                }
                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()

        num_actions = experience.action_mask.size(1)
        # actor loss
        action_log_probs = self.actor(experience.sequences, num_actions, attention_mask=experience.attention_mask)
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            experience.action_log_probs,
            experience.advantages,
            action_mask=experience.action_mask,
        )
        self.strategy.backward(actor_loss, self.actor, self.actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                inputs.eq(self.tokenizer.pad_token_id),
                self.ptx_loss_fn.IGNORE_INDEX,
                inputs,
            )

            ptx_log_probs = self.actor(inputs, attention_mask=attention_mask, return_output=True)["logits"]
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            self.strategy.backward(ptx_loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # critic loss
        values = self.critic(
            experience.sequences,
            action_mask=experience.action_mask,
            attention_mask=experience.attention_mask,
        )
        critic_loss = self.critic_loss_fn(
            values,
            experience.values,
            experience.returns,
            action_mask=experience.action_mask,
        )

        self.strategy.backward(critic_loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "policy_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask, dim=(0, 1)).item(),
        }
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        return status
