import abc
import ray
import actor
import critic
import initial
import reward
import Dict

import openllama2.utils.ray as ray_utils
from openllama2.trainer.ppo_utils import (
    Experience,
    RayExperienceMaker,
    FixedKLController,
)

import tqdm

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



    def fit(self, num_episodes: 1) -> None:
        self._prompts_dataloader = prompts_dataloader
        self._pretrain_dataloader = pretrain_dataloader

        def tokenize_fn(texts):
            batch = self._tokenizer(
                texts,
                return_tensors='pt',
                max_length=self._prompt_max_len,
                padding=True,
                truncation=True,
            )
            # TODO(qwang): 这里返回什么比较合适？因为有device?
            # return {k: v.to(torch.cuda.current_device()) for k, v in batch.items() 

        update_timesteps = rollout_batch_size // self.strategy.world_size * self.micro_rollout_batch_size
        time_step = 0

        for episode in range(num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)

            pbar = tqdm.tqdm(self.prompts_dataloader,
                    desc=f'Episode [{episode+1}/{num_episodes}]',
                    disable=not self.strategy.is_rank_0())

            for rand_prompts in pbar:
                inputs = tokenize_fn(rand_prompts)
                experience = self._experience_maker.make_experience(**inputs, **self.generate_kwargs)
                # TODO(qwang): The replay buffer should be re-designed for avoiding extra copy overhead.
                self._replay_buffer.append(experience)

                time_step = (time_step + 1) % update_timesteps
                if time_step % update_timesteps == 0:
                    self._replay_buffer.normalize('advantages', self.strategy)
                    status = self._ppo_train()
                    self._replay_buffer.clear() 

                    self.kl_ctl.update(status['kl'], rollout_batch_size)
                    status['k_coef'] = self.kl_ctl.value
                    # TODO(qwang): This strategy should be re-designed.
                    self.strategy.print(status)



    def _ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(self.replay_buffer,
                          batch_size=self.replay_buffer.sample_batch_size,
                          shuffle=True,
                          drop_last=False,
                          pin_memory=self.dataloader_pin_memory,
                          collate_fn=self.replay_buffer.collate_fn)
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(dataloader, desc=f'Train epoch [{epoch+1}/{self.max_epochs}]', disable=not self.strategy.is_rank_0())
            for experience in pbar:
                experience.to_device(device)
                status = self._training_step(experience)

                # for DP
                # weighted mean for kl
                status['kl'] *= status['glen']
                status = self.strategy.all_reduce(status)
                status['kl'] /= status['glen']

                status_list.append(status)
                pbar.set_postfix(status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def _training_step(self, experience: Experience) -> Dict[str, float]:
        self._remote_actor.train.remote()
        self._critic_actor.train.remote()

        num_actions = experience.action_mask.size(1)
        for _ in tqdm(range(1), desc=f'Actor forward'):
            action_log_probs_obj = self._remote_actor.forward.remote(
                experience.sequences,
                num_actions,
                attention_mask=experience.attention_mask,
            )
            action_log_probs = ray.get(action_log_probs_obj)

            actor_loss = raw_actor_loss = self.actor_loss_fn(action_log_probs,
                                            experience.action_log_probs,
                                            experience.advantages,
                                            action_mask=experience.action_mask)

            # ptx loss
            if self.pretrain_dataloader is not None:
                data = next(self.pretrain_dataloader)
                inputs = data[1].squeeze(1).to(torch.cuda.current_device())
                attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
                label = torch.where(inputs.eq(self.tokenizer.pad_token_id), self.ptx_loss_fn.IGNORE_INDEX, inputs)

                ptx_log_probs = self.actor(inputs, attention_mask=attention_mask, return_output=True)['logits']
                ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
                actor_loss = ptx_loss * self.ptx_coef + raw_actor_loss

        for _ in tqdm(range(1), desc=f'Actor backward'):
            self.strategy.backward(actor_loss, self.actor, self.actor_optim)
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

            if self.ema_model:
                self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, 'cpu')

        for _ in tqdm(range(1), desc=f'Critic forward'):
            values = self.critic(experience.sequences,
                                action_mask=experience.action_mask,
                                attention_mask=experience.attention_mask)
            critic_loss = self.critic_loss_fn(values,
                                            experience.values,
                                            experience.returns,
                                            action_mask=experience.action_mask)

        for _ in tqdm(range(1), desc=f'Critic backward'):
            self.strategy.backward(critic_loss, self.critic, self.critic_optim)
            self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {'pg': raw_actor_loss.item(), 
                'cri': critic_loss.item(),
                'vals':  masked_mean(values, experience.action_mask, dim=(0, 1)).item(),
                }
        if self.pretrain_dataloader is not None:
            status['ptx'] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == 'kl':
                status[k] = ((v * experience.info['glen']).sum() / experience.info['glen'].sum()).item()
            else:
                status[k] = v.mean().item()
        return status
