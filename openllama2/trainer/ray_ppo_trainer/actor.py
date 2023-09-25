import itertools
import math

import ray
from transformers.trainer import get_scheduler
from utils import blending_datasets, get_strategy, get_tokenizer

from openllama2.datasets import PromptDataset, SFTDataset
from openllama2.models import Actor, Critic, RewardModel
from openllama2.utils.ray import DistributedTorchRayActor


@ray.remote
class RayActor(DistributedTorchRayActor):
    def __init__(self, world_info, args) -> None:
        super().__init__(
            world_info.world_size,
            world_info.rank,
            world_info.local_rank,
            world_info.master_addr,
            world_info.master_port,
        )
        self._is_ready = False

        # configure strategy
        self.strategy = get_strategy(args)

        # configure flash attention
        if args.flash_attn:
            from openllama2.models.llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

            replace_llama_attn_with_flash_attn()

        # init model
        actor_from_config = bool(args.sft_model_path or args.load_checkpoint)
        self.actor = Actor(args.pretrain, actor_from_config, revision=args.model_revision)
        if args.sft_model_path:
            self.strategy.load_model(self.actor, args.sft_model_path)

        # init tokenizer
        tokenizer = get_tokenizer(args.pretrain, self.actor.model, "left", self.strategy)

        actor_optim = self.strategy.create_optimizer(
            self.actor, lr=args.actor_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
        )

        # init dataloader
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            self.strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        prompts_dataset = PromptDataset(prompts_data, self.strategy)
        self.prompts_dataloader = self.strategy.setup_dataloader(
            prompts_dataset, args.micro_rollout_batch_size, True, True
        )

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                self.strategy,
                args.seed,
                return_eval=False,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))),
                tokenizer,
                pretrain_max_len,
                self.strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    self.strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

        # configure scheduler
        num_update_steps_per_episodes = (
            len(self.prompts_dataloader) * args.max_epochs // self.strategy.accumulated_gradient
        )
        max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

        # init scheduler
        actor_scheduler = get_scheduler(
            "constant_with_warmup",
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
        )

        # deepspeed prepare
        (self.actor, actor_optim, actor_scheduler) = self.self.strategy.prepare(
            (self.actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        self._is_ready = True

    def is_ready(self) -> bool:
        return self._is_ready
