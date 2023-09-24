from typing import Optional, List

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from transformers import AutoConfig, AutoModel

from openllama2.reward import (
    DirectPreferenceRewardModel,
    MockRewardModel,
    NSFWRewardModel,
    OpenAssistantRewardModel,
    ReciprocateRewardModel,
    RelevanceRewardModel,
)
from openllama2.reward.blacklist import Blacklist
from openllama2.reward.task_validator import TaskValidator
from openllama2.types import RewardInput, ResponseModel
# Now, I'll redefine the RewardModel class
class RewardModel(nn.Module):
    def __init__(self, model, tokenizer, observation_input, max_length, compare_sample, **kwargs):
        super().__init__()
        
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        mock = kwargs.get("mock", False)
        
        self.dpo_weight: float = 0.3 if not mock else 0.0
        self.rlhf_weight: float = 0.4 if not mock else 0.0
        self.reciprocate_weight: float = 0.3 if not mock else 0.0

        self.model = model
        self.tokenizer = tokenizer

        self.reward_functions = [
            OpenAssistantRewardModel(device=self.device) if self.rlhf_weight > 0 else MockRewardModel("rlhf"),
            ReciprocateRewardModel(device=self.device) if self.reciprocate_weight > 0 else None,
            DirectPreferenceRewardModel(device=self.device) if self.dpo_weight > 0 else MockRewardModel("dpo")
        ]

        self.reward_weights = [
            self.rlhf_weight,
            self.reciprocate_weight,
            self.dpo_weight,
        ]

        self.blacklist = Blacklist()
        self.task_validator = TaskValidator()
        self.relevance_model = RelevanceRewardModel(device=self.device)
        self.nsfw_model = NSFWRewardModel(device=self.device)

        self.masking_functions = [
            self.blacklist,
            self.task_validator,
            self.relevance_model,
            self.nsfw_model
        ]
        
    def compute_rewards(self, prompt: str, responses: List[str]) -> torch.FloatTensor:
        name = "augment"
        
        # Compute the rewards for the responses given the prompt.
        rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(self.device)
        for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
            if reward_fn_i:
                reward_i, reward_i_normalized = reward_fn_i.apply(prompt, responses, name)
                rewards += weight_i * reward_i_normalized.to(self.device)

        for masking_fn_i in self.masking_functions:
            mask_i, mask_i_normalized = masking_fn_i.apply(prompt, responses, name)
            rewards *= mask_i_normalized.to(self.device)

        return rewards

    def format_response(self, responses: List[str]) -> List[ResponseModel]:
        data = [ResponseModel(completion=r, is_success=True) for r in responses]
        return data

    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Convert sequences to text for the reward model
        predicted_list = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]
        input_item = {"input": "dummy_prompt"}  # This needs to be provided or generated in some way
        
        rewards = self.get_reward(input_item, predicted_list, False)
        return torch.tensor(rewards)

    def get_reward(self, input_item, predicted_list, finish):  # predicted will be the list of predicted token
        prompt = RewardInput(prompt=input_item['input'], responses=predicted_list)
        responses = self.format_response(predicted_list)
        rewards = self.compute_rewards(prompt.prompt, [res.completion for res in responses])
        
        if finish:
            reward = [1] * len(predicted_list)  # calculate reward score based on predicted_list
        return list(rewards[ :, -1 ])

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def lora_enable(self, lora_rank=0, lora_train_bias="none"):
        if lora_rank > 0:
            lora_config = LoraConfig(
                inference_mode=False,
                r=lora_rank,
                lora_alpha=16,
                lora_dropout=0.05,
                bias=lora_train_bias,
            )
            self.model = get_peft_model(self.model, lora_config)
