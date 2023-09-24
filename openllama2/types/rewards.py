from pydantic import BaseModel
from typing import List

class RewardInput(BaseModel):
    prompt: str
    responses: List[str]


class ResponseModel(BaseModel):
    completion: str
    is_success: bool


class Item(BaseModel):
    roles: List[str]
    messages: List[str]
    successful_completions: List[str]