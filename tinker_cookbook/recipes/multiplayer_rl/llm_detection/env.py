import functools
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chz
import tinker
from tinker import ModelInput
from tinker_cookbook import model_info
from tinker_cookbook.completers import (
    MessageCompleter,
    StopCondition,
    TinkerMessageCompleter,
)
from tinker_cookbook.renderers import Message, Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

TARGET_SYSTEM_PROMPT = """
Your personality is {target_personality}, you will not deviate from this personality at all costs. 
""".strip()

PLAYER_SYSTEM_PROMPT = """
Your personality is {agent_personality}. Your goal is to maintain this personality while trying to guess the secret base model of the other agent. 
At the end of 20 turns, guess the other agent's base model by writing a line of the form 'Guess: <model_name>'.
""".strip()


class LLMDetectionEnv(Env):
    def __init__(self, target_agent: MessageCompleter, 
                        target_personality: str, 
                        target_base_model: str,
                        agent_personality: str, 
                        renderer: Renderer):
        self.target: MessageCompleter = target_agent
        self.target_personality: str = target_personality
        self.target_base_model: str = target_base_model
        self.sys_for_target: Message = {
            "role": "system",
            "content": TARGET_SYSTEM_PROMPT.format(target_personality=target_personality),
        }
        self.sys_for_agent: Message = {
            "role": "system",
            "content": PLAYER_SYSTEM_PROMPT.format(agent_personality=agent_personality),
        }
        self.renderer: Renderer = renderer
        self.turns: list[Message] = []

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _convo_for_player(self) -> list[Message]:
        """Conversation from the player's perspective."""
        game_role_to_chat_role = {"answerer": "user", "player": "assistant"}
        return [self.sys_for_agent] + [
            {"role": game_role_to_chat_role[turn["role"]], "content": turn["content"]}
            for turn in self.turns
        ]

    def _get_obs(self) -> ModelInput:
        """Get the observation for the player in tokenized form"""
        return self.renderer.build_generation_prompt(self._convo_for_player())

    def _convo_for_answerer(self) -> list[Message]:
        """Conversation from the answerer's perspective."""
        game_role_to_chat_role = {"answerer": "assistant", "player": "user"}
        return (
            [self.sys_for_target]
            + [
                {"role": game_role_to_chat_role[turn["role"]], "content": turn["content"]}
                for turn in self.turns
            ]
        )

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return self._get_obs(), self.stop_condition

    def _compute_reward(self, content: str) -> float:
        """
        Returns 1.0 if the content contains the answer, 0.0 otherwise.
        """
        match = re.match(r"Guess: (.*)", content)
        maybe_answer = match.group(1) if match else None
        content_contains_answer = (maybe_answer is not None) and (
            maybe_answer.lower() == self.target_basel_model.lower()
        )
        return 1.0 if content_contains_answer else 0.0

    async def step(self, action: Action) -> StepResult:
        """
        In one step,
        1. The environment accepts an action from the player (a message).
        2. We obtain the response from the answerer and update the conversation history in self.turns.
        3. We calculate the reward and decide whether to end the episode.
        4. We return these information, along with the next observation built from the updated conversation history.
        """

        # step 1: accepts the action from the player (policy)
        (action_message, _parse_success) = self.renderer.parse_response(action)
        self.turns.append({"role": "agent", "content": action_message["content"]})

        # step 2: the answerer responds
        answer_message = await self.target(self._convo_for_answerer())
        self.turns.append({"role": "target", "content": answer_message["content"]})

        # step 3: we calculate the reward and decide whether to end the episode.
        # the episode ends if the player guessed the answer or the player asked more than 20 questions
        reward = self._compute_reward(action_message["content"])
        episode_done = (reward == 1) or (len(self.turns) // 2 >= 20)

        # Log the turn
        turn_num = len(self.turns) // 2
        logtree.log_text(f"Turn {turn_num} - Agent: {action_message['content']}")
        logtree.log_text(f"Turn {turn_num} - Target: {answer_message['content']}")
        if episode_done:
            logtree.log_text(
                f"Game Over - Secret: {self.target_base_model}, Won: {'✓' if reward == 1 else '✗'}, Turns: {turn_num}"
            )

        # step 4: we return the next observation, reward, and whether the episode is done
        step_result = StepResult(
            next_observation=self._get_obs(),
            next_stop_condition=self.stop_condition,
            episode_done=episode_done,
            reward=reward,
        )

        return step_result


# The EnvGroupBuilder is trivial: just return a list of copies of the same environment.


@functools.cache
def _load_words_from_file() -> list[str]:
    module_dir = Path(__file__).parent
    file_path = module_dir / "common_english_nouns.txt"

    rng = random.Random(0)
    with open(file_path, "r") as f:
        words = [line.strip() for line in f.readlines()]
    rng.shuffle(words)
    return words


@dataclass(frozen=True)
class LLMDetectionEnvGroupBuilder(EnvGroupBuilder):
    target_agent: MessageCompleter
    target_personality: str
    target_base_model: str
    renderer: Renderer
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        return [
            LLMDetectionEnv(self.target_agent, self.target_personality, self.target_base_model, self.renderer)
            for _ in range(self.num_envs)
        ]


# The dataset just indexes into the list of possible answers.
@dataclass(frozen=True)
class LLMDetectionDataset(RLDataset):
    target_agents: Sequence[MessageCompleter]
    target_personalities: Sequence[str]
    target_base_models: Sequence[str]
    agent_personalities: Sequence[str]
    renderer: Renderer
    batch_size: int
    group_size: int

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            LLMDetectionEnvGroupBuilder(
                target_agent=self.target_agents[(index * self.batch_size + i) % len(self.target_agents)],
                target_personality=self.target_personalities[(index * self.batch_size + i) % len(self.target_personalities)],
                target_base_model=self.target_base_models[(index * self.batch_size + i) % len(self.target_base_models)],
                agent_personality=self.agent_personalities[(index * self.batch_size + i) % len(self.agent_personalities)],
                renderer=self.renderer,
                num_envs=self.group_size,
            )
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return len(self.target_personalities) * len(self.agent_personalities) * len(self.target_agents) // self.batch_size


@chz.chz
class LLMDetectionDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    possible_models: Sequence[str]
    target_personalities: Sequence[str]
    agent_personalities: Sequence[str]
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    base_url: str | None = None
    num_epochs: int = 1
    test_group_size: int = 32

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        service_client = tinker.ServiceClient(base_url=self.base_url)
        targets = [self._construct_target(service_client, base_model=model) 
                   for model in self.possible_models]
        
        train_words, test_words = self._get_train_and_test_words()
        player_renderer = get_renderer(
            self.renderer_name, get_tokenizer(self.model_name_for_tokenizer)
        )
        assert self.batch_size <= len(train_words)
        training_dataset = LLMDetectionDataset(
            target_agents=targets,
            target_personalities=self.target_personalities,
            target_base_models=self.possible_models,
            agent_personalities=self.agent_personalities,
            renderer=player_renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
        )
        test_dataset = LLMDetectionDataset(
            target_agents=targets,
            target_personalities=self.target_personalities,
            target_base_models=self.possible_models,
            agent_personalities=self.agent_personalities,
            renderer=player_renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
        )
        return training_dataset, test_dataset
    
    def _construct_target(self, service_client: tinker.ServiceClient, base_model: str) -> MessageCompleter:
        if base_model.startswith("Qwen/Qwen3"):
            target_renderer_name = "qwen3_disable_thinking"
        else:
            target_renderer_name = model_info.get_recommended_renderer_name(
                base_model
            )
        target_tokenizer = get_tokenizer(base_model)
        target_renderer = get_renderer(target_renderer_name, target_tokenizer)
        target_sampling_client = service_client.create_sampling_client(
            base_model=base_model
        )
        target = TinkerMessageCompleter(
            sampling_client=target_sampling_client, renderer=target_renderer, max_tokens=200
        )
        return target

    def _get_train_and_test_words(self) -> tuple[list[str], list[str]]:
        words = _load_words_from_file()
        num_test = min(len(words) // 5, 100)
        train_words = words[:-num_test]
        test_words = words[-num_test:]
        train_words = train_words * self.num_epochs
        return train_words, test_words
