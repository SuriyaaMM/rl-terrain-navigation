import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os 
from typing import Tuple, List
from terrain import Terrain

class ActorCriticNetwork(nn.Module):
    """
    Constructs the Actor Critic network for Agent
    """

    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        num_actions: int
    ):
        super().__init__()
        self.device_gpu = torch.device("cuda")
        self.device_cpu = torch.device("cpu")

        self.shared_net = nn.Sequential(
            # input: (4, 256, 256)
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # -> (32, 128, 128)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # -> (64, 64, 64)
            
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.LeakyReLU(),
            # nn.MaxPool2d(2, 2), # -> (128, 32, 32)
            
            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.LeakyReLU(),
            # nn.MaxPool2d(2, 2), # -> (256, 16, 16)
            
            nn.Flatten()
        )

        with torch.no_grad():
            shared_output_dim = self.shared_net(torch.zeros((1, *state_shape))).shape[1]
        
        self.actor_net = nn.Sequential(
            nn.Linear(in_features=shared_output_dim,
                      out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512,
                      out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512,
                      out_features=num_actions)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(in_features=shared_output_dim,
                      out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512,
                      out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512,
                      out_features=1)
        )

    def forward(self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        shared_output = self.shared_net(state)
        actor_logits = self.actor_net(shared_output)
        state_value = self.critic_net(shared_output)
        return actor_logits, state_value

    def act(
        self,
        state: torch.Tensor
    ):
        """
        Description
        -----------
        invokes the actor network & samples an action 
        """
        actor_logits, state_value = self.forward(state)
        actor_dist = torch.distributions.Categorical(logits=actor_logits)
        action = actor_dist.sample()
        logprob = actor_dist.log_prob(action)
        entropy = actor_dist.entropy()

        return action, state_value, logprob, entropy

    def criticize(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ):
        R"""
        Description
        -----------
        invokes the critic network & critizes the states and actions in memory
        """
        actor_logits, state_value = self.forward(states)
        actor_dist = torch.distributions.Categorical(logits=actor_logits)
        logprobs = actor_dist.log_prob(actions)
        entropy = actor_dist.entropy()

        return state_value, logprobs, entropy
    
class MemoryTensor(object):

    def __init__(self):
        self.states: List[torch.Tensor]
        self.actions: List[torch.Tensor]
        self.dones: List[torch.Tensor]
        self.rewards: List[torch.Tensor]
        self.state_values: List[torch.Tensor]
        self.logprobs: List[torch.Tensor]
        self.entropies: List[torch.Tensor]

        self.states = []
        self.actions = []
        self.dones = []
        self.rewards = []
        self.state_values = []
        self.logprobs = []
        self.entropies = []

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        reward: torch.Tensor,
        state_value: torch.Tensor,
        logprob: torch.Tensor,
        entropy: torch.Tensor
    ):
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.logprobs.append(logprob)
        self.entropies.append(entropy)


def _zscore_norm(x: torch.Tensor, eps: float = 1e-12):
    mean = x.mean()
    std = x.std()
    z = (x - mean)/(std + eps)
    return z, mean, std

def _calculate_gae_and_returns_norm(
    rewards: torch.Tensor,
    state_values: torch.Tensor,
    dones: torch.Tensor,
    last_advantage: torch.Tensor,
    next_state_value: torch.Tensor,
    device: torch.device,
    gamma: float,
    lmbda: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description
    -----------
    Calculates the Generalized Advantage (Normalized) & Returns (Un-Normalized)
    """
    advantages = torch.zeros(len(rewards)).to(device)
    for t in reversed(range(len(rewards))):
        temporal_diff = rewards[t] + gamma * next_state_value * (1 - dones[t].item()) - state_values[t]
        advantages[t] = last_advantage = temporal_diff + gamma * lmbda * (1 - dones[t].item()) * last_advantage
        next_state_value = state_values[t]
    returns = advantages + state_values
    returns.to(device)
    advantages, _, _ = _zscore_norm(advantages)
    return advantages, returns

def train(
    env: Terrain,
    model: ActorCriticNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    train_iterations: int,
    replay_iterations: int,
    max_replay_iterations: int,
    ppo_epochs: int,
    gamma: float,
    lmbda: float,
    clip_coeff: float,
    value_loss_coeff: float,
    entropy_initial: float,
    entropy_min: float,
    save_model_path: str = "./terrain_models",
    save_data_path: str = "./terrain_data"
):
    actions_hist = []
    dones_hist = []
    rewards_hist = []
    state_values_hist = []
    logprobs_hist = []
    entropies_hist = []

    ratios_hist = []
    policy_loss1_hist = []
    policy_loss2_hist = []
    policy_loss_hist = []
    value_loss_hist = []
    loss_hist = []

    best_trajectory = []
    best_episode_reward = -float('inf')

    entropy_coeff = entropy_initial

    for i in range(train_iterations):
        print("-"*80)
        print(f"Training Iteration {i + 1}")
        print("-"*80)
        # replay memory tensor
        memory = MemoryTensor()

        for i in range(replay_iterations):
            current_trajectory = []
            iteration = 0
            state = env.reset()
            done = False
            cumulative_episode_reward = torch.tensor(0.0, dtype=torch.float32)
            start_info = {
                'x': env.agent_position[0].item(),
                'y': env.agent_position[1].item(),
                'action': -1,  # no action was taken
                'reward': 0.0,
                'points': 0.0,
                'fuel': env.fuel.item()
            }
            current_trajectory.append(start_info)
            # we will either quit if done early or if iteration exceeds max_iterations
            while not done and iteration < max_replay_iterations:
                iteration += 1
                action, state_value, logprob, entropy = model.act(state.unsqueeze(0))
                next_state, reward, done, info = env.step(int(action.item()))
                current_trajectory.append(info._asdict())
                memory.push(state, 
                            action, torch.tensor(done, dtype=torch.bool), reward, 
                            state_value.squeeze().detach(), logprob.detach(), entropy.detach())
                state = next_state
                cumulative_episode_reward += reward
            
            if cumulative_episode_reward > best_episode_reward:
                best_episode_reward = cumulative_episode_reward
                best_trajectory = current_trajectory

            print(f"Replay Iteration({i}) | Average Reward = {cumulative_episode_reward}")

        states = torch.stack(memory.states).to(device)
        actions = torch.stack(memory.actions).to(device)
        dones = torch.stack(memory.dones).to(device)
        rewards = torch.stack(memory.rewards).to(device)
        state_values = torch.stack(memory.state_values).to(device)

        # this next state is calculated for advantage calculation
        # becasue advantage depends on future episodes
        with torch.no_grad():
            _, next_state_value, _, _ = model.act(state.unsqueeze(0))
        
        logprobs = torch.stack(memory.logprobs)
        entropies = torch.stack(memory.entropies)
        
        advantages, returns = _calculate_gae_and_returns_norm(
            rewards, 
            state_values, 
            dones, 
            torch.tensor(0.0), 
            next_state_value.squeeze(), 
            device,
            gamma, 
            lmbda
        )

        # history tracking
        _, most_taken_actions = torch.unique(actions, return_counts=True)
        _, most_done_decisions = torch.unique(dones, return_counts=True)
        actions_hist.append(max(most_taken_actions).item())
        dones_hist.append(max(most_done_decisions).item())
        rewards_hist.append(rewards.mean().item())
        state_values_hist.append(state_values.mean().item())
        logprobs_hist.append(logprobs.mean().item())
        entropies_hist.append(entropies.mean().item())

        
        for i in range(ppo_epochs):
            
            new_values, new_logprobs, _ = model.criticize(states, actions)

            ratio = torch.exp(new_logprobs - logprobs)
            policy_loss1 = advantages * ratio
            policy_loss2 = advantages * torch.clamp(ratio, 1 - clip_coeff, 1 + clip_coeff)
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

            value_loss = F.smooth_l1_loss(new_values.squeeze(), returns)

            loss = policy_loss + value_loss * value_loss_coeff - entropy_coeff * entropies.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            # history tracking
            loss_hist.append(loss.item())
            policy_loss_hist.append(policy_loss.item())
            policy_loss1_hist.append(policy_loss1.mean().item())
            policy_loss2_hist.append(policy_loss2.mean().item())
            value_loss_hist.append(value_loss.item())
            ratios_hist.append(ratio.mean().item())

            print(f"PPO Epoch ({i + 1}) | Loss = {loss}")

        entropy_coeff = entropy_coeff = entropy_min + (entropy_initial - entropy_min) / 2 * (
                    1 + torch.cos(torch.tensor(torch.pi * i / train_iterations))
            )
        entropy_coeff = entropy_coeff.item()
        scheduler.step()
        print(f"Entropy = {entropy_coeff}")
        print(f"")
    os.makedirs(save_data_path, exist_ok=True)
    os.makedirs(save_model_path, exist_ok=True)

    print(f"Saved model to {save_model_path}")
    torch.save(model.state_dict(), os.path.join(save_model_path, "model.pth"))

    replay_df = pd.DataFrame.from_dict({
        "actions" : actions_hist,
        "rewards" : rewards_hist,
        "state_values" : state_values_hist,
        "logprobs" : logprobs_hist,
        "entropies" : entropies_hist
    })

    ppo_df = pd.DataFrame.from_dict({
        "ratios" : ratios_hist,
        "policy_loss1" : policy_loss1_hist,
        "policy_loss2" : policy_loss2_hist,
        "policy_loss" : policy_loss_hist,
        "value_loss" : value_loss_hist,
        "loss_hist" : loss_hist
    })  

    trajectory_df = pd.DataFrame(best_trajectory)

    trajectory_df.to_csv(os.path.join(save_data_path, "best_trajectory.csv"), index=False)
    replay_df.to_csv(os.path.join(save_data_path, "reply_df.csv"))
    ppo_df.to_csv(os.path.join(save_data_path, "ppo_df.csv"))

    replay_df.describe()

    return replay_df, ppo_df