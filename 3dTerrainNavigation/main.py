import torch
from terrain import Terrain
from network import ActorCriticNetwork, train

terrain = Terrain()
# terrain.render()
print(terrain.get_state_shape())
terrain.render()

TRAIN_ITERATIONS = 50
REPLAY_ITERATIONS = 32
MAX_REPLAY_ITERATIONS = 25000
PPO_EPOCHS = 30
GAMMA = 0.98
LAMBDA = 0.98
CLIP_COEFF = 0.1
VALUE_LOSS_COEFF = 0.38
ENTROPY_COEFF = 0.05


device = torch.device("cuda")
model = ActorCriticNetwork(terrain.get_state_shape(), num_actions=len(terrain.actions)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
replay_df, ppo_df = train(
    terrain,
    model,
    optimizer,
    device,
    TRAIN_ITERATIONS,
    REPLAY_ITERATIONS,
    MAX_REPLAY_ITERATIONS,
    PPO_EPOCHS,
    GAMMA, 
    LAMBDA,
    CLIP_COEFF,
    VALUE_LOSS_COEFF,
    ENTROPY_COEFF
)

print(replay_df)
print(ppo_df)