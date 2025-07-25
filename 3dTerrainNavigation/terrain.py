import torch
import plotly.graph_objects as go

from typing import Tuple, Dict, NamedTuple
from noise import pnoise2

class TerrainInfo(NamedTuple):
    x: int
    y: int
    action: int
    reward: float
    points: float
    fuel: float

class Terrain(object):
    """
    Terrain Environment

    Initializes a seeded Terrain Environment
    """

    def _generate_terrain(
        self,
        shape: Tuple[int, int], 
        scale: float,
        octaves: int, 
        persistence: float, 
        lacunarity: float, 
        seed: int
    ) -> torch.Tensor:
        """
        Description
        -----------
        Generates Smooth terrain using Fractal Noise
        """
        terrain = torch.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                terrain[i, j] = pnoise2(i / scale, 
                                    j / scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    base=seed)
        return terrain

    def __init__(
        self,
        shape: Tuple[int, int] = (32, 32),
        initial_fuel: int = 30, 
        fuel_exhaustion_rate: float = 1.0,
        num_goals: int = 5,
        scale: float = 125.0, 
        octaves: int = 6, 
        persistence: float = 0.6, 
        lacunarity: float = 2.0, 
        seed: int = 69
    ):  
        self.gpu = torch.device("cuda")
        self.cpu = torch.device("cpu")
        

        self.terrain = self._generate_terrain(
            shape, 
            scale, 
            octaves, 
            persistence, 
            lacunarity, 
            seed
        )
        
        goals = []
        for _ in range(num_goals):
            randx = torch.randint(low=0, high=min(shape), size=(1,)).squeeze()
            randy = torch.randint(low=0, high=min(shape), size=(1,)).squeeze()
            goals.append(torch.tensor([randx, randy]))
        
        self.goals: torch.Tensor
        self.start: torch.Tensor
        self.agent_position: torch.Tensor
        self.fuel: torch.Tensor
        self.points: torch.Tensor

        randx = torch.randint(low=0, high=min(shape), size=(1,)).squeeze()
        randy = torch.randint(low=0, high=min(shape), size=(1,)).squeeze()

        self.goals = torch.stack(goals)
        self.visited_goals = torch.zeros(num_goals, dtype=torch.bool)
        self.start = torch.tensor([randx, randy])

        self.agent_position = self.start
        self.initial_fuel = initial_fuel
        self.fuel_exhaustion_rate = fuel_exhaustion_rate
        self.num_goals = num_goals
        self.shape = shape

        self.points = torch.tensor(0, dtype=torch.float32)
        self.fuel = torch.tensor(initial_fuel, dtype=torch.float32)
        self.actions = torch.tensor([0, 1, 2, 3])

        # agent's view
        self.agent_map = self.terrain.clone()
        self.agent_map[self.agent_position[0], self.agent_position[1]] = self.points.item() 

        # goals spatial view
        self.goal_map = self.terrain.clone()
        for i in range(self.num_goals):
            self.goal_map[self.goals[i,0], self.goals[i,1]] = self.points.item() + 1.0

    
    def _action_helper(
        self,
        current_elevation: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Description
        -----------
        Helper function for performing an action 
        """

        reward = 0.0
        points = self.points.item()
        
        next_elevation = self.terrain[self.agent_position[0], self.agent_position[1]]
        fuel_exhausted = 0.2 * min(0.0, (next_elevation - current_elevation).item())
        self.fuel -= fuel_exhausted
        
        # penalty for fuel exhaustion
        reward -= 0.5 * fuel_exhausted
        
        # additional positive rewards 
        is_at_goal = (self.agent_position == self.goals).all(dim=1)
        if is_at_goal.any():
            goal_idx = int(is_at_goal.nonzero(as_tuple=True)[0].item())

            if not self.visited_goals[goal_idx]:
                points += 1.0
                self.visited_goals[goal_idx] = True

        
        reward += 0.5 * points
        return reward, points
    
    def reset(self) -> torch.Tensor:
        # reset agent_position, fuel, points and visited_goals
        self.agent_position = self.start
        self.fuel = torch.tensor(self.initial_fuel, dtype=torch.float32)
        self.points = torch.tensor(0.0, dtype=torch.float32)
        self.visited_goals.fill_(False)
        return self.get_state(self.agent_position)

    def get_state(
        self, 
        previous_agent_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Description
        -----------
        Returns the Overlayed state of the agent, This tensor resides in GPU
        """

        # constructs the fuel channel
        fuel_map = torch.ones_like(self.terrain) * self.fuel.item()
        # modifies the agent channel        
        self.agent_map[previous_agent_position[0], previous_agent_position[1]] = 0.0
        self.agent_map[self.agent_position[0], self.agent_position[1]] = self.points.item() 
        # (4, *self.terrain.shape)
        state = torch.stack([self.terrain, fuel_map, self.agent_map, self.goal_map,])
        return state.to(self.gpu)
    
    def get_state_shape(self): return (4, *self.terrain.shape)

    def step(
        self,
        action: int
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, TerrainInfo]: 

        done = False 
        reward = torch.tensor(0.0, dtype=torch.float32)
        agentx = int(self.agent_position[0].item())
        agenty = int(self.agent_position[1].item())
        current_elevation = self.terrain[agentx, agenty]
        previous_agent_position = torch.tensor([agentx, agenty])
        self.fuel -= self.fuel_exhaustion_rate

        info = TerrainInfo(
            x=agentx,
            y=agenty,
            action=action,
            reward=reward.item(),
            points=self.points.item(),
            fuel=self.fuel.item()
        )

        # invalid action
        if action not in self.actions:
            reward -= 10
            return self.get_state(self.agent_position), reward, done, info
        # fuel exhausted
        if self.fuel <= 0:
            done = True
            reward -= 5
            return self.get_state(self.agent_position), reward, done, info
        # valid actions
        if action == 0:  # move up
            agentx = max(agentx - 1, 0)
        elif action == 1:  # move down
            agentx = min(agentx + 1, self.shape[0] - 1)
        elif action == 2:  # move left
            agenty = max(agenty - 1, 0)
        elif action == 3:  # move right
            agenty = min(agenty + 1, self.shape[1] - 1)

        self.agent_position = torch.tensor([agentx, agenty])
        reward_value, points = self._action_helper(current_elevation)

        # we have reached the maximum number of goals we can acheive
        if(int(points) == self.num_goals):
            done = True
        
        reward += reward_value
        self.points = torch.tensor(points, dtype=torch.float32)

        info = TerrainInfo(
            x=agentx,
            y=agenty,
            action=action,
            reward=reward.item(),
            points=self.points.item(),
            fuel=self.fuel.item()
        )

        return self.get_state(previous_agent_position), reward, done, info

    def render(self):
        """
        Description
        -----------
        Renders the terrain using plotly & opens it in localhost with start/goal markers.
        """

        print(self.goals)

        terrain_cpu = self.terrain.to(self.cpu)

        start_x, start_y = int(self.start[0].item()), int(self.start[1].item())
        start_z = terrain_cpu[start_x, start_y].item()
        
        fig = go.Figure(data=[
            go.Surface(
                z=terrain_cpu.numpy(),
                colorscale='earth',
                showscale=False
            )
        ])

        fig.add_trace(go.Scatter3d(
            x=[start_y], y=[start_x], z=[start_z],
            mode='markers',
            marker=dict(
                size=4,
                color='green',
                symbol='circle'
            ),
            name='Start'
        ))
        
        for i in range(self.num_goals):
            goal_x, goal_y = int(self.goals[i, 0].item()), int(self.goals[i, 1].item())
            goal_z = terrain_cpu[goal_x, goal_y].item()
            fig.add_trace(go.Scatter3d(
                x=[goal_y], y=[goal_x], z=[goal_z],
                mode='markers',
                marker=dict(
                    size=4,
                    color='red',
                    symbol='circle'
                ),
                name='Goal'
            ))

        fig.update_layout(
            title='3D Terrain Map',
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Elevation'
            ),
            autosize=False,
            width=800,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        fig.show()
        fig.write_image("./3dTerrainNavigation/terrain.svg", scale=1, format="svg", width=4096, height=4096)