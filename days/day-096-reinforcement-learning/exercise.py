"""Day 96: Reinforcement Learning - Exercises

NOTE: Uses mock implementations for learning RL concepts.
"""

import numpy as np
from typing import Tuple, List, Dict, Any


# Exercise 1: Q-Learning
class QLearningAgent:
    """Q-Learning agent."""
    
    def __init__(self, n_states: int, n_actions: int, 
                 alpha: float = 0.1, gamma: float = 0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
    
    def get_action(self, state: int, epsilon: float = 0.1) -> int:
        """
        Select action using epsilon-greedy policy.
        
        TODO: Implement epsilon-greedy
        TODO: Explore with probability epsilon
        TODO: Exploit best action otherwise
        """
        pass
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        """
        Update Q-values using Q-learning rule.
        
        TODO: Calculate TD target
        TODO: Calculate TD error
        TODO: Update Q-value
        """
        pass


# Exercise 2: Grid World Environment
class GridWorld:
    """Simple grid world environment."""
    
    def __init__(self, size: int = 5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, right, down, left
        self.state = 0
        self.goal = size * size - 1
    
    def reset(self) -> int:
        """
        Reset environment to initial state.
        
        TODO: Set state to start position
        TODO: Return initial state
        """
        pass
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take action and return next state, reward, done.
        
        Actions: 0=up, 1=right, 2=down, 3=left
        
        TODO: Update state based on action
        TODO: Calculate reward
        TODO: Check if done
        TODO: Return (next_state, reward, done)
        """
        pass
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to (x, y) position."""
        return state % self.size, state // self.size
    
    def _pos_to_state(self, x: int, y: int) -> int:
        """Convert (x, y) position to state index."""
        return y * self.size + x


# Exercise 3: Training Loop
def train_agent(agent: QLearningAgent, env: GridWorld, 
                episodes: int = 1000, epsilon_start: float = 1.0,
                epsilon_end: float = 0.01, epsilon_decay: float = 0.995) -> List[float]:
    """
    Train agent on environment.
    
    TODO: Loop over episodes
    TODO: Decay epsilon
    TODO: Collect experience
    TODO: Update agent
    TODO: Track rewards
    TODO: Return reward history
    """
    pass


# Exercise 4: Policy Gradient
class PolicyGradientAgent:
    """Simple policy gradient agent."""
    
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        # Policy: probability distribution over actions
        self.policy = np.ones((n_states, n_actions)) / n_actions
    
    def get_action(self, state: int) -> int:
        """
        Sample action from policy.
        
        TODO: Get action probabilities
        TODO: Sample action
        TODO: Return action
        """
        pass
    
    def update(self, trajectory: Dict[str, List], gamma: float = 0.99):
        """
        Update policy using REINFORCE.
        
        Trajectory contains:
        - states: list of states
        - actions: list of actions
        - rewards: list of rewards
        
        TODO: Calculate returns
        TODO: Update policy
        """
        pass


# Exercise 5: Agent Evaluation
def evaluate_agent(agent: QLearningAgent, env: GridWorld, 
                   episodes: int = 100) -> Dict[str, float]:
    """
    Evaluate trained agent.
    
    TODO: Run episodes without exploration
    TODO: Track rewards and steps
    TODO: Calculate statistics
    TODO: Return metrics
    """
    pass


# Bonus: Experience Replay
class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, state: int, action: int, reward: float, 
            next_state: int, done: bool):
        """
        Add experience to buffer.
        
        TODO: Store experience
        TODO: Handle buffer overflow
        """
        pass
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample random batch.
        
        TODO: Sample experiences
        TODO: Return batch
        """
        pass
    
    def __len__(self) -> int:
        return len(self.buffer)


if __name__ == "__main__":
    print("Day 96: Reinforcement Learning - Exercises")
    print("=" * 50)
    
    # Test Exercise 1
    print("\nExercise 1: Q-Learning Agent")
    agent = QLearningAgent(n_states=25, n_actions=4)
    print(f"Agent created: {agent is not None}")
    print(f"Q-table shape: {agent.Q.shape}")
    
    # Test Exercise 2
    print("\nExercise 2: Grid World")
    env = GridWorld(size=5)
    print(f"Environment created: {env is not None}")
    print(f"States: {env.n_states}, Actions: {env.n_actions}")
    
    # Test Exercise 3
    print("\nExercise 3: Training")
    print("Training function defined")
    
    # Test Exercise 4
    print("\nExercise 4: Policy Gradient")
    pg_agent = PolicyGradientAgent(n_states=25, n_actions=4)
    print(f"Policy agent created: {pg_agent is not None}")
    
    # Test Exercise 5
    print("\nExercise 5: Evaluation")
    print("Evaluation function defined")
    
    print("\n" + "=" * 50)
    print("Complete the TODOs to finish the exercises!")
    print("\nNote: These are simplified RL implementations.")
    print("For production RL, use Stable Baselines3 or RLlib.")
