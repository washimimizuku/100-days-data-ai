"""Day 96: Reinforcement Learning - Solutions

NOTE: Simplified RL implementations for learning.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
import random


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
        """Select action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        """Update Q-values using Q-learning rule."""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error


# Exercise 2: Grid World Environment
class GridWorld:
    """Simple grid world environment."""
    
    def __init__(self, size: int = 5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.state = 0
        self.goal = size * size - 1
    
    def reset(self) -> int:
        """Reset environment to initial state."""
        self.state = 0
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """Take action and return next state, reward, done."""
        x, y = self._state_to_pos(self.state)
        
        # Update position based on action
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # right
            x = min(self.size - 1, x + 1)
        elif action == 2:  # down
            y = min(self.size - 1, y + 1)
        elif action == 3:  # left
            x = max(0, x - 1)
        
        self.state = self._pos_to_state(x, y)
        
        # Calculate reward
        if self.state == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01  # Small penalty for each step
            done = False
        
        return self.state, reward, done
    
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
    """Train agent on environment."""
    rewards_history = []
    epsilon = epsilon_start
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = agent.get_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        rewards_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.3f}, Epsilon: {epsilon:.3f}")
    
    return rewards_history


# Exercise 4: Policy Gradient
class PolicyGradientAgent:
    """Simple policy gradient agent."""
    
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.policy = np.ones((n_states, n_actions)) / n_actions
    
    def get_action(self, state: int) -> int:
        """Sample action from policy."""
        probs = self.policy[state]
        probs = probs / probs.sum()  # Normalize
        return np.random.choice(self.n_actions, p=probs)
    
    def update(self, trajectory: Dict[str, List], gamma: float = 0.99):
        """Update policy using REINFORCE."""
        states = trajectory['states']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        
        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        # Normalize returns
        returns = np.array(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / returns.std()
        
        # Update policy
        for state, action, G in zip(states, actions, returns):
            # Increase probability of actions with positive returns
            self.policy[state, action] += self.alpha * G
            # Ensure probabilities stay positive
            self.policy[state] = np.maximum(self.policy[state], 1e-8)
            # Normalize
            self.policy[state] /= self.policy[state].sum()


# Exercise 5: Agent Evaluation
def evaluate_agent(agent: QLearningAgent, env: GridWorld, 
                   episodes: int = 100) -> Dict[str, float]:
    """Evaluate trained agent."""
    rewards = []
    steps_list = []
    successes = 0
    
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = agent.get_action(state, epsilon=0)  # No exploration
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        steps_list.append(steps)
        if done:
            successes += 1
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_steps': np.mean(steps_list),
        'success_rate': successes / episodes
    }


# Bonus: Experience Replay
class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, state: int, action: int, reward: float, 
            next_state: int, done: bool):
        """Add experience to buffer."""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


def demo_reinforcement_learning():
    """Demonstrate RL concepts."""
    print("Day 96: Reinforcement Learning - Solutions Demo\n" + "=" * 60)
    
    print("\n1. Q-Learning Agent")
    agent = QLearningAgent(n_states=25, n_actions=4, alpha=0.1, gamma=0.99)
    print(f"   Q-table shape: {agent.Q.shape}")
    print(f"   Initial Q-values: all zeros")
    
    print("\n2. Grid World Environment")
    env = GridWorld(size=5)
    state = env.reset()
    print(f"   Grid size: {env.size}x{env.size}")
    print(f"   States: {env.n_states}, Actions: {env.n_actions}")
    print(f"   Start: {state}, Goal: {env.goal}")
    
    print("\n3. Training Agent")
    print("   Training for 500 episodes...")
    rewards = train_agent(agent, env, episodes=500, epsilon_start=1.0, 
                         epsilon_end=0.01, epsilon_decay=0.995)
    print(f"   Final 100 episodes avg reward: {np.mean(rewards[-100:]):.3f}")
    
    print("\n4. Policy Gradient Agent")
    pg_agent = PolicyGradientAgent(n_states=25, n_actions=4, alpha=0.01)
    print(f"   Policy shape: {pg_agent.policy.shape}")
    print(f"   Initial policy: uniform distribution")
    
    print("\n5. Agent Evaluation")
    metrics = evaluate_agent(agent, env, episodes=100)
    print(f"   Mean reward: {metrics['mean_reward']:.3f}")
    print(f"   Mean steps: {metrics['mean_steps']:.1f}")
    print(f"   Success rate: {metrics['success_rate']:.1%}")
    
    print("\n6. Experience Replay")
    buffer = ReplayBuffer(capacity=1000)
    for i in range(10):
        buffer.add(i, i % 4, 0.1, i+1, False)
    print(f"   Buffer size: {len(buffer)}")
    batch = buffer.sample(5)
    print(f"   Sampled batch size: {len(batch)}")
    
    print("\n" + "=" * 60)
    print("All RL concepts demonstrated!")


if __name__ == "__main__":
    demo_reinforcement_learning()
