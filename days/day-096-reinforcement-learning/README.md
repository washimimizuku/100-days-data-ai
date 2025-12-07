# Day 96: Reinforcement Learning

## Learning Objectives

**Time**: 1 hour

- Understand reinforcement learning fundamentals
- Learn about agents, environments, and rewards
- Implement Q-learning and policy gradients
- Apply RL to simple problems

## Theory (15 minutes)

### What is Reinforcement Learning?

RL is a machine learning paradigm where agents learn to make decisions by interacting with an environment to maximize cumulative rewards.

**Key Concepts**:
- Agent: Decision maker
- Environment: World the agent interacts with
- State: Current situation
- Action: What the agent can do
- Reward: Feedback signal
- Policy: Strategy for choosing actions

### RL Framework

**Agent-Environment Loop**:
```
1. Agent observes state s
2. Agent takes action a
3. Environment returns reward r and new state s'
4. Repeat
```

**Goal**: Learn policy π that maximizes expected cumulative reward.

### Markov Decision Process (MDP)

**Components**:
- S: Set of states
- A: Set of actions
- P: Transition probabilities
- R: Reward function
- γ: Discount factor (0 < γ < 1)

**Value Function**:
```python
V(s) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s]
```

### Q-Learning

**Q-Function**: Expected return for taking action a in state s.

**Update Rule**:
```python
Q(s, a) = Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

**Implementation**:
```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
    
    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
    
    def get_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])
```

### Exploration vs Exploitation

**Epsilon-Greedy**:
```python
def epsilon_greedy(Q, state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return random_action()  # Explore
    return best_action(Q, state)  # Exploit
```

**Decay Schedule**:
```python
epsilon = epsilon_start * (epsilon_decay ** episode)
```

### Policy Gradient

**Policy**: Probability distribution over actions.

**REINFORCE Algorithm**:
```python
def reinforce_update(policy, trajectory, gamma=0.99):
    returns = []
    G = 0
    
    # Calculate returns
    for reward in reversed(trajectory['rewards']):
        G = reward + gamma * G
        returns.insert(0, G)
    
    # Update policy
    for state, action, G in zip(trajectory['states'], 
                                 trajectory['actions'], 
                                 returns):
        policy.update(state, action, G)
```

### Deep Q-Network (DQN)

**Neural Network Q-Function**:
```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)
```

**Experience Replay**:
```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

### Simple Environment

**Grid World**:
```python
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = (0, 0)
        self.goal = (size-1, size-1)
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        x, y = self.state
        
        if action == 0: y = max(0, y-1)
        elif action == 1: x = min(self.size-1, x+1)
        elif action == 2: y = min(self.size-1, y+1)
        elif action == 3: x = max(0, x-1)
        
        self.state = (x, y)
        reward = 1 if self.state == self.goal else -0.1
        done = self.state == self.goal
        
        return self.state, reward, done
```

### Training Loop

**Basic Training**:
```python
def train(agent, env, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
```

### Reward Shaping

**Design Good Rewards**:
```python
def shaped_reward(state, action, next_state):
    # Distance to goal
    distance = manhattan_distance(next_state, goal)
    
    # Reward components
    goal_reward = 100 if at_goal(next_state) else 0
    step_penalty = -1
    distance_reward = -distance * 0.1
    
    return goal_reward + step_penalty + distance_reward
```

### Common Algorithms

**Value-Based**:
- Q-Learning
- DQN
- Double DQN
- Dueling DQN

**Policy-Based**:
- REINFORCE
- Actor-Critic
- A3C
- PPO

**Model-Based**:
- Dyna-Q
- MCTS
- AlphaZero

### Evaluation

**Metrics**:
- Average reward per episode
- Success rate
- Steps to goal
- Learning curve

**Testing**:
```python
def evaluate(agent, env, episodes=100):
    rewards = []
    
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state, epsilon=0)  # No exploration
            state, reward, done = env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards)
```

### Use Cases

**Robotics**:
- Robot navigation
- Manipulation tasks
- Locomotion

**Games**:
- Board games (Chess, Go)
- Video games
- Card games

**Optimization**:
- Resource allocation
- Scheduling
- Traffic control

**Finance**:
- Trading strategies
- Portfolio management
- Risk assessment

### Best Practices

1. **Start Simple**: Test on simple environments
2. **Tune Hyperparameters**: Learning rate, discount factor
3. **Monitor Learning**: Track rewards and losses
4. **Use Baselines**: Compare with random policy
5. **Stabilize Training**: Experience replay, target networks
6. **Reward Design**: Shape rewards carefully

### Why This Matters

RL enables agents to learn complex behaviors through trial and error. It powers game-playing AI, robotics, and autonomous systems. Understanding RL fundamentals is essential for building adaptive AI systems.

## Exercise (40 minutes)

Complete the exercises in `exercise.py`:

1. **Q-Learning**: Implement Q-learning algorithm
2. **Grid World**: Create simple environment
3. **Training**: Train agent on environment
4. **Policy**: Implement policy-based method
5. **Evaluation**: Evaluate trained agent

## Resources

- [OpenAI Gym](https://gym.openai.com/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [RL Course by David Silver](https://www.davidsilver.uk/teaching/)

## Next Steps

- Complete the exercises
- Review the solution
- Take the quiz
- Move to Day 97: Model Optimization

Tomorrow you'll learn about model optimization including quantization, pruning, distillation, and deployment optimization techniques.
