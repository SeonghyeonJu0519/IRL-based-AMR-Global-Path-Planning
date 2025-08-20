import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from irl_training.amr_path_planning_irl import AMRGridworld, ComputationalProfiler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImprovedDQN8Directions(nn.Module):
    """개선된 8방향 DQN 네트워크"""
    
    def __init__(self, input_size, hidden_size=256, output_size=8):
        super(ImprovedDQN8Directions, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class ImprovedDQN8DirectionsAgent:
    """개선된 8방향 DQN 에이전트"""
    
    def __init__(self, state_size, action_size=8, lr=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # 더 큰 메모리
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # 더 높은 최소값
        self.epsilon_decay = 0.9995  # 더 느린 감소
        self.gamma = 0.99
        self.learning_rate = lr
        self.batch_size = 64  # 더 큰 배치
        
        # 8방향 이동 정의
        self.directions = [
            (-1, 0),   # 상
            (1, 0),    # 하
            (0, -1),   # 좌
            (0, 1),    # 우
            (-1, -1),  # 좌상
            (-1, 1),   # 우상
            (1, -1),   # 좌하
            (1, 1)     # 우하
        ]
        
        # 네트워크 초기화
        self.q_network = ImprovedDQN8Directions(state_size, 256, action_size).to(device)
        self.target_network = ImprovedDQN8Directions(state_size, 256, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        self.update_target_counter = 0
        self.update_target_freq = 50  # 더 자주 업데이트
        
    def get_state_features(self, env, pos):
        """개선된 상태 특성 추출"""
        x, y = pos
        goal_x, goal_y = env.goal_pos
        
        # 기본 특성 (4개)
        features = [
            x / env.grid_size,
            y / env.grid_size,
            goal_x / env.grid_size,
            goal_y / env.grid_size,
        ]
        
        # 거리 정보 (3개)
        dx = goal_x - x
        dy = goal_y - y
        manhattan_dist = abs(dx) + abs(dy)
        euclidean_dist = np.sqrt(dx**2 + dy**2)
        diagonal_dist = max(abs(dx), abs(dy)) + (1.414 - 1) * min(abs(dx), abs(dy))
        
        features.extend([
            manhattan_dist / (2 * env.grid_size),
            euclidean_dist / (np.sqrt(2) * env.grid_size),
            diagonal_dist / (np.sqrt(2) * env.grid_size),
        ])
        
        # 방향 정보 (2개)
        features.extend([
            dx / env.grid_size,
            dy / env.grid_size,
        ])
        
        # 8방향 주변 장애물 정보 (8개)
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size:
                features.append(float(env.grid[nx, ny] > 0))
            else:
                features.append(1.0)
        
        # 추가 특성 (4개) - 총 21개
        features.extend([
            float(x == 0 or x == env.grid_size - 1),  # 경계 여부
            float(y == 0 or y == env.grid_size - 1),  # 경계 여부
            float(x == goal_x),  # x 위치 일치
            float(y == goal_y),  # y 위치 일치
        ])
        
        return np.array(features, dtype=np.float32)
    
    def act(self, state, training=True):
        """행동 선택"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """경험 재플레이"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.LongTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_target_counter += 1
        if self.update_target_counter % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

def calculate_improved_reward_8d(env, current_pos, next_pos, goal_pos, action_idx):
    """개선된 8방향 보상 함수"""
    if next_pos == goal_pos:
        return 2000  # 더 큰 목표 보상
    
    if env.grid[next_pos[0], next_pos[1]] > 0:
        return -200  # 더 큰 장애물 페널티
    
    # 이동 비용 계산
    if action_idx < 4:  # 상하좌우
        move_cost = 1.0
    else:  # 대각선
        move_cost = 1.414
    
    # 거리 기반 보상 (8방향 고려)
    old_dist = max(abs(current_pos[0] - goal_pos[0]), abs(current_pos[1] - goal_pos[1])) + \
               (1.414 - 1) * min(abs(current_pos[0] - goal_pos[0]), abs(current_pos[1] - goal_pos[1]))
    new_dist = max(abs(next_pos[0] - goal_pos[0]), abs(next_pos[1] - goal_pos[1])) + \
               (1.414 - 1) * min(abs(next_pos[0] - goal_pos[0]), abs(next_pos[1] - goal_pos[1]))
    
    if new_dist < old_dist:
        reward = 50  # 더 큰 거리 감소 보상
    elif new_dist == old_dist:
        reward = -1  # 더 작은 페널티
    else:
        reward = -20  # 더 큰 거리 증가 페널티
    
    # 대각선 이동 보상
    if action_idx >= 4:  # 대각선 이동
        reward += 15  # 대각선 이동 추가 보상
    
    # 목표 방향으로의 이동 보상
    goal_dx = goal_pos[0] - current_pos[0]
    goal_dy = goal_pos[1] - current_pos[1]
    move_dx = next_pos[0] - current_pos[0]
    move_dy = next_pos[1] - current_pos[1]
    
    # x 방향이 올바르면 보상
    if goal_dx > 0 and move_dx > 0:  # 목표가 오른쪽에 있고 오른쪽으로 이동
        reward += 20
    elif goal_dx < 0 and move_dx < 0:  # 목표가 왼쪽에 있고 왼쪽으로 이동
        reward += 20
    elif goal_dx == 0 and move_dx == 0:  # x 위치가 맞으면 보상
        reward += 10
    
    # y 방향이 올바르면 보상
    if goal_dy > 0 and move_dy > 0:  # 목표가 아래쪽에 있고 아래쪽으로 이동
        reward += 20
    elif goal_dy < 0 and move_dy < 0:  # 목표가 위쪽에 있고 위쪽으로 이동
        reward += 20
    elif goal_dy == 0 and move_dy == 0:  # y 위치가 맞으면 보상
        reward += 10
    
    # 잘못된 방향으로 가면 큰 페널티
    if (goal_dx > 0 and move_dx < 0) or (goal_dx < 0 and move_dx > 0):
        reward -= 30
    if (goal_dy > 0 and move_dy < 0) or (goal_dy < 0 and move_dy > 0):
        reward -= 30
    
    # 이동 비용 페널티 (더 작게)
    reward -= move_cost * 0.5
    
    return reward

def train_improved_dqn_8d(env, episodes=2000, max_steps=1000):
    """개선된 8방향 DQN 훈련"""
    state_size = 21  # 4(위치) + 3(거리) + 2(방향) + 8(주변장애물) + 4(추가특성)
    agent = ImprovedDQN8DirectionsAgent(state_size)
    
    print(f"Training Improved 8-direction DQN for {episodes} episodes...")
    training_start_time = time.time()
    scores = []
    success_count = 0
    
    for episode in range(episodes):
        current_pos = env.start_pos
        state = agent.get_state_features(env, current_pos)
        total_reward = 0
        visited_positions = set([current_pos])
        reached_goal = False
        
        for step in range(max_steps):
            action = agent.act(state, training=True)
            
            # 8방향 이동 실행
            dx, dy = agent.directions[action]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # 경계 체크
            if (0 <= next_pos[0] < env.grid_size and 
                0 <= next_pos[1] < env.grid_size):
                
                reward = calculate_improved_reward_8d(env, current_pos, next_pos, env.goal_pos, action)
                
                # 재방문 페널티 (더 작게)
                if next_pos in visited_positions:
                    reward -= 2
                else:
                    visited_positions.add(next_pos)
                
                next_state = agent.get_state_features(env, next_pos)
                done = (next_pos == env.goal_pos) or (step == max_steps - 1)
                
                if next_pos == env.goal_pos:
                    reached_goal = True
                
                agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                current_pos = next_pos
                total_reward += reward
                
                if done:
                    break
            else:
                # 경계 밖으로 나가면 페널티
                reward = -100
                agent.remember(state, action, reward, state, False)
                total_reward += reward
        
        if reached_goal:
            success_count += 1
        
        if len(agent.memory) > agent.batch_size:
            agent.replay()
        
        scores.append(total_reward)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            success_rate = success_count / (episode + 1) * 100
            print(f"Episode {episode}: Avg Score = {avg_score:.2f}, Epsilon = {agent.epsilon:.3f}, Success Rate = {success_rate:.1f}%")
    
    training_time = time.time() - training_start_time
    print(f"Improved 8-direction DQN training completed in {training_time:.2f}s")
    print(f"Final Success Rate: {success_count / episodes * 100:.1f}%")
    
    return agent, training_time

def improved_dqn_8d_path_planning(env, agent, profiler=None):
    """개선된 8방향 DQN 경로 계획"""
    if profiler:
        profiler.start_profiling()
    
    current_pos = env.start_pos
    path = [current_pos]
    max_steps = env.grid_size * 4  # 더 긴 최대 스텝
    visited = set([current_pos])
    
    for step in range(max_steps):
        if current_pos == env.goal_pos:
            print(f"Goal reached at step {step}")
            break
        
        state = agent.get_state_features(env, current_pos)
        action = agent.act(state, training=False)
        
        if profiler:
            profiler.increment_neural_inferences(1)
        
        dx, dy = agent.directions[action]
        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        if (0 <= next_pos[0] < env.grid_size and 
            0 <= next_pos[1] < env.grid_size and
            env.grid[next_pos[0], next_pos[1]] == 0):
            
            current_pos = next_pos
            path.append(current_pos)
            visited.add(current_pos)
        else:
            # 유효하지 않은 이동이면 다른 방향 시도
            for alt_action in range(8):
                if alt_action != action:
                    dx, dy = agent.directions[alt_action]
                    alt_pos = (current_pos[0] + dx, current_pos[1] + dy)
                    if (0 <= alt_pos[0] < env.grid_size and 
                        0 <= alt_pos[1] < env.grid_size and
                        env.grid[alt_pos[0], alt_pos[1]] == 0):
                        current_pos = alt_pos
                        path.append(current_pos)
                        visited.add(current_pos)
                        break
            else:
                # 모든 방향이 막혀있으면 중단
                print(f"Stuck at step {step}, no valid moves")
                break
    
    if profiler:
        profiler.end_profiling()
    
    return path, profiler.get_results() if profiler else {}

def test_improved_dqn_8d():
    """개선된 8방향 DQN 테스트"""
    print("🧠 Testing Improved 8-Direction DQN")
    print("=" * 80)
    
    # 테스트 환경 생성
    env = AMRGridworld(grid_size=20, obstacle_density=0.15, dynamic_obstacles=0)
    
    # 시작점과 목표점이 장애물에 막히지 않도록 보장
    env.grid[env.start_pos[0], env.start_pos[1]] = 0
    env.grid[env.goal_pos[0], env.goal_pos[1]] = 0
    
    print(f"Environment: {env.grid_size}x{env.grid_size}, Start={env.start_pos}, Goal={env.goal_pos}")
    
    # 개선된 DQN 훈련
    agent, training_time = train_improved_dqn_8d(env, episodes=2000)
    
    # 경로 계획
    profiler = ComputationalProfiler()
    start_time = time.time()
    path, stats = improved_dqn_8d_path_planning(env, agent, profiler)
    inference_time = time.time() - start_time
    
    print(f"\nImproved DQN 8D Results:")
    print(f"  Path Length: {len(path)}")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Inference Time: {inference_time:.4f}s")
    print(f"  Reached Goal: {path[-1] == env.goal_pos if path else False}")
    
    # 시각화
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 그리드 표시
    for i in range(env.grid_size + 1):
        ax.axhline(y=i, color='gray', alpha=0.3)
        ax.axvline(x=i, color='gray', alpha=0.3)
    
    # 장애물 표시
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if env.grid[i, j] > 0:
                ax.add_patch(plt.Rectangle((j, env.grid_size-1-i), 1, 1, 
                                        facecolor='black', alpha=0.7))
    
    # 시작점과 목표점
    start_x, start_y = env.start_pos[1], env.grid_size-1-env.start_pos[0]
    goal_x, goal_y = env.goal_pos[1], env.grid_size-1-env.goal_pos[0]
    
    ax.plot(start_x + 0.5, start_y + 0.5, 'go', markersize=15, label='Start')
    ax.plot(goal_x + 0.5, goal_y + 0.5, 'ro', markersize=15, label='Goal')
    
    # 경로 표시
    if path and len(path) > 1:
        x = [p[1] + 0.5 for p in path]
        y = [env.grid_size-1-p[0] + 0.5 for p in path]
        ax.plot(x, y, 'b-', linewidth=3, label='Improved DQN 8D', alpha=0.8)
        ax.plot(x, y, 'bo', markersize=6)
    
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_title('Improved 8-Direction DQN Path Planning')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('improved_8_direction_dqn.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as: improved_8_direction_dqn.png")
    plt.close()

if __name__ == "__main__":
    test_improved_dqn_8d()
