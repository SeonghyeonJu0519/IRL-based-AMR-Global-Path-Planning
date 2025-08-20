import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import time
import copy
import os

from irl_training.amr_path_planning_irl import AMRGridworld

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImprovedDQN(nn.Module):
    """참고 저장소 기반 개선된 DQN 네트워크"""
    
    def __init__(self, input_size, hidden_size=128, output_size=4):
        super(ImprovedDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  # 과적합 방지
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class ImprovedDQNAgent:
    """참고 저장소 기반 개선된 DQN 에이전트"""
    
    def __init__(self, state_size, action_size=4, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # 경험 리플레이 버퍼
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # 할인율
        self.learning_rate = lr
        self.batch_size = 32
        
        # 네트워크 초기화
        self.q_network = ImprovedDQN(state_size, 128, action_size).to(device)
        self.target_network = ImprovedDQN(state_size, 128, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.update_target_counter = 0
        self.update_target_freq = 100  # 타겟 네트워크 업데이트 빈도
        
    def get_state_features(self, env, pos):
        """참고 저장소 기반 상태 특성 추출"""
        x, y = pos
        goal_x, goal_y = env.goal_pos
        
        # 기본 특성
        features = [
            x / env.grid_size,  # 현재 x 위치 (정규화)
            y / env.grid_size,  # 현재 y 위치 (정규화)
            goal_x / env.grid_size,  # 목표 x 위치 (정규화)
            goal_y / env.grid_size,  # 목표 y 위치 (정규화)
        ]
        
        # 거리 정보
        manhattan_dist = abs(x - goal_x) + abs(y - goal_y)
        euclidean_dist = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        features.extend([
            manhattan_dist / (2 * env.grid_size),  # 맨해튼 거리
            euclidean_dist / (np.sqrt(2) * env.grid_size),  # 유클리드 거리
        ])
        
        # 방향 정보
        dx = goal_x - x
        dy = goal_y - y
        features.extend([
            dx / env.grid_size,  # x 방향
            dy / env.grid_size,  # y 방향
        ])
        
        # 주변 장애물 정보 (8방향)
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size:
                features.append(float(env.grid[nx, ny] > 0))
            else:
                features.append(1.0)  # 경계는 장애물로 처리
        
        return np.array(features, dtype=np.float32)
    
    def act(self, state, training=True):
        """행동 선택 - ε-greedy 정책"""
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
        
        # 현재 Q 값
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 다음 Q 값 (타겟 네트워크 사용)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 손실 계산 및 최적화
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # 그래디언트 클리핑
        self.optimizer.step()
        
        # ε 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 타겟 네트워크 업데이트
        self.update_target_counter += 1
        if self.update_target_counter % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

def calculate_reward(env, current_pos, next_pos, goal_pos):
    """참고 저장소 기반 보상 함수"""
    if next_pos == goal_pos:
        return 1000  # 목표 도달 - 매우 큰 보상
    
    # 장애물 체크
    if env.grid[next_pos[0], next_pos[1]] > 0:
        return -100  # 장애물 충돌 - 큰 페널티
    
    # 거리 기반 보상
    old_dist = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
    new_dist = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])
    
    if new_dist < old_dist:
        reward = 20  # 목표에 가까워지면 큰 양의 보상
    elif new_dist == old_dist:
        reward = -2  # 같은 거리는 페널티
    else:
        reward = -10  # 목표에서 멀어지면 큰 페널티
    
    # 목표 방향으로의 이동 보상 (더 강화)
    goal_dx = goal_pos[0] - current_pos[0]
    goal_dy = goal_pos[1] - current_pos[1]
    move_dx = next_pos[0] - current_pos[0]
    move_dy = next_pos[1] - current_pos[1]
    
    # x 방향이 올바르면 보상
    if goal_dx > 0 and move_dx > 0:  # 목표가 오른쪽에 있고 오른쪽으로 이동
        reward += 15
    elif goal_dx < 0 and move_dx < 0:  # 목표가 왼쪽에 있고 왼쪽으로 이동
        reward += 15
    elif goal_dx == 0 and move_dx == 0:  # x 위치가 맞으면 보상
        reward += 5
    
    # y 방향이 올바르면 보상
    if goal_dy > 0 and move_dy > 0:  # 목표가 아래쪽에 있고 아래쪽으로 이동
        reward += 15
    elif goal_dy < 0 and move_dy < 0:  # 목표가 위쪽에 있고 위쪽으로 이동
        reward += 15
    elif goal_dy == 0 and move_dy == 0:  # y 위치가 맞으면 보상
        reward += 5
    
    # 잘못된 방향으로 가면 큰 페널티
    if (goal_dx > 0 and move_dx < 0) or (goal_dx < 0 and move_dx > 0):
        reward -= 20
    if (goal_dy > 0 and move_dy < 0) or (goal_dy < 0 and move_dy > 0):
        reward -= 20
    
    return reward

def train_improved_dqn(env, episodes=1000, max_steps=500):
    """개선된 DQN 훈련"""
    state_size = 16  # 4(위치) + 2(거리) + 2(방향) + 8(주변장애물)
    agent = ImprovedDQNAgent(state_size)
    
    print(f"Training Improved DQN for {episodes} episodes...")
    training_start_time = time.time()
    scores = []
    
    for episode in range(episodes):
        current_pos = env.start_pos
        state = agent.get_state_features(env, current_pos)
        total_reward = 0
        steps = 0
        # 재방문 추적(학습 전용)
        visited_positions = set([current_pos])
        
        for step in range(max_steps):
            # 행동 선택
            action = agent.act(state, training=True)
            
            # 행동 실행
            if action == 0:  # 상
                next_pos = (max(0, current_pos[0] - 1), current_pos[1])
            elif action == 1:  # 하
                next_pos = (min(env.grid_size - 1, current_pos[0] + 1), current_pos[1])
            elif action == 2:  # 좌
                next_pos = (current_pos[0], max(0, current_pos[1] - 1))
            else:  # 우
                next_pos = (current_pos[0], min(env.grid_size - 1, current_pos[1] + 1))
            
            # 보상 계산
            reward = calculate_reward(env, current_pos, next_pos, env.goal_pos)

            # 재방문 페널티(우회 허용하되 루프 억제)
            if next_pos in visited_positions:
                reward -= 10.0  # 필요 시 -5~-15 범위에서 조정
            else:
                visited_positions.add(next_pos)
            
            # 다음 상태
            next_state = agent.get_state_features(env, next_pos)
            done = (next_pos == env.goal_pos) or (step == max_steps - 1)
            
            # 경험 저장
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            current_pos = next_pos
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # 경험 재플레이
        if len(agent.memory) > agent.batch_size:
            agent.replay()
        
        scores.append(total_reward)
        
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            print(f"Episode {episode}: Avg Score = {avg_score:.2f}, Epsilon = {agent.epsilon:.3f}")
    
    training_time = time.time() - training_start_time
    print(f"Improved DQN training completed in {training_time:.2f}s")
    
    return agent, training_time

def improved_dqn_path_planning(env, agent, profiler=None):
    """개선된 DQN 경로 계획 - 연산량 측정 포함"""
    if profiler:
        profiler.start_profiling()
    
    current_pos = env.start_pos
    path = [current_pos]
    max_steps = env.grid_size * 3
    visited = set([current_pos])
    
    print(f"Starting path planning from {current_pos} to {env.goal_pos}")
    
    for step in range(max_steps):
        if current_pos == env.goal_pos:
            print(f"Goal reached at step {step}")
            break
        
        state = agent.get_state_features(env, current_pos)
        action = agent.act(state, training=False)
        
        if profiler:
            profiler.increment_neural_inferences(1)  # 신경망 추론 횟수 증가
        
        # 후보 이동(우선: 미방문 유효칸 → 없으면 재방문 유효칸)
        def moved(pos, a):
            if a == 0:
                return (max(0, pos[0] - 1), pos[1])
            if a == 1:
                return (min(env.grid_size - 1, pos[0] + 1), pos[1])
            if a == 2:
                return (pos[0], max(0, pos[1] - 1))
            return (pos[0], min(env.grid_size - 1, pos[1] + 1))

        candidates = [action] + [a for a in range(4) if a != action]
        next_pos = None
        # 1) 미방문 유효칸
        for a in candidates:
            cand = moved(current_pos, a)
            if env.grid[cand[0], cand[1]] == 0 and cand not in visited:
                next_pos = cand
                break
        # 2) 재방문 유효칸 허용
        if next_pos is None:
            for a in candidates:
                cand = moved(current_pos, a)
                if env.grid[cand[0], cand[1]] == 0:
                    next_pos = cand
                    break
        if next_pos is None:
            print(f"Stuck at step {step}, no valid moves")
            break

        current_pos = next_pos
        path.append(current_pos)
        visited.add(current_pos)
        
        # 진행 상황 출력
        if step % 10 == 0:
            print(f"Step {step}: Current pos = {current_pos}, Path length = {len(path)}")
    
    print(f"Final path length: {len(path)}")
    print(f"Path: {path}")
    
    if profiler:
        profiler.end_profiling()
    
    # 경로가 너무 짧으면 None 반환
    if len(path) <= 1:
        print("Warning: Path too short, returning None")
        return None
    
    return path

def test_improved_dqn():
    """개선된 DQN 테스트"""
    print("Testing Improved DQN based on GitHub repository...")
    
    # 환경 생성
    env = AMRGridworld(grid_size=10, obstacle_density=0.15, dynamic_obstacles=0)
    print(f"Environment: Start={env.start_pos}, Goal={env.goal_pos}")
    
    # 개선된 DQN 훈련
    agent, training_time = train_improved_dqn(env, episodes=1000)
    
    # 경로 계획
    start_time = time.time()
    path = improved_dqn_path_planning(env, agent)
    inference_time = time.time() - start_time
    
    print(f"Improved DQN Results:")
    print(f"  Path Length: {len(path)}")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Inference Time: {inference_time:.4f}s")
    print(f"  Path: {path}")
    
    # 시각화
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 그리드 표시
    for i in range(env.grid_size + 1):
        ax.axhline(y=i, color='gray', alpha=0.3)
        ax.axvline(x=i, color='gray', alpha=0.3)
    
    # 장애물 표시
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if env.grid[i, j] > 0:
                ax.add_patch(plt.Rectangle((j, env.grid_size-1-i), 1, 1, facecolor='black', alpha=0.7))
    
    # 시작점과 목표점
    start_x, start_y = env.start_pos[1], env.grid_size-1-env.start_pos[0]
    goal_x, goal_y = env.goal_pos[1], env.grid_size-1-env.goal_pos[0]
    
    ax.plot(start_x + 0.5, start_y + 0.5, 'go', markersize=15, label='Start')
    ax.plot(goal_x + 0.5, goal_y + 0.5, 'ro', markersize=15, label='Goal')
    
    # 경로 표시
    if path and len(path) > 1:
        x = [p[1] + 0.5 for p in path]
        y = [env.grid_size-1-p[0] + 0.5 for p in path]
        ax.plot(x, y, 'b-', linewidth=3, label='Improved DQN', alpha=0.8)
        ax.plot(x, y, 'bo', markersize=6)
    
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_title('Improved DQN Path Planning')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('improved_dqn_path.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as: improved_dqn_path.png")
    plt.close()

if __name__ == "__main__":
    test_improved_dqn()
