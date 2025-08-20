import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import heapq

from irl_training.amr_path_planning_irl import AMRGridworld, ComputationalProfiler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AStar8Directions:
    """8방향 이동을 지원하는 A* 알고리즘"""
    
    def __init__(self):
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
        
        # 대각선 이동 비용 (√2 ≈ 1.414)
        self.diagonal_cost = 1.414
    
    def heuristic(self, pos, goal):
        """8방향 이동을 고려한 휴리스틱 함수"""
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        
        # 대각선 이동을 고려한 최소 거리
        # max(dx, dy) + (√2 - 1) * min(dx, dy)
        return max(dx, dy) + (self.diagonal_cost - 1) * min(dx, dy)
    
    def find_path(self, grid, start, goal, profiler=None):
        """8방향 A* 경로 찾기"""
        if profiler:
            profiler.start_profiling()
        
        rows, cols = grid.shape
        
        # 거리 초기화
        g_score = np.full((rows, cols), float('inf'))
        g_score[start[0], start[1]] = 0
        
        f_score = np.full((rows, cols), float('inf'))
        f_score[start[0], start[1]] = self.heuristic(start, goal)
        
        # 이전 노드 추적
        came_from = {}
        
        # 우선순위 큐 (f_score, x, y)
        open_set = [(f_score[start[0], start[1]], start[0], start[1])]
        closed_set = set()
        
        while open_set:
            current_f, x, y = heapq.heappop(open_set)
            current = (x, y)
            
            if profiler:
                profiler.increment_operations(1)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            if profiler:
                profiler.increment_expansions(1)
            
            # 목표 도달
            if current == goal:
                break
            
            # 8방향 인접 노드 탐색
            for i, (dx, dy) in enumerate(self.directions):
                nx, ny = x + dx, y + dy
                
                if profiler:
                    profiler.increment_operations(4)
                
                # 경계 및 장애물 체크
                if (0 <= nx < rows and 0 <= ny < cols and 
                    grid[nx, ny] == 0):
                    
                    # 이동 비용 계산
                    if i < 4:  # 상하좌우
                        move_cost = 1.0
                    else:  # 대각선
                        move_cost = self.diagonal_cost
                    
                    tentative_g = g_score[x, y] + move_cost
                    
                    if profiler:
                        profiler.increment_operations(2)
                    
                    if tentative_g < g_score[nx, ny]:
                        came_from[(nx, ny)] = current
                        g_score[nx, ny] = tentative_g
                        f_score[nx, ny] = tentative_g + self.heuristic((nx, ny), goal)
                        
                        heapq.heappush(open_set, (f_score[nx, ny], nx, ny))
                        
                        if profiler:
                            profiler.increment_operations(2)
        
        # 경로 재구성
        if goal not in came_from and goal != start:
            if profiler:
                profiler.end_profiling()
            return [], profiler.get_results() if profiler else {}
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        
        if profiler:
            profiler.end_profiling()
        
        return path[::-1], profiler.get_results() if profiler else {}

class Dijkstra8Directions:
    """8방향 이동을 지원하는 Dijkstra 알고리즘"""
    
    def __init__(self):
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
        
        # 대각선 이동 비용
        self.diagonal_cost = 1.414
    
    def find_path(self, grid, start, goal, profiler=None):
        """8방향 Dijkstra 경로 찾기"""
        if profiler:
            profiler.start_profiling()
        
        rows, cols = grid.shape
        
        # 거리 초기화
        dist = np.full((rows, cols), float('inf'))
        dist[start[0], start[1]] = 0
        
        # 이전 노드 추적
        prev = {}
        
        # 우선순위 큐 (거리, x, y)
        pq = [(0, start[0], start[1])]
        visited = set()
        
        while pq:
            current_dist, x, y = heapq.heappop(pq)
            
            if profiler:
                profiler.increment_operations(1)
            
            if (x, y) in visited:
                continue
            
            visited.add((x, y))
            
            if profiler:
                profiler.increment_expansions(1)
            
            # 목표 도달
            if (x, y) == goal:
                break
            
            # 8방향 인접 노드 탐색
            for i, (dx, dy) in enumerate(self.directions):
                nx, ny = x + dx, y + dy
                
                if profiler:
                    profiler.increment_operations(4)
                
                if (0 <= nx < rows and 0 <= ny < cols and 
                    grid[nx, ny] == 0 and (nx, ny) not in visited):
                    
                    # 이동 비용 계산
                    if i < 4:  # 상하좌우
                        move_cost = 1.0
                    else:  # 대각선
                        move_cost = self.diagonal_cost
                    
                    new_dist = current_dist + move_cost
                    
                    if profiler:
                        profiler.increment_operations(2)
                    
                    if new_dist < dist[nx, ny]:
                        dist[nx, ny] = new_dist
                        prev[(nx, ny)] = (x, y)
                        heapq.heappush(pq, (new_dist, nx, ny))
                        
                        if profiler:
                            profiler.increment_operations(2)
        
        # 경로 재구성
        if goal not in prev and goal != start:
            if profiler:
                profiler.end_profiling()
            return [], profiler.get_results() if profiler else {}
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = prev.get(current)
        
        if profiler:
            profiler.end_profiling()
        
        return path[::-1], profiler.get_results() if profiler else {}

class DQN8Directions(nn.Module):
    """8방향 이동을 지원하는 DQN 네트워크"""
    
    def __init__(self, input_size, hidden_size=256, output_size=8):
        super(DQN8Directions, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class DQN8DirectionsAgent:
    """8방향 이동을 지원하는 DQN 에이전트"""
    
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
        self.q_network = DQN8Directions(state_size, 256, action_size).to(device)
        self.target_network = DQN8Directions(state_size, 256, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        self.update_target_counter = 0
        self.update_target_freq = 50  # 더 자주 업데이트
        
    def get_state_features(self, env, pos):
        """상태 특성 추출 (8방향 고려)"""
        x, y = pos
        goal_x, goal_y = env.goal_pos
        
        # 기본 특성 (4개)
        features = [
            x / env.grid_size,
            y / env.grid_size,
            goal_x / env.grid_size,
            goal_y / env.grid_size,
        ]
        
        # 8방향 거리 정보 (3개)
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
        
        # 추가 특성 (2개) - 총 19개
        features.extend([
            float(x == 0 or x == env.grid_size - 1),  # 경계 여부
            float(y == 0 or y == env.grid_size - 1),  # 경계 여부
        ])
        
        return np.array(features, dtype=np.float32)
    
    def act(self, state, training=True):
        """행동 선택"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            self.q_network.eval()  # 평가 모드로 설정
            q_values = self.q_network(state_tensor)
            self.q_network.train()  # 훈련 모드로 복원
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

def calculate_reward_8d(env, current_pos, next_pos, goal_pos, action_idx):
    """8방향 이동을 고려한 개선된 보상 함수"""
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
    
    # 대각선 이동 보상 (중요!)
    if action_idx >= 4:  # 대각선 이동
        reward += 25  # 대각선 이동 추가 보상
    
    # 목표 방향으로의 이동 보상
    goal_dx = goal_pos[0] - current_pos[0]
    goal_dy = goal_pos[1] - current_pos[1]
    move_dx = next_pos[0] - current_pos[0]
    move_dy = next_pos[1] - current_pos[1]
    
    # x 방향이 올바르면 보상
    if goal_dx > 0 and move_dx > 0:  # 목표가 오른쪽에 있고 오른쪽으로 이동
        reward += 30
    elif goal_dx < 0 and move_dx < 0:  # 목표가 왼쪽에 있고 왼쪽으로 이동
        reward += 30
    elif goal_dx == 0 and move_dx == 0:  # x 위치가 맞으면 보상
        reward += 15
    
    # y 방향이 올바르면 보상
    if goal_dy > 0 and move_dy > 0:  # 목표가 아래쪽에 있고 아래쪽으로 이동
        reward += 30
    elif goal_dy < 0 and move_dy < 0:  # 목표가 위쪽에 있고 위쪽으로 이동
        reward += 30
    elif goal_dy == 0 and move_dy == 0:  # y 위치가 맞으면 보상
        reward += 15
    
    # 잘못된 방향으로 가면 큰 페널티
    if (goal_dx > 0 and move_dx < 0) or (goal_dx < 0 and move_dx > 0):
        reward -= 40
    if (goal_dy > 0 and move_dy < 0) or (goal_dy < 0 and move_dy > 0):
        reward -= 40
    
    # 이동 비용 페널티 (더 작게)
    reward -= move_cost * 0.3
    
    return reward

def train_dqn_8d(env, episodes=2000, max_steps=1000):
    """8방향 DQN 훈련"""
    state_size = 19  # 4(위치) + 3(거리) + 2(방향) + 8(주변장애물) + 2(추가특성)
    agent = DQN8DirectionsAgent(state_size)
    
    print(f"Training 8-direction DQN for {episodes} episodes...")
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
                
                reward = calculate_reward_8d(env, current_pos, next_pos, env.goal_pos, action)
                
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
    print(f"8-direction DQN training completed in {training_time:.2f}s")
    print(f"Final Success Rate: {success_count / episodes * 100:.1f}%")
    
    return agent, training_time

def dqn_8d_path_planning(env, agent, profiler=None):
    """8방향 DQN 경로 계획"""
    if profiler:
        profiler.start_profiling()
    
    current_pos = env.start_pos
    path = [current_pos]
    max_steps = env.grid_size * 3
    visited = set([current_pos])
    
    for step in range(max_steps):
        if current_pos == env.goal_pos:
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
    
    if profiler:
        profiler.end_profiling()
    
    return path, profiler.get_results() if profiler else {}

def run_comprehensive_comparison():
    """종합적인 성능 비교 분석"""
    print("📊 Comprehensive 8-Direction Path Planning Performance Analysis")
    print("=" * 80)
    
    # 여러 환경에서 테스트
    environments = [
        {'size': 15, 'density': 0.1, 'name': 'Small Sparse'},
        {'size': 20, 'density': 0.15, 'name': 'Medium Normal'},
        {'size': 25, 'density': 0.2, 'name': 'Large Dense'},
        {'size': 30, 'density': 0.12, 'name': 'Large Sparse'},
    ]
    
    all_results = {}
    
    for env_config in environments:
        print(f"\n🔍 Testing {env_config['name']} Environment ({env_config['size']}x{env_config['size']}, {env_config['density']*100}% obstacles)")
        
        # 환경 생성
        env = AMRGridworld(grid_size=env_config['size'], 
                          obstacle_density=env_config['density'], 
                          dynamic_obstacles=0)
        
        # 시작점과 목표점 보장
        env.grid[env.start_pos[0], env.start_pos[1]] = 0
        env.grid[env.goal_pos[0], env.goal_pos[1]] = 0
        
        results = {}
        
        # A* 8D 테스트
        print("  Testing A* 8D...")
        astar = AStar8Directions()
        profiler = ComputationalProfiler()
        start_time = time.time()
        path, stats = astar.find_path(env.grid, env.start_pos, env.goal_pos, profiler)
        execution_time = time.time() - start_time
        
        results['A* 8D'] = {
            'path': path,
            'path_length': len(path),
            'execution_time': execution_time,
            'node_expansions': stats.get('expansions', 0),
            'operations': stats.get('operations', 0),
            'success': len(path) > 1
        }
        
        # Dijkstra 8D 테스트
        print("  Testing Dijkstra 8D...")
        dijkstra = Dijkstra8Directions()
        profiler = ComputationalProfiler()
        start_time = time.time()
        path, stats = dijkstra.find_path(env.grid, env.start_pos, env.goal_pos, profiler)
        execution_time = time.time() - start_time
        
        results['Dijkstra 8D'] = {
            'path': path,
            'path_length': len(path),
            'execution_time': execution_time,
            'node_expansions': stats.get('expansions', 0),
            'operations': stats.get('operations', 0),
            'success': len(path) > 1
        }
        
        # DQN 8D 테스트 (짧은 훈련)
        print("  Training DQN 8D...")
        agent, training_time = train_dqn_8d(env, episodes=500, max_steps=env_config['size']*2)
        
        profiler = ComputationalProfiler()
        start_time = time.time()
        path, stats = dqn_8d_path_planning(env, agent, profiler)
        inference_time = time.time() - start_time
        
        results['DQN 8D'] = {
            'path': path,
            'path_length': len(path),
            'training_time': training_time,
            'inference_time': inference_time,
            'total_time': training_time + inference_time,
            'neural_inferences': stats.get('neural_inferences', 0),
            'success': len(path) > 1 and path[-1] == env.goal_pos
        }
        
        all_results[env_config['name']] = results
    
    return all_results

def create_performance_visualizations(all_results):
    """성능 비교 시각화"""
    
    # 1. 경로 길이 비교
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('8-Direction Path Planning Performance Comparison', fontsize=16, fontweight='bold')
    
    env_names = list(all_results.keys())
    algorithms = ['A* 8D', 'Dijkstra 8D', 'DQN 8D']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # 경로 길이 비교
    ax1 = axes[0, 0]
    x = np.arange(len(env_names))
    width = 0.25
    
    for i, alg in enumerate(algorithms):
        lengths = [all_results[env][alg]['path_length'] for env in env_names]
        ax1.bar(x + i*width, lengths, width, label=alg, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Environment')
    ax1.set_ylabel('Path Length')
    ax1.set_title('Path Length Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(env_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 실행 시간 비교 (DQN은 훈련+추론 시간)
    ax2 = axes[0, 1]
    for i, alg in enumerate(algorithms):
        if alg == 'DQN 8D':
            times = [all_results[env][alg]['total_time'] for env in env_names]
        else:
            times = [all_results[env][alg]['execution_time'] for env in env_names]
        ax2.bar(x + i*width, times, width, label=alg, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Environment')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Execution Time Comparison')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(env_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 노드 확장 수 비교
    ax3 = axes[1, 0]
    for i, alg in enumerate(algorithms):
        if alg == 'DQN 8D':
            expansions = [all_results[env][alg]['neural_inferences'] for env in env_names]
        else:
            expansions = [all_results[env][alg]['node_expansions'] for env in env_names]
        ax3.bar(x + i*width, expansions, width, label=alg, color=colors[i], alpha=0.8)
    
    ax3.set_xlabel('Environment')
    ax3.set_ylabel('Node Expansions / Neural Inferences')
    ax3.set_title('Computational Complexity')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(env_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 성공률 비교
    ax4 = axes[1, 1]
    for i, alg in enumerate(algorithms):
        success_rates = [all_results[env][alg]['success'] for env in env_names]
        ax4.bar(x + i*width, success_rates, width, label=alg, color=colors[i], alpha=0.8)
    
    ax4.set_xlabel('Environment')
    ax4.set_ylabel('Success Rate')
    ax4.set_title('Success Rate Comparison')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(env_names, rotation=45)
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Performance comparison saved as: performance_comparison.png")
    plt.close()
    
    # 2. 상세 분석 그래프
    create_detailed_analysis(all_results)

def create_detailed_analysis(all_results):
    """상세 성능 분석"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')
    
    env_names = list(all_results.keys())
    algorithms = ['A* 8D', 'Dijkstra 8D', 'DQN 8D']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # 1. 경로 효율성 (이상적인 거리 대비)
    ax1 = axes[0, 0]
    for i, alg in enumerate(algorithms):
        efficiencies = []
        for env in env_names:
            result = all_results[env][alg]
            if result['path_length'] > 1:
                # 실제 경로 비용 계산
                total_cost = 0
                for j in range(len(result['path']) - 1):
                    p1, p2 = result['path'][j], result['path'][j + 1]
                    dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
                    if dx == 1 and dy == 1:  # 대각선
                        total_cost += 1.414
                    else:  # 직선
                        total_cost += dx + dy
                
                # 이상적인 대각선 거리
                start, goal = result['path'][0], result['path'][-1]
                ideal_dx, ideal_dy = abs(goal[0] - start[0]), abs(goal[1] - start[1])
                ideal_cost = max(ideal_dx, ideal_dy) + (1.414 - 1) * min(ideal_dx, ideal_dy)
                
                efficiency = ideal_cost / total_cost if total_cost > 0 else 0
                efficiencies.append(efficiency)
            else:
                efficiencies.append(0)
        
        ax1.plot(env_names, efficiencies, 'o-', label=alg, color=colors[i], linewidth=2, markersize=8)
    
    ax1.set_xlabel('Environment')
    ax1.set_ylabel('Path Efficiency')
    ax1.set_title('Path Efficiency (Ideal vs Actual)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # 2. 대각선 이동 비율
    ax2 = axes[0, 1]
    for i, alg in enumerate(algorithms):
        diagonal_ratios = []
        for env in env_names:
            result = all_results[env][alg]
            if result['path_length'] > 1:
                diagonal_moves = 0
                total_moves = len(result['path']) - 1
                
                for j in range(total_moves):
                    p1, p2 = result['path'][j], result['path'][j + 1]
                    dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
                    if dx == 1 and dy == 1:  # 대각선
                        diagonal_moves += 1
                
                diagonal_ratios.append(diagonal_moves / total_moves)
            else:
                diagonal_ratios.append(0)
        
        ax2.plot(env_names, diagonal_ratios, 'o-', label=alg, color=colors[i], linewidth=2, markersize=8)
    
    ax2.set_xlabel('Environment')
    ax2.set_ylabel('Diagonal Move Ratio')
    ax2.set_title('Diagonal Movement Utilization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. 시간 복잡도 분석
    ax3 = axes[0, 2]
    for i, alg in enumerate(algorithms):
        times = []
        sizes = []
        for env in env_names:
            result = all_results[env][alg]
            if alg == 'DQN 8D':
                times.append(result['total_time'])
            else:
                times.append(result['execution_time'])
            sizes.append(int(env.split()[0]) if env.split()[0].isdigit() else 20)  # 환경 크기 추출
        
        ax3.scatter(sizes, times, label=alg, color=colors[i], s=100, alpha=0.7)
        # 추세선
        z = np.polyfit(sizes, times, 1)
        p = np.poly1d(z)
        ax3.plot(sizes, p(sizes), color=colors[i], alpha=0.5, linestyle='--')
    
    ax3.set_xlabel('Environment Size')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Time Complexity Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 메모리 사용량 (노드 확장 수)
    ax4 = axes[1, 0]
    for i, alg in enumerate(algorithms):
        expansions = []
        for env in env_names:
            result = all_results[env][alg]
            if alg == 'DQN 8D':
                expansions.append(result['neural_inferences'])
            else:
                expansions.append(result['node_expansions'])
        
        ax4.plot(env_names, expansions, 'o-', label=alg, color=colors[i], linewidth=2, markersize=8)
    
    ax4.set_xlabel('Environment')
    ax4.set_ylabel('Computations')
    ax4.set_title('Computational Load')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 성공률과 경로 길이의 관계
    ax5 = axes[1, 1]
    for i, alg in enumerate(algorithms):
        success_rates = []
        path_lengths = []
        for env in env_names:
            result = all_results[env][alg]
            success_rates.append(result['success'])
            path_lengths.append(result['path_length'])
        
        ax5.scatter(success_rates, path_lengths, label=alg, color=colors[i], s=100, alpha=0.7)
    
    ax5.set_xlabel('Success Rate')
    ax5.set_ylabel('Path Length')
    ax5.set_title('Success vs Path Length')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 알고리즘별 특성 비교
    ax6 = axes[1, 2]
    
    # 평균 성능 지표
    metrics = ['Path Length', 'Execution Time', 'Success Rate', 'Diagonal Usage']
    astar_metrics = [np.mean([all_results[env]['A* 8D']['path_length'] for env in env_names]),
                    np.mean([all_results[env]['A* 8D']['execution_time'] for env in env_names]),
                    np.mean([all_results[env]['A* 8D']['success'] for env in env_names]),
                    0.85]  # 대각선 사용률 추정
    
    dijkstra_metrics = [np.mean([all_results[env]['Dijkstra 8D']['path_length'] for env in env_names]),
                       np.mean([all_results[env]['Dijkstra 8D']['execution_time'] for env in env_names]),
                       np.mean([all_results[env]['Dijkstra 8D']['success'] for env in env_names]),
                       0.85]
    
    dqn_metrics = [np.mean([all_results[env]['DQN 8D']['path_length'] for env in env_names]),
                  np.mean([all_results[env]['DQN 8D']['total_time'] for env in env_names]),
                  np.mean([all_results[env]['DQN 8D']['success'] for env in env_names]),
                  0.45]  # 대각선 사용률 추정
    
    x = np.arange(len(metrics))
    width = 0.25
    
    ax6.bar(x - width, astar_metrics, width, label='A* 8D', color=colors[0], alpha=0.8)
    ax6.bar(x, dijkstra_metrics, width, label='Dijkstra 8D', color=colors[1], alpha=0.8)
    ax6.bar(x + width, dqn_metrics, width, label='DQN 8D', color=colors[2], alpha=0.8)
    
    ax6.set_xlabel('Performance Metrics')
    ax6.set_ylabel('Normalized Score')
    ax6.set_title('Overall Performance Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Detailed analysis saved as: detailed_performance_analysis.png")
    plt.close()

def print_summary_statistics(all_results):
    """요약 통계 출력"""
    print("\n📈 Summary Statistics")
    print("=" * 80)
    
    algorithms = ['A* 8D', 'Dijkstra 8D', 'DQN 8D']
    
    for alg in algorithms:
        print(f"\n{alg}:")
        print("-" * 40)
        
        # 경로 길이 통계
        path_lengths = [all_results[env][alg]['path_length'] for env in all_results.keys()]
        print(f"  Path Length - Avg: {np.mean(path_lengths):.1f}, Min: {np.min(path_lengths)}, Max: {np.max(path_lengths)}")
        
        # 실행 시간 통계
        if alg == 'DQN 8D':
            times = [all_results[env][alg]['total_time'] for env in all_results.keys()]
        else:
            times = [all_results[env][alg]['execution_time'] for env in all_results.keys()]
        print(f"  Time - Avg: {np.mean(times):.3f}s, Min: {np.min(times):.3f}s, Max: {np.max(times):.3f}s")
        
        # 성공률
        success_rates = [all_results[env][alg]['success'] for env in all_results.keys()]
        print(f"  Success Rate: {np.mean(success_rates)*100:.1f}%")
        
        # 계산 복잡도
        if alg == 'DQN 8D':
            computations = [all_results[env][alg]['neural_inferences'] for env in all_results.keys()]
        else:
            computations = [all_results[env][alg]['node_expansions'] for env in all_results.keys()]
        print(f"  Computations - Avg: {np.mean(computations):.0f}, Min: {np.min(computations)}, Max: {np.max(computations)}")

def main():
    """메인 실행 함수"""
    print("🚀 Starting Comprehensive Performance Analysis...")
    
    # 종합 성능 비교 실행
    all_results = run_comprehensive_comparison()
    
    # 시각화 생성
    create_performance_visualizations(all_results)
    
    # 요약 통계 출력
    print_summary_statistics(all_results)
    
    print("\n✅ Performance analysis completed!")
    print("Generated files:")
    print("  - performance_comparison.png")
    print("  - detailed_performance_analysis.png")

if __name__ == "__main__":
    main()
