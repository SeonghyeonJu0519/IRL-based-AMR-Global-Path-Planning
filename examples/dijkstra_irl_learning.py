import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import gc
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim

from amr_path_planning_irl import AMRGridworld, extract_features, maxent_irl_gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ComputationalProfiler:
    """연산 복잡도 및 성능 분석기"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.operations_count = 0
        self.state_expansions = 0
        self.neural_inferences = 0
    
    def start_profiling(self):
        gc.collect()
        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.operations_count = 0
        self.state_expansions = 0
        self.neural_inferences = 0
    
    def end_profiling(self):
        self.end_time = time.perf_counter()
        self.end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def get_results(self):
        execution_time = self.end_time - self.start_time if self.start_time and self.end_time else 0
        memory_usage = self.end_memory - self.start_memory if self.start_memory and self.end_memory else 0
        ops_per_second = self.operations_count / execution_time if execution_time > 0 else 0
        return {
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'operations_count': self.operations_count,
            'state_expansions': self.state_expansions,
            'neural_inferences': self.neural_inferences,
            'ops_per_second': ops_per_second
        }
    
    def increment_operations(self, count=1):
        self.operations_count += count
    
    def increment_expansions(self, count=1):
        self.state_expansions += count
    
    def increment_neural_inferences(self, count=1):
        self.neural_inferences += count

class RewardPredictor(nn.Module):
    def __init__(self, input_dim=10):  # 10차원 feature에 맞춤
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

def train_irl_model(algorithm="dijkstra", grid_size=20, obstacle_density=0.15, dynamic_obstacles=4):
    """
    다익스트라/Bellman-Ford 기반 IRL 모델 학습
    
    Args:
        algorithm: "dijkstra" 또는 "bellman_ford"
        grid_size: 학습 환경 크기
        obstacle_density: 장애물 밀도
        dynamic_obstacles: 동적 장애물 수
    
    Returns:
        reward_predictor: 훈련된 보상 예측기
        training_stats: 훈련 통계
    """
    
    print(f"Training IRL model using {algorithm.upper()} algorithm")
    print(f"Environment: {grid_size}x{grid_size}, Obstacle density: {obstacle_density}")
    print("=" * 80)
    
    # 학습 환경 생성
    train_env = AMRGridworld(grid_size=grid_size, obstacle_density=obstacle_density, dynamic_obstacles=dynamic_obstacles)
    
    # 최적 경로 생성
    profiler = ComputationalProfiler()
    profiler.start_profiling()
    
    expert_trajectories = train_env.generate_trajectories_from_optimal_path(
        n_trajectories=100, algorithm=algorithm
    )
    
    profiler.end_profiling()
    path_generation_stats = profiler.get_results()
    
    print(f"   {algorithm.upper()} path generation completed in {path_generation_stats['execution_time']:.3f}s")
    print(f"   Memory usage: {path_generation_stats['memory_usage']:.2f} MB")
    
    # IRL 학습
    profiler.reset()
    profiler.start_profiling()
    
    learned_reward = maxent_irl_gpu(
        feature_matrix=train_env.feature_matrix(),
        n_actions=train_env.n_actions,
        discount=0.9,
        transition_probability=train_env.transition_probability,
        trajectories=expert_trajectories,
        epochs=150,
        learning_rate=0.01,
        grid_size=train_env.grid_size,
        goal_pos=train_env.goal_pos,
        grid=train_env.grid
    )
    
    profiler.end_profiling()
    irl_stats = profiler.get_results()
    
    print(f"   IRL learning completed in {irl_stats['execution_time']:.3f}s")
    print(f"   Memory usage: {irl_stats['memory_usage']:.2f} MB")
    
    # 신경망 보상 예측기 훈련
    features_tensor = torch.tensor(train_env.feature_matrix(), dtype=torch.float32, device=device)
    rewards_tensor = torch.tensor(learned_reward, dtype=torch.float32, device=device).unsqueeze(1)
    feature_dim = features_tensor.shape[1]
    reward_predictor = RewardPredictor(input_dim=feature_dim).to(device)
    optimizer = optim.Adam(reward_predictor.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Feature vector shape:", features_tensor.shape)
    print("RewardPredictor input_dim:", feature_dim)
    
    print("   Training neural network reward predictor...")
    for epoch in range(100):
        optimizer.zero_grad()
        predicted = reward_predictor(features_tensor)
        loss = criterion(predicted, rewards_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"      Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print("   Neural network training completed")
    
    # 훈련 통계 수집
    training_stats = {
        'algorithm': algorithm,
        'grid_size': grid_size,
        'obstacle_density': obstacle_density,
        'dynamic_obstacles': dynamic_obstacles,
        'path_generation_time': path_generation_stats['execution_time'],
        'path_generation_memory': path_generation_stats['memory_usage'],
        'irl_training_time': irl_stats['execution_time'],
        'irl_training_memory': irl_stats['memory_usage'],
        'total_training_time': path_generation_stats['execution_time'] + irl_stats['execution_time'],
        'expert_trajectories_count': len(expert_trajectories),
        'feature_dim': feature_dim,
        'reward_dim': rewards_tensor.shape[0]
    }
    
    return reward_predictor, training_stats

def save_model(reward_predictor, training_stats, filename_prefix="irl_model"):
    """모델과 통계 저장"""
    
    # 모델 저장
    model_filename = f"{filename_prefix}.pth"
    torch.save(reward_predictor.state_dict(), model_filename)
    print(f"   Model saved as: {model_filename}")
    
    # 통계 저장
    stats_filename = f"{filename_prefix}_stats.json"
    with open(stats_filename, 'w') as f:
        json.dump(training_stats, f, indent=2)
    print(f"   Training stats saved as: {stats_filename}")
    
    return model_filename, stats_filename

def load_model(model_filename, stats_filename):
    """모델과 통계 로드"""
    
    # 통계 로드
    with open(stats_filename, 'r') as f:
        training_stats = json.load(f)
    
    # 모델 로드
    feature_dim = training_stats.get('feature_dim', 10)
    print("Loading model with feature_dim:", feature_dim)
    reward_predictor = RewardPredictor(input_dim=feature_dim).to(device)
    reward_predictor.load_state_dict(torch.load(model_filename, map_location=device))
    reward_predictor.eval()
    
    return reward_predictor, training_stats

def irl_guided_search(env, reward_predictor):
    """IRL 기반 경로 계획"""
    start_state = env.start_pos[0] * env.grid_size + env.start_pos[1]
    goal_state = env.goal_pos[0] * env.grid_size + env.goal_pos[1]
    
    open_set = [(0, start_state)]
    came_from = {}
    g_score = {start_state: 0}
    
    while open_set:
        current = open_set.pop(0)[1]
        
        if current == goal_state:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_state)
            return path[::-1]
        
        x, y = current // env.grid_size, current % env.grid_size
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor_x, neighbor_y = x + dx, y + dy
            
            if (0 <= neighbor_x < env.grid_size and 
                0 <= neighbor_y < env.grid_size and 
                env.grid[neighbor_x, neighbor_y] == 0):
                
                neighbor_state = neighbor_x * env.grid_size + neighbor_y
                tentative_g_score = g_score[current] + 1
                
                if neighbor_state not in g_score or tentative_g_score < g_score[neighbor_state]:
                    came_from[neighbor_state] = current
                    g_score[neighbor_state] = tentative_g_score
                    
                    # IRL 보상 예측
                    features = extract_features(neighbor_state, env)
                    device = next(reward_predictor.parameters()).device
                    with torch.no_grad():
                        predicted_reward = reward_predictor(features.unsqueeze(0).to(device)).item()
                    
                    # 휴리스틱에 보상 반영
                    heuristic = abs(neighbor_x - env.goal_pos[0]) + abs(neighbor_y - env.goal_pos[1])
                    f_score = tentative_g_score + heuristic - predicted_reward * 0.1
                    
                    if neighbor_state not in [item[1] for item in open_set]:
                        open_set.append((f_score, neighbor_state))
        
        open_set.sort()
    
    return [start_state]

def calculate_manhattan_distance(path):
    """경로의 실제 Manhattan 거리 계산"""
    if not path or len(path) < 2:
        return 0
    
    total_distance = 0
    for i in range(len(path) - 1):
        # 경로가 좌표 튜플인지 상태 인덱스인지 확인
        if isinstance(path[i], tuple):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
        else:
            # 상태 인덱스를 좌표로 변환
            grid_size = int(np.sqrt(max(path) + 1))  # 근사치
            x1, y1 = path[i] // grid_size, path[i] % grid_size
            x2, y2 = path[i + 1] // grid_size, path[i + 1] % grid_size
        
        # Manhattan 거리 계산
        distance = abs(x2 - x1) + abs(y2 - y1)
        total_distance += distance
    
    return total_distance

def calculate_path_efficiency(path, start_pos, goal_pos):
    """경로 효율성 계산 (Manhattan 거리 기반)"""
    if not path:
        return 0.0
    
    # 시작점과 목표점 사이의 직선 Manhattan 거리
    ideal_distance = abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])
    
    # 실제 경로의 Manhattan 거리
    actual_distance = calculate_manhattan_distance(path)
    
    if actual_distance == 0:
        return 0.0
    
    # 효율성 = 이상적인 거리 / 실제 거리 (1.0에 가까울수록 효율적)
    efficiency = ideal_distance / actual_distance
    
    return efficiency

def analyze_path_quality(path, start_pos, goal_pos, grid_size):
    """경로 품질 상세 분석"""
    if not path:
        return {
            'node_count': 0,
            'actual_distance': 0,
            'ideal_distance': 0,
            'efficiency': 0.0,
            'detour_ratio': 0.0,
            'path_quality': 'No path found'
        }
    
    # 기본 정보
    node_count = len(path)
    ideal_distance = abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])
    actual_distance = calculate_manhattan_distance(path)
    
    # 효율성 계산
    efficiency = ideal_distance / actual_distance if actual_distance > 0 else 0.0
    
    # 우회 비율 (실제 거리 / 이상적 거리)
    detour_ratio = actual_distance / ideal_distance if ideal_distance > 0 else float('inf')
    
    # 경로 품질 평가
    if efficiency >= 0.95:
        path_quality = "Excellent"
    elif efficiency >= 0.85:
        path_quality = "Good"
    elif efficiency >= 0.70:
        path_quality = "Fair"
    elif efficiency >= 0.50:
        path_quality = "Poor"
    else:
        path_quality = "Very Poor"
    
    return {
        'node_count': node_count,
        'actual_distance': actual_distance,
        'ideal_distance': ideal_distance,
        'efficiency': efficiency,
        'detour_ratio': detour_ratio,
        'path_quality': path_quality
    }

def dijkstra_irl_learning_experiment():
    """
    10x10 환경에서 Dijkstra 기반 IRL 학습 및 모델 저장 (폴더 분리)
    """
    print("Dijkstra-based IRL Learning Experiment (10x10 only)")
    print("=" * 80)
    
    # 저장 폴더
    model_dir = "saved_models/10x10_dijkstra"
    result_dir = "experiment_results/10x10_dijkstra"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # 실험 설정: 10x10 환경, Dijkstra만
    experiments = [
        {"algorithm": "dijkstra", "grid_size": 10, "obstacle_density": 0.15, "dynamic_obstacles": 2, "name": "10x10_Dijkstra"},
    ]
    
    results = {}
    
    for exp_config in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_config['name']}")
        print(f"Algorithm: {exp_config['algorithm']}")
        print(f"Grid Size: {exp_config['grid_size']}x{exp_config['grid_size']}")
        print(f"{'='*60}")
        
        # IRL 모델 훈련
        reward_predictor, training_stats = train_irl_model(
            algorithm=exp_config['algorithm'],
            grid_size=exp_config['grid_size'],
            obstacle_density=exp_config['obstacle_density'],
            dynamic_obstacles=exp_config['dynamic_obstacles']
        )
        
        # 모델 저장 (폴더 내)
        model_filename = os.path.join(model_dir, f"irl_model_{exp_config['name']}.pth")
        stats_filename = os.path.join(model_dir, f"irl_model_{exp_config['name']}_stats.json")
        torch.save(reward_predictor.state_dict(), model_filename)
        with open(stats_filename, 'w') as f:
            json.dump(training_stats, f, indent=2)
        print(f"   Model saved as: {model_filename}")
        print(f"   Training stats saved as: {stats_filename}")
        
        results[exp_config['name']] = {
            'model_filename': model_filename,
            'stats_filename': stats_filename,
            'training_stats': training_stats,
            'config': exp_config
        }
        print(f"   Experiment {exp_config['name']} completed successfully")
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print("Model | Algorithm | Grid Size | Training Time | Memory Usage")
    print("-" * 70)
    for name, result in results.items():
        stats = result['training_stats']
        print(f"{name:<15} | {stats['algorithm']:<9} | {stats['grid_size']:>9} | {stats['total_training_time']:>12.2f}s | {stats['irl_training_memory']:>11.2f} MB")
    
    # Manhattan 거리 기반 경로 품질 분석
    print(f"\n{'='*80}")
    print("MANHATTAN DISTANCE BASED PATH QUALITY ANALYSIS")
    print(f"{'='*80}")
    
    # 테스트 환경에서 경로 품질 분석
    test_env = AMRGridworld(grid_size=10, obstacle_density=0.15, dynamic_obstacles=2)
    
    print("Testing path quality on 10x10 environment...")
    
    # Dijkstra 경로 계산
    dijkstra_path, _, _, _ = test_env.get_optimal_path_mathematical("dijkstra")
    
    # IRL 가이드 경로 계산
    reward_predictor, _ = load_model(
        results['10x10_Dijkstra']['model_filename'],
        results['10x10_Dijkstra']['stats_filename']
    )
    irl_path = irl_guided_search(test_env, reward_predictor)
    
    # 경로 품질 분석
    dijkstra_analysis = analyze_path_quality(dijkstra_path, test_env.start_pos, test_env.goal_pos, test_env.grid_size)
    irl_analysis = analyze_path_quality(irl_path, test_env.start_pos, test_env.goal_pos, test_env.grid_size)
    
    print("\nPath Quality Comparison (Manhattan Distance Based):")
    print("-" * 80)
    print("Algorithm | Nodes | Actual Dist | Ideal Dist | Efficiency | Detour Ratio | Quality")
    print("-" * 80)
    
    print(f"Dijkstra   | {dijkstra_analysis['node_count']:>5} | {dijkstra_analysis['actual_distance']:>11} | {dijkstra_analysis['ideal_distance']:>10} | {dijkstra_analysis['efficiency']:>9.3f} | {dijkstra_analysis['detour_ratio']:>11.2f} | {dijkstra_analysis['path_quality']}")
    print(f"IRL        | {irl_analysis['node_count']:>5} | {irl_analysis['actual_distance']:>11} | {irl_analysis['ideal_distance']:>10} | {irl_analysis['efficiency']:>9.3f} | {irl_analysis['detour_ratio']:>11.2f} | {irl_analysis['path_quality']}")
    
    # 개선율 계산
    if dijkstra_analysis['actual_distance'] > 0 and irl_analysis['actual_distance'] > 0:
        distance_improvement = (dijkstra_analysis['actual_distance'] - irl_analysis['actual_distance']) / dijkstra_analysis['actual_distance'] * 100
        efficiency_improvement = (irl_analysis['efficiency'] - dijkstra_analysis['efficiency']) / dijkstra_analysis['efficiency'] * 100
        
        print(f"\nImprovement Analysis:")
        print(f"  • Distance improvement: {distance_improvement:+.1f}%")
        print(f"  • Efficiency improvement: {efficiency_improvement:+.1f}%")
        
        if distance_improvement > 0:
            print(f"  ✅ IRL found shorter actual distance!")
        else:
            print(f"  📏 Dijkstra found shorter actual distance")
            
        if efficiency_improvement > 0:
            print(f"  ✅ IRL achieved higher efficiency!")
        else:
            print(f"  📉 Dijkstra achieved higher efficiency")
    
    print(f"\nKey Insights:")
    print(f"  • Ideal distance: {dijkstra_analysis['ideal_distance']} (Manhattan distance without obstacles)")
    print(f"  • Detour ratio > 1.0 means path is longer than ideal")
    print(f"  • Efficiency closer to 1.0 means more optimal path")
    print(f"  • Node count ≠ actual distance (due to diagonal vs orthogonal movement)")
    
    # 시각화 (폴더 내 저장)
    create_training_visualization(results, save_dir=result_dir)
    
    return results

def create_training_visualization(results, save_dir=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    model_names = list(results.keys())
    training_times = [results[name]['training_stats']['total_training_time'] for name in model_names]
    memory_usages = [results[name]['training_stats']['irl_training_memory'] for name in model_names]
    grid_sizes = [results[name]['training_stats']['grid_size'] for name in model_names]
    algorithms = [results[name]['training_stats']['algorithm'] for name in model_names]
    colors = ['red' for _ in model_names]
    bars1 = ax1.bar(range(len(model_names)), training_times, color=colors, alpha=0.7)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.grid(True, alpha=0.3)
    for bar, time in zip(bars1, training_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{time:.1f}s', ha='center', va='bottom')
    bars2 = ax2.bar(range(len(model_names)), memory_usages, color=colors, alpha=0.7)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage Comparison')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.grid(True, alpha=0.3)
    for bar, memory in zip(bars2, memory_usages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{memory:.1f}MB', ha='center', va='bottom')
    dijkstra_times = [time for time, alg in zip(training_times, algorithms) if 'dijkstra' in alg]
    ax3.boxplot([dijkstra_times], tick_labels=['Dijkstra'])
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Algorithm Performance Comparison')
    ax3.grid(True, alpha=0.3)
    small_times = dijkstra_times
    ax4.boxplot([small_times], tick_labels=['10x10'])
    ax4.set_ylabel('Training Time (seconds)')
    ax4.set_title('Environment Size Performance Comparison')
    ax4.grid(True, alpha=0.3)
    plt.suptitle('IRL Model Training: Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    filename = "irl_training_analysis.png"
    if save_dir:
        filename = os.path.join(save_dir, filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nTraining analysis saved as: {filename}")
    plt.close()

if __name__ == "__main__":
    dijkstra_irl_learning_experiment() 