import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import copy
import time
import os
import gc
from amr_path_planning_irl import AMRGridworld, extract_features, irl_guided_search
from dijkstra_irl_learning import load_model

# GPU 메모리 관리 함수
def clear_gpu_memory():
    """GPU 메모리를 완전히 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def create_connected_environment(grid_size, obstacle_density, dynamic_obstacles=0):
    """연결 가능한 환경 생성 (경로 차단 방지)"""
    print(f"Creating connected environment: {grid_size}x{grid_size}, obstacle density: {obstacle_density:.1%}")
    
    env = AMRGridworld(grid_size=grid_size, obstacle_density=obstacle_density, dynamic_obstacles=dynamic_obstacles)
    
    # 시작점과 목표점이 장애물이 아닌지 확인
    while env.grid[env.start_pos[0], env.start_pos[1]] > 0:
        env.start_pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
    while env.grid[env.goal_pos[0], env.goal_pos[1]] > 0:
        env.goal_pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
    
    # 경로 연결성 확인 및 수정
    def check_connectivity():
        """시작점에서 목표점까지 경로가 있는지 확인"""
        try:
            from amr_path_planning_irl import ComputationalProfiler
            profiler = ComputationalProfiler()
            path, _, _, _ = env.get_optimal_path_mathematical("dijkstra", profiler)
            return len(path) > 1  # 경로가 있고 길이가 1보다 큼
        except:
            return False
    
    # 경로가 없으면 장애물을 일부 제거
    max_attempts = 10
    for attempt in range(max_attempts):
        if check_connectivity():
            print(f"✅ Connected environment created (attempt {attempt + 1})")
            return env
        
        print(f"⚠️ Path blocked, removing some obstacles (attempt {attempt + 1})")
        
        # 장애물 중 일부를 제거 (가장자리부터)
        obstacles_to_remove = max(1, int(grid_size * 0.05))  # 5% 제거
        
        # 시작점과 목표점 주변의 장애물 우선 제거
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = env.start_pos[0] + dx, env.start_pos[1] + dy
                if (0 <= x < grid_size and 0 <= y < grid_size and 
                    env.grid[x, y] > 0):
                    env.grid[x, y] = 0
                    obstacles_to_remove -= 1
                    if obstacles_to_remove <= 0:
                        break
            if obstacles_to_remove <= 0:
                break
        
        # 목표점 주변도 제거
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = env.goal_pos[0] + dx, env.goal_pos[1] + dy
                if (0 <= x < grid_size and 0 <= y < grid_size and 
                    env.grid[x, y] > 0):
                    env.grid[x, y] = 0
                    obstacles_to_remove -= 1
                    if obstacles_to_remove <= 0:
                        break
            if obstacles_to_remove <= 0:
                break
        
        # 남은 장애물을 랜덤하게 제거
        if obstacles_to_remove > 0:
            obstacle_positions = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if env.grid[i, j] > 0:
                        obstacle_positions.append((i, j))
            
            if obstacle_positions:
                import random
                random.shuffle(obstacle_positions)
                for i, j in obstacle_positions[:obstacles_to_remove]:
                    env.grid[i, j] = 0
    
    print(f"❌ Failed to create connected environment after {max_attempts} attempts")
    print("Creating environment with lower obstacle density...")
    
    # 마지막 시도: 낮은 장애물 밀도로 재생성
    return create_connected_environment(grid_size, max(0.1, obstacle_density * 0.5), dynamic_obstacles)

def run_algorithms_safe(grid_size=10, obstacle_density=0.3, dynamic_obstacles=0):
    """안전한 알고리즘 실행 (연결 가능한 환경 보장)"""
    print(f"Running algorithms on {grid_size}x{grid_size} environment...")
    
    clear_gpu_memory()
    
    # 환경 생성
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    # 연결 가능한 환경 생성 (경로 차단 방지)
    env = create_connected_environment(grid_size, obstacle_density, dynamic_obstacles)
    
    print(f"Environment: Start={env.start_pos}, Goal={env.goal_pos}")
    print(f"Final obstacle count: {np.sum(env.grid > 0)}/{grid_size*grid_size} ({np.sum(env.grid > 0)/(grid_size*grid_size)*100:.1f}%)")
    
    results = {}
    
    # 1. Dijkstra (수학적 최적 알고리즘)
    try:
        print("Running Dijkstra...")
        start_time = time.time()
        
        # 연산량 측정을 위한 프로파일러 생성
        from amr_path_planning_irl import ComputationalProfiler
        profiler = ComputationalProfiler()
        
        dijkstra_path, _, _, profiler_results = env.get_optimal_path_mathematical("dijkstra", profiler=profiler)
        dijkstra_time = time.time() - start_time
        
        # 연산량 정보 추출
        dijkstra_operations = profiler_results.get('operations_count', 0)
        dijkstra_expansions = profiler_results.get('state_expansions', 0)
        
        results['dijkstra'] = {
            'path': dijkstra_path, 
            'time': dijkstra_time, 
            'training_time': 0,
            'operations': dijkstra_operations,
            'expansions': dijkstra_expansions
        }
        print(f"Dijkstra: Length={len(dijkstra_path) if dijkstra_path else 'N/A'}, Time={dijkstra_time:.3f}s, Operations={dijkstra_operations}")
    except Exception as e:
        print(f"Dijkstra error: {e}")
        results['dijkstra'] = {'path': None, 'time': 0, 'training_time': 0, 'operations': 0, 'expansions': 0}
    
    # 2. IRL (사전 학습된 모델 사용)
    try:
        print("Running IRL...")
        start_time = time.time()
        # IRL 모델 경로 확인
        irl_model_path = "saved_models/10x10_dijkstra/irl_model_10x10_Dijkstra.pth"
        irl_stats_path = "saved_models/10x10_dijkstra/irl_model_10x10_Dijkstra_stats.json"
        
        if os.path.exists(irl_model_path) and os.path.exists(irl_stats_path):
            irl_reward_predictor, _ = load_model(irl_model_path, irl_stats_path)
            
            # IRL 연산량 측정
            irl_profiler = ComputationalProfiler()
            irl_path = irl_guided_search(copy.deepcopy(env), irl_reward_predictor, profiler=irl_profiler)
            irl_time = time.time() - start_time
            
            # IRL 연산량 정보 추출
            irl_operations = irl_profiler.get_results().get('operations_count', 0)
            irl_inferences = irl_profiler.get_results().get('neural_inferences', 0)
            
            # IRL은 사전 학습된 모델이므로 학습 시간은 0으로 설정
            results['irl'] = {
                'path': irl_path, 
                'time': irl_time, 
                'training_time': 0,
                'operations': irl_operations,
                'inferences': irl_inferences
            }
            print(f"IRL: Length={len(irl_path) if irl_path else 'N/A'}, Time={irl_time:.3f}s, Operations={irl_operations}, Inferences={irl_inferences}")
        else:
            print(f"IRL model not found: {irl_model_path}")
            results['irl'] = {'path': None, 'time': 0, 'training_time': 0, 'operations': 0, 'inferences': 0}
    except Exception as e:
        print(f"IRL error: {e}")
        results['irl'] = {'path': None, 'time': 0, 'training_time': 0, 'operations': 0, 'inferences': 0}
    
    # 3. DQN (개선된 DQN)
    try:
        print("Training DQN...")
        start_time = time.time()
        from improved_dqn_path_planning import train_improved_dqn, improved_dqn_path_planning
        
        # 환경 복잡도에 따른 DQN 학습 파라미터 조정
        obstacle_count = np.sum(env.grid > 0)
        obstacle_ratio = obstacle_count / (grid_size * grid_size)
        
        print(f"Environment complexity: {obstacle_ratio:.1%} obstacles")
        
        # 장애물 밀도에 따른 에피소드 수 조정
        if obstacle_ratio <= 0.15:
            episodes = 300  # 낮은 복잡도
        elif obstacle_ratio <= 0.25:
            episodes = 600  # 중간 복잡도
        elif obstacle_ratio <= 0.35:
            episodes = 1000  # 높은 복잡도
        else:
            episodes = 1500  # 매우 높은 복잡도
        
        # 환경 크기에 따른 추가 조정
        if grid_size <= 10:
            episodes = int(episodes * 0.8)
        elif grid_size <= 20:
            episodes = episodes
        elif grid_size <= 50:
            episodes = int(episodes * 1.2)
        elif grid_size <= 70:
            episodes = int(episodes * 1.5)
        else:
            episodes = int(episodes * 2.0)
        
        print(f"DQN training episodes: {episodes} (adjusted for complexity)")
        
        # DQN 학습 시간 측정
        dqn_training_start = time.time()
        improved_dqn_agent, _ = train_improved_dqn(copy.deepcopy(env), episodes=episodes)
        dqn_training_time = time.time() - dqn_training_start
        
        # DQN 경로 계획 시간 및 연산량 측정
        dqn_path_start = time.time()
        dqn_profiler = ComputationalProfiler()
        improved_dqn_path = improved_dqn_path_planning(copy.deepcopy(env), improved_dqn_agent, profiler=dqn_profiler)
        dqn_path_time = time.time() - dqn_path_start
        
        # DQN 연산량 정보 추출
        dqn_operations = dqn_profiler.get_results().get('operations_count', 0)
        dqn_inferences = dqn_profiler.get_results().get('neural_inferences', 0)
        
        total_dqn_time = time.time() - start_time
        results['dqn'] = {
            'path': improved_dqn_path, 
            'time': dqn_path_time, 
            'training_time': dqn_training_time,
            'total_time': total_dqn_time,
            'operations': dqn_operations,
            'inferences': dqn_inferences
        }
        print(f"DQN: Length={len(improved_dqn_path) if improved_dqn_path else 'N/A'}, Path Time={dqn_path_time:.3f}s, Training Time={dqn_training_time:.3f}s, Operations={dqn_operations}, Inferences={dqn_inferences}")
    except Exception as e:
        print(f"DQN error: {e}")
        results['dqn'] = {'path': None, 'time': 0, 'training_time': 0, 'total_time': 0, 'operations': 0, 'inferences': 0}
    
    return results, env

def visualize_paths_simple(env, results, grid_size):
    """간단한 경로 시각화"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
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
    
    # 경로들 표시
    colors = ['blue', 'purple', 'green', 'red', 'orange']
    algorithms = ['dijkstra', 'irl', 'dqn']  # DQN으로 변경
    
    for i, alg in enumerate(algorithms):
        if alg in results and results[alg]['path']:
            path = results[alg]['path']
            if path and len(path) > 1:  # 경로가 있고 길이가 1보다 클 때만
                # 경로가 튜플인지 확인
                if isinstance(path[0], tuple):
                    x = [p[1] + 0.5 for p in path]
                    y = [env.grid_size-1-p[0] + 0.5 for p in path]
                else:
                    # 정수인 경우 (x, y) 좌표로 변환
                    x = [(p % env.grid_size) + 0.5 for p in path]
                    y = [env.grid_size-1-(p // env.grid_size) + 0.5 for p in path]
                ax.plot(x, y, color=colors[i], linewidth=2, label=alg.upper(), alpha=0.8)
    
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_title(f'Path Comparison: {grid_size}x{grid_size}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    filename = f"paths_{grid_size}x{grid_size}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {filename}")
    plt.close()  # show() 대신 close()로 변경

def calculate_path_quality(dijkstra_path, other_path):
    """경로 품질 계산 (Dijkstra 대비)"""
    if not dijkstra_path or not other_path:
        return 0.0
    
    dijkstra_length = len(dijkstra_path)
    other_length = len(other_path)
    
    # 경로 품질 = Dijkstra 길이 / 다른 알고리즘 길이 (1에 가까울수록 좋음)
    quality = dijkstra_length / other_length if other_length > 0 else 0.0
    return quality

def create_performance_summary(all_results, grid_sizes):
    """성능 요약 생성"""
    print(f"\n{'='*80}")
    print("PERFORMANCE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    # 각 환경별 결과 요약
    for grid_size in grid_sizes:
        if grid_size in all_results:
            print(f"\n{grid_size}x{grid_size} Environment:")
            print("-" * 50)
            
            results = all_results[grid_size]
            
            # Dijkstra 결과 (기준)
            if 'dijkstra' in results and results['dijkstra']['path']:
                dijkstra_length = len(results['dijkstra']['path'])
                dijkstra_time = results['dijkstra']['time']
                dijkstra_operations = results['dijkstra']['operations']
                dijkstra_expansions = results['dijkstra']['expansions']
                print(f"Dijkstra (Optimal):")
                print(f"  Path Length: {dijkstra_length}")
                print(f"  Execution Time: {dijkstra_time:.4f}s")
                print(f"  Total Operations: {dijkstra_operations:,}")
                print(f"  Node Expansions: {dijkstra_expansions:,}")
                print(f"  Training Time: N/A (Mathematical Algorithm)")
                
                # IRL 결과
                if 'irl' in results and results['irl']['path']:
                    irl_length = len(results['irl']['path'])
                    irl_time = results['irl']['time']
                    irl_operations = results['irl']['operations']
                    irl_inferences = results['irl']['inferences']
                    irl_quality = calculate_path_quality(results['dijkstra']['path'], results['irl']['path'])
                    print(f"\nIRL (Inverse Reinforcement Learning):")
                    print(f"  Path Length: {irl_length}")
                    print(f"  Path Quality: {irl_quality:.3f} (vs Dijkstra)")
                    print(f"  Execution Time: {irl_time:.4f}s")
                    print(f"  Total Operations: {irl_operations:,}")
                    print(f"  Neural Inferences: {irl_inferences:,}")
                    if irl_time > 0:
                        print(f"  Speed Ratio: {dijkstra_time/irl_time:.2f}x faster than Dijkstra")
                    else:
                        print(f"  Speed Ratio: IRL time = 0 (no valid path found)")
                    if irl_operations > 0:
                        print(f"  Efficiency Ratio: {dijkstra_operations/irl_operations:.2f}x fewer operations than Dijkstra")
                    else:
                        print(f"  Efficiency Ratio: IRL operations = 0 (no valid path found)")
                    print(f"  Training Time: N/A (Pre-trained Model)")
                
                # DQN 결과
                if 'dqn' in results and results['dqn']['path']:
                    dqn_length = len(results['dqn']['path'])
                    dqn_time = results['dqn']['time']
                    dqn_training_time = results['dqn']['training_time']
                    dqn_total_time = results['dqn']['total_time']
                    dqn_operations = results['dqn']['operations']
                    dqn_inferences = results['dqn']['inferences']
                    dqn_quality = calculate_path_quality(results['dijkstra']['path'], results['dqn']['path'])
                    print(f"\nDQN (Deep Q-Network):")
                    print(f"  Path Length: {dqn_length}")
                    print(f"  Path Quality: {dqn_quality:.3f} (vs Dijkstra)")
                    print(f"  Execution Time: {dqn_time:.4f}s")
                    print(f"  Total Operations: {dqn_operations:,}")
                    print(f"  Neural Inferences: {dqn_inferences:,}")
                    print(f"  Training Time: {dqn_training_time:.4f}s")
                    print(f"  Total Time: {dqn_total_time:.4f}s")
                    if dqn_time > 0:
                        print(f"  Speed Ratio: {dijkstra_time/dqn_time:.2f}x faster than Dijkstra")
                    else:
                        print(f"  Speed Ratio: DQN time = 0 (no valid path found)")
                    if dqn_operations > 0:
                        print(f"  Efficiency Ratio: {dijkstra_operations/dqn_operations:.2f}x fewer operations than Dijkstra")
                    else:
                        print(f"  Efficiency Ratio: DQN operations = 0 (no valid path found)")
                    
                    # IRL vs DQN 비교
                    if 'irl' in results and results['irl']['path']:
                        print(f"\nIRL vs DQN Comparison:")
                        print(f"  Path Quality: IRL({irl_quality:.3f}) vs DQN({dqn_quality:.3f})")
                        print(f"  Execution Speed: IRL({irl_time:.4f}s) vs DQN({dqn_time:.4f}s)")
                        print(f"  Operation Efficiency: IRL({irl_operations:,}) vs DQN({dqn_operations:,})")
                        print(f"  Training Required: IRL(No) vs DQN({dqn_training_time:.4f}s)")
            else:
                print("Dijkstra failed - cannot calculate quality metrics")

def create_comparison_plots(all_results, grid_sizes):
    """비교 그래프 생성"""
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*80}")
    
    # 데이터 준비
    algorithms = ['dijkstra', 'irl', 'dqn']
    metrics = ['path_length', 'operations', 'path_quality']
    
    # 각 메트릭별로 데이터 수집
    data = {metric: {alg: [] for alg in algorithms} for metric in metrics}
    
    for grid_size in grid_sizes:
        if grid_size in all_results:
            results = all_results[grid_size]
            
            for alg in algorithms:
                if alg in results and results[alg]['path']:
                    # 경로 길이
                    data['path_length'][alg].append(len(results[alg]['path']))
                    # 연산량
                    if alg == 'dijkstra':
                        data['operations'][alg].append(results[alg]['operations'])
                    else:
                        data['operations'][alg].append(results[alg]['operations'])
                    # 경로 품질 (Dijkstra 대비)
                    if alg == 'dijkstra':
                        data['path_quality'][alg].append(1.0)  # Dijkstra는 항상 1.0
                    else:
                        quality = calculate_path_quality(results['dijkstra']['path'], results[alg]['path'])
                        data['path_quality'][alg].append(quality)
                else:
                    data['path_length'][alg].append(0)
                    data['operations'][alg].append(0)
                    data['path_quality'][alg].append(0)
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Algorithm Performance Comparison', fontsize=16)
    
    # 1. 경로 길이 비교
    ax1 = axes[0, 0]
    x = np.arange(len(grid_sizes))
    width = 0.25
    
    for i, alg in enumerate(algorithms):
        ax1.bar(x + i*width, data['path_length'][alg], width, label=alg.upper(), alpha=0.8)
    
    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Path Length')
    ax1.set_title('Path Length Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'{size}x{size}' for size in grid_sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 연산량 비교 (실행 시간 대신)
    ax2 = axes[0, 1]
    for i, alg in enumerate(algorithms):
        ax2.bar(x + i*width, data['operations'][alg], width, label=alg.upper(), alpha=0.8)
    
    ax2.set_xlabel('Grid Size')
    ax2.set_ylabel('Total Operations')
    ax2.set_title('Computational Complexity Comparison')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'{size}x{size}' for size in grid_sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 경로 품질 비교
    ax3 = axes[1, 0]
    for i, alg in enumerate(algorithms):
        ax3.bar(x + i*width, data['path_quality'][alg], width, label=alg.upper(), alpha=0.8)
    
    ax3.set_xlabel('Grid Size')
    ax3.set_ylabel('Path Quality (vs Dijkstra)')
    ax3.set_title('Path Quality Comparison')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f'{size}x{size}' for size in grid_sizes])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. IRL vs DQN 학습 시간 비교
    ax4 = axes[1, 1]
    irl_training_times = []
    dqn_training_times = []
    
    for grid_size in grid_sizes:
        if grid_size in all_results:
            results = all_results[grid_size]
            irl_training_times.append(0)  # IRL은 사전 학습된 모델
            if 'dqn' in results:
                dqn_training_times.append(results['dqn']['training_time'])
            else:
                dqn_training_times.append(0)
    
    ax4.bar(x, irl_training_times, width, label='IRL', alpha=0.8, color='purple')
    ax4.bar(x + width, dqn_training_times, width, label='DQN', alpha=0.8, color='green')
    
    ax4.set_xlabel('Grid Size')
    ax4.set_ylabel('Training Time (s)')
    ax4.set_title('Training Time Comparison (IRL vs DQN)')
    ax4.set_xticks(x + width/2)
    ax4.set_xticklabels([f'{size}x{size}' for size in grid_sizes])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Performance comparison plots saved as: performance_comparison.png")
    plt.close()

if __name__ == "__main__":
    # GPU 초기화
    clear_gpu_memory()
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 장애물 밀도 설정 (경로 차단 문제 해결을 위해 조정)
    print("\n🎯 Obstacle Density Settings:")
    print("• 0.15 (15%): Easy - All algorithms should work well")
    print("• 0.20 (20%): Medium - Good balance for comparison")
    print("• 0.25 (25%): Hard - Challenging but solvable")
    print("• 0.30 (30%): Very Hard - May cause path blocking issues")
    
    obstacle_density = 0.20  # 기본값을 20%로 조정 (더 안정적)
    print(f"Using obstacle density: {obstacle_density:.1%}")
    
    grid_sizes = [10, 50, 70, 100]
    all_results = {}
    
    for grid_size in grid_sizes:
        print(f"\n{'='*50}")
        print(f"Testing {grid_size}x{grid_size} environment")
        print(f"{'='*50}")
        
        try:
            results, env = run_algorithms_safe(
                grid_size=grid_size,
                obstacle_density=obstacle_density,  # 조정된 장애물 밀도 사용
                dynamic_obstacles=0
            )
            all_results[grid_size] = results
            
            # 시각화
            visualize_paths_simple(env, results, grid_size)
            
        except Exception as e:
            print(f"Error in {grid_size}x{grid_size}: {e}")
            all_results[grid_size] = {}
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for grid_size in grid_sizes:
        if grid_size in all_results:
            print(f"\n{grid_size}x{grid_size}:")
            for alg, data in all_results[grid_size].items():
                try:
                    path_length = len(data['path']) if data['path'] else 'N/A'
                    time_taken = f"{data['time']:.3f}s"
                    print(f"  {alg.upper()}: Length={path_length}, Time={time_taken}")
                except Exception as e:
                    print(f"  {alg.upper()}: Error processing data - {e}")
    
    # 성능 요약 및 비교 그래프 생성
    create_performance_summary(all_results, grid_sizes)
    create_comparison_plots(all_results, grid_sizes) 