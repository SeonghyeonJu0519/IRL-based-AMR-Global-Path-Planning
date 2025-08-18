import numpy as np
import torch
import time
import random
from collections import deque
import math
import matplotlib.pyplot as plt

class Node:
    """RRT 노드 클래스"""
    def __init__(self, position, parent=None):
        self.position = position  # (x, y) 좌표
        self.parent = parent
        self.children = []

class Tree:
    """RRT 트리 클래스"""
    def __init__(self, root_position):
        self.root = Node(root_position)
        self.nodes = [self.root]
    
    def add_node(self, node, parent):
        node.parent = parent
        parent.children.append(node)
        self.nodes.append(node)
    
    def get_all_nodes(self):
        return self.nodes

def distance(pos1, pos2):
    """두 위치 간의 유클리드 거리"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def find_nearest_node(tree, target_position):
    """트리에서 가장 가까운 노드 찾기"""
    min_dist = float('inf')
    nearest_node = None
    
    for node in tree.get_all_nodes():
        dist = distance(node.position, target_position)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node

def extend_towards(from_node, to_position, step_size=1.0):
    """한 노드에서 목표 위치 방향으로 확장 (그리드 기반 4방향)"""
    direction = np.array(to_position) - np.array(from_node.position)
    dist = np.linalg.norm(direction)
    
    if dist == 0:
        return None
    
    # 4방향 이동으로 제한 (상, 하, 좌, 우)
    dx = direction[0]
    dy = direction[1]
    
    # 가장 큰 변화량을 가진 방향으로만 이동
    if abs(dx) > abs(dy):
        # x 방향 이동
        new_x = from_node.position[0] + (1 if dx > 0 else -1)
        new_y = from_node.position[1]
    else:
        # y 방향 이동
        new_x = from_node.position[0]
        new_y = from_node.position[1] + (1 if dy > 0 else -1)
    
    return Node((new_x, new_y))

def is_valid_path(from_node, to_node, env):
    """두 노드 간 경로가 유효한지 확인 (그리드 기반)"""
    if to_node is None:
        return False
    
    # 목표 노드가 장애물이 아닌지 확인
    x, y = to_node.position
    if not (0 <= x < env.grid_size and 0 <= y < env.grid_size and env.grid[x, y] == 0):
        return False
    
    # 두 노드 사이의 경로 전체를 검사 (그리드 기반)
    start_x, start_y = from_node.position
    end_x, end_y = to_node.position
    
    # 그리드 기반 Bresenham 알고리즘
    dx = abs(end_x - start_x)
    dy = abs(end_y - start_y)
    
    if dx > dy:
        # x 방향으로 더 긴 경우
        if start_x > end_x:
            start_x, end_x = end_x, start_x
            start_y, end_y = end_y, start_y
        
        y_step = 1 if end_y >= start_y else -1
        error = dx / 2
        
        y = start_y
        for x in range(start_x, end_x + 1):
            # 경로상의 각 점이 장애물이 아닌지 확인
            if not (0 <= x < env.grid_size and 0 <= y < env.grid_size and env.grid[x, y] == 0):
                return False
            
            error -= dy
            if error < 0:
                y += y_step
                error += dx
    else:
        # y 방향으로 더 긴 경우
        if start_y > end_y:
            start_x, end_x = end_x, start_x
            start_y, end_y = end_y, start_y
        
        x_step = 1 if end_x >= start_x else -1
        error = dy / 2
        
        x = start_x
        for y in range(start_y, end_y + 1):
            # 경로상의 각 점이 장애물이 아닌지 확인
            if not (0 <= x < env.grid_size and 0 <= y < env.grid_size and env.grid[x, y] == 0):
                return False
            
            error -= dx
            if error < 0:
                x += x_step
                error += dy
    
    return True

def construct_path(tree, goal_node):
    """트리에서 시작점부터 목표점까지의 경로 구성"""
    path = []
    current = goal_node
    
    while current is not None:
        path.append(current.position)
        current = current.parent
    
    return path[::-1]  # 역순으로 뒤집기

def basic_rrt(start, goal, env, max_iterations=5000, step_size=1.0, goal_bias=0.2):
    """그리드 기반 기본 RRT 알고리즘 (4방향 이동)"""
    tree = Tree(start)
    
    for i in range(max_iterations):
        # 목표 지향적 샘플링
        if random.random() < goal_bias:
            # 목표점 근처에서 랜덤 샘플링
            random_point = (
                goal[0] + random.randint(-2, 2),
                goal[1] + random.randint(-2, 2)
            )
            # 경계 내로 제한
            random_point = (
                max(0, min(env.grid_size-1, random_point[0])),
                max(0, min(env.grid_size-1, random_point[1]))
            )
        else:
            # 일반적인 랜덤 샘플링
            random_point = (
                random.randint(0, env.grid_size-1),
                random.randint(0, env.grid_size-1)
            )
        
        # 장애물이 아닌 경우에만 처리
        if env.grid[random_point[0], random_point[1]] == 0:
            nearest_node = find_nearest_node(tree, random_point)
            new_node = extend_towards(nearest_node, random_point, step_size)
            
            if new_node and is_valid_path(nearest_node, new_node, env):
                tree.add_node(new_node, nearest_node)
                
                # 목표점에 도달했는지 확인
                if distance(new_node.position, goal) < 2.0:
                    return construct_path(tree, new_node)
                
                # 목표점 근처에 도달했는지 확인 (더 강력한 연결)
                if distance(new_node.position, goal) < 5.0:
                    # 목표점까지 단계별로 연결 시도
                    current_pos = list(new_node.position)
                    goal_x, goal_y = goal
                    
                    # x 방향으로 목표점까지 이동
                    while current_pos[0] != goal_x:
                        if current_pos[0] < goal_x:
                            next_x = current_pos[0] + 1
                        else:
                            next_x = current_pos[0] - 1
                        
                        if (0 <= next_x < env.grid_size and 
                            env.grid[next_x, current_pos[1]] == 0):
                            next_node = Node((next_x, current_pos[1]))
                            tree.add_node(next_node, tree.nodes[-1] if tree.nodes else new_node)
                            current_pos[0] = next_x
                            
                            if distance(next_node.position, goal) < 2.0:
                                return construct_path(tree, next_node)
                        else:
                            break
                    
                    # y 방향으로 목표점까지 이동
                    while current_pos[1] != goal_y:
                        if current_pos[1] < goal_y:
                            next_y = current_pos[1] + 1
                        else:
                            next_y = current_pos[1] - 1
                        
                        if (0 <= next_y < env.grid_size and 
                            env.grid[current_pos[0], next_y] == 0):
                            next_node = Node((current_pos[0], next_y))
                            tree.add_node(next_node, tree.nodes[-1] if tree.nodes else new_node)
                            current_pos[1] = next_y
                            
                            if distance(next_node.position, goal) < 2.0:
                                return construct_path(tree, next_node)
                        else:
                            break
    
    # 목표점에 도달하지 못한 경우, 마지막 노드에서 목표점으로 직접 연결 시도
    if tree.nodes:
        last_node = tree.nodes[-1]
        if distance(last_node.position, goal) < 3.0:
            # 목표점까지 최단 경로 찾기
            current_pos = list(last_node.position)
            goal_x, goal_y = goal
            
            # x 방향 먼저
            while current_pos[0] != goal_x:
                if current_pos[0] < goal_x:
                    next_x = current_pos[0] + 1
                else:
                    next_x = current_pos[0] - 1
                
                if (0 <= next_x < env.grid_size and 
                    env.grid[next_x, current_pos[1]] == 0):
                    next_node = Node((next_x, current_pos[1]))
                    tree.add_node(next_node, tree.nodes[-1])
                    current_pos[0] = next_x
                else:
                    break
            
            # y 방향
            while current_pos[1] != goal_y:
                if current_pos[1] < goal_y:
                    next_y = current_pos[1] + 1
                else:
                    next_y = current_pos[1] - 1
                
                if (0 <= next_y < env.grid_size and 
                    env.grid[current_pos[0], next_y] == 0):
                    next_node = Node((current_pos[0], next_y))
                    tree.add_node(next_node, tree.nodes[-1])
                    current_pos[1] = next_y
                else:
                    break
            
            return construct_path(tree, tree.nodes[-1])
    
    return None

def irl_rrt(start, goal, env, irl_reward_predictor, max_iterations=5000, step_size=1.0, goal_bias=0.2):
    """그리드 기반 IRL 보상을 활용한 RRT (4방향 이동)"""
    tree = Tree(start)
    
    for i in range(max_iterations):
        # 목표 지향적 샘플링
        if random.random() < goal_bias:
            random_point = (
                goal[0] + random.randint(-2, 2),
                goal[1] + random.randint(-2, 2)
            )
            random_point = (
                max(0, min(env.grid_size-1, random_point[0])),
                max(0, min(env.grid_size-1, random_point[1]))
            )
        else:
            random_point = (
                random.randint(0, env.grid_size-1),
                random.randint(0, env.grid_size-1)
            )
        
        if env.grid[random_point[0], random_point[1]] == 0:
            nearest_node = find_nearest_node(tree, random_point)
            new_node = extend_towards(nearest_node, random_point, step_size)
            
            if new_node and is_valid_path(nearest_node, new_node, env):
                # IRL 보상 계산
                state = new_node.position[0] * env.grid_size + new_node.position[1]
                features = extract_features(state, env)
                device = next(irl_reward_predictor.parameters()).device
                with torch.no_grad():
                    irl_reward = irl_reward_predictor(features.unsqueeze(0).to(device)).item()
                
                # IRL 보상이 일정 임계값 이상일 때만 노드 추가
                if irl_reward > -1.5:
                    tree.add_node(new_node, nearest_node)
                    
                    if distance(new_node.position, goal) < 2.0:
                        return construct_path(tree, new_node)
                    
                    # 목표점 근처에 도달했는지 확인 (더 강력한 연결)
                    if distance(new_node.position, goal) < 5.0:
                        # 목표점까지 단계별로 연결 시도
                        current_pos = list(new_node.position)
                        goal_x, goal_y = goal
                        
                        # x 방향으로 목표점까지 이동
                        while current_pos[0] != goal_x:
                            if current_pos[0] < goal_x:
                                next_x = current_pos[0] + 1
                            else:
                                next_x = current_pos[0] - 1
                            
                            if (0 <= next_x < env.grid_size and 
                                env.grid[next_x, current_pos[1]] == 0):
                                next_node = Node((next_x, current_pos[1]))
                                tree.add_node(next_node, tree.nodes[-1] if tree.nodes else new_node)
                                current_pos[0] = next_x
                                
                                if distance(next_node.position, goal) < 2.0:
                                    return construct_path(tree, next_node)
                            else:
                                break
                        
                        # y 방향으로 목표점까지 이동
                        while current_pos[1] != goal_y:
                            if current_pos[1] < goal_y:
                                next_y = current_pos[1] + 1
                            else:
                                next_y = current_pos[1] - 1
                            
                            if (0 <= next_y < env.grid_size and 
                                env.grid[current_pos[0], next_y] == 0):
                                next_node = Node((current_pos[0], next_y))
                                tree.add_node(next_node, tree.nodes[-1] if tree.nodes else new_node)
                                current_pos[1] = next_y
                                
                                if distance(next_node.position, goal) < 2.0:
                                    return construct_path(tree, next_node)
                            else:
                                break
    
    # 목표점에 도달하지 못한 경우, 마지막 노드에서 목표점으로 직접 연결 시도
    if tree.nodes:
        last_node = tree.nodes[-1]
        if distance(last_node.position, goal) < 3.0:
            # 목표점까지 최단 경로 찾기
            current_pos = list(last_node.position)
            goal_x, goal_y = goal
            
            # x 방향 먼저
            while current_pos[0] != goal_x:
                if current_pos[0] < goal_x:
                    next_x = current_pos[0] + 1
                else:
                    next_x = current_pos[0] - 1
                
                if (0 <= next_x < env.grid_size and 
                    env.grid[next_x, current_pos[1]] == 0):
                    next_node = Node((next_x, current_pos[1]))
                    tree.add_node(next_node, tree.nodes[-1])
                    current_pos[0] = next_x
                else:
                    break
            
            # y 방향
            while current_pos[1] != goal_y:
                if current_pos[1] < goal_y:
                    next_y = current_pos[1] + 1
                else:
                    next_y = current_pos[1] - 1
                
                if (0 <= next_y < env.grid_size and 
                    env.grid[current_pos[0], next_y] == 0):
                    next_node = Node((current_pos[0], next_y))
                    tree.add_node(next_node, tree.nodes[-1])
                    current_pos[1] = next_y
                else:
                    break
            
            return construct_path(tree, tree.nodes[-1])
    
    return None

def extract_features(state, env):
    """상태를 특성 벡터로 변환 (IRL과 동일한 함수 사용)"""
    from amr_path_planning_irl import extract_features
    return extract_features(state, env)

def visualize_rrt_paths(env, basic_path, irl_path, grid_size):
    """RRT 경로 시각화"""
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
    if basic_path:
        x = [p[1] + 0.5 for p in basic_path]
        y = [env.grid_size-1-p[0] + 0.5 for p in basic_path]
        ax.plot(x, y, color='blue', linewidth=2, label='Basic RRT', alpha=0.8)
    
    if irl_path:
        x = [p[1] + 0.5 for p in irl_path]
        y = [env.grid_size-1-p[0] + 0.5 for p in irl_path]
        ax.plot(x, y, color='purple', linewidth=2, label='IRL-RRT', alpha=0.8)
    
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_title(f'RRT Path Comparison: {grid_size}x{grid_size}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    filename = f"rrt_paths_{grid_size}x{grid_size}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"RRT visualization saved as: {filename}")
    plt.close()

def rrt_path_planning(env, max_iterations=1000, use_irl=False, irl_reward_predictor=None):
    """RRT 경로 계획 메인 함수"""
    start = env.start_pos
    goal = env.goal_pos
    
    if use_irl and irl_reward_predictor is not None:
        path = irl_rrt(start, goal, env, irl_reward_predictor, max_iterations)
    else:
        path = basic_rrt(start, goal, env, max_iterations)
    
    return path

if __name__ == "__main__":
    # 테스트
    from amr_path_planning_irl import AMRGridworld
    from dijkstra_irl_learning import load_model
    import os
    
    # 테스트 환경 생성
    env = AMRGridworld(grid_size=10, obstacle_density=0.1)
    
    print(f"Start: {env.start_pos}")
    print(f"Goal: {env.goal_pos}")
    
    # 기본 RRT 테스트
    start_time = time.time()
    basic_path = rrt_path_planning(env, max_iterations=1000, use_irl=False)
    basic_time = time.time() - start_time
    
    print(f"Basic RRT: {len(basic_path) if basic_path else 'No path'} steps, {basic_time:.3f}s")
    
    if basic_path:
        print(f"Path: {basic_path[:5]}...{basic_path[-5:] if len(basic_path) > 10 else basic_path}")
    
    # IRL-RRT 테스트
    irl_model_path = "saved_models/10x10_dijkstra/irl_model_10x10_Dijkstra.pth"
    irl_stats_path = "saved_models/10x10_dijkstra/irl_model_10x10_Dijkstra_stats.json"
    
    if os.path.exists(irl_model_path) and os.path.exists(irl_stats_path):
        print("\nTesting IRL-RRT...")
        irl_reward_predictor, _ = load_model(irl_model_path, irl_stats_path)
        
        start_time = time.time()
        irl_path = rrt_path_planning(env, max_iterations=1000, use_irl=True, irl_reward_predictor=irl_reward_predictor)
        irl_time = time.time() - start_time
        
        print(f"IRL-RRT: {len(irl_path) if irl_path else 'No path'} steps, {irl_time:.3f}s")
        
        if irl_path:
            print(f"Path: {irl_path[:5]}...{irl_path[-5:] if len(irl_path) > 10 else irl_path}")
        
        # 경로 길이 비교
        if basic_path and irl_path:
            print(f"\nComparison:")
            print(f"Basic RRT: {len(basic_path)} steps")
            print(f"IRL-RRT: {len(irl_path)} steps")
            print(f"Improvement: {len(basic_path) - len(irl_path)} steps shorter")
            
            # 시각화
            visualize_rrt_paths(env, basic_path, irl_path, env.grid_size)
    else:
        print(f"\nIRL model not found: {irl_model_path}")
        print("Please run dijkstra_irl_learning.py first to train the IRL model.")
        
        # Basic RRT만 시각화
        if basic_path:
            visualize_rrt_paths(env, basic_path, None, env.grid_size)
