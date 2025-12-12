"""Ejercicio 1 - Q-learning en Grid 3x4"""

import numpy as np
import random
from typing import Tuple, Dict, List
from collections import defaultdict


class QLearningGrid:
    """Q-learning para grid 3x4. Start: (0,0), Goal: (2,3), Obstacle: (1,1)"""
    
    def __init__(self, grid_size: Tuple[int, int] = (3, 4)):
        self.grid_size = grid_size
        self.actions = ['up', 'down', 'left', 'right']
        # Valores Q inicializados a 0 por defecto
        self.q_table = defaultdict(float)
        
        # Parámetros de Q-learning
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        
        # Estados especiales
        self.start_state = (0, 0)
        self.goal_state = (2, 3)
        self.obstacle = (1, 1)
        
        # Estadísticas
        self.episodes_history = []
        self.q_table_snapshots = []
        
    def get_next_state(self, state: Tuple[int, int], action: str, 
                       stochastic: bool = False, success_prob: float = 0.99) -> Tuple[int, int]:
        row, col = state
        
        # Marinero borracho: a veces se mueve aleatoriamente
        if stochastic and random.random() > success_prob:
            possible = [a for a in self.actions if a != action]
            action = random.choice(possible)
        
        next_row, next_col = row, col
        
        # Calcular nueva posición respetando límites del grid
        if action == 'up':
            next_row = min(row + 1, self.grid_size[0] - 1)
        elif action == 'down':
            next_row = max(row - 1, 0)
        elif action == 'left':
            next_col = max(col - 1, 0)
        elif action == 'right':
            next_col = min(col + 1, self.grid_size[1] - 1)
        
        next_state = (next_row, next_col)
        
        if next_state == self.obstacle:
            return state
            
        return next_state
    
    def get_reward(self, state: Tuple[int, int], reward_type: str = 'simple') -> float:
        if state == self.goal_state:
            return 100.0
        
        if reward_type == 'simple':
            return -1.0
        
        elif reward_type == 'distance':
            # Recompensas más altas cuanto más cerca del objetivo
            reward_grid = np.array([
                [-5, -4, -3, -2],
                [-4, -99, -2, -1],
                [-3, -2, -1, 100]
            ])
            return reward_grid[state[0]][state[1]]
        
        return -1.0
    
    def choose_action(self, state: Tuple[int, int]) -> str:
        # Epsilon-greedy: explorar vs. explotar
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = {action: self.q_table[(state, action)] for action in self.actions}
            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state: Tuple[int, int], action: str, 
                       reward: float, next_state: Tuple[int, int]):
        # Ecuación de Bellman para actualizar Q-values
        current_q = self.q_table[(state, action)]
        max_next_q = max([self.q_table[(next_state, a)] for a in self.actions])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[(state, action)] = new_q
    
    def train(self, num_episodes: int, reward_type: str = 'simple', 
              stochastic: bool = False, success_prob: float = 0.99,
              snapshot_episodes: List[int] = None) -> Dict:
        if snapshot_episodes is None:
            snapshot_episodes = [0, num_episodes // 3, 2 * num_episodes // 3, num_episodes - 1]
        
        self.q_table_snapshots = []
        steps_per_episode = []
        
        for episode in range(num_episodes):
            state = self.start_state
            steps = 0
            episode_done = False
            
            # Bucle de un episodio: elegir acción, moverse, actualizar Q
            while not episode_done:
                action = self.choose_action(state)
                next_state = self.get_next_state(state, action, stochastic, success_prob)
                reward = self.get_reward(next_state, reward_type)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                steps += 1
                
                if state == self.goal_state or steps >= 1000:
                    episode_done = True
            
            steps_per_episode.append(steps)
            
            # Guardar Q-table en ciertos episodios para análisis
            if episode in snapshot_episodes:
                self.q_table_snapshots.append((episode, dict(self.q_table)))
        
        return {
            'steps_per_episode': steps_per_episode,
            'final_q_table': dict(self.q_table),
            'snapshots': self.q_table_snapshots
        }
    
    def get_optimal_path(self) -> List[Tuple[int, int]]:
        path = [self.start_state]
        state = self.start_state
        visited = set()
        
        # Seguir la mejor acción según Q-table (sin exploración)
        while state != self.goal_state and len(path) < 100:
            if state in visited:  # Evitar bucles
                break
            visited.add(state)
            
            q_values = {action: self.q_table[(state, action)] for action in self.actions}
            best_action = max(q_values, key=q_values.get)
            state = self.get_next_state(state, best_action)
            path.append(state)
        
        return path
    
    def print_q_table(self, episode_num: int = None):
        print("\n" + "="*70)
        print(f"Q-TABLE - Episodio {episode_num}" if episode_num is not None else "Q-TABLE FINAL")
        print("="*70)
        
        for row in range(self.grid_size[0] - 1, -1, -1):
            for col in range(self.grid_size[1]):
                state = (row, col)
                print(f"\nEstado ({row},{col}):")
                for action in self.actions:
                    q_val = self.q_table[(state, action)]
                    print(f"  {action:>5}: {q_val:7.2f}")
    
    def print_optimal_policy(self):
        print("\n" + "="*50)
        print("POLÍTICA ÓPTIMA")
        print("="*50)
        
        # Símbolos para mostrar dirección de cada acción
        action_symbols = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→'
        }
        
        for row in range(self.grid_size[0] - 1, -1, -1):
            row_str = ""
            for col in range(self.grid_size[1]):
                state = (row, col)
                if state == self.goal_state:
                    row_str += " G "
                elif state == self.start_state:
                    row_str += " S "
                elif state == self.obstacle:
                    row_str += " X "
                else:
                    q_values = {action: self.q_table[(state, action)] for action in self.actions}
                    best_action = max(q_values, key=q_values.get)
                    row_str += f" {action_symbols[best_action]} "
            print(row_str)
        print()


def ejercicio_1a():
    print("\n" + "="*70)
    print("EJERCICIO 1.a - Q-learning con recompensa simple")
    print("="*70)
    
    env = QLearningGrid()
    env.alpha = 0.1
    env.gamma = 0.9
    env.epsilon = 0.1
    
    print(f"\nParámetros: alpha={env.alpha}, gamma={env.gamma}, epsilon={env.epsilon}")
    
    num_episodes = 1000
    print(f"Entrenando {num_episodes} episodios...")
    
    results = env.train(
        num_episodes=num_episodes,
        reward_type='simple',
        snapshot_episodes=[0, 100, 500, 999]
    )
    
    env.print_q_table()
    
    optimal_path = env.get_optimal_path()
    print(f"\nCamino óptimo: {optimal_path}")
    print(f"Longitud: {len(optimal_path) - 1} pasos")
    
    env.print_optimal_policy()
    
    steps = results['steps_per_episode']
    avg_last_100 = np.mean(steps[-100:])
    print(f"\nPromedio últimos 100 episodios: {avg_last_100:.2f} pasos")
    
    return env


def ejercicio_1b():
    print("\n\n" + "="*70)
    print("EJERCICIO 1.b - Q-learning con recompensa basada en distancia")
    print("="*70)
    
    env = QLearningGrid()
    # Alpha y gamma más altos porque la recompensa guía mejor
    env.alpha = 0.15
    env.gamma = 0.95
    env.epsilon = 0.1
    
    print(f"\nParámetros: alpha={env.alpha}, gamma={env.gamma}, epsilon={env.epsilon}")
    
    num_episodes = 500
    print(f"Entrenando {num_episodes} episodios...")
    
    results = env.train(
        num_episodes=num_episodes,
        reward_type='distance',
        snapshot_episodes=[0, 50, 250, 499]
    )
    
    env.print_q_table()
    
    optimal_path = env.get_optimal_path()
    print(f"\nCamino óptimo: {optimal_path}")
    print(f"Longitud: {len(optimal_path) - 1} pasos")
    
    env.print_optimal_policy()
    
    steps = results['steps_per_episode']
    avg_last_100 = np.mean(steps[-100:])
    print(f"\nPromedio últimos 100 episodios: {avg_last_100:.2f} pasos")
    
    return env


def ejercicio_1c():
    print("\n\n" + "="*70)
    print("EJERCICIO 1.c - Marinero Borracho (Entorno Estocástico)")
    print("="*70)
    
    env = QLearningGrid()
    env.alpha = 0.1
    env.gamma = 0.9
    env.epsilon = 0.15  # Más exploración para compensar el ruido
    
    print(f"\nParámetros: alpha={env.alpha}, gamma={env.gamma}, epsilon={env.epsilon}")
    
    num_episodes = 2000
    print(f"Entrenando {num_episodes} episodios...")
    
    results = env.train(
        num_episodes=num_episodes,
        reward_type='simple',
        stochastic=True,
        success_prob=0.99,
        snapshot_episodes=[0, 500, 1000, 1999]
    )
    
    env.print_q_table()
    
    optimal_path = env.get_optimal_path()
    print(f"\nCamino óptimo: {optimal_path}")
    
    env.print_optimal_policy()
    
    steps = results['steps_per_episode']
    avg_last_200 = np.mean(steps[-200:])
    print(f"\nPromedio últimos 200 episodios: {avg_last_200:.2f} pasos")
    
    return env


if __name__ == "__main__":
    env_1a = ejercicio_1a()
    env_1b = ejercicio_1b()
    env_1c = ejercicio_1c()
    
    print("\n" + "="*70)
    print("EJERCICIO 1 COMPLETADO")
    print("="*70)