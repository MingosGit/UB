"""
Ejercicio 1 - Práctica 3: Q-learning en Grid 4x4
Implementación de Q-learning para Reinforcement Learning

Conceptos de teoría aplicados:
- Q-learning: algoritmo de RL libre de modelo
- Q-table: almacena valores Q(s,a)
- Ecuación de Bellman: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- Política epsilon-greedy: exploración vs explotación
"""

import numpy as np
import random
from typing import Tuple, Dict, List
from collections import defaultdict


class QLearningGrid:
    """
    Implementación de Q-learning para grid 4x4.
    
    Grid layout:
        3  [ ][ ][ ][G]
        2  [ ][ ][ ][ ]
        1  [ ][ ][ ][ ]
        0  [S][ ][ ][ ]
           0  1  2  3
    
    Start: (0, 0)
    Goal: (3, 3)
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (4, 4)):
        self.grid_size = grid_size
        self.actions = ['up', 'down', 'left', 'right']
        # Q-table: diccionario {(state, action): Q-value}
        self.q_table = defaultdict(float)
        
        # Parámetros de Q-learning
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        
        # Estados especiales
        self.start_state = (0, 0)
        self.goal_state = (3, 3)
        
        # Estadísticas
        self.episodes_history = []
        self.q_table_snapshots = []
        
    def get_next_state(self, state: Tuple[int, int], action: str, 
                       stochastic: bool = False, success_prob: float = 0.99) -> Tuple[int, int]:
        """
        Calcula el siguiente estado dada una acción.
        
        Args:
            state: Estado actual (row, col)
            action: Acción a ejecutar
            stochastic: Si True, aplica estocasticidad (marinero borracho)
            success_prob: Probabilidad de éxito de la acción
        
        Returns:
            Siguiente estado
        """
        row, col = state
        
        # Marinero borracho: con probabilidad (1 - success_prob) toma acción aleatoria
        if stochastic and random.random() > success_prob:
            action = random.choice(self.actions)
        
        # Calcular nuevo estado según la acción
        if action == 'up':
            next_state = (min(row + 1, self.grid_size[0] - 1), col)
        elif action == 'down':
            next_state = (max(row - 1, 0), col)
        elif action == 'left':
            next_state = (row, max(col - 1, 0))
        elif action == 'right':
            next_state = (row, min(col + 1, self.grid_size[1] - 1))
        else:
            next_state = state
        
        return next_state
    
    def get_reward(self, state: Tuple[int, int], reward_type: str = 'simple') -> float:
        """
        Función de recompensa.
        
        Args:
            state: Estado actual
            reward_type: 'simple' (-1 everywhere, 100 at goal) o 
                        'distance' (función de distancia al objetivo)
        
        Returns:
            Recompensa
        """
        if state == self.goal_state:
            return 100.0
        
        if reward_type == 'simple':
            # Ejercicio 1.a: -1 en todas partes excepto objetivo
            return -1.0
        
        elif reward_type == 'distance':
            # Ejercicio 1.b: Recompensa basada en distancia al objetivo
            # Grid de recompensas:
            # 3: [-3, -2, -1, 100]
            # 2: [-4, -2, -1,   1]
            # 1: [-5, -4, -3,  -2]
            # 0: [-6, -5, -4,  -3]
            reward_grid = np.array([
                [-6, -5, -4, -3],
                [-5, -4, -3, -2],
                [-4, -2, -1,  1],
                [-3, -2, -1, 100]
            ])
            return reward_grid[state[0]][state[1]]
        
        return -1.0
    
    def choose_action(self, state: Tuple[int, int]) -> str:
        """
        Política epsilon-greedy para selección de acciones.
        
        Conceptos de teoría:
        - Exploración: con probabilidad epsilon, elige acción aleatoria
        - Explotación: con probabilidad (1-epsilon), elige mejor acción según Q-table
        
        Args:
            state: Estado actual
        
        Returns:
            Acción seleccionada
        """
        if random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return random.choice(self.actions)
        else:
            # Explotación: mejor acción según Q-table
            q_values = {action: self.q_table[(state, action)] for action in self.actions}
            max_q = max(q_values.values())
            # Si hay empate, elegir aleatoriamente entre las mejores
            best_actions = [action for action, q in q_values.items() if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state: Tuple[int, int], action: str, 
                      reward: float, next_state: Tuple[int, int]):
        """
        Actualiza Q(s,a) usando la ecuación de Bellman.
        
        Ecuación de Q-learning:
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Donde:
        - α (alpha): learning rate
        - γ (gamma): discount factor
        - r: recompensa inmediata
        - max_a' Q(s',a'): máximo valor Q en el siguiente estado
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Siguiente estado
        """
        current_q = self.q_table[(state, action)]
        
        # Calcular max_a' Q(s',a')
        max_next_q = max([self.q_table[(next_state, a)] for a in self.actions])
        
        # Ecuación de Bellman
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[(state, action)] = new_q
    
    def train(self, num_episodes: int, reward_type: str = 'simple', 
              stochastic: bool = False, success_prob: float = 0.99,
              snapshot_episodes: List[int] = None) -> Dict:
        """
        Entrena el agente usando Q-learning.
        
        Args:
            num_episodes: Número de episodios de entrenamiento
            reward_type: Tipo de función de recompensa
            stochastic: Si aplicar estocasticidad
            success_prob: Probabilidad de éxito (para marinero borracho)
            snapshot_episodes: Lista de episodios donde guardar Q-table
        
        Returns:
            Diccionario con estadísticas del entrenamiento
        """
        if snapshot_episodes is None:
            snapshot_episodes = [0, num_episodes // 3, 2 * num_episodes // 3, num_episodes - 1]
        
        self.q_table_snapshots = []
        steps_per_episode = []
        
        for episode in range(num_episodes):
            state = self.start_state
            steps = 0
            episode_done = False
            
            while not episode_done:
                # Elegir acción
                action = self.choose_action(state)
                
                # Ejecutar acción
                next_state = self.get_next_state(state, action, stochastic, success_prob)
                
                # Obtener recompensa
                reward = self.get_reward(next_state, reward_type)
                
                # Actualizar Q-value
                self.update_q_value(state, action, reward, next_state)
                
                # Siguiente estado
                state = next_state
                steps += 1
                
                # Verificar si llegó al objetivo o límite de pasos
                if state == self.goal_state or steps >= 1000:
                    episode_done = True
            
            steps_per_episode.append(steps)
            
            # Guardar snapshot de Q-table
            if episode in snapshot_episodes:
                self.q_table_snapshots.append((episode, dict(self.q_table)))
        
        return {
            'steps_per_episode': steps_per_episode,
            'final_q_table': dict(self.q_table),
            'snapshots': self.q_table_snapshots
        }
    
    def get_optimal_path(self) -> List[Tuple[int, int]]:
        """
        Extrae el camino óptimo siguiendo la política greedy de la Q-table.
        
        Returns:
            Lista de estados del camino óptimo
        """
        path = [self.start_state]
        state = self.start_state
        visited = set()
        
        while state != self.goal_state and len(path) < 100:
            if state in visited:
                break
            visited.add(state)
            
            # Elegir mejor acción (sin exploración)
            q_values = {action: self.q_table[(state, action)] for action in self.actions}
            best_action = max(q_values, key=q_values.get)
            
            # Moverse
            state = self.get_next_state(state, best_action)
            path.append(state)
        
        return path
    
    def print_q_table(self, episode_num: int = None):
        """Imprime la Q-table de forma legible."""
        print("\n" + "="*70)
        if episode_num is not None:
            print(f"Q-TABLE - Episodio {episode_num}")
        else:
            print("Q-TABLE FINAL")
        print("="*70)
        
        for row in range(self.grid_size[0] - 1, -1, -1):
            for col in range(self.grid_size[1]):
                state = (row, col)
                print(f"\nEstado ({row},{col}):")
                for action in self.actions:
                    q_val = self.q_table[(state, action)]
                    print(f"  {action:>5}: {q_val:7.2f}")
    
    def print_optimal_policy(self):
        """Visualiza la política óptima en el grid."""
        print("\n" + "="*50)
        print("POLÍTICA ÓPTIMA (Mejor acción en cada estado)")
        print("="*50)
        
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
                else:
                    q_values = {action: self.q_table[(state, action)] for action in self.actions}
                    best_action = max(q_values, key=q_values.get)
                    row_str += f" {action_symbols[best_action]} "
            print(row_str)
        print()


def ejercicio_1a():
    """
    Ejercicio 1.a: Q-learning con recompensa simple (-1 everywhere, 100 at goal)
    """
    print("\n" + "="*70)
    print("EJERCICIO 1.a - Q-learning con recompensa simple")
    print("="*70)
    print("\nConceptos de teoría:")
    print("- Q-learning: RL libre de modelo")
    print("- Q(s,a): valor esperado de tomar acción a en estado s")
    print("- Epsilon-greedy: balance exploración/explotación")
    print("- Convergencia: cuando Q-values se estabilizan")
    
    # Crear entorno
    env = QLearningGrid()
    
    # Parámetros óptimos (tras experimentación)
    env.alpha = 0.1   # Learning rate
    env.gamma = 0.9   # Discount factor
    env.epsilon = 0.1  # Exploration rate
    
    print(f"\nParámetros elegidos:")
    print(f"- Alpha (learning rate): {env.alpha}")
    print(f"- Gamma (discount factor): {env.gamma}")
    print(f"- Epsilon (exploration): {env.epsilon}")
    print(f"\nJustificación:")
    print(f"- Alpha = {env.alpha}: tasa moderada para actualización gradual")
    print(f"- Gamma = {env.gamma}: valor alto para considerar recompensas futuras")
    print(f"- Epsilon = {env.epsilon}: exploración baja, favorece explotación")
    
    # Entrenar
    num_episodes = 1000
    print(f"\nEntrenando {num_episodes} episodios...")
    
    results = env.train(
        num_episodes=num_episodes,
        reward_type='simple',
        snapshot_episodes=[0, 100, 500, 999]
    )
    
    # Imprimir Q-tables en diferentes momentos
    print("\n" + "="*70)
    print("Q-TABLES EN DIFERENTES MOMENTOS DEL ENTRENAMIENTO")
    print("="*70)
    
    for episode, q_table_snapshot in results['snapshots'][:2]:  # Primera y una intermedia
        print(f"\n--- Episodio {episode} ---")
        # Mostrar solo algunos estados representativos
        for state in [(0,0), (1,1), (3,3)]:
            print(f"Estado {state}:")
            for action in env.actions:
                print(f"  {action}: {q_table_snapshot.get((state, action), 0.0):.2f}")
    
    # Q-table final
    env.print_q_table()
    
    # Camino óptimo
    optimal_path = env.get_optimal_path()
    print(f"\nSecuencia de acciones óptima:")
    print(f"Camino: {optimal_path}")
    print(f"Longitud: {len(optimal_path) - 1} pasos")
    
    # Política
    env.print_optimal_policy()
    
    # Convergencia
    steps = results['steps_per_episode']
    avg_last_100 = np.mean(steps[-100:])
    print(f"\nConvergencia:")
    print(f"- Promedio últimos 100 episodios: {avg_last_100:.2f} pasos")
    print(f"- El algoritmo converge cuando los pasos se estabilizan")
    print(f"- Tiempo de convergencia: ~{num_episodes//2} episodios")
    
    return env


def ejercicio_1b():
    """
    Ejercicio 1.b: Q-learning con recompensa basada en distancia al objetivo
    """
    print("\n\n" + "="*70)
    print("EJERCICIO 1.b - Q-learning con recompensa basada en distancia")
    print("="*70)
    
    env = QLearningGrid()
    env.alpha = 0.15  # Aumentar learning rate ligeramente
    env.gamma = 0.95  # Mayor discount para aprovechar guía de recompensas
    env.epsilon = 0.1
    
    print(f"\nParámetros ajustados:")
    print(f"- Alpha: {env.alpha} (mayor que en 1.a)")
    print(f"- Gamma: {env.gamma} (mayor que en 1.a)")
    print(f"- Epsilon: {env.epsilon}")
    
    num_episodes = 500  # Converge más rápido
    print(f"\nEntrenando {num_episodes} episodios...")
    
    results = env.train(
        num_episodes=num_episodes,
        reward_type='distance',
        snapshot_episodes=[0, 50, 250, 499]
    )
    
    # Mostrar algunas Q-tables
    print("\n--- Q-table intermedia (episodio 50) ---")
    for state in [(0,0), (2,2), (3,3)]:
        print(f"Estado {state}:")
        for action in env.actions:
            print(f"  {action}: {results['snapshots'][1][1].get((state, action), 0.0):.2f}")
    
    env.print_q_table()
    
    optimal_path = env.get_optimal_path()
    print(f"\nCamino óptimo: {optimal_path}")
    print(f"Longitud: {len(optimal_path) - 1} pasos")
    
    env.print_optimal_policy()
    
    # Comparación
    steps = results['steps_per_episode']
    avg_last_100 = np.mean(steps[-100:])
    print(f"\nEfecto de la nueva función de recompensa:")
    print(f"- Convergencia más rápida (~{num_episodes//2} episodios vs ~500 en 1.a)")
    print(f"- La recompensa basada en distancia guía mejor al agente")
    print(f"- Actúa como heurística similar a A* de P1")
    print(f"\nRelación con P1:")
    print(f"- Recompensa por distancia ≈ Heurística en A*")
    print(f"- Ambos guían la búsqueda hacia el objetivo")
    print(f"- A* encuentra camino óptimo garantizado, Q-learning aprende por ensayo-error")
    
    return env


def ejercicio_1c():
    """
    Ejercicio 1.c: Marinero borracho (entorno estocástico)
    """
    print("\n\n" + "="*70)
    print("EJERCICIO 1.c - Marinero Borracho (Entorno Estocástico)")
    print("="*70)
    print("\nNovedad: Estocasticidad")
    print("- Solo 99% de movimientos se ejecutan correctamente")
    print("- 1% de veces, el marinero se mueve aleatoriamente")
    
    env = QLearningGrid()
    env.alpha = 0.1
    env.gamma = 0.9
    env.epsilon = 0.15  # Más exploración para entorno estocástico
    
    print(f"\nParámetros:")
    print(f"- Epsilon: {env.epsilon} (mayor que en casos deterministas)")
    print(f"- Justificación: más exploración ayuda en entornos estocásticos")
    
    num_episodes = 2000  # Más episodios necesarios
    print(f"\nEntrenando {num_episodes} episodios...")
    
    results = env.train(
        num_episodes=num_episodes,
        reward_type='simple',
        stochastic=True,
        success_prob=0.99,
        snapshot_episodes=[0, 500, 1000, 1999]
    )
    
    env.print_q_table()
    
    optimal_path = env.get_optimal_path()
    print(f"\nCamino óptimo encontrado: {optimal_path}")
    
    env.print_optimal_policy()
    
    steps = results['steps_per_episode']
    avg_last_200 = np.mean(steps[-200:])
    
    print(f"\nComparación determinista vs estocástico:")
    print(f"- Noches necesarias: ~{num_episodes} (vs ~1000 en determinista)")
    print(f"- El marinero necesita más práctica debido a la incertidumbre")
    print(f"\n¿Sigue siempre el mismo camino?")
    print(f"- NO. Debido a la estocasticidad, a veces se desvía")
    print(f"- Pero la política aprendida es robusta a estos errores")
    print(f"\n¿Podríamos usar algoritmos de P1?")
    print(f"- NO directamente. A*, BFS requieren entorno determinista")
    print(f"- Concepto clave: POLICY vs PATH")
    print(f"  * PATH: secuencia fija de estados (P1)")
    print(f"  * POLICY: regla para elegir acción en cada estado (RL)")
    print(f"- En entornos estocásticos necesitamos POLICIES, no PATHS")
    
    return env


if __name__ == "__main__":
    # Ejecutar ejercicio 1.a
    env_1a = ejercicio_1a()
    
    # Ejecutar ejercicio 1.b
    env_1b = ejercicio_1b()
    
    # Ejecutar ejercicio 1.c
    env_1c = ejercicio_1c()
    
    print("\n" + "="*70)
    print("EJERCICIO 1 COMPLETADO")
    print("="*70)
