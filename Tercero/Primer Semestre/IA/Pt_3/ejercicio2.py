"""
Ejercicio 2 - Práctica 3: Q-learning aplicado al ajedrez
Q-learning para Rey + Torre blancas vs Rey negro

IMPORTANTE: Rey negro es ESTÁTICO (no se mueve nunca)

Conceptos de teoría aplicados:
- Q-learning en espacio de estados complejo
- Representación de estados en ajedrez
- Función de recompensa para jaque mate
- Convergencia en espacios grandes
"""

import numpy as np
import random
import sys
import os
from typing import Tuple, Dict, List
from collections import defaultdict

# Añadir directorio chess al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'chess'))

import chess
import board


class QLearningChess:
    """
    Implementación de Q-learning para ajedrez (K+R vs K).
    
    Conceptos aplicados:
    - Estado: posiciones de Rey blanco, Torre blanca (Rey negro ESTÁTICO)
    - Acciones: movimientos válidos de piezas blancas
    - Recompensa: -1 por movimiento, 100 por jaque mate
    - Q-learning para aprender política óptima de mate
    - IMPORTANTE: Rey negro NO se mueve (posición estática)
    """
    
    def __init__(self, black_king_pos: Tuple[int, int]):
        # Parámetros de Q-learning
        self.alpha = 0.3   # Learning rate (aumentado para aprendizaje más rápido)
        self.gamma = 0.99  # Discount factor (muy alto para planificación a largo plazo)
        self.epsilon = 0.3 # Exploration rate (aumentado para mejor exploración)
        
        # Posición ESTÁTICA del rey negro
        self.black_king_pos = black_king_pos
        
        # Q-table: {(state_string, action_string): Q-value}
        self.q_table = defaultdict(float)
        
        # Estadísticas
        self.q_table_snapshots = []
        self.episodes_history = []
        self.mates_found = 0
        
    def state_to_string(self, white_state: List[List[int]]) -> str:
        """
        Convierte estado de piezas blancas a string para usar como key.
        
        Args:
            white_state: Lista [[row, col, piece_type], ...] para piezas blancas
        
        Returns:
            String representando el estado
        """
        # Obtener rey blanco (tipo 6)
        wk = [p for p in white_state if p[2] == 6][0] if any(p[2] == 6 for p in white_state) else None
        # Obtener torre blanca (tipo 2)
        wr = [p for p in white_state if p[2] == 2][0] if any(p[2] == 2 for p in white_state) else None
        
        state_str = f"{wk[0]},{wk[1]}" if wk else ""
        if wr:
            state_str += f",{wr[0]},{wr[1]}"
        
        return state_str
    
    def action_to_string(self, piece_state: List[int], next_pos: List[int]) -> str:
        """
        Convierte una acción a string.
        
        Args:
            piece_state: Estado de la pieza a mover
            next_pos: Siguiente posición
        
        Returns:
            String de la acción
        """
        return f"{piece_state[0]},{piece_state[1]}->{next_pos[0]},{next_pos[1]}"
    
    def is_checkmate(self, white_state: List[List[int]]) -> bool:
        """
        Verifica si hay jaque mate.
        Rey negro es ESTÁTICO - solo verificamos si está en jaque y sin escapes.
        
        Args:
            white_state: Estado de piezas blancas
        
        Returns:
            True si hay jaque mate
        """
        # Usar posición estática del rey negro
        black_king = self.black_king_pos
        
        # Verificar si está en jaque
        def is_square_attacked(row, col, white_pieces):
            # Obtener rey y torre blanca
            wk = [p for p in white_pieces if p[2] == 6][0] if any(p[2] == 6 for p in white_pieces) else None
            wr = [p for p in white_pieces if p[2] == 2][0] if any(p[2] == 2 for p in white_pieces) else None
            
            # Verificar ataque de torre (líneas rectas)
            if wr:
                # Torre en misma fila
                if wr[0] == row:
                    # Verificar que no hay piezas bloqueando
                    min_col = min(wr[1], col)
                    max_col = max(wr[1], col)
                    blocked = False
                    for c in range(min_col + 1, max_col):
                        if wk and wk[0] == row and wk[1] == c:
                            blocked = True
                            break
                    if not blocked:
                        return True
                
                # Torre en misma columna
                if wr[1] == col:
                    min_row = min(wr[0], row)
                    max_row = max(wr[0], row)
                    blocked = False
                    for r in range(min_row + 1, max_row):
                        if wk and wk[0] == r and wk[1] == col:
                            blocked = True
                            break
                    if not blocked:
                        return True
            
            # Verificar adyacencia al rey blanco
            if wk:
                if abs(wk[0] - row) <= 1 and abs(wk[1] - col) <= 1:
                    return True
            
            return False
        
        # Si no está en jaque, no es mate
        if not is_square_attacked(black_king[0], black_king[1], white_state):
            return False
        
        # Verificar si tiene movimientos de escape
        # (aunque el rey NO se mueve, debemos verificar que NO TENGA salidas posibles)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row = black_king[0] + dr
                new_col = black_king[1] + dc
                
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    # Verificar si hay pieza blanca en esa casilla
                    occupied_by_white = False
                    for wp in white_state:
                        if wp[0] == new_row and wp[1] == new_col:
                            occupied_by_white = True
                            break
                    
                    if occupied_by_white:
                        continue
                    
                    # Verificar si esa casilla estaría atacada
                    if not is_square_attacked(new_row, new_col, white_state):
                        return False
        
        return True
    
    def get_possible_actions(self, white_state: List[List[int]], black_king_pos: Tuple[int, int]) -> List[Tuple[List[int], List[int]]]:
        """
        Obtiene acciones posibles para las blancas.
        Recrea el tablero en cada llamada para asegurar estado correcto.
        
        Args:
            white_state: Estado actual de piezas blancas
            black_king_pos: Posición del rey negro
        
        Returns:
            Lista de tuplas (pieza_origen, posición_destino)
        """
        # Recrear tablero con estado actual
        board_array = np.zeros((8, 8))
        board_array[black_king_pos[0]][black_king_pos[1]] = 12
        for piece in white_state:
            board_array[piece[0]][piece[1]] = piece[2]
        
        board_obj = board.Board(board_array, False)
        board_obj.getListNextStatesW(white_state)
        next_states = board_obj.listNextStates
        
        actions = []
        for next_state in next_states:
            # Encontrar qué pieza se movió comparando estados
            moved_piece = None
            new_position = None
            
            for orig_piece in white_state:
                found = False
                for new_piece in next_state:
                    if (orig_piece[0] == new_piece[0] and 
                        orig_piece[1] == new_piece[1] and 
                        orig_piece[2] == new_piece[2]):
                        found = True
                        break
                
                if not found:
                    moved_piece = orig_piece
                    # Buscar nueva posición de esta pieza
                    for new_piece in next_state:
                        if new_piece[2] == orig_piece[2]:
                            # Verificar que no es otra pieza del mismo tipo
                            is_other_piece = False
                            for other in white_state:
                                if (other != orig_piece and 
                                    other[0] == new_piece[0] and 
                                    other[1] == new_piece[1] and 
                                    other[2] == new_piece[2]):
                                    is_other_piece = True
                                    break
                            
                            if not is_other_piece:
                                new_position = new_piece
                                break
                    break
            
            if moved_piece and new_position:
                actions.append((moved_piece, new_position))
        
        return actions
    
    def choose_action(self, state: List[List[int]], state_str: str) -> Tuple[List[int], List[int]]:
        """
        Política epsilon-greedy para elegir acción.
        
        Args:
            state: Estado actual
            state_str: String del estado
        
        Returns:
            Tupla (pieza_origen, posición_destino)
        """
        possible_actions = self.get_possible_actions(state, self.black_king_pos)
        
        if not possible_actions:
            return None
        
        if random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return random.choice(possible_actions)
        else:
            # Explotación: mejor acción según Q-table
            best_action = None
            max_q = -float('inf')
            
            for action in possible_actions:
                action_str = self.action_to_string(action[0], action[1])
                q_val = self.q_table[(state_str, action_str)]
                if q_val > max_q:
                    max_q = q_val
                    best_action = action
            
            return best_action if best_action else random.choice(possible_actions)
    
    def execute_action(self, white_state: List[List[int]], action: Tuple[List[int], List[int]]) -> List[List[int]]:
        """
        Ejecuta una acción (mueve una pieza blanca).
        Rey negro NO se mueve - permanece estático.
        
        Args:
            white_state: Estado actual de piezas blancas
            action: Tupla (pieza_origen, posición_destino)
        
        Returns:
            Nuevo estado de piezas blancas
        """
        new_state = [p.copy() for p in white_state]
        
        # Encontrar pieza que se mueve
        for i, piece in enumerate(new_state):
            if piece[0] == action[0][0] and piece[1] == action[0][1] and piece[2] == action[0][2]:
                # Mover a nueva posición
                new_state[i] = [action[1][0], action[1][1], piece[2]]
                break
        
        return new_state
    
    def get_reward(self, white_state: List[List[int]], reward_type: str = 'simple') -> float:
        """
        Función de recompensa.
        
        Args:
            white_state: Estado de piezas blancas
            reward_type: 'simple' o 'heuristic'
        
        Returns:
            Recompensa
        """
        if self.is_checkmate(white_state):
            return 100.0
        
        if reward_type == 'simple':
            # Ejercicio 2.a: -1 por cada movimiento
            return -1.0
        
        elif reward_type == 'heuristic':
            # Ejercicio 2.b: Recompensa basada en heurística mejorada
            bk_pos = self.black_king_pos
            
            # Buscar rey y torre blanca
            wk = [p for p in white_state if p[2] == 6][0] if any(p[2] == 6 for p in white_state) else None
            wr = [p for p in white_state if p[2] == 2][0] if any(p[2] == 2 for p in white_state) else None
            
            if not wk or not wr:
                return -10.0
            
            reward = -1.0  # Penalización base por movimiento
            
            # 1. Rey blanco cerca del rey negro (distancia Chebyshev)
            wk_dist_cheb = max(abs(wk[0] - bk_pos[0]), abs(wk[1] - bk_pos[1]))
            reward += (7 - wk_dist_cheb) * 0.5  # Bonus por estar cerca
            
            # 2. Torre alineada con rey negro (crítico para jaque)
            if wr[0] == bk_pos[0] or wr[1] == bk_pos[1]:
                reward += 3.0
                
                # 3. Si torre da jaque, bonus extra
                def is_in_check():
                    if wr[0] == bk_pos[0]:
                        min_c, max_c = min(wr[1], bk_pos[1]), max(wr[1], bk_pos[1])
                        for c in range(min_c + 1, max_c):
                            if wk[0] == wr[0] and wk[1] == c:
                                return False
                        return True
                    if wr[1] == bk_pos[1]:
                        min_r, max_r = min(wr[0], bk_pos[0]), max(wr[0], bk_pos[0])
                        for r in range(min_r + 1, max_r):
                            if wk[0] == r and wk[1] == wr[1]:
                                return False
                        return True
                    return False
                
                if is_in_check():
                    reward += 5.0
            
            # 4. Rey blanco controlando casillas cerca del rey negro
            king_control = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = bk_pos[0] + dr, bk_pos[1] + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        if abs(wk[0] - nr) <= 1 and abs(wk[1] - nc) <= 1:
                            king_control += 0.5
            
            reward += king_control
            
            return reward
        
        return -1.0
    
    def update_q_value(self, state_str: str, action_str: str, reward: float, 
                      next_state_str: str, possible_next_actions: List[str]):
        """
        Actualiza Q(s,a) usando ecuación de Bellman.
        
        Args:
            state_str: Estado actual como string
            action_str: Acción tomada como string
            reward: Recompensa recibida
            next_state_str: Siguiente estado como string
            possible_next_actions: Acciones posibles desde siguiente estado
        """
        current_q = self.q_table[(state_str, action_str)]
        
        # Calcular max Q del siguiente estado
        max_next_q = -float('inf')
        if possible_next_actions:
            for next_action_str in possible_next_actions:
                q_val = self.q_table[(next_state_str, next_action_str)]
                if q_val > max_next_q:
                    max_next_q = q_val
        else:
            max_next_q = 0.0
        
        # Ecuación de Bellman
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state_str, action_str)] = new_q
    
    def train(self, initial_white_state: List[List[int]], num_episodes: int, reward_type: str = 'simple',
              snapshot_episodes: List[int] = None) -> Dict:
        """
        Entrena el agente con Q-learning.
        Rey negro permanece ESTÁTICO durante todo el entrenamiento.
        
        Args:
            initial_white_state: Estado inicial de piezas blancas
            num_episodes: Número de episodios
            reward_type: Tipo de recompensa ('simple' o 'heuristic')
            snapshot_episodes: Episodios donde guardar Q-table
        
        Returns:
            Diccionario con estadísticas
        """
        if snapshot_episodes is None:
            snapshot_episodes = [0, num_episodes // 3, 2 * num_episodes // 3, num_episodes - 1]
        
        self.q_table_snapshots = []
        steps_per_episode = []
        self.mates_found = 0
        
        for episode in range(num_episodes):
            # Decaimiento de epsilon (exploración -> explotación)
            self.epsilon = max(0.1, 0.3 - (episode / num_episodes) * 0.2)
            
            # Resetear estado inicial
            state = [p.copy() for p in initial_white_state]
            state_str = self.state_to_string(state)
            
            steps = 0
            max_steps = 100  # Aumentado para permitir más exploración
            episode_done = False
            
            while not episode_done and steps < max_steps:
                # Elegir acción
                action = self.choose_action(state, state_str)
                
                if action is None:
                    break
                
                action_str = self.action_to_string(action[0], action[1])
                
                # Ejecutar acción (solo mueve piezas blancas)
                next_state = self.execute_action(state, action)
                next_state_str = self.state_to_string(next_state)
                
                # Verificar jaque mate
                is_mate = self.is_checkmate(next_state)
                
                # Obtener recompensa
                reward = self.get_reward(next_state, reward_type)
                
                # Obtener posibles acciones del siguiente estado
                possible_next_actions = self.get_possible_actions(next_state, self.black_king_pos)
                next_action_strs = [self.action_to_string(a[0], a[1]) for a in possible_next_actions]
                
                # Actualizar Q-value
                self.update_q_value(state_str, action_str, reward, next_state_str, next_action_strs)
                
                # Siguiente estado
                state = next_state
                state_str = next_state_str
                steps += 1
                
                # Verificar jaque mate
                if is_mate:
                    episode_done = True
                    self.mates_found += 1
            
            steps_per_episode.append(steps)
            
            # Guardar snapshot
            if episode in snapshot_episodes:
                self.q_table_snapshots.append((episode, dict(self.q_table)))
            
            # Mostrar progreso
            if (episode + 1) % 500 == 0:
                avg_steps = np.mean(steps_per_episode[-500:])
                mates_last_500 = sum(1 for s in steps_per_episode[-500:] if s < max_steps)
                print(f"Episodio {episode + 1}/{num_episodes} - Avg steps: {avg_steps:.2f} - Mates: {mates_last_500}/500")
        
        return {
            'steps_per_episode': steps_per_episode,
            'final_q_table': dict(self.q_table),
            'snapshots': self.q_table_snapshots,
            'mates_found': self.mates_found
        }
    
    def get_optimal_sequence(self, initial_white_state: List[List[int]], max_moves: int = 50) -> List:
        """
        Extrae secuencia óptima de movimientos usando política greedy.
        
        Args:
            initial_white_state: Estado inicial de piezas blancas
            max_moves: Máximo número de movimientos
        
        Returns:
            Lista de estados en la secuencia
        """
        sequence = []
        state = [p.copy() for p in initial_white_state]
        
        for _ in range(max_moves):
            state_str = self.state_to_string(state)
            sequence.append(state.copy())
            
            # Verificar mate
            if self.is_checkmate(state):
                print(f"¡Jaque mate alcanzado en {len(sequence)} movimientos!")
                break
            
            # Mejor acción (sin exploración)
            possible_actions = self.get_possible_actions(state, self.black_king_pos)
            if not possible_actions:
                break
            
            best_action = None
            max_q = -float('inf')
            
            for action in possible_actions:
                action_str = self.action_to_string(action[0], action[1])
                q_val = self.q_table[(state_str, action_str)]
                if q_val > max_q:
                    max_q = q_val
                    best_action = action
            
            if best_action is None:
                break
            
            # Ejecutar mejor acción
            state = self.execute_action(state, best_action)
        
        return sequence
    
    def print_q_table_sample(self, num_samples: int = 3):
        """Imprime muestra de la Q-table."""
        print("\n" + "="*70)
        print(f"MUESTRA DE Q-TABLE (primeros {num_samples} estados)")
        print("="*70)
        
        states_printed = 0
        current_state = None
        
        for (state, action), q_val in list(self.q_table.items())[:num_samples * 5]:
            if state != current_state:
                if states_printed >= num_samples:
                    break
                print(f"\nEstado: {state}")
                current_state = state
                states_printed += 1
            print(f"  Acción {action}: Q = {q_val:.3f}")


def ejercicio_2a():
    """
    Ejercicio 2.a: Q-learning en ajedrez con recompensa simple
    """
    print("\n" + "="*70)
    print("EJERCICIO 2.a - Q-learning en Ajedrez (Recompensa Simple)")
    print("="*70)
    print("\nConceptos aplicados:")
    print("- Q-learning en espacio de estados complejo")
    print("- Estado: posiciones de K blanco, R blanca (K negro ESTÁTICO)")
    print("- Recompensa: -1 por movimiento, 100 por jaque mate")
    print("- Objetivo: aprender secuencia de mate")
    print("- IMPORTANTE: Rey negro NO se mueve (posición fija)")
    
    # Configuración inicial (igual que P1)
    TA = np.zeros((8, 8))
    TA[7][0] = 2   # Torre blanca
    TA[7][4] = 6   # Rey blanco
    TA[0][4] = 12  # Rey negro (ESTÁTICO)
    
    print("\nConfiguración inicial del tablero:")
    temp_chess = chess.Chess(TA.copy(), True)
    temp_chess.board.print_board()
    
    # Posición estática del rey negro
    black_king_pos = (0, 4)
    
    # Estado inicial de piezas blancas
    initial_white_state = [
        [7, 4, 6],  # Rey blanco
        [7, 0, 2]   # Torre blanca
    ]
    
    # Crear agente con rey negro estático
    agent = QLearningChess(black_king_pos)
    
    print(f"\nParámetros de Q-learning:")
    print(f"- Alpha (learning rate): {agent.alpha}")
    print(f"- Gamma (discount factor): {agent.gamma}")
    print(f"- Epsilon inicial (exploration): {agent.epsilon}")
    print(f"\nJustificación:")
    print(f"- Alpha = {agent.alpha}: actualización rápida para convergencia")
    print(f"- Gamma = {agent.gamma}: muy alto para recompensas futuras (mate lejano)")
    print(f"- Epsilon = {agent.epsilon}: exploración adaptativa (decae con episodios)")
    
    # Entrenar
    num_episodes = 5000
    print(f"\nEntrenando {num_episodes} episodios con epsilon decreciente...")
    print("(Esto puede tardar 1-2 minutos)")
    
    results = agent.train(
        initial_white_state=initial_white_state,
        num_episodes=num_episodes,
        reward_type='simple',
        snapshot_episodes=[0, 1000, 2500, 4999]
    )
    
    # Mostrar estadísticas
    print("\n" + "="*70)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("="*70)
    print(f"Total de mates encontrados: {results['mates_found']}/{num_episodes}")
    
    # Convergencia
    steps = results['steps_per_episode']
    avg_last_500 = np.mean(steps[-500:])
    min_steps = min(steps[-500:])
    
    print(f"\nConvergencia:")
    print(f"- Promedio pasos últimos 500 episodios: {avg_last_500:.2f}")
    print(f"- Mínimo de pasos alcanzado: {min_steps}")
    
    # Q-table sample
    print("\n" + "="*70)
    print("MUESTRA DE Q-TABLE")
    print("="*70)
    agent.print_q_table_sample(num_samples=2)
    
    # Secuencia óptima
    print("\n" + "="*70)
    print("SECUENCIA ÓPTIMA DE MOVIMIENTOS (Política Greedy)")
    print("="*70)
    
    sequence = agent.get_optimal_sequence(initial_white_state, max_moves=30)
    
    if len(sequence) > 0:
        print(f"\nSecuencia de {len(sequence)} movimientos:")
        for i, state in enumerate(sequence):
            wk = [p for p in state if p[2] == 6][0]
            wr = [p for p in state if p[2] == 2][0]
            print(f"  Mov {i}: Rey({wk[0]},{wk[1]}) Torre({wr[0]},{wr[1]})")
            if agent.is_checkmate(state):
                print(f"  >>> ¡JAQUE MATE EN {i} MOVIMIENTOS! <<<")
                break
            if i >= 14:  # Mostrar solo primeros 15
                print(f"  ... (continúa)")
                break
    
    return agent


def ejercicio_2b():
    """
    Ejercicio 2.b: Q-learning con recompensa heurística
    """
    print("\n\n" + "="*70)
    print("EJERCICIO 2.b - Q-learning con Recompensa Heurística")
    print("="*70)
    print("\nNovedad: Función de recompensa basada en heurística mejorada")
    print("- Proximidad del rey blanco al rey negro (distancia Chebyshev)")
    print("- Torre alineada con rey negro (horizontal/vertical)")
    print("- Bonificación por dar jaque")
    print("- Rey blanco controlando casillas de escape")
    
    # Configuración inicial
    black_king_pos = (0, 4)
    initial_white_state = [
        [7, 4, 6],  # Rey blanco
        [7, 0, 2]   # Torre blanca
    ]
    
    # Crear agente con rey negro estático
    agent = QLearningChess(black_king_pos)
    agent.alpha = 0.3
    agent.gamma = 0.99
    
    print(f"\nParámetros optimizados:")
    print(f"- Alpha: {agent.alpha} (aprendizaje rápido)")
    print(f"- Gamma: {agent.gamma} (planificación a largo plazo)")
    print(f"- Epsilon: {agent.epsilon} inicial con decaimiento")
    print(f"- Razón: la heurística guía hacia configuraciones de mate")
    
    # Entrenar
    num_episodes = 3000
    print(f"\nEntrenando {num_episodes} episodios con recompensa heurística...")
    print("(La heurística acelera significativamente el aprendizaje)")
    
    results = agent.train(
        initial_white_state=initial_white_state,
        num_episodes=num_episodes,
        reward_type='heuristic',
        snapshot_episodes=[0, 500, 1500, 2999]
    )
    
    # Resultados
    print("\n" + "="*70)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("="*70)
    print(f"Total de mates encontrados: {results['mates_found']}/{num_episodes}")
    
    # Convergencia
    steps = results['steps_per_episode']
    avg_last_500 = np.mean(steps[-500:])
    min_steps = min(steps[-500:])
    
    print(f"\nConvergencia:")
    print(f"- Promedio pasos últimos 500 episodios: {avg_last_500:.2f}")
    print(f"- Mínimo de pasos alcanzado: {min_steps}")
    
    # Muestra de Q-table
    print("\n" + "="*70)
    print("MUESTRA DE Q-TABLE")
    print("="*70)
    agent.print_q_table_sample(num_samples=2)
    
    # Secuencia óptima
    print("\n" + "="*70)
    print("SECUENCIA ÓPTIMA (Política Greedy)")
    print("="*70)
    
    sequence = agent.get_optimal_sequence(initial_white_state, max_moves=30)
    
    if len(sequence) > 0:
        print(f"\nSecuencia de {len(sequence)} movimientos:")
        for i, state in enumerate(sequence):
            wk = [p for p in state if p[2] == 6][0]
            wr = [p for p in state if p[2] == 2][0]
            print(f"  Mov {i}: Rey({wk[0]},{wk[1]}) Torre({wr[0]},{wr[1]})")
            if agent.is_checkmate(state):
                print(f"  >>> ¡JAQUE MATE EN {i} MOVIMIENTOS! <<<")
                break
            if i >= 14:
                print(f"  ... (continúa)")
                break
    
    print(f"\nComparación con ejercicio 2.a:")
    print(f"- La recompensa heurística acelera el aprendizaje")
    print(f"- Requiere menos episodios para encontrar soluciones óptimas")
    print(f"- Guía al agente hacia configuraciones ganadoras más rápidamente")
    
    return agent


if __name__ == "__main__":
    # Ejecutar ejercicio 2.a
    agent_2a = ejercicio_2a()
    
    # Ejecutar ejercicio 2.b
    agent_2b = ejercicio_2b()
    
    print("\n" + "="*70)
    print("EJERCICIOS 2.a y 2.b COMPLETADOS")
    print("="*70)
