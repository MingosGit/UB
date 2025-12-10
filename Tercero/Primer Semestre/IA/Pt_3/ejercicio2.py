"""
Ejercicio 2 - Práctica 3: Q-learning aplicado al ajedrez
Q-learning para Rey + Torre blancas vs Rey negro

2.a y 2.b: Rey negro ESTÁTICO (no se mueve nunca)
2.c: Rey negro MÓVIL (se mueve para escapar del mate)

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
import piece


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
            # Ejercicio 2.b: Heurística como GUÍA, no recompensa dominante
            bk_pos = self.black_king_pos
            
            # Buscar rey y torre blanca
            wk = [p for p in white_state if p[2] == 6][0] if any(p[2] == 6 for p in white_state) else None
            wr = [p for p in white_state if p[2] == 2][0] if any(p[2] == 2 for p in white_state) else None
            
            if not wk or not wr:
                return -50.0
            
            # Penalización base IGUAL que simple (mantiene escala correcta)
            reward = -1.0
            
            # COMPONENTE 1: Proximidad del rey blanco (bonificación PEQUEÑA)
            # La heurística REDUCE la penalización, NO la elimina
            king_dist = max(abs(wk[0] - bk_pos[0]), abs(wk[1] - bk_pos[1]))
            if king_dist == 2:
                reward += 0.4  # Óptimo: distancia de mate
            elif king_dist == 1:
                reward += 0.3  # Oposición directa
            elif king_dist == 3:
                reward += 0.2  # Cerca
            elif king_dist <= 4:
                reward += 0.1  # Acercándose
            # Sin penalización extra por lejos (ya tenemos -1)
            
            # COMPONENTE 2: Torre atacando (bonificación MODERADA)
            def tower_attacks_king():
                if wr[0] == bk_pos[0]:  # Misma fila
                    min_c, max_c = min(wr[1], bk_pos[1]), max(wr[1], bk_pos[1])
                    for c in range(min_c + 1, max_c):
                        if wk[0] == wr[0] and wk[1] == c:
                            return False
                    return True
                if wr[1] == bk_pos[1]:  # Misma columna
                    min_r, max_r = min(wr[0], bk_pos[0]), max(wr[0], bk_pos[0])
                    for r in range(min_r + 1, max_r):
                        if wk[0] == r and wk[1] == wr[1]:
                            return False
                    return True
                return False
            
            if tower_attacks_king():
                reward += 0.5  # Bonus por jaque (señal fuerte)
            elif wr[0] == bk_pos[0] or wr[1] == bk_pos[1]:
                reward += 0.2  # Bonus por alineación
            
            # COMPONENTE 3: Control de casillas de escape (bonificación PEQUEÑA)
            escape_squares_controlled = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    er, ec = bk_pos[0] + dr, bk_pos[1] + dc
                    if 0 <= er < 8 and 0 <= ec < 8:
                        # ¿Rey blanco controla esta casilla?
                        if abs(wk[0] - er) <= 1 and abs(wk[1] - ec) <= 1:
                            escape_squares_controlled += 1
                        # ¿Torre controla esta casilla?
                        elif wr[0] == er or wr[1] == ec:
                            # Verificar si no está bloqueada
                            if wr[0] == er:
                                min_c, max_c = min(wr[1], ec), max(wr[1], ec)
                                blocked = any(wk[0] == er and wk[1] == c for c in range(min_c + 1, max_c))
                                if not blocked:
                                    escape_squares_controlled += 0.5
                            elif wr[1] == ec:
                                min_r, max_r = min(wr[0], er), max(wr[0], er)
                                blocked = any(wk[0] == r and wk[1] == ec for r in range(min_r + 1, max_r))
                                if not blocked:
                                    escape_squares_controlled += 0.5
            
            reward += escape_squares_controlled * 0.05  # Bonus pequeño por control
            
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
        
        for move_num in range(max_moves):
            state_str = self.state_to_string(state)
            sequence.append(state.copy())
            
            # Verificar mate
            if self.is_checkmate(state):
                # len(sequence)-1 porque incluimos el estado inicial
                actual_moves = len(sequence) - 1
                print(f"¡Jaque mate encontrado! Secuencia de {len(sequence)} estados ({actual_moves} movimientos)")
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
    
    print(f"\nParámetros: α={agent.alpha}, γ={agent.gamma}, ε={agent.epsilon}")
    print("Justificación: α alto→convergencia rápida | γ muy alto→planificación largo plazo | ε→decae")
    
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
        # La secuencia contiene ESTADOS (incluyendo inicial), los MOVIMIENTOS son len-1
        num_states = len(sequence)
        mate_step = None
        
        print(f"\nSecuencia de {num_states} estados (Estado inicial + {num_states-1} movimientos):")
        for i, state in enumerate(sequence):
            wk = [p for p in state if p[2] == 6][0]
            wr = [p for p in state if p[2] == 2][0]
            print(f"  Estado {i}: Rey({wk[0]},{wk[1]}) Torre({wr[0]},{wr[1]})")
            if agent.is_checkmate(state):
                mate_step = i
                print(f"  >>> ¡JAQUE MATE! (alcanzado en {i} movimientos desde el estado inicial) <<<")
                break
            if i >= 14:  # Mostrar solo primeros 15
                print(f"  ... (continúa hasta estado {num_states-1})")
                break
    
    return agent, results


def ejercicio_2b(results_2a=None):
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
    agent.alpha = 0.3  # Mismo que 2.a - evita inestabilidad con heurística
    agent.gamma = 0.95  # Ligeramente menor - la heurística da señal inmediata
    
    print(f"\nParámetros: α={agent.alpha}, γ={agent.gamma}, ε={agent.epsilon} (con decaimiento)")
    print("Justificación: α igual que 2.a | γ menor→señal heurística inmediata | estabilidad en aprendizaje")
    
    # Entrenar CON MÁS EPISODIOS para demostrar convergencia
    num_episodes = 5000  # REDUCIDO: heurística no necesita más episodios
    print(f"\nEntrenando {num_episodes} episodios con recompensa heurística...")
    print("(La heurística debería acelerar el aprendizaje inicial)")
    
    results = agent.train(
        initial_white_state=initial_white_state,
        num_episodes=num_episodes,
        reward_type='heuristic',
        snapshot_episodes=[0, 1000, 2500, 4999]
    )
    
    # Resultados
    print("\n" + "="*70)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("="*70)
    print(f"Total de mates encontrados: {results['mates_found']}/{num_episodes} ({results['mates_found']/num_episodes*100:.1f}%)")
    
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
        # La secuencia contiene ESTADOS (incluyendo inicial), los MOVIMIENTOS son len-1
        num_states = len(sequence)
        mate_step = None
        
        print(f"\nSecuencia de {num_states} estados (Estado inicial + {num_states-1} movimientos):")
        for i, state in enumerate(sequence):
            wk = [p for p in state if p[2] == 6][0]
            wr = [p for p in state if p[2] == 2][0]
            print(f"  Estado {i}: Rey({wk[0]},{wk[1]}) Torre({wr[0]},{wr[1]})")
            if agent.is_checkmate(state):
                mate_step = i
                print(f"  >>> ¡JAQUE MATE! (alcanzado en {i} movimientos desde el estado inicial) <<<")
                break
            if i >= 14:
                print(f"  ... (continúa hasta estado {num_states-1})")
                break
    
    # Comparación con ejercicio 2.a
    print(f"\nComparación con ejercicio 2.a:")
    if results_2a:
        mates_2a = results_2a['mates_found']
        pct_2a = (mates_2a / 5000) * 100
        print(f"- 2.a encontró {mates_2a} mates en 5000 episodios ({pct_2a:.1f}%)")
    else:
        print(f"- 2.a: ~90-91% mates (recompensa simple)")
    
    pct_2b = (results['mates_found'] / num_episodes) * 100
    print(f"- 2.b encontró {results['mates_found']} mates en {num_episodes} episodios ({pct_2b:.1f}%)")
    
    return agent, results


def ejercicio_2c():
    """
    Ejercicio 2.c: Q-learning con Rey Negro MÓVIL.
    
    Concepto: El rey negro se mueve ALEATORIAMENTE después de cada turno blanco.
    El estado INCLUYE la posición del rey negro (espacio de estados expandido).
    Modelo MDP correcto: s -> a -> s' (donde s' incluye movimiento del rey negro).
    """
    print("\n" + "="*70)
    print("EJERCICIO 2.c - Q-learning con Rey Negro MÓVIL")
    print("="*70)
    print("\nDescripción:")
    print("- Rey blanco + Torre vs Rey negro")
    print("- REY NEGRO SE MUEVE aleatoriamente cada turno")
    print("- Espacio de estados ampliado (incluye posición rey negro)")
    print("- Demuestra Q-learning contra oponente dinámico")
    print("- MDP correcto: estado incluye posición rey negro tras su movimiento")
    
    # Parámetros optimizados
    alpha = 0.4
    gamma = 0.95
    epsilon_inicial = 0.5
    episodios = 5000
    max_pasos = 150
    
    print(f"\nParámetros: α={alpha}, γ={gamma}, ε={epsilon_inicial}→0.05, episodios={episodios}, max_pasos={max_pasos}")
    print("Justificación: α reducido→estabilidad vs oponente estocástico | γ→balance | posiciones iniciales: esquinas/bordes")
    
    # Q-table expandida (incluye posición rey negro en el estado)
    q_table = defaultdict(float)
    mates_encontrados = 0
    pasos_por_episodio = []
    
    # Agente auxiliar para reutilizar funciones (evita duplicación de código)
    agent_helper = QLearningChess((0, 0))
    
    print(f"\nEntrenando contra rey negro móvil...")
    
    for episodio in range(episodios):
        # Epsilon decay
        epsilon = max(0.05, epsilon_inicial - (episodio / episodios) * 0.45)
        
        # Posiciones iniciales realistas
        posiciones_iniciales = [(0,0), (0,4), (0,7), (4,0), (4,7), (7,0), (7,7)]
        black_king_row, black_king_col = random.choice(posiciones_iniciales)
        
        # Posición fija para blancas
        white_state = [[7, 4, 6], [7, 0, 2]]
        
        # Asegurar que rey negro no colisiona
        while any(p[0] == black_king_row and p[1] == black_king_col for p in white_state):
            black_king_row, black_king_col = random.choice(posiciones_iniciales)
        
        for paso in range(max_pasos):
            # Estado expandido (incluye rey negro)
            estado_str = f"{white_state[0][0]},{white_state[0][1]},{white_state[1][0]},{white_state[1][1]},{black_king_row},{black_king_col}"
            
            # Verificar jaque mate (reutilizando función de la clase)
            agent_helper.black_king_pos = (black_king_row, black_king_col)
            if agent_helper.is_checkmate(white_state):
                mates_encontrados += 1
                pasos_por_episodio.append(paso + 1)
                break
            
            # Obtener acciones posibles (reutilizando función de la clase)
            possible_actions = agent_helper.get_possible_actions(white_state, (black_king_row, black_king_col))
            
            if not possible_actions:
                pasos_por_episodio.append(max_pasos)
                break
            
            # Epsilon-greedy (reutilizando lógica)
            if random.random() < epsilon:
                action = random.choice(possible_actions)
            else:
                # Greedy: mejor Q-value
                best_action = None
                max_q = -float('inf')
                for act in possible_actions:
                    action_str = agent_helper.action_to_string(act[0], act[1])
                    q_val = q_table[(estado_str, action_str)]
                    if q_val > max_q:
                        max_q = q_val
                        best_action = act
                action = best_action if best_action else random.choice(possible_actions)
            
            action_str = agent_helper.action_to_string(action[0], action[1])
            
            # Ejecutar acción (reutilizando función)
            white_state = agent_helper.execute_action(white_state, action)
            
            # Rey negro se mueve (parte específica del ejercicio 2.c)
            movimientos_rey_negro = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = black_king_row + dr, black_king_col + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        if not any(p[0] == nr and p[1] == nc for p in white_state):
                            movimientos_rey_negro.append((nr, nc))
            
            if movimientos_rey_negro:
                black_king_row, black_king_col = random.choice(movimientos_rey_negro)
            
            # Estado siguiente (tras movimiento rey negro)
            next_estado_str = f"{white_state[0][0]},{white_state[0][1]},{white_state[1][0]},{white_state[1][1]},{black_king_row},{black_king_col}"
            
            # Recompensa
            recompensa = -1.0
            
            # Max Q siguiente estado
            next_actions = agent_helper.get_possible_actions(white_state, (black_king_row, black_king_col))
            max_q_siguiente = 0
            if next_actions:
                for next_act in next_actions:
                    next_act_str = agent_helper.action_to_string(next_act[0], next_act[1])
                    q_val = q_table[(next_estado_str, next_act_str)]
                    if q_val > max_q_siguiente:
                        max_q_siguiente = q_val
            
            # Actualizar Q-value (Bellman)
            q_actual = q_table[(estado_str, action_str)]
            q_nuevo = q_actual + alpha * (recompensa + gamma * max_q_siguiente - q_actual)
            q_table[(estado_str, action_str)] = q_nuevo
        
        # Si no terminó, registrar max_pasos
        if len(pasos_por_episodio) <= episodio:
            pasos_por_episodio.append(max_pasos)
        
        # Progreso
        if (episodio + 1) % 1000 == 0:
            mates_recientes = sum(1 for p in pasos_por_episodio[max(0, episodio-999):episodio+1] if p < max_pasos)
            print(f"Episodio {episodio + 1}/{episodios} - Mates últimos 1000: {mates_recientes} - Epsilon: {epsilon:.3f} - Q-table: {len(q_table)}")
    
    # Resultados finales
    print("\n" + "-"*70)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("-"*70)
    print(f"Mates encontrados: {mates_encontrados}/{episodios} ({mates_encontrados/episodios*100:.1f}%)")
    print(f"Tamaño Q-table: {len(q_table)} estados-acción")
    print(f"Pasos promedio: {np.mean(pasos_por_episodio):.1f}")
    print(f"Pasos mínimo: {min(pasos_por_episodio)}")
    
    # Análisis de convergencia
    print("\n" + "-"*70)
    print("ANÁLISIS DE CONVERGENCIA")
    print("-"*70)
    for i in [1000, 2000, 3000, 4000, 5000]:
        recent = pasos_por_episodio[max(0, i-1000):i]
        mates_interval = sum(1 for p in recent if p < max_pasos)
        print(f"Episodios {max(1, i-999):5d}-{i:5d}: {mates_interval:4d} mates ({mates_interval/len(recent)*100:5.1f}%)")
    
    return q_table


# ======================================================================
# FUNCIÓN PRINCIPAL
# ======================================================================

if __name__ == "__main__":
    # Ejecutar ejercicio 2.a
    agent_2a, results_2a = ejercicio_2a()
    
    # Ejecutar ejercicio 2.b (pasando resultados de 2.a para comparación)
    agent_2b, results_2b = ejercicio_2b(results_2a)
    
    # Ejecutar ejercicio 2.c
    agent_2c = ejercicio_2c()
