"""
Ejercicio 2 - Práctica 3: Q-learning aplicado al ajedrez
Q-learning para Rey + Torre blancas vs Rey negro
@author: Jose Candon y Daniel Barcelo
"""

import numpy as np
import random
import sys
import os
from typing import Tuple, Dict, List
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), 'chess'))

import chess
import board
import piece

class QLearningChess:
    
    def __init__(self, black_king_pos: Tuple[int, int]):
        self.alpha = 0.3
        self.gamma = 0.99
        self.epsilon = 0.3
        self.black_king_pos = black_king_pos
        self.q_table = defaultdict(float)
        self.q_table_snapshots = []
        self.episodes_history = []
        self.mates_found = 0
        
    def state_to_string(self, white_state: List[List[int]]) -> str:
        wk = [p for p in white_state if p[2] == 6][0] if any(p[2] == 6 for p in white_state) else None
        wr = [p for p in white_state if p[2] == 2][0] if any(p[2] == 2 for p in white_state) else None
        
        state_str = f"{wk[0]},{wk[1]}" if wk else ""
        if wr:
            state_str += f",{wr[0]},{wr[1]}"
        
        return state_str
    
    def action_to_string(self, piece_state: List[int], next_pos: List[int]) -> str:
        return f"{piece_state[0]},{piece_state[1]}->{next_pos[0]},{next_pos[1]}"
    
    def is_checkmate(self, white_state: List[List[int]]) -> bool:
        black_king = self.black_king_pos
        
        def is_square_attacked(row, col, white_pieces):
            wk = [p for p in white_pieces if p[2] == 6][0] if any(p[2] == 6 for p in white_pieces) else None
            wr = [p for p in white_pieces if p[2] == 2][0] if any(p[2] == 2 for p in white_pieces) else None
            
            if wr:
                if wr[0] == row:
                    min_col = min(wr[1], col)
                    max_col = max(wr[1], col)
                    blocked = False
                    for c in range(min_col + 1, max_col):
                        if wk and wk[0] == row and wk[1] == c:
                            blocked = True
                            break
                    if not blocked:
                        return True
                
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
            
            if wk:
                if abs(wk[0] - row) <= 1 and abs(wk[1] - col) <= 1:
                    return True
            
            return False
        
        if not is_square_attacked(black_king[0], black_king[1], white_state):
            return False
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row = black_king[0] + dr
                new_col = black_king[1] + dc
                
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    occupied_by_white = False
                    for wp in white_state:
                        if wp[0] == new_row and wp[1] == new_col:
                            occupied_by_white = True
                            break
                    
                    if occupied_by_white:
                        continue
                    
                    if not is_square_attacked(new_row, new_col, white_state):
                        return False
        
        return True
    
    def get_possible_actions(self, white_state: List[List[int]], black_king_pos: Tuple[int, int]) -> List[Tuple[List[int], List[int]]]:
        board_array = np.zeros((8, 8))
        board_array[black_king_pos[0]][black_king_pos[1]] = 12
        for piece in white_state:
            board_array[piece[0]][piece[1]] = piece[2]
        
        board_obj = board.Board(board_array, False)
        board_obj.getListNextStatesW(white_state)
        next_states = board_obj.listNextStates
        
        actions = []
        for next_state in next_states:
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
                    for new_piece in next_state:
                        if new_piece[2] == orig_piece[2]:
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
        possible_actions = self.get_possible_actions(state, self.black_king_pos)
        
        if not possible_actions:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
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
        new_state = [p.copy() for p in white_state]
        
        for i, piece in enumerate(new_state):
            if piece[0] == action[0][0] and piece[1] == action[0][1] and piece[2] == action[0][2]:
                new_state[i] = [action[1][0], action[1][1], piece[2]]
                break
        
        return new_state
    
    def get_reward(self, white_state: List[List[int]], reward_type: str = 'simple') -> float:
        if self.is_checkmate(white_state):
            return 100.0
        
        if reward_type == 'simple':
            return -1.0
        
        elif reward_type == 'heuristic':
            bk_pos = self.black_king_pos
            
            wk = [p for p in white_state if p[2] == 6][0] if any(p[2] == 6 for p in white_state) else None
            wr = [p for p in white_state if p[2] == 2][0] if any(p[2] == 2 for p in white_state) else None
            
            if not wk or not wr:
                return -50.0
            
            reward = -1.0
            
            king_dist = max(abs(wk[0] - bk_pos[0]), abs(wk[1] - bk_pos[1]))
            if king_dist == 2:
                reward += 0.4
            elif king_dist == 1:
                reward += 0.3
            elif king_dist == 3:
                reward += 0.2
            elif king_dist <= 4:
                reward += 0.1
            
            def tower_attacks_king():
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
            
            if tower_attacks_king():
                reward += 0.5
            elif wr[0] == bk_pos[0] or wr[1] == bk_pos[1]:
                reward += 0.2
            
            escape_squares_controlled = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    er, ec = bk_pos[0] + dr, bk_pos[1] + dc
                    if 0 <= er < 8 and 0 <= ec < 8:
                        if abs(wk[0] - er) <= 1 and abs(wk[1] - ec) <= 1:
                            escape_squares_controlled += 1
                        elif wr[0] == er or wr[1] == ec:
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
            
            reward += escape_squares_controlled * 0.05
            
            return reward
        
        return -1.0
    
    def update_q_value(self, state_str: str, action_str: str, reward: float, 
                      next_state_str: str, possible_next_actions: List[str]):
        current_q = self.q_table[(state_str, action_str)]
        
        max_next_q = -float('inf')
        if possible_next_actions:
            for next_action_str in possible_next_actions:
                q_val = self.q_table[(next_state_str, next_action_str)]
                if q_val > max_next_q:
                    max_next_q = q_val
        else:
            max_next_q = 0.0
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state_str, action_str)] = new_q
    
    def train(self, initial_white_state: List[List[int]], num_episodes: int, reward_type: str = 'simple',
              snapshot_episodes: List[int] = None, stochastic_prob: float = 1.0) -> Dict:
        if snapshot_episodes is None:
            snapshot_episodes = [0, num_episodes // 3, 2 * num_episodes // 3, num_episodes - 1]
        
        self.q_table_snapshots = []
        steps_per_episode = []
        self.mates_found = 0
        
        for episode in range(num_episodes):
            self.epsilon = max(0.1, 0.3 - (episode / num_episodes) * 0.2)
            
            state = [p.copy() for p in initial_white_state]
            state_str = self.state_to_string(state)
            
            steps = 0
            max_steps = 100
            episode_done = False
            
            while not episode_done and steps < max_steps:
                action = self.choose_action(state, state_str)
                
                if action is None:
                    break
                
                if random.random() < stochastic_prob:
                    actual_action = action
                else:
                    possible_actions = self.get_possible_actions(state, self.black_king_pos)
                    other_actions = [a for a in possible_actions if a != action]
                    if other_actions:
                        actual_action = random.choice(other_actions)
                    else:
                        actual_action = action
                
                action_str = self.action_to_string(action[0], action[1])
                
                next_state = self.execute_action(state, actual_action)
                next_state_str = self.state_to_string(next_state)
                
                is_mate = self.is_checkmate(next_state)
                
                reward = self.get_reward(next_state, reward_type)
                
                possible_next_actions = self.get_possible_actions(next_state, self.black_king_pos)
                next_action_strs = [self.action_to_string(a[0], a[1]) for a in possible_next_actions]
                
                self.update_q_value(state_str, action_str, reward, next_state_str, next_action_strs)
                
                state = next_state
                state_str = next_state_str
                steps += 1
                
                if is_mate:
                    episode_done = True
                    self.mates_found += 1
            
            steps_per_episode.append(steps)
            
            if episode in snapshot_episodes:
                self.q_table_snapshots.append((episode, dict(self.q_table)))
            
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
        sequence = []
        state = [p.copy() for p in initial_white_state]
        
        for move_num in range(max_moves):
            state_str = self.state_to_string(state)
            sequence.append(state.copy())
            
<<<<<<< HEAD
            # Mostrar tablero si está activado
            if self.show_board:
                print(f"\n--- Movimiento {move_num} ---")
                board_array = np.zeros((8, 8))
                board_array[self.black_king_pos[0]][self.black_king_pos[1]] = 12
                for piece in state:
                    board_array[piece[0]][piece[1]] = piece[2]
                temp_board = board.Board(board_array, False)
                temp_board.print_board()
            
            # Verificar mate
=======
>>>>>>> nuevo
            if self.is_checkmate(state):
                actual_moves = len(sequence) - 1
                print(f"¡Jaque mate encontrado! Secuencia de {len(sequence)} estados ({actual_moves} movimientos)")
                break
            
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
            
            state = self.execute_action(state, best_action)
        
        return sequence
    
    def print_q_table_sample(self, num_samples: int = 3):
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
    
    def print_q_table_snapshots(self):
        print("\n" + "="*70)
        print("Q-TABLE SNAPSHOTS (First, Two Intermediate, Final)")
        print("="*70)
        
        if not self.q_table_snapshots:
            print("No hay snapshots disponibles.")
            return
        
        for episode, q_table_snapshot in self.q_table_snapshots:
            print(f"\n{'─'*70}")
            print(f"EPISODIO {episode}")
            print(f"{'─'*70}")
            print(f"Tamaño de Q-table: {len(q_table_snapshot)} pares (estado, acción)")
            
            if len(q_table_snapshot) == 0:
                print("Q-table vacía (sin exploración aún)")
                continue
            
            q_values = list(q_table_snapshot.values())
            print(f"Estadísticas de Q-values:")
            print(f"  - Min: {min(q_values):.3f}")
            print(f"  - Max: {max(q_values):.3f}")
            print(f"  - Mean: {np.mean(q_values):.3f}")
            print(f"  - Std: {np.std(q_values):.3f}")
            
            sorted_items = sorted(q_table_snapshot.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 5 mejores Q-values:")
            for i, ((state, action), q_val) in enumerate(sorted_items[:5]):
                print(f"  {i+1}. Estado={state}, Acción={action}")
                print(f"     Q-value={q_val:.3f}")
            
            print(f"\nTop 5 peores Q-values:")
            for i, ((state, action), q_val) in enumerate(sorted_items[-5:]):
                print(f"  {i+1}. Estado={state}, Acción={action}")
                print(f"     Q-value={q_val:.3f}")


def ejercicio_2a():
    print("\n" + "="*70)
    print("EJERCICIO 2.a - Q-learning en Ajedrez (Recompensa Simple)")
    print("="*70)
    
    TA = np.zeros((8, 8))
    TA[7][0] = 2
    TA[7][4] = 6
    TA[0][5] = 12
    
    print("\nConfiguración inicial del tablero:")
    temp_chess = chess.Chess(TA.copy(), True)
    temp_chess.board.print_board()
    
    black_king_pos = (0, 5)
    
    initial_white_state = [
        [7, 4, 6],
        [7, 0, 2]
    ]
    
<<<<<<< HEAD
    # Crear agente con rey negro estático
    agent = QLearningChess(black_king_pos, show_board=True)
    
    print(f"\nParámetros: α={agent.alpha}, γ={agent.gamma}, ε={agent.epsilon}")
    print("Justificación: α alto→convergencia rápida | γ muy alto→planificación largo plazo | ε→decae")
=======
    agent = QLearningChess(black_king_pos)
    
    print(f"\nParámetros de Q-learning:")
    print(f"- Alpha (learning rate): {agent.alpha}")
    print(f"- Gamma (discount factor): {agent.gamma}")
    print(f"- Epsilon inicial (exploration): {agent.epsilon}")
>>>>>>> nuevo
    
    num_episodes = 5000
    print(f"\nEntrenando {num_episodes} episodios con epsilon decreciente...")
    print("(Esto puede tardar 1-2 minutos)")
    
    results = agent.train(
        initial_white_state=initial_white_state,
        num_episodes=num_episodes,
        reward_type='simple',
        snapshot_episodes=[0, 1000, 2500, 4999]
    )
    
    print("\n" + "="*70)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("="*70)
    print(f"Total de mates encontrados: {results['mates_found']}/{num_episodes}")
    
    steps = results['steps_per_episode']
    avg_last_500 = np.mean(steps[-500:])
    min_steps = min(steps[-500:])
    
    print(f"\nEstadísticas de convergencia:")
    print(f"- Promedio pasos últimos 500 episodios: {avg_last_500:.2f}")
    print(f"- Mínimo de pasos alcanzado: {min_steps}")
    
<<<<<<< HEAD
    # Q-table sample
    agent.print_q_table_sample(num_samples=2)
=======
    window_size = 100
    convergence_episode = None
    for i in range(window_size, len(steps)):
        if np.mean(steps[i-window_size:i]) < 25:
            convergence_episode = i
            break
    
    if convergence_episode:
        print(f"- Punto de convergencia: episodio ~{convergence_episode} ({convergence_episode/num_episodes*100:.1f}% del total)")
    else:
        print(f"- No converge en {num_episodes} episodios")
    
    agent.print_q_table_snapshots()
>>>>>>> nuevo
    
    print("\n" + "="*70)
    print("SECUENCIA ÓPTIMA DE MOVIMIENTOS (Política Greedy)")
    print("="*70)
    
    sequence = agent.get_optimal_sequence(initial_white_state, max_moves=30)
    
    if len(sequence) > 0:
        num_states = len(sequence)
        mate_step = None
        
        print(f"\nSecuencia de {num_states} estados (Estado inicial + {num_states-1} movimientos):")
        for i, state in enumerate(sequence):
            wk = [p for p in state if p[2] == 6][0]
            wr = [p for p in state if p[2] == 2][0]
            print(f"  Estado {i}: Rey({wk[0]},{wk[1]}) Torre({wr[0]},{wr[1]})")
            if agent.is_checkmate(state):
                mate_step = i
                print(f"  ¡JAQUE MATE! (alcanzado en {i} movimientos desde el estado inicial)")
                break
            if i >= 14:
                print(f"  ... (continúa hasta estado {num_states-1})")
                break
    
    return agent, results


def ejercicio_2b(results_2a=None):
    print("\n\n" + "="*70)
    print("EJERCICIO 2.b - Q-learning con Recompensa Heurística")
    print("="*70)
    
    black_king_pos = (0, 4)
    initial_white_state = [
        [7, 4, 6],
        [7, 0, 2]
    ]
    
<<<<<<< HEAD
    # Crear agente con rey negro estático
    agent = QLearningChess(black_king_pos, show_board=True)
    agent.alpha = 0.3  # Mismo que 2.a - evita inestabilidad con heurística
    agent.gamma = 0.95  # Ligeramente menor - la heurística da señal inmediata
    
    print(f"\nParámetros: α={agent.alpha}, γ={agent.gamma}, ε={agent.epsilon} (con decaimiento)")
    print("Justificación: α igual que 2.a | γ menor→señal heurística inmediata | estabilidad en aprendizaje")
=======
    agent = QLearningChess(black_king_pos)
    agent.alpha = 0.3
    agent.gamma = 0.95
    
    print(f"\nParámetros:")
    print(f"- Alpha: {agent.alpha}")
    print(f"- Gamma: {agent.gamma}")
    print(f"- Epsilon: {agent.epsilon} inicial con decaimiento")
>>>>>>> nuevo
    
    num_episodes = 5000
    print(f"\nEntrenando {num_episodes} episodios con recompensa heurística...")
    
    results = agent.train(
        initial_white_state=initial_white_state,
        num_episodes=num_episodes,
        reward_type='heuristic',
        snapshot_episodes=[0, 1000, 2500, 4999]
    )
    
    print("\n" + "="*70)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("="*70)
    print(f"Total de mates encontrados: {results['mates_found']}/{num_episodes} ({results['mates_found']/num_episodes*100:.1f}%)")
    
    steps = results['steps_per_episode']
    avg_last_500 = np.mean(steps[-500:])
    min_steps = min(steps[-500:])
    
    print(f"\nEstadísticas de convergencia:")
    print(f"- Promedio pasos últimos 500 episodios: {avg_last_500:.2f}")
    print(f"- Mínimo de pasos alcanzado: {min_steps}")
    
<<<<<<< HEAD
    # Muestra de Q-table
    agent.print_q_table_sample(num_samples=2)
=======
    window_size = 100
    convergence_episode = None
    for i in range(window_size, len(steps)):
        if np.mean(steps[i-window_size:i]) < 25:
            convergence_episode = i
            break
    
    if convergence_episode:
        print(f"- Punto de convergencia: episodio ~{convergence_episode}")
    else:
        print(f"- No converge en {num_episodes} episodios")
    
    agent.print_q_table_snapshots()
>>>>>>> nuevo
    
    print("\n" + "="*70)
    print("SECUENCIA ÓPTIMA (Política Greedy)")
    print("="*70)
    
    sequence = agent.get_optimal_sequence(initial_white_state, max_moves=30)
    
    if len(sequence) > 0:
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
    
    print(f"\n" + "="*70)
    print("COMPARACIÓN CON EJERCICIO 2.a")
    print("="*70)
    
    if results_2a:
        mates_2a = results_2a['mates_found']
        pct_2a = (mates_2a / 5000) * 100
        print(f"- 2.a: {mates_2a} mates en 5000 episodios ({pct_2a:.1f}%)")
    
    pct_2b = (results['mates_found'] / num_episodes) * 100
    print(f"- 2.b: {results['mates_found']} mates en {num_episodes} episodios ({pct_2b:.1f}%)")
    
    return agent, results


def ejercicio_2c():
    print("\n\n" + "="*70)
    print("EJERCICIO 2.c - Marinero Borracho (Stochastic Q-learning)")
    print("="*70)
    
<<<<<<< HEAD
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
=======
    black_king_pos = (0, 4)
    initial_white_state = [
        [7, 4, 6],
        [7, 0, 2]
    ]
    
    TA = np.zeros((8, 8))
    TA[7][0] = 2
    TA[7][4] = 6
    TA[0][5] = 12
    
    print("\nConfiguración inicial del tablero:")
    temp_chess = chess.Chess(TA.copy(), True)
    temp_chess.board.print_board()
    
    print("\n" + "="*70)
    print("PARTE i: ESTOCASTICIDAD (Probabilidad de éxito)")
    print("="*70)
>>>>>>> nuevo
    
    stochastic_probs = [1.0, 0.9, 0.8, 0.7]
    results_all = {}
    
<<<<<<< HEAD
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
=======
    for prob in stochastic_probs:
        print(f"\n{'─'*70}")
        if prob == 1.0:
            print(f"Entrenando con probabilidad {prob} (DETERMINISTA - baseline)")
        else:
            print(f"Entrenando con probabilidad {prob} ({prob*100:.0f}% éxito, {(1-prob)*100:.0f}% aleatorio)")
        print(f"{'─'*70}")
        
        agent = QLearningChess(black_king_pos)
        
        if prob < 1.0:
            agent.alpha = 0.2
            agent.gamma = 0.95
            agent.epsilon = 0.3
        
        num_episodes = 8000 if prob < 1.0 else 5000
        
        print(f"Parámetros ajustados:")
        print(f"  - Alpha: {agent.alpha}")
        print(f"  - Gamma: {agent.gamma}")
        print(f"  - Epsilon inicial: {agent.epsilon}")
        print(f"  - Episodios: {num_episodes}")
        
        results = agent.train(
            initial_white_state=initial_white_state,
            num_episodes=num_episodes,
            reward_type='heuristic',
            snapshot_episodes=[0, num_episodes//3, 2*num_episodes//3, num_episodes-1],
            stochastic_prob=prob
        )
        
        results_all[prob] = {
            'agent': agent,
            'results': results,
            'num_episodes': num_episodes
        }
        steps = results['steps_per_episode']
        avg_last_500 = np.mean(steps[-500:])
        min_steps = min(steps[-500:])
        mates_pct = (results['mates_found'] / num_episodes) * 100
        
        print(f"\nResultados:")
        print(f"  - Mates encontrados: {results['mates_found']}/{num_episodes} ({mates_pct:.1f}%)")
        print(f"  - Promedio pasos (últimos 500): {avg_last_500:.2f}")
        print(f"  - Mínimo de pasos: {min_steps}")
        
        print(f"\n{'─'*70}")
        print(f"Q-TABLE SNAPSHOTS para probabilidad {prob}")
        print(f"{'─'*70}")
        agent.print_q_table_snapshots()
    
    print("\n\n" + "="*70)
    print("PARTE ii: ANÁLISIS DE RESULTADOS")
    print("="*70)
    
    print("\n" + "─"*70)
    print("PARÁMETROS USADOS EN ENTORNO ESTOCÁSTICO")
    print("─"*70)
    print("  - Alpha = 0.2")
    print("  - Gamma = 0.95")
    print("  - Epsilon = 0.3 inicial con decaimiento")
    print("  - Recompensa: heurística")
    
    print("\n" + "─"*70)
    print("COMPARACIÓN DE CONVERGENCIA")
    print("─"*70)
    print(f"\n{'Probabilidad':<15} {'Episodios':<12} {'Mates %':<10} {'Avg pasos':<12} {'Convergencia'}")
    print("─"*70)
    
    for prob in stochastic_probs:
        data = results_all[prob]
        results = data['results']
        num_ep = data['num_episodes']
        steps = results['steps_per_episode']
        
        mates_pct = (results['mates_found'] / num_ep) * 100
        avg_last_500 = np.mean(steps[-500:])
        
        window_size = 100
        convergence_ep = "No converge"
        for i in range(window_size, len(steps)):
            if np.mean(steps[i-window_size:i]) < 25:
                convergence_ep = f"~{i} eps"
                break
        
        prob_str = f"{prob} ({prob*100:.0f}%)" if prob < 1.0 else f"{prob} (det.)"
        print(f"{prob_str:<15} {num_ep:<12} {mates_pct:>6.1f}%    {avg_last_500:>6.2f}       {convergence_ep}")
    
    print("\n" + "─"*70)
    print("CAMINO ÓPTIMO (probabilidad 0.8)")
    print("─"*70)
    
    agent_08 = results_all[0.8]['agent']
    
    sequence = agent_08.get_optimal_sequence(initial_white_state, max_moves=30)
    
    if len(sequence) > 0:
        num_states = len(sequence)
        print(f"\nSecuencia de {num_states} estados ({num_states-1} movimientos):")
        for i, state in enumerate(sequence):
            wk = [p for p in state if p[2] == 6][0]
            wr = [p for p in state if p[2] == 2][0]
            print(f"  Estado {i}: Rey({wk[0]},{wk[1]}) Torre({wr[0]},{wr[1]})")
            if agent_08.is_checkmate(state):
                print(f"  >>> ¡JAQUE MATE! (en {i} movimientos) <<<")
                break
            if i >= 9:
                print(f"  ... (continúa)")
                break
    
    print("\n" + "─"*70)
    print("SIMULACIÓN DE EJECUCIÓN DE POLÍTICA")
    print("─"*70)
    
    print("\nSimulando 10 intentos de ejecutar la política aprendida:")
    successful_mates = 0
    path_lengths = []
    
    for trial in range(10):
        state = [p.copy() for p in initial_white_state]
        steps = 0
        max_steps = 50
        
        while steps < max_steps:
            state_str = agent_08.state_to_string(state)
            
            if agent_08.is_checkmate(state):
                successful_mates += 1
                path_lengths.append(steps)
                print(f"  Intento {trial+1}: MATE en {steps} movimientos (OK)")
                break
            
            possible_actions = agent_08.get_possible_actions(state, black_king_pos)
            if not possible_actions:
                print(f"  Intento {trial+1}: Sin movimientos (pasos={steps}) (FALLO)")
                break
            
            best_action = None
            max_q = -float('inf')
            for action in possible_actions:
                action_str = agent_08.action_to_string(action[0], action[1])
                q_val = agent_08.q_table[(state_str, action_str)]
                if q_val > max_q:
                    max_q = q_val
                    best_action = action
            
            if best_action is None:
                print(f"  Intento {trial+1}: Sin acción válida (FALLO)")
                break
            
            if random.random() < 0.8:
                actual_action = best_action
            else:
                other_actions = [a for a in possible_actions if a != best_action]
                actual_action = random.choice(other_actions) if other_actions else best_action
            
            state = agent_08.execute_action(state, actual_action)
            steps += 1
        else:
            print(f"  Intento {trial+1}: No alcanzó mate en {max_steps} pasos (FALLO)")
    
    print(f"\nResultados de simulación:")
    print(f"  - Mates exitosos: {successful_mates}/10 ({successful_mates*10}%)")
    if path_lengths:
        print(f"  - Longitud promedio: {np.mean(path_lengths):.1f} movimientos")
        print(f"  - Rango: {min(path_lengths)}-{max(path_lengths)} movimientos")
    
    return results_all


def ejercicio_2f():
    print("\n\n" + "="*70)
    print("EJERCICIO 2.f - Robustez de Parámetros (Configuración P1.2)")
    print("="*70)
    
    black_king_pos = (0, 5)
    initial_white_state = [
        [7, 7, 6],
        [7, 0, 2]
    ]
    
    TA = np.zeros((8, 8))
    TA[7][0] = 2
    TA[7][7] = 6
    TA[0][5] = 12
    
    print("\nConfiguración P1.2 (más compleja):")
    temp_chess = chess.Chess(TA.copy(), True)
    temp_chess.board.print_board()
    print("\nNota: Rey negro en (0,5), Rey blanco en (7,7), Torre en (7,0)")
    print("      Configuración diferente para evaluar robustez de parámetros")
    
    print("\n" + "="*70)
    print("PARTE i: GRID SEARCH DE PARÁMETROS")
    print("="*70)
    
    alphas = [0.1, 0.2, 0.3, 0.5]
    gammas = [0.9, 0.95, 0.99]
    epsilons = [0.1, 0.2, 0.3, 0.4]
    stochastic_prob = 0.8
    num_episodes = 6000
    
    print(f"\nParámetros de búsqueda:")
    print(f"  - Alphas: {alphas}")
    print(f"  - Gammas: {gammas}")
    print(f"  - Epsilons: {epsilons}")
    print(f"  - Probabilidad estocástica: {stochastic_prob}")
    print(f"  - Episodios por prueba: {num_episodes}")
    print(f"  - Total combinaciones: {len(alphas)*len(gammas)*len(epsilons)} = {len(alphas)*len(gammas)*len(epsilons)}")
    
    results_grid = []
    best_result = None
    best_score = -float('inf')
    
    print("\nIniciando grid search (esto puede tardar varios minutos)...")
    
    combination_num = 0
    total_combinations = len(alphas) * len(gammas) * len(epsilons)
    
    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                combination_num += 1
                print(f"\n[{combination_num}/{total_combinations}] Probando α={alpha}, γ={gamma}, ε={epsilon}...", end=" ")
                
                agent = QLearningChess(black_king_pos)
                agent.alpha = alpha
                agent.gamma = gamma
                agent.epsilon = epsilon
                
                results = agent.train(
                    initial_white_state=initial_white_state,
                    num_episodes=num_episodes,
                    reward_type='heuristic',
                    snapshot_episodes=[],
                    stochastic_prob=stochastic_prob
                )
                
                steps = results['steps_per_episode']
                avg_last_500 = np.mean(steps[-500:])
                mates_pct = (results['mates_found'] / num_episodes) * 100
                
                window_size = 100
                convergence_ep = num_episodes
                for i in range(window_size, len(steps)):
                    if np.mean(steps[i-window_size:i]) < 25:
                        convergence_ep = i
                        break
                
                score = mates_pct - (avg_last_500 * 0.5) - (convergence_ep / 100)
                
                result_data = {
                    'alpha': alpha,
                    'gamma': gamma,
                    'epsilon': epsilon,
                    'mates_pct': mates_pct,
                    'avg_steps': avg_last_500,
                    'convergence': convergence_ep,
                    'score': score
                }
                results_grid.append(result_data)
                
                print(f" Mates={mates_pct:.1f}%, Avg={avg_last_500:.1f}, Conv={convergence_ep}")
                
                if score > best_score:
                    best_score = score
                    best_result = result_data
                    best_result['agent'] = agent
>>>>>>> nuevo
    
    print("\n" + "="*70)
    print("PARTE ii: RESULTADOS DE GRID SEARCH")
    print("="*70)
    
<<<<<<< HEAD
    # Análisis de convergencia
    print("\n" + "-"*70)
    print("ANÁLISIS DE CONVERGENCIA")
    print("-"*70)
    for i in [1000, 2000, 3000, 4000, 5000]:
        recent = pasos_por_episodio[max(0, i-1000):i]
        mates_interval = sum(1 for p in recent if p < max_pasos)
        print(f"Episodios {max(1, i-999):5d}-{i:5d}: {mates_interval:4d} mates ({mates_interval/len(recent)*100:5.1f}%)")
    
    # DEMOSTRACIÓN: Partida de ejemplo con rey negro móvil
    print("\n" + "="*70)
    print("PARTIDA DE DEMOSTRACIÓN (Rey Negro Móvil)")
    print("="*70)
    print("Usando política greedy aprendida contra rey negro que se mueve aleatoriamente\n")
    
    # Posición inicial para demostración
    demo_black_king = (0, 4)
    demo_white_state = [[7, 4, 6], [7, 0, 2]]
    
    print(f"Posición inicial: Rey blanco (7,4), Torre blanca (7,0), Rey negro (0,4)")
    
    for movimiento in range(30):
        # Mostrar tablero actual
        print(f"\n--- Movimiento {movimiento} ---")
        board_array = np.zeros((8, 8))
        board_array[demo_black_king[0]][demo_black_king[1]] = 12
        for piece in demo_white_state:
            board_array[piece[0]][piece[1]] = piece[2]
        temp_board = board.Board(board_array, False)
        temp_board.print_board()
        
        # Verificar mate
        agent_helper.black_king_pos = demo_black_king
        if agent_helper.is_checkmate(demo_white_state):
            print(f"\n¡JAQUE MATE en {movimiento} movimientos!")
            break
        
        # Estado actual
        estado_str = f"{demo_white_state[0][0]},{demo_white_state[0][1]},{demo_white_state[1][0]},{demo_white_state[1][1]},{demo_black_king[0]},{demo_black_king[1]}"
        
        # Obtener acciones posibles
        possible_actions = agent_helper.get_possible_actions(demo_white_state, demo_black_king)
        
        if not possible_actions:
            print("\nNo hay movimientos posibles. Partida terminada.")
            break
        
        # Elegir mejor acción según Q-table
        best_action = None
        max_q = -float('inf')
        for act in possible_actions:
            action_str = agent_helper.action_to_string(act[0], act[1])
            q_val = q_table[(estado_str, action_str)]
            if q_val > max_q:
                max_q = q_val
                best_action = act
        
        if best_action is None:
            best_action = random.choice(possible_actions)
        
        # Mostrar movimiento blanco
        print(f"\nBlancas: {agent_helper.action_to_string(best_action[0], best_action[1])} (Q={max_q:.2f})")
        
        # Ejecutar movimiento blanco
        demo_white_state = agent_helper.execute_action(demo_white_state, best_action)
        
        # Rey negro se mueve aleatoriamente
        movimientos_posibles = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = demo_black_king[0] + dr, demo_black_king[1] + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    if not any(p[0] == nr and p[1] == nc for p in demo_white_state):
                        movimientos_posibles.append((nr, nc))
        
        if movimientos_posibles:
            old_pos = demo_black_king
            demo_black_king = random.choice(movimientos_posibles)
            print(f"Negras: Rey de ({old_pos[0]},{old_pos[1]}) a ({demo_black_king[0]},{demo_black_king[1]}) (aleatorio)")
        else:
            print("Negras: Rey sin movimientos legales")
    
    return q_table
=======
    results_grid.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "─"*70)
    print("TOP 8 MEJORES COMBINACIONES")
    print("─"*70)
    print(f"{'Alpha':<8} {'Gamma':<8} {'Epsilon':<9} {'Conv (eps)':<12} {'Mates %':<10} {'Avg Pasos':<11} {'Score'}")
    print("─"*70)
    
    for i, res in enumerate(results_grid[:8]):
        print(f"{res['alpha']:<8} {res['gamma']:<8} {res['epsilon']:<9} "
              f"~{res['convergence']:<11} {res['mates_pct']:>6.1f}%    "
              f"{res['avg_steps']:>6.1f}      {res['score']:>6.1f}")
    
    print("\n" + "─"*70)
    print("MEJOR COMBINACIÓN ENCONTRADA")
    print("─"*70)
    print(f"  Alpha:        {best_result['alpha']}")
    print(f"  Gamma:        {best_result['gamma']}")
    print(f"  Epsilon:      {best_result['epsilon']}")
    print(f"  Convergencia: ~{best_result['convergence']} episodios")
    print(f"  Mates:        {best_result['mates_pct']:.1f}%")
    print(f"  Pasos medio:  {best_result['avg_steps']:.2f}")
    
    print("\n" + "="*70)
    print("PARTE iii: COMPARACIÓN CON PARÁMETROS ORIGINALES (2.c)")
    print("="*70)
    
    print("\nProbando parámetros de Ejercicio 2.c en configuración P1.2...")
    agent_original = QLearningChess(black_king_pos)
    agent_original.alpha = 0.2
    agent_original.gamma = 0.95
    agent_original.epsilon = 0.3
    
    results_original = agent_original.train(
        initial_white_state=initial_white_state,
        num_episodes=num_episodes,
        reward_type='heuristic',
        snapshot_episodes=[],
        stochastic_prob=stochastic_prob
    )
    
    steps_orig = results_original['steps_per_episode']
    avg_last_500_orig = np.mean(steps_orig[-500:])
    mates_pct_orig = (results_original['mates_found'] / num_episodes) * 100
    
    window_size = 100
    convergence_orig = num_episodes
    for i in range(window_size, len(steps_orig)):
        if np.mean(steps_orig[i-window_size:i]) < 25:
            convergence_orig = i
            break
    
    print("\n" + "─"*70)
    print("COMPARACIÓN CONFIGURACIÓN ORIGINAL vs P1.2")
    print("─"*70)
    print(f"\n{'Configuración':<20} {'Parámetros':<20} {'Convergencia':<15} {'Mates %':<12} {'Avg Pasos'}")
    print("─"*70)
    print(f"{'Original (0,4)':<20} {'α=0.2,γ=0.95,ε=0.3':<20} {'~1465 eps':<15} {'94.2%':<12} {'15.22'}")
    print(f"{'P1.2 (0,5) [2.c]':<20} {'α=0.2,γ=0.95,ε=0.3':<20} "
          f"{'~'+str(convergence_orig)+' eps':<15} {f'{mates_pct_orig:.1f}%':<12} {f'{avg_last_500_orig:.2f}'}")
    print(f"{'P1.2 (0,5) [opt]':<20} {'α={:.1f},γ={:.2f},ε={:.1f}'.format(best_result['alpha'],best_result['gamma'],best_result['epsilon']):<20} "
          f"{'~'+str(best_result['convergence'])+' eps':<15} {f"{best_result['mates_pct']:.1f}%":<12} {f"{best_result['avg_steps']:.2f}"}")
    
    degradacion_conv = ((convergence_orig - 1465) / 1465) * 100 if convergence_orig != num_episodes else 100
    degradacion_mates = ((mates_pct_orig - 94.2) / 94.2) * 100
    degradacion_pasos = ((avg_last_500_orig - 15.22) / 15.22) * 100
    
    mejora_conv = ((convergence_orig - best_result['convergence']) / convergence_orig) * 100
    mejora_mates = ((best_result['mates_pct'] - mates_pct_orig) / mates_pct_orig) * 100
    mejora_pasos = ((avg_last_500_orig - best_result['avg_steps']) / avg_last_500_orig) * 100
    
    print("\n" + "─"*70)
    print("ANÁLISIS DE ROBUSTEZ")
    print("─"*70)
    
    print(f"\nDegradación parámetros 2.c en configuración P1.2:")
    print(f"  - Convergencia: {degradacion_conv:+.1f}% (más lento)")
    print(f"  - Tasa de mates: {degradacion_mates:+.1f}%")
    print(f"  - Pasos promedio: {degradacion_pasos:+.1f}%")
    
    print(f"\nMejora con parámetros optimizados:")
    print(f"  - Convergencia: {mejora_conv:+.1f}%")
    print(f"  - Tasa de mates: {mejora_mates:+.1f}%")
    print(f"  - Pasos promedio: {mejora_pasos:+.1f}%")
    
    print("\n" + "─"*70)
    print("EVALUACIÓN DE ROBUSTEZ DE PARÁMETROS 2.c")
    print("─"*70)
    
    param_scores = {}
    param_scores['alpha'] = " BAJA" if abs(best_result['alpha'] - 0.2) > 0.1 else " ALTA"
    param_scores['gamma'] = " ALTA" if abs(best_result['gamma'] - 0.95) < 0.02 else " MEDIA"
    param_scores['epsilon'] = " ALTA" if abs(best_result['epsilon'] - 0.3) < 0.1 else " MEDIA"
    
    robustez_score = sum([1 if "ALTA" in v else 0.5 if "MEDIA" in v else 0 for v in param_scores.values()])
    robustez_score = (robustez_score / 3) * 10
    
    print(f"\n{'Parámetro':<12} {'Valor 2.c':<12} {'Valor Óptimo':<12} {'Robustez'}")
    print("─"*70)
    print(f"{'Alpha':<12} {'0.2':<12} {str(best_result['alpha']):<12} {param_scores['alpha']}")
    print(f"{'Gamma':<12} {'0.95':<12} {str(best_result['gamma']):<12} {param_scores['gamma']}")
    print(f"{'Epsilon':<12} {'0.3':<12} {str(best_result['epsilon']):<12} {param_scores['epsilon']}")
    
    print(f"\nPuntuación de robustez global: {robustez_score:.1f}/10")
    
    print("\n" + "─"*70)
    print("CONCLUSIONES")
    print("─"*70)
    print("\n1. Robustez de parámetros:")
    print(f"   - Los parámetros de 2.c son {'robustos' if robustez_score >= 7 else 'poco robustos'}")
    print(f"   - Alpha=0.2 es {'demasiado conservador' if best_result['alpha'] > 0.25 else 'adecuado'} para problemas complejos")
    print(f"   - Gamma=0.95 es {'óptimo' if abs(best_result['gamma']-0.95)<0.02 else 'subóptimo'} universalmente")
    print(f"   - Epsilon=0.3 {'funciona bien' if abs(best_result['epsilon']-0.3)<0.1 else 'necesita ajuste'}")
    
    print("\n2. Impacto de la complejidad:")
    print(f"   - Config. P1.2 es ~{(degradacion_pasos/100*15.22):.0f}% más difícil (pasos adicionales)")
    print(f"   - Convergencia {abs(degradacion_conv):.0f}% {'más lenta' if degradacion_conv > 0 else 'más rápida'}")
    print(f"   - Alpha debe {'aumentar' if best_result['alpha'] > 0.2 else 'mantener'} para problemas complejos")
    
    print("\n3. Recomendaciones:")
    print(f"   - Usar α={best_result['alpha']}, γ={best_result['gamma']}, ε={best_result['epsilon']} para config. difíciles")
    print("   - Implementar alpha adaptativo basado en complejidad estimada")
    print("   - Heurística de recompensa debería considerar esquinas vs centro")
    
    return {
        'grid_results': results_grid,
        'best_result': best_result,
        'original_result': {
            'mates_pct': mates_pct_orig,
            'avg_steps': avg_last_500_orig,
            'convergence': convergence_orig
        },
        'robustez_score': robustez_score
    }
>>>>>>> nuevo


if __name__ == "__main__":
<<<<<<< HEAD
    # Ejecutar ejercicio 2.a
    #agent_2a, results_2a = ejercicio_2a()
    
    ## Ejecutar ejercicio 2.b (pasando resultados de 2.a para comparación)
    #agent_2b, results_2b = ejercicio_2b(results_2a)
=======
    agent_2a, results_2a = ejercicio_2a()
    
    agent_2b, results_2b = ejercicio_2b(results_2a)
>>>>>>> nuevo
    
    results_2c = ejercicio_2c()
    
    print("\n" + "="*70)
    print("¿Desea ejecutar el Ejercicio 2.f (Grid Search)?")
    print("Nota: Esto puede tardar 1-2 horas dependiendo del hardware.")
    print("="*70)
    respuesta = input("Ejecutar 2.f? (s/n): ")
    
    if respuesta.lower() == 's':
        results_2f = ejercicio_2f()
    else:
        print("\nEjercicio 2.f omitido.")
