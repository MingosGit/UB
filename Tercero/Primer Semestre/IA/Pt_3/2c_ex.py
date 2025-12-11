"""
Ejercicio 2.c - Práctica 3: Q-learning con Rey Negro MÓVIL
Q-learning para Rey + Torre blancas vs Rey negro MÓVIL

Concepto: El rey negro se mueve ALEATORIAMENTE después de cada turno blanco.
El estado INCLUYE la posición del rey negro (espacio de estados expandido).
Modelo MDP correcto: s -> a -> s' (donde s' incluye movimiento del rey negro).
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
    Clase auxiliar para reutilizar funciones de ajedrez.
    (Versión simplificada para el ejercicio 2.c)
    """
    
    def __init__(self, black_king_pos: Tuple[int, int]):
        self.black_king_pos = black_king_pos
    
    def action_to_string(self, piece_state: List[int], next_pos: List[int]) -> str:
        """Convierte una acción a string."""
        return f"{piece_state[0]},{piece_state[1]}->{next_pos[0]},{next_pos[1]}"
    
    def is_checkmate(self, white_state: List[List[int]]) -> bool:
        """Verifica si hay jaque mate (OPTIMIZADO)."""
        black_king = self.black_king_pos
        
        # Cache de piezas blancas
        wk = wr = None
        for p in white_state:
            if p[2] == 6:
                wk = p
            elif p[2] == 2:
                wr = p
        
        # Función optimizada para verificar ataque
        def is_attacked(row, col):
            if wr:
                if wr[0] == row:
                    if not (wk and wk[0] == row and min(wr[1], col) < wk[1] < max(wr[1], col)):
                        return True
                if wr[1] == col:
                    if not (wk and wk[1] == col and min(wr[0], row) < wk[0] < max(wr[0], row)):
                        return True
            if wk and abs(wk[0] - row) <= 1 and abs(wk[1] - col) <= 1:
                return True
            return False
        
        # ¿Rey negro en jaque?
        if not is_attacked(black_king[0], black_king[1]):
            return False
        
        # Verificar si tiene escapes
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = black_king[0] + dr, black_king[1] + dc
                
                if 0 <= nr < 8 and 0 <= nc < 8:
                    # Ocupada por pieza blanca (excepto torre capturable sin jaque)
                    if wk and wk[0] == nr and wk[1] == nc:
                        continue
                    
                    # Simular captura de torre
                    temp_wk = wk
                    temp_wr = wr if not (wr and wr[0] == nr and wr[1] == nc) else None
                    
                    # Verificar si nuevo cuadro está atacado
                    is_safe = True
                    if temp_wr:
                        if temp_wr[0] == nr:
                            if not (temp_wk and temp_wk[0] == nr and min(temp_wr[1], nc) < temp_wk[1] < max(temp_wr[1], nc)):
                                is_safe = False
                        if temp_wr[1] == nc:
                            if not (temp_wk and temp_wk[1] == nc and min(temp_wr[0], nr) < temp_wk[0] < max(temp_wr[0], nr)):
                                is_safe = False
                    if temp_wk and abs(temp_wk[0] - nr) <= 1 and abs(temp_wk[1] - nc) <= 1:
                        is_safe = False
                    
                    if is_safe:
                        return False
        
        return True
    
    def get_possible_actions(self, white_state: List[List[int]], black_king_pos: Tuple[int, int]) -> List[Tuple[List[int], List[int]]]:
        """Obtiene acciones posibles para las blancas."""
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
                if moved_piece[2] == 6:
                    dist_row = abs(new_position[0] - black_king_pos[0])
                    dist_col = abs(new_position[1] - black_king_pos[1])
                    max_dist = max(dist_row, dist_col)
                    
                    if max_dist <= 1:
                        continue
                
                actions.append((moved_piece, new_position))
        
        return actions
    
    def execute_action(self, white_state: List[List[int]], action: Tuple[List[int], List[int]]) -> List[List[int]]:
        """Ejecuta una acción (mueve una pieza blanca)."""
        new_state = [p.copy() for p in white_state]
        
        for i, piece in enumerate(new_state):
            if piece[0] == action[0][0] and piece[1] == action[0][1] and piece[2] == action[0][2]:
                new_state[i] = [action[1][0], action[1][1], piece[2]]
                break
        
        return new_state


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
    alpha = 0.7  # Learning rate MUY ALTO
    gamma = 0.85  # REDUCIDO para evitar inflación de Q-values
    epsilon_inicial = 0.9  # Exploración muy alta
    episodios = 10000  # Más episodios
    max_pasos = 40  # REDUCIDO para forzar mates rápidos
    
    print(f"\nParametros: alpha={alpha}, gamma={gamma}, epsilon={epsilon_inicial}->0.05, episodios={episodios}, max_pasos={max_pasos}")
    print("Gamma BAJO (0.85) evita inflacion de Q-values | Penalizacion FUERTE por movimiento (-2)")
    
    # Q-table expandida (incluye posición rey negro en el estado)
    q_table = defaultdict(float)
    mates_encontrados = 0
    pasos_por_episodio = []
    
    # Agente auxiliar para reutilizar funciones (evita duplicación de código)
    agent_helper = QLearningChess((0, 0))
    
    print(f"\nEntrenando contra rey negro móvil...")
    
    for episodio in range(episodios):
        # Epsilon decay MUY lento (más exploración)
        epsilon = max(0.05, epsilon_inicial * (0.9995 ** episodio))
        
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
            
            # Verificar jaque mate ANTES de que el rey negro se mueva
            agent_helper.black_king_pos = (black_king_row, black_king_col)
            if agent_helper.is_checkmate(white_state):
                # ¡JAQUE MATE! Recompensa ENORME
                recompensa_mate = 2000.0  # DUPLICADO para dominar cualquier Q-value
                q_actual = q_table[(estado_str, action_str)]
                q_nuevo = q_actual + alpha * (recompensa_mate - q_actual)
                q_table[(estado_str, action_str)] = q_nuevo
                mates_encontrados += 1
                pasos_por_episodio.append(paso + 1)
                break
            
            # ==============================================================
            # CALCULAR MOVIMIENTOS LEGALES DEL REY NEGRO (sin jaque)
            # ==============================================================
            wk = [p for p in white_state if p[2] == 6][0]
            wr = [p for p in white_state if p[2] == 2][0]
            
            movimientos_rey_negro = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = black_king_row + dr, black_king_col + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        # NO puede estar junto al rey blanco
                        if abs(wk[0] - nr) <= 1 and abs(wk[1] - nc) <= 1:
                            continue
                        
                        # Simular movimiento (puede capturar torre)
                        puede_capturar_torre = wr[0] == nr and wr[1] == nc
                        
                        # Verificar si quedaría en jaque
                        en_jaque = False
                        
                        # Ataque del rey blanco
                        if abs(wk[0] - nr) <= 1 and abs(wk[1] - nc) <= 1:
                            en_jaque = True
                        
                        # Ataque de la torre (si no la captura)
                        if not en_jaque and not puede_capturar_torre:
                            if wr[0] == nr:  # Misma fila
                                min_c, max_c = min(wr[1], nc), max(wr[1], nc)
                                bloqueada = any(wk[0] == nr and wk[1] == c for c in range(min_c + 1, max_c))
                                if not bloqueada:
                                    en_jaque = True
                            elif wr[1] == nc:  # Misma columna
                                min_r, max_r = min(wr[0], nr), max(wr[0], nr)
                                bloqueada = any(wk[0] == r and wk[1] == nc for r in range(min_r + 1, max_r))
                                if not bloqueada:
                                    en_jaque = True
                        
                        # Solo agregar si NO estaría en jaque
                        if not en_jaque:
                            movimientos_rey_negro.append((nr, nc))
            
            if movimientos_rey_negro:
                black_king_row, black_king_col = random.choice(movimientos_rey_negro)
            
            # Estado siguiente (tras movimiento rey negro)
            next_estado_str = f"{white_state[0][0]},{white_state[0][1]},{white_state[1][0]},{white_state[1][1]},{black_king_row},{black_king_col}"
            
            # ========================================================
            # RECOMPENSA ULTRA MINIMALISTA
            # ========================================================
            recompensa = -2.0  # Penalización FUERTE por cada movimiento
            
            bk_pos = (black_king_row, black_king_col)
            movimientos_legales_bk = len(movimientos_rey_negro)
            
            # CRÍTICO: Verificar jaque
            def esta_en_jaque_ahora():
                if wr[0] == bk_pos[0]:
                    min_c, max_c = min(wr[1], bk_pos[1]), max(wr[1], bk_pos[1])
                    if not any(wk[0] == wr[0] and wk[1] == c for c in range(min_c + 1, max_c)):
                        return True
                if wr[1] == bk_pos[1]:
                    min_r, max_r = min(wr[0], bk_pos[0]), max(wr[0], bk_pos[0])
                    if not any(wk[0] == r and wk[1] == wr[1] for r in range(min_r + 1, max_r)):
                        return True
                return False
            
            torre_da_jaque = esta_en_jaque_ahora()
            
            # AHOGADO: penalización MASIVA
            if movimientos_legales_bk == 0 and not torre_da_jaque:
                recompensa = -100.0
            else:
                # SOLO bonificaciones MUY PEQUEÑAS
                # Movilidad: objetivo principal
                if movimientos_legales_bk <= 2:
                    recompensa += (3 - movimientos_legales_bk) * 1.5  # Max +4.5
                
                # Jaque + casi mate
                if torre_da_jaque and movimientos_legales_bk == 1:
                    recompensa += 15.0  # CASI MATE es lo único importante
                elif torre_da_jaque and movimientos_legales_bk == 2:
                    recompensa += 5.0
                elif torre_da_jaque:
                    recompensa += 1.0  # Jaque genérico casi no vale nada
            
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
            
            # Si hay ahogado, terminar episodio (es un empate, no queremos seguir)
            if movimientos_legales_bk == 0 and not torre_da_jaque:
                pasos_por_episodio.append(max_pasos)  # Contar como fallido
                break
        
        # Si no terminó, registrar max_pasos
        if len(pasos_por_episodio) <= episodio:
            pasos_por_episodio.append(max_pasos)
        
        # Progreso
        if (episodio + 1) % 1000 == 0:
            mates_recientes = sum(1 for p in pasos_por_episodio[max(0, episodio-999):episodio+1] if p < max_pasos)
            tasa = mates_recientes / min(1000, episodio+1) * 100
            print(f"Ep {episodio + 1}/{episodios} | Mates: {mates_recientes}/1000 ({tasa:.1f}%) | e={epsilon:.2f} | Q={len(q_table)}")
    
    # Resultados finales
    print("\n" + "-"*70)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("-"*70)
    tasa_final = mates_encontrados/episodios*100
    print(f"Mates encontrados: {mates_encontrados}/{episodios} ({tasa_final:.1f}%)")
    print(f"Tamano Q-table: {len(q_table)} estados-accion")
    print(f"Pasos promedio: {np.mean(pasos_por_episodio):.1f}")
    print(f"Pasos minimo: {min(pasos_por_episodio)}")
    
    # Análisis de convergencia
    print("\n" + "-"*70)
    print("ANÁLISIS DE CONVERGENCIA")
    print("-"*70)
    intervalos = [2000, 4000, 6000, 8000, 10000]
    
    for i in intervalos:
        if i > episodios:
            continue
        recent = pasos_por_episodio[max(0, i-2000):i]
        if not recent:
            continue
        mates_interval = sum(1 for p in recent if p < max_pasos)
        print(f"Episodios {max(1, i-1999):4d}-{i:4d}: {mates_interval:4d} mates ({mates_interval/len(recent)*100:5.1f}%)")
    
    # ======================================================================
    # DEMOSTRACIONES: 10 Partidas de ejemplo con rey negro móvil
    # ======================================================================
    print("\n" + "="*70)
    print("PARTIDAS DE DEMOSTRACION (3 partidas - Rey Negro Movil)")
    print("="*70)
    print("Usando politica GREEDY pura de Q-learning")
    
    resultados_demos = {"mates": 0, "ahogados": 0, "max_movs": 0}
    
    for num_demo in range(1, 4):
        print("\n" + "-"*70)
        print(f"PARTIDA {num_demo}/3")
        print("-"*70)
        
        # Variar posiciones iniciales del rey negro para diversidad
        posiciones_iniciales_bk = [(0,0), (0,4), (0,7), (4,0), (4,7), (7,0), (7,7)]
        demo_black_king = random.choice(posiciones_iniciales_bk)
        
        # Posiciones blancas fijas
        demo_white_state = [[7, 4, 6], [7, 0, 2]]
        
        # Historial de estados COMPLETOS (incluyendo rey negro) para detectar ciclos
        estados_recientes = []
        
        print(f"Posicion inicial: Rey blanco (7,4), Torre blanca (7,0), Rey negro {demo_black_king}")
        
        resultado_partida = None
        
        for movimiento in range(60):
            # Solo mostrar tablero inicial y final (para no saturar output)
            if movimiento == 0:
                print(f"\n--- Movimiento {movimiento} (inicial) ---")
                board_array = np.zeros((8, 8))
                board_array[demo_black_king[0]][demo_black_king[1]] = 12
                for piece in demo_white_state:
                    board_array[piece[0]][piece[1]] = piece[2]
                temp_board = board.Board(board_array, False)
                temp_board.print_board()
            
            # Verificar mate
            agent_helper.black_king_pos = demo_black_king
            if agent_helper.is_checkmate(demo_white_state):
                print(f"\n--- Movimiento {movimiento} (JAQUE MATE) ---")
                board_array = np.zeros((8, 8))
                board_array[demo_black_king[0]][demo_black_king[1]] = 12
                for piece in demo_white_state:
                    board_array[piece[0]][piece[1]] = piece[2]
                temp_board = board.Board(board_array, False)
                temp_board.print_board()
                print(f"¡JAQUE MATE en {movimiento} movimientos!")
                resultado_partida = "mate"
                resultados_demos["mates"] += 1
                break
            
            # Estado actual COMPLETO (incluyendo rey negro)
            estado_actual = (demo_white_state[0][0], demo_white_state[0][1], 
                           demo_white_state[1][0], demo_white_state[1][1],
                           demo_black_king[0], demo_black_king[1])
            
            # Detectar ciclos: si este estado se repite 3+ veces, hacer movimiento aleatorio
            cuenta_repeticiones = estados_recientes.count(estado_actual)
            
            estado_str = f"{demo_white_state[0][0]},{demo_white_state[0][1]},{demo_white_state[1][0]},{demo_white_state[1][1]},{demo_black_king[0]},{demo_black_king[1]}"
            
            # Obtener acciones posibles
            possible_actions = agent_helper.get_possible_actions(demo_white_state, demo_black_king)
            
            if not possible_actions:
                print(f"\nNo hay movimientos posibles en movimiento {movimiento}. Partida terminada.")
                resultado_partida = "sin_movimientos"
                break
            
            # Elegir acción: si hay ciclo, aleatoria; si no, mejor Q-value
            if cuenta_repeticiones >= 2:
                # ROMPER CICLO: movimiento aleatorio
                best_action = random.choice(possible_actions)
                q_val_display = 0
                print(f"\n¡Ciclo detectado! Movimiento aleatorio para romperlo.")
            else:
                # Política GREEDY: mejor Q-value
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
                    q_val_display = 0
                else:
                    action_str = agent_helper.action_to_string(best_action[0], best_action[1])
                    q_val_display = q_table[(estado_str, action_str)]
            
            # Mostrar movimiento blanco
            print(f"\nBlancas: {agent_helper.action_to_string(best_action[0], best_action[1])} (Q={q_val_display:.2f})")
            
            # Ejecutar movimiento blanco
            demo_white_state = agent_helper.execute_action(demo_white_state, best_action)
            
            # Registrar estado completo
            estados_recientes.append(estado_actual)
            
            # Rey negro se mueve aleatoriamente (SOLO a casillas NO en jaque)
            # IMPORTANTE: Puede capturar la torre si es seguro
            movimientos_posibles = []
            wk = [p for p in demo_white_state if p[2] == 6][0]
            wr = [p for p in demo_white_state if p[2] == 2][0]
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = demo_black_king[0] + dr, demo_black_king[1] + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        # Verificar si hay pieza blanca en destino
                        pieza_blanca_en_destino = None
                        for p in demo_white_state:
                            if p[0] == nr and p[1] == nc:
                                pieza_blanca_en_destino = p
                                break
                        
                        # NO puede capturar/acercarse al rey blanco
                        if pieza_blanca_en_destino and pieza_blanca_en_destino[2] == 6:
                            continue
                        if abs(wk[0] - nr) <= 1 and abs(wk[1] - nc) <= 1:
                            continue
                        
                        # Simular el movimiento/captura
                        temp_white_state = [p for p in demo_white_state if not (p[0] == nr and p[1] == nc)]
                        
                        # Verificar si quedaría en jaque en esa casilla
                        en_jaque = False
                        temp_wk = [p for p in temp_white_state if p[2] == 6][0] if any(p[2] == 6 for p in temp_white_state) else None
                        temp_wr = [p for p in temp_white_state if p[2] == 2][0] if any(p[2] == 2 for p in temp_white_state) else None
                        
                        # Verificar ataque del rey blanco
                        if temp_wk and abs(temp_wk[0] - nr) <= 1 and abs(temp_wk[1] - nc) <= 1:
                            en_jaque = True
                        
                        # Verificar ataque de la torre (si aún existe)
                        if not en_jaque and temp_wr:
                            if temp_wr[0] == nr:  # Misma fila
                                min_c, max_c = min(temp_wr[1], nc), max(temp_wr[1], nc)
                                bloqueada = temp_wk and any(temp_wk[0] == nr and temp_wk[1] == c for c in range(min_c + 1, max_c))
                                if not bloqueada:
                                    en_jaque = True
                            elif temp_wr[1] == nc:  # Misma columna
                                min_r, max_r = min(temp_wr[0], nr), max(temp_wr[0], nr)
                                bloqueada = temp_wk and any(temp_wk[0] == r and temp_wk[1] == nc for r in range(min_r + 1, max_r))
                                if not bloqueada:
                                    en_jaque = True
                        
                        # Solo agregar si NO estaría en jaque
                        if not en_jaque:
                            movimientos_posibles.append((nr, nc))
            
            if movimientos_posibles:
                old_pos = demo_black_king
                # Rey negro se mueve ALEATORIAMENTE (no defensivamente)
                demo_black_king = random.choice(movimientos_posibles)
                # No imprimir cada movimiento para no saturar
            else:
                # ¡Rey negro sin movimientos legales!
                # Verificar si es ahogado (sin jaque) o mate (con jaque)
                agent_helper.black_king_pos = demo_black_king
                
                # Verificar si está en jaque
                def esta_en_jaque_demo():
                    wr = [p for p in demo_white_state if p[2] == 2][0]
                    wk = [p for p in demo_white_state if p[2] == 6][0]
                    if wr[0] == demo_black_king[0]:  # Misma fila
                        min_c, max_c = min(wr[1], demo_black_king[1]), max(wr[1], demo_black_king[1])
                        if not any(wk[0] == wr[0] and wk[1] == c for c in range(min_c + 1, max_c)):
                            return True
                    if wr[1] == demo_black_king[1]:  # Misma columna
                        min_r, max_r = min(wr[0], demo_black_king[0]), max(wr[0], demo_black_king[0])
                        if not any(wk[0] == r and wk[1] == wr[1] for r in range(min_r + 1, max_r)):
                            return True
                    return False
                
                if esta_en_jaque_demo():
                    print(f"\n--- Movimiento {movimiento} (JAQUE MATE) ---")
                    board_array = np.zeros((8, 8))
                    board_array[demo_black_king[0]][demo_black_king[1]] = 12
                    for piece in demo_white_state:
                        board_array[piece[0]][piece[1]] = piece[2]
                    temp_board = board.Board(board_array, False)
                    temp_board.print_board()
                    print(f"¡JAQUE MATE! Las blancas ganan en {movimiento} movimientos.")
                    resultado_partida = "mate"
                    resultados_demos["mates"] += 1
                else:
                    print(f"\n--- Movimiento {movimiento} (AHOGADO) ---")
                    board_array = np.zeros((8, 8))
                    board_array[demo_black_king[0]][demo_black_king[1]] = 12
                    for piece in demo_white_state:
                        board_array[piece[0]][piece[1]] = piece[2]
                    temp_board = board.Board(board_array, False)
                    temp_board.print_board()
                    print(f"¡AHOGADO! Es empate en {movimiento} movimientos (las blancas no lograron el mate).")
                    resultado_partida = "ahogado"
                    resultados_demos["ahogados"] += 1
                break
        
        # Límite de movimientos alcanzado
        if resultado_partida is None:
            print(f"\n--- Movimiento {movimiento} (final) ---")
            board_array = np.zeros((8, 8))
            board_array[demo_black_king[0]][demo_black_king[1]] = 12
            for piece in demo_white_state:
                board_array[piece[0]][piece[1]] = piece[2]
            temp_board = board.Board(board_array, False)
            temp_board.print_board()
            print(f"\nAlcanzado maximo de 60 movimientos sin resultado.")
            resultado_partida = "max_movimientos"
            resultados_demos["max_movs"] += 1
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE LAS 3 PARTIDAS DE DEMOSTRACION")
    print("="*70)
    total = resultados_demos['mates'] + resultados_demos['ahogados'] + resultados_demos['max_movs']
    if total > 0:
        print(f"Mates: {resultados_demos['mates']}/3 ({resultados_demos['mates']/3*100:.0f}%)")
        print(f"Ahogados: {resultados_demos['ahogados']}/3 ({resultados_demos['ahogados']/3*100:.0f}%)")
        print(f"Sin terminar: {resultados_demos['max_movs']}/3 ({resultados_demos['max_movs']/3*100:.0f}%)")
    
    return q_table


# ======================================================================
# FUNCIÓN PRINCIPAL
# ======================================================================

if __name__ == "__main__":
    ejercicio_2c()
