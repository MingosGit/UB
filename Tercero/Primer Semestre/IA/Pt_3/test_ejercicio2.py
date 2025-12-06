"""
Test para verificar el funcionamiento del ejercicio 2
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'chess'))
import chess
import board

# Test 1: Verificar que se pueden generar movimientos
print("="*70)
print("TEST 1: Verificar generación de movimientos")
print("="*70)

TA = np.zeros((8, 8))
TA[7][0] = 2   # Torre blanca
TA[7][4] = 6   # Rey blanco
TA[0][4] = 12  # Rey negro

board_obj = board.Board(TA.copy(), False)

white_pieces = [[7, 4, 6], [7, 0, 2]]
board_obj.getListNextStatesW(white_pieces)
print(f"Número de estados siguientes: {len(board_obj.listNextStates)}")
print(f"Primeros 5 estados siguientes:")
for i, state in enumerate(board_obj.listNextStates[:5]):
    print(f"  {i+1}. {state}")

# Test 2: Verificar detección de jaque mate
print("\n" + "="*70)
print("TEST 2: Verificar detección de jaque mate")
print("="*70)

from ejercicio2 import QLearningChess

# Posición de jaque mate: Rey blanco en (1,3), Torre en (0,3), Rey negro en (0,4)
checkmate_state = [[1, 3, 6], [0, 3, 2]]
black_king_pos = (0, 4)

agent = QLearningChess(black_king_pos)
is_mate = agent.is_checkmate(checkmate_state)
print(f"Estado: Rey blanco (1,3), Torre (0,3), Rey negro (0,4)")
print(f"¿Es jaque mate? {is_mate}")

# Test 3: Posición sin jaque mate
print("\n" + "="*70)
print("TEST 3: Posición sin jaque mate")
print("="*70)

no_mate_state = [[7, 4, 6], [7, 0, 2]]
is_mate = agent.is_checkmate(no_mate_state)
print(f"Estado: Rey blanco (7,4), Torre (7,0), Rey negro (0,4)")
print(f"¿Es jaque mate? {is_mate}")

# Test 4: Verificar si hay acciones disponibles
print("\n" + "="*70)
print("TEST 4: Acciones disponibles desde posición inicial")
print("="*70)

initial_state = [[7, 4, 6], [7, 0, 2]]
actions = agent.get_possible_actions(initial_state, black_king_pos)
print(f"Número de acciones posibles: {len(actions)}")
print(f"Primeras 5 acciones:")
for i, action in enumerate(actions[:5]):
    print(f"  {i+1}. Mover {action[0]} -> {action[1]}")

# Test 5: Verificar que el tablero se muestra correctamente
print("\n" + "="*70)
print("TEST 5: Visualización del tablero inicial")
print("="*70)

TA = np.zeros((8, 8))
TA[7][0] = 2   # Torre blanca
TA[7][4] = 6   # Rey blanco
TA[0][4] = 12  # Rey negro

chess_board = chess.Chess(TA.copy(), True)
chess_board.board.print_board()
