"""
Debug de la función is_checkmate
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'chess'))

from ejercicio2 import QLearningChess

# Posición de jaque mate: Rey blanco en (1,3), Torre en (0,3), Rey negro en (0,4)
checkmate_state = [[1, 3, 6], [0, 3, 2]]
black_king_pos = (0, 4)

agent = QLearningChess(black_king_pos)

# Verificar si la torre ataca al rey
wk = checkmate_state[0]  # Rey blanco (1,3)
wr = checkmate_state[1]  # Torre (0,3)
bk = black_king_pos  # Rey negro (0,4)

print(f"Rey blanco: {wk}")
print(f"Torre: {wr}")
print(f"Rey negro: {bk}")
print()

# Torre en misma fila que rey negro?
print(f"Torre fila: {wr[0]}, Rey negro fila: {bk[0]}")
print(f"¿Misma fila? {wr[0] == bk[0]}")
print()

# Torre en misma columna que rey negro?
print(f"Torre col: {wr[1]}, Rey negro col: {bk[1]}")
print(f"¿Misma columna? {wr[1] == bk[1]}")
print()

# Si están en la misma fila, verificar bloqueo
if wr[0] == bk[0]:
    min_col = min(wr[1], bk[1])
    max_col = max(wr[1], bk[1])
    print(f"Rango de columnas entre torre y rey negro: {min_col} a {max_col}")
    print(f"Columnas a verificar: {list(range(min_col + 1, max_col))}")
    
    blocked = False
    for c in range(min_col + 1, max_col):
        print(f"  Verificando columna {c}...")
        if wk[0] == wr[0] and wk[1] == c:
            blocked = True
            print(f"    ¡Rey blanco bloqueando en ({wk[0]}, {wk[1]})!")
            break
        else:
            print(f"    No bloqueado")
    
    print(f"\n¿Bloqueado? {blocked}")
    print(f"¿Torre ataca rey negro? {not blocked}")

print("\n" + "="*70)
# Ahora verificar todos los movimientos de escape del rey negro
print("Movimientos de escape del rey negro:")
for dr in [-1, 0, 1]:
    for dc in [-1, 0, 1]:
        if dr == 0 and dc == 0:
            continue
        new_row = bk[0] + dr
        new_col = bk[1] + dc
        
        if 0 <= new_row < 8 and 0 <= new_col < 8:
            # Verificar si hay pieza blanca en esa casilla
            occupied = False
            for wp in checkmate_state:
                if wp[0] == new_row and wp[1] == new_col:
                    occupied = True
                    break
            
            if occupied:
                print(f"  ({new_row},{new_col}): OCUPADA por pieza blanca")
                continue
            
            # Verificar si estaría atacada
            # Torre ataca?
            attacked_by_rook = False
            if wr[0] == new_row:  # Misma fila
                min_col_check = min(wr[1], new_col)
                max_col_check = max(wr[1], new_col)
                blocked_check = False
                for c in range(min_col_check + 1, max_col_check):
                    if wk[0] == new_row and wk[1] == c:
                        blocked_check = True
                        break
                if not blocked_check:
                    attacked_by_rook = True
            
            if wr[1] == new_col:  # Misma columna
                min_row_check = min(wr[0], new_row)
                max_row_check = max(wr[0], new_row)
                blocked_check = False
                for r in range(min_row_check + 1, max_row_check):
                    if wk[0] == r and wk[1] == new_col:
                        blocked_check = True
                        break
                if not blocked_check:
                    attacked_by_rook = True
            
            # Rey blanco ataca?
            attacked_by_king = abs(wk[0] - new_row) <= 1 and abs(wk[1] - new_col) <= 1
            
            attacked = attacked_by_rook or attacked_by_king
            
            print(f"  ({new_row},{new_col}): Atacada={attacked} (Torre={attacked_by_rook}, Rey={attacked_by_king})")

print("\n" + "="*70)
is_mate = agent.is_checkmate(checkmate_state)
print(f"\n¿Es jaque mate? {is_mate}")
