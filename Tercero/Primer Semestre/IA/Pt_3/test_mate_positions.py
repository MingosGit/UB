"""
Verificar posiciones reales de jaque mate
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'chess'))

from ejercicio2 import QLearningChess

black_king_pos = (0, 4)
agent = QLearningChess(black_king_pos)

print("="*70)
print("TEST DE POSICIONES DE JAQUE MATE")
print("="*70)

# Caso 1: Jaque mate real - Rey negro en (0,4), Torre en (1,0), Rey blanco en (2,4)
print("\nCaso 1: Rey blanco (2,4), Torre (1,0), Rey negro (0,4)")
print("  El rey negro está en jaque por la torre (misma columna no, pero...")
# Espera, la torre en (1,0) no ataca al rey en (0,4), revisemos mejor

# Caso 2: Mate del pasillo - Rey negro en (0,4), Torre en (0,0), Rey blanco en (1,4)
print("\nCaso 2: Rey blanco (1,4), Torre (0,0), Rey negro (0,4)")
state2 = [[1, 4, 6], [0, 0, 2]]
is_mate2 = agent.is_checkmate(state2)
print(f"  ¿Es mate? {is_mate2}")
print("  Análisis: Torre en (0,0) y Rey negro en (0,4) - misma fila")
print("  Casillas entre torre y rey negro: (0,1), (0,2), (0,3)")
print("  Rey blanco en (1,4) protege las casillas (0,3), (0,4), (0,5), (1,3), (1,4), (1,5)")

# Caso 3: Torre en (0,3), Rey blanco en (1,4), Rey negro en (0,4)
print("\nCaso 3: Rey blanco (1,4), Torre (0,3), Rey negro (0,4)")
state3 = [[1, 4, 6], [0, 3, 2]]
is_mate3 = agent.is_checkmate(state3)
print(f"  ¿Es mate? {is_mate3}")

# Caso 4: Torre en (1,4), Rey blanco en (0,3), Rey negro en (0,4)
print("\nCaso 4: Rey blanco (0,3), Torre (1,4), Rey negro (0,4)")
state4 = [[0, 3, 6], [1, 4, 2]]
is_mate4 = agent.is_checkmate(state4)
print(f"  ¿Es mate? {is_mate4}")

# Caso 5: Torre en (0,5), Rey blanco en (1,4), Rey negro en (0,4)
print("\nCaso 5: Rey blanco (1,4), Torre (0,5), Rey negro (0,4)")
state5 = [[1, 4, 6], [0, 5, 2]]
is_mate5 = agent.is_checkmate(state5)
print(f"  ¿Es mate? {is_mate5}")

# Busquemos un mate válido manualmente
print("\n" + "="*70)
print("BUSCANDO MATE VÁLIDO")
print("="*70)
print("Rey negro en (0,4) - está en la primera fila (borde)")
print("Para mate necesitamos:")
print("  1. Torre atacando al rey (jaque)")
print("  2. Rey blanco controlando casillas de escape")
print()
print("Casillas de escape del rey negro en (0,4):")
print("  (-1,3), (-1,4), (-1,5) -> fuera del tablero")
print("  (0,3), (0,5) -> en la misma fila")
print("  (1,3), (1,4), (1,5) -> en la siguiente fila")
print()
print("Para mate necesitamos:")
print("  - Torre en (0,3) o (0,5) o (algo,4) atacando al rey")
print("  - Rey blanco controlando las otras casillas")
print()

# Caso 6: Torre en (2,4), Rey blanco en (1,3), Rey negro en (0,4)
print("Caso 6: Rey blanco (1,3), Torre (2,4), Rey negro (0,4)")
state6 = [[1, 3, 6], [2, 4, 2]]
is_mate6 = agent.is_checkmate(state6)
print(f"  Torre ataca columna 4: rey negro en (0,4) está en jaque")
print(f"  Rey blanco en (1,3) controla: (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4)")
print(f"  Casillas de escape: (0,3)✗rey, (0,5)?, (1,4)✗rey, (1,5)?")
print(f"  ¿Es mate? {is_mate6}")

# Caso 7: Torre en (2,4), Rey blanco en (0,5), Rey negro en (0,4)
print("\nCaso 7: Rey blanco (0,5), Torre (2,4), Rey negro (0,4)")
state7 = [[0, 5, 6], [2, 4, 2]]
is_mate7 = agent.is_checkmate(state7)
print(f"  ¿Es mate? {is_mate7}")
