"""
Ejercicio 1 - Práctica 3
Implementación de conceptos de búsqueda informada con A*

Basado en los conceptos de teoría:
- Búsqueda informada
- Algoritmo A*
- Función de evaluación f(n) = g(n) + h(n)
- Heurística admisible
"""

import heapq
from typing import List, Tuple, Set, Dict, Optional


class Node:
    """
    Representa un nodo en el espacio de búsqueda.
    
    Atributos:
    - state: Estado del nodo (posición o configuración)
    - g: Coste real desde el nodo inicial (coste acumulado)
    - h: Heurística (estimación del coste hasta el objetivo)
    - f: Función de evaluación f(n) = g(n) + h(n)
    - parent: Nodo padre para reconstruir el camino
    """
    
    def __init__(self, state: Tuple[int, int], g: float, h: float, parent: Optional['Node'] = None):
        self.state = state
        self.g = g  # Coste real desde el inicio
        self.h = h  # Heurística estimada hasta el objetivo
        self.f = g + h  # Función de evaluación
        self.parent = parent
    
    def __lt__(self, other):
        """Comparador para la cola de prioridad (menor f tiene prioridad)"""
        return self.f < other.f
    
    def __eq__(self, other):
        return self.state == other.state
    
    def __hash__(self):
        return hash(self.state)


class AStarSearch:
    """
    Implementación del algoritmo A* para búsqueda informada.
    
    Conceptos de teoría aplicados:
    - Búsqueda en grafos con heurística
    - Lista abierta (open list): nodos por explorar
    - Lista cerrada (closed list): nodos ya explorados
    - Heurística admisible: h(n) ≤ coste real
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (8, 8)):
        """
        Inicializa el espacio de búsqueda.
        
        Args:
            grid_size: Tamaño del tablero/grid (filas, columnas)
        """
        self.grid_size = grid_size
        self.obstacles: Set[Tuple[int, int]] = set()
    
    def add_obstacle(self, position: Tuple[int, int]):
        """Añade un obstáculo en la posición especificada."""
        self.obstacles.add(position)
    
    def heuristic(self, state: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Heurística admisible: Distancia de Manhattan.
        
        h(n) = |x1 - x2| + |y1 - y2|
        
        Es admisible porque nunca sobreestima el coste real
        (en un grid sin movimientos diagonales).
        
        Args:
            state: Estado actual (x, y)
            goal: Estado objetivo (x, y)
        
        Returns:
            Estimación del coste hasta el objetivo
        """
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])
    
    def get_neighbors(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Obtiene los estados sucesores (vecinos) de un estado.
        
        Movimientos permitidos: arriba, abajo, izquierda, derecha
        
        Args:
            state: Estado actual (x, y)
        
        Returns:
            Lista de estados vecinos válidos
        """
        x, y = state
        neighbors = []
        
        # Movimientos posibles: arriba, abajo, izquierda, derecha
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            # Verificar que esté dentro del grid
            if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                # Verificar que no sea un obstáculo
                if (new_x, new_y) not in self.obstacles:
                    neighbors.append((new_x, new_y))
        
        return neighbors
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Ejecuta el algoritmo A* para encontrar el camino óptimo.
        
        Algoritmo A*:
        1. Inicializar open_list con el nodo inicial
        2. Mientras open_list no esté vacía:
            a. Extraer nodo n con menor f(n)
            b. Si n es el objetivo, reconstruir y retornar el camino
            c. Añadir n a closed_list
            d. Para cada sucesor s de n:
                - Calcular g(s) = g(n) + coste(n, s)
                - Calcular h(s) usando la heurística
                - Si s no está en open_list o closed_list, añadirlo a open_list
                - Si s está en open_list con mayor g, actualizar
        
        Args:
            start: Estado inicial (x, y)
            goal: Estado objetivo (x, y)
        
        Returns:
            Lista de estados que forman el camino óptimo, o None si no existe
        """
        # Lista abierta (cola de prioridad): nodos por explorar
        open_list: List[Node] = []
        
        # Nodo inicial
        start_node = Node(start, g=0, h=self.heuristic(start, goal))
        heapq.heappush(open_list, start_node)
        
        # Lista cerrada: nodos ya explorados
        closed_set: Set[Tuple[int, int]] = set()
        
        # Diccionario para almacenar el mejor g conocido para cada estado
        g_scores: Dict[Tuple[int, int], float] = {start: 0}
        
        while open_list:
            # Extraer nodo con menor f(n)
            current = heapq.heappop(open_list)
            
            # Si alcanzamos el objetivo, reconstruir el camino
            if current.state == goal:
                return self.reconstruct_path(current)
            
            # Añadir a la lista cerrada
            closed_set.add(current.state)
            
            # Explorar vecinos (sucesores)
            for neighbor_state in self.get_neighbors(current.state):
                # Ignorar si ya fue explorado
                if neighbor_state in closed_set:
                    continue
                
                # Calcular g(s) = g(n) + coste(n, s)
                # Asumimos coste uniforme de 1 por movimiento
                tentative_g = current.g + 1
                
                # Si encontramos un mejor camino a este vecino
                if neighbor_state not in g_scores or tentative_g < g_scores[neighbor_state]:
                    g_scores[neighbor_state] = tentative_g
                    h = self.heuristic(neighbor_state, goal)
                    neighbor_node = Node(neighbor_state, g=tentative_g, h=h, parent=current)
                    heapq.heappush(open_list, neighbor_node)
        
        # No se encontró camino
        return None
    
    def reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """
        Reconstruye el camino desde el nodo inicial hasta el nodo dado.
        
        Args:
            node: Nodo final (objetivo)
        
        Returns:
            Lista de estados que forman el camino
        """
        path = []
        current = node
        
        while current is not None:
            path.append(current.state)
            current = current.parent
        
        # Invertir para obtener el camino desde inicio a objetivo
        path.reverse()
        return path
    
    def print_path(self, path: Optional[List[Tuple[int, int]]], start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Visualiza el camino encontrado en el grid.
        
        Args:
            path: Camino a visualizar
            start: Estado inicial
            goal: Estado objetivo
        """
        if path is None:
            print("No se encontró camino")
            return
        
        print(f"\nCamino encontrado de {start} a {goal}:")
        print(f"Longitud del camino: {len(path)}")
        print(f"Camino: {' -> '.join(str(state) for state in path)}")
        
        # Crear visualización del grid
        print("\nVisualización del grid:")
        for y in range(self.grid_size[1] - 1, -1, -1):
            row = []
            for x in range(self.grid_size[0]):
                if (x, y) in self.obstacles:
                    row.append('█')  # Obstáculo
                elif (x, y) == start:
                    row.append('S')  # Start
                elif (x, y) == goal:
                    row.append('G')  # Goal
                elif (x, y) in path:
                    row.append('*')  # Camino
                else:
                    row.append('·')  # Vacío
            print(' '.join(row))
        print()


def ejemplo_uso():
    """
    Ejemplo de uso del algoritmo A* aplicando conceptos de teoría.
    """
    print("=" * 60)
    print("EJERCICIO 1 - Búsqueda Informada con A*")
    print("=" * 60)
    print("\nConceptos de teoría aplicados:")
    print("- Búsqueda informada (usa heurística)")
    print("- Algoritmo A* con f(n) = g(n) + h(n)")
    print("- g(n): coste real desde el inicio")
    print("- h(n): heurística admisible (distancia Manhattan)")
    print("- Garantiza optimalidad con heurística admisible")
    
    # Crear instancia de búsqueda A*
    astar = AStarSearch(grid_size=(8, 8))
    
    # Añadir algunos obstáculos
    obstacles = [(2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (5, 3), (5, 2)]
    for obs in obstacles:
        astar.add_obstacle(obs)
    
    # Definir inicio y objetivo
    start = (0, 0)
    goal = (7, 7)
    
    print(f"\nProblema: Encontrar camino desde {start} hasta {goal}")
    print(f"Obstáculos: {len(obstacles)}")
    
    # Ejecutar búsqueda A*
    path = astar.search(start, goal)
    
    # Mostrar resultado
    astar.print_path(path, start, goal)
    
    if path:
        # Calcular coste total
        print(f"Coste total del camino: {len(path) - 1}")


if __name__ == "__main__":
    ejemplo_uso()
