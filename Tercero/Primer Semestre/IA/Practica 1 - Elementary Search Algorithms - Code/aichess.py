#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi

Edited and completed by: Jose Candon and Daniel Barceló on Fri Oct 03 14:04 2025

"""
import copy
import math
import chess
import board
import numpy as np
import sys
import queue
from typing import List
RawStateType = List[List[List[int]]]
from itertools import permutations

class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game
        
    listNextStates : list
        List of next possible states for the current player.

    listVisitedStates : list
        List of all visited states during A*.

    listVisitedSituations : list
        List of visited game situations (state + color) for minimax/alpha-beta pruning.

    pathToTarget : list
        Sequence of states from the initial state to the target (used by A*).

    depthMax : int
        Maximum search depth for minimax/alpha-beta searches.

    dictPath : dict
        Dictionary used to reconstruct the path in A* search.

    Methods:
    --------
    copyState(state) -> list
        Returns a deep copy of the given state.

    isVisitedSituation(color, mystate) -> bool
        Checks whether a given state with a specific color has already been visited.

    getListNextStatesW(myState) -> list
        Returns a list of possible next states for the white pieces.

    getListNextStatesB(myState) -> list
        Returns a list of possible next states for the black pieces.

    isSameState(a, b) -> bool
        Checks whether two states represent the same board configuration.

    isVisited(mystate) -> bool
        Checks if a given state has been visited in search algorithms.

    getCurrentState() -> list
        Returns the combined state of both white and black pieces.

    isCheckMate(mystate) -> bool
        Determines if a state represents a checkmate configuration.

    heuristica(currentState, color) -> int
        Calculates a heuristic value for the current state from the perspective of the given color.

    movePieces(start, depthStart, to, depthTo) -> None
        Moves all pieces along the path between two states.

    changeState(start, to) -> None
        Moves a single piece from start state to to state.

    reconstructPath(state, depth) -> None
        Reconstructs the path from initial state to the target state for A*.

    h(state) -> int       
        Heuristic function for A* search.

    DepthFirstSearch(currentState, depth) -> bool
        Depth-first search algorithm.

    worthExploring(state, depth) -> bool
        Checks if a state is worth exploring during search using the optimised DFS algorithm.

    DepthFirstSearchOptimized(currentState, depth) -> bool
        Optimized depth-first search algorithm.

    BreadthFirstSearch(currentState, depth) -> None
        Breadth-first search algorithm.

    AStarSearch(currentState) 
        A* search algorithm -> To be implemented by you

    """

    # ============================== NO MODIFICADAS ==============================
    
    def changeState(self, start, to):
        # Determine which piece has moved from the start state to the next state
        if start[0] == to[0]:
            movedPieceStart = 1
            movedPieceTo = 1
        elif start[0] == to[1]:
            movedPieceStart = 1
            movedPieceTo = 0
        elif start[1] == to[0]:
            movedPieceStart = 0
            movedPieceTo = 1
        else:
            movedPieceStart = 0
            movedPieceTo = 0
        # Move the piece that changed
        self.chess.moveSim(start[movedPieceStart], to[movedPieceTo])       

    def DepthFirstSearch(self, currentState, depth):
        # We visited the node, therefore we add it to the list
        # In DF, when we add a node to the list of visited, and when we have
        # visited all nodes, we remove it from the list of visited ones
        self.listVisitedStates.append(currentState)

        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True

        if depth + 1 <= self.depthMax:
            for son in self.getListNextStatesW(currentState):
                if not self.isVisited(son):
                    # in the state 'son', the first piece is the one just moved
                    # We check which piece in currentState matches the one moved
                    if son[0][2] == currentState[0][2]:
                        movedPieceIndex = 0
                    else:
                        movedPieceIndex = 1

                    # we move the piece to the new position
                    self.chess.moveSim(currentState[movedPieceIndex], son[0])
                    # We call the method again with 'son', increasing depth
                    if self.DepthFirstSearch(son, depth + 1):
                        # If the method returns True, this means that there has
                        # been a checkmate
                        # We add the state to the pathToTarget
                        self.pathToTarget.insert(0, currentState)
                        return True
                    # we reset the board to the previous state
                    self.chess.moveSim(son[0], currentState[movedPieceIndex])

        # We remove the node from the list of visited nodes
        # since we explored all successors
        self.listVisitedStates.remove(currentState)

    def worthExploring(self, state, depth):
        # First of all, check that the depth is not bigger than depthMax
        if depth > self.depthMax:
            return False
        visited = False
        # check if the state has been visited
        for perm in list(permutations(state)):
            permStr = str(perm)
            if permStr in list(self.dictVisitedStates.keys()):
                visited = True
                # If the state has been visited at a larger depth,
                # we are interested in visiting it again
                if depth < self.dictVisitedStates[perm]:
                    # Update the depth associated with the state
                    self.dictVisitedStates[permStr] = depth
                    return True
        # If never visited, add it to the dictionary at the current depth
        if not visited:
            permStr = str(state)
            self.dictVisitedStates[permStr] = depth
            return True

    def DepthFirstSearchOptimized(self, currentState, depth):
        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True

        for son in self.getListNextStatesW(currentState):
            if self.worthExploring(son, depth + 1):
                # in state 'son', the first piece is the one just moved
                # we check which piece of currentState matches the one just moved
                if son[0][2] == currentState[0][2]:
                    movedPieceIndex = 0
                else:
                    movedPieceIndex = 1
                # move the piece to the new position
                self.chess.moveSim(currentState[movedPieceIndex], son[0])
                # recursive call with increased depth
                if self.DepthFirstSearchOptimized(son, depth + 1):
                    # If the method returns True, this means there was a checkmate
                    # add the state to the pathToTarget
                    self.pathToTarget.insert(0, currentState)
                    return True
                # restore the board to its previous state
                self.chess.moveSim(son[0], currentState[movedPieceIndex])

    def BreadthFirstSearch(self, currentState, depth):
        """
        Checkmate from currentStateW
        """
        BFSQueue = queue.Queue()
        # The root node has no parent, thus we add None, and -1 as the parent's depth
        self.dictPath[str(currentState)] = (None, -1)
        depthCurrentState = 0
        BFSQueue.put(currentState)
        self.listVisitedStates.append(currentState)
        # iterate until there are no more candidate nodes
        while BFSQueue.qsize() > 0:
            # Get the next node
            node = BFSQueue.get()
            depthNode = self.dictPath[str(node)][1] + 1
            if depthNode > self.depthMax:
                break
            # If not the root node, move the pieces from the previous to the current state
            if depthNode > 0:
                self.movePieces(currentState, depthCurrentState, node, depthNode)
            if self.isCheckMate(node):
                # If it is checkmate, reconstruct the optimal path found
                self.reconstructPath(node, depthNode)
                break
            for son in self.getListNextStatesW(node):
                if not self.isVisited(son):
                    self.listVisitedStates.append(son)
                    BFSQueue.put(son)
                    self.dictPath[str(son)] = (node, depthNode)
            currentState = node
            depthCurrentState = depthNode

    # getListNextStatesW: Wrapper que llama a la generación de sucesores sobre
    # `boardSim` y devuelve una copia segura de la lista de sucesores.
    def getListNextStatesW(self, myState):
        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()
        return self.listNextStates


    # getListNextStatesB: Igual que su versión blanca, pero para piezas negras.
    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()
        return self.listNextStates


    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.depthMax = 8
        # Dictionary to reconstruct the visited path
        self.dictPath = {}
        # Prepare a dictionary to control the visited state and at which
        # depth they were found for DepthFirstSearchOptimized
        self.dictVisitedStates = {}

    def copyState(self, state):
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState

    def isVisitedSituation(self, color, mystate):
        if (len(self.listVisitedSituations) > 0):
            perm_state = list(permutations(mystate))
            isVisited = False
            for j in range(len(perm_state)):
                for k in range(len(self.listVisitedSituations)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedSituations.__getitem__(k)[1]) and color == \
                            self.listVisitedSituations.__getitem__(k)[0]:
                        isVisited = True
            return isVisited
        else:
            return False

    def isSameState(self, a, b):
        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):
            if a[k] not in b:
                isSameState1 = False
        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):
            if b[k] not in a:
                isSameState2 = False
        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):
        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))
            isVisited = False
            for j in range(len(perm_state)):
                for k in range(len(self.listVisitedStates)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True
            return isVisited
        else:
            return False

    # newBoardSim(listStates)
    # Reconstruye `boardSim` a partir de una lista de estados compacta [row,col,code].
    # Usado por A* para evitar sincronizar movimientos incrementalmente.
    def newBoardSim(self, listStates):
        # crear un nuevo `boardSim` a partir del mapa codificado
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]
        self.chess.newBoardSim(TA)

    # getCurrentState()
    # Devuelve la lista combinada de estados actuales de blancas y negras
    # (formato usado por las funciones de búsqueda).
    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

    def getNextPositions(self, state):
        # Given a state, we check the next possible states
        # From these, we return a list with position, i.e., [row, column]
        if state == None:
            return None
        if state[2] > 6:
            nextStates = self.getListNextStatesB([state])
        else:
            nextStates = self.getListNextStatesW([state])
        nextPositions = []
        for i in nextStates:
            nextPositions.append(i[0][0:2])
        return nextPositions

    def movePieces(self, start, depthStart, to, depthTo):
        
        # To move from one state to the next we will need to find
        # the state in common, and then move until the node 'to'
        moveList = []
        # We want that the depths are equal to find a common ancestor
        nodeTo = to
        nodeStart = start
        # if the depth of the node To is larger than that of start, 
        # we pick the ancesters of the node until being at the same
        # depth
        while(depthTo > depthStart):
            moveList.insert(0,to)
            nodeTo = self.dictPath[str(nodeTo)][0]
            depthTo-=1
        # Analogous to the previous case, but we trace back the ancestors
        #until the node 'start'
        while(depthStart > depthTo):
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # We move the piece the the parerent state of nodeStart
            self.changeState(nodeStart, ancestreStart)
            nodeStart = ancestreStart
            depthStart -= 1

        moveList.insert(0,nodeTo)
        # We seek for common node
        while nodeStart != nodeTo:
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # Move the piece the the parerent state of nodeStart
            self.changeState(nodeStart,ancestreStart)
            # pick the parent of nodeTo
            nodeTo = self.dictPath[str(nodeTo)][0]
            # store in the list
            moveList.insert(0,nodeTo)
            nodeStart = ancestreStart
        # Move the pieces from the node in common
        # until the node 'to'
        for i in range(len(moveList)):
            if i < len(moveList) - 1:
                self.changeState(moveList[i],moveList[i+1])

    def reconstructPath(self, state, depth):
        # Once the solution is found, reconstruct the path taken to reach it
        for i in range(depth):
            self.pathToTarget.insert(0, state)
            # For each node, retrieve its parent from dictPath
            state = self.dictPath[str(state)][0]
        # Insert the root node at the beginning
        self.pathToTarget.insert(0, state)

    # getWhiteState(currentState)
    # Normaliza y extrae la representación canónica de blancas: [rey, torre]
    # (sirve para A* que trabaja con un estado reducido y consistente).

    def getWhiteState(self, currentState):
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState != None:
            whiteState.append(wrState)
        return whiteState

    # getBlackState(currentState)
    # Igual que `getWhiteState` pero para negras (rey y torre si existe).

    def getBlackState(self, currentState):
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState != None:
            blackState.append(brState)
        return blackState


    def getMovement(self, state, nextState):
        # Given a state and a successor state, return the postiion of the piece that has been moved in both states
        pieceState = None
        pieceNextState = None
        for piece in state:
            if piece not in nextState:
                movedPiece = piece[2]
                pieceNext = self.getPieceState(nextState, movedPiece)
                if pieceNext != None:
                    pieceState = piece
                    pieceNextState = pieceNext
                    break
        return [pieceState, pieceNextState]


# ================================================================================
# ============================== FIN NO MODIFICADAS ==============================
# ================================================================================

        
    # Función para detectar jaque mate
    def isCheckMate(self, mystate=None, return_escapes=False):
        """
        Comprueba si la posición actual en `self.chess.boardSim` es mate para las negras.
        Si `return_escapes` es True devuelve una tupla (is_mate, escapes) donde `escapes` es
        la lista de jugadas negras que eliminan el jaque; si es False devuelve solo el booleano.
        """
        board_sim = self.chess.boardSim

        # Detector puro de ataques (sin efectos secundarios)
        def square_attacked_pure(r, c):
            b = board_sim.board
            for i in range(8):
                for j in range(8):
                    p = b[i][j]
                    if p is None or not p.color:
                        continue
                    name = p.name
                    di = r - i
                    dj = c - j
                    adi = abs(di)
                    adj = abs(dj)
                    if name == 'P':
                        if (i - 1 == r) and (j - 1 == c or j + 1 == c):
                            return True
                        continue
                    if name == 'N':
                        if (adi == 2 and adj == 1) or (adi == 1 and adj == 2):
                            return True
                        continue
                    if name == 'K':
                        if max(adi, adj) == 1:
                            return True
                        continue
                    if name == 'R' or name == 'Q':
                        if i == r:
                            step = 1 if j < c else -1
                            blocked = False
                            x = j + step
                            while x != c:
                                if b[i][x] is not None:
                                    blocked = True
                                    break
                                x += step
                            if not blocked:
                                return True
                        if j == c:
                            step = 1 if i < r else -1
                            blocked = False
                            x = i + step
                            while x != r:
                                if b[x][j] is not None:
                                    blocked = True
                                    break
                                x += step
                            if not blocked:
                                return True
                    if name == 'B' or name == 'Q':
                        if adi == adj and adi != 0:
                            step_i = 1 if i < r else -1
                            step_j = 1 if j < c else -1
                            x = i + step_i
                            y = j + step_j
                            blocked = False
                            while x != r and y != c:
                                if b[x][y] is not None:
                                    blocked = True
                                    break
                                x += step_i
                                y += step_j
                            if not blocked:
                                return True
            return False

        # localizar la posición del rey negro
        bk_pos = None
        for i in range(8):
            for j in range(8):
                p = board_sim.board[i][j]
                if p is not None and p.name == 'K' and not p.color:
                    bk_pos = (i, j)
                    break
            if bk_pos is not None:
                break

        if bk_pos is None:
            if return_escapes:
                return (False, [])
            return False

        # si el rey no está en jaque, no es mate
        if not square_attacked_pure(bk_pos[0], bk_pos[1]):
            if return_escapes:
                return (False, [])
            return False

        # simular todas las jugadas legales negras y coleccionar escapes
        escapes = []
        for i in range(8):
            for j in range(8):
                bp = board_sim.board[i][j]
                if bp is None or bp.color:
                    continue
                for r in range(8):
                    for c in range(8):
                        dest = board_sim.board[r][c]
                        if dest is not None and dest.color == False:
                            continue
                        try:
                            if not bp.is_valid_move(board_sim, (i, j), (r, c)):
                                continue
                        except Exception:
                            continue

                        # simular movimiento
                        orig_from = board_sim.board[i][j]
                        orig_to = board_sim.board[r][c]
                        board_sim.board[i][j] = None
                        board_sim.board[r][c] = bp

                        # localizar nueva posición del rey negro si cambia
                        if bp.name == 'K':
                            new_bk_pos = (r, c)
                        else:
                            new_bk_pos = None
                            for x in range(8):
                                for y in range(8):
                                    p = board_sim.board[x][y]
                                    if p is not None and p.name == 'K' and not p.color:
                                        new_bk_pos = (x, y)
                                        break
                                if new_bk_pos is not None:
                                    break

                        attacked_after = True
                        if new_bk_pos is not None:
                            attacked_after = square_attacked_pure(new_bk_pos[0], new_bk_pos[1])

                        # revertir simulación
                        board_sim.board[i][j] = orig_from
                        board_sim.board[r][c] = orig_to

                        if not attacked_after:
                            escapes.append(((i, j), (r, c)))
                            if not return_escapes:
                                # existe una defensa -> no es mate
                                return False

        is_mate = len(escapes) == 0
        if return_escapes:
            return (is_mate, escapes)
        return is_mate


    # Devuelve una lista de jugadas negras que eliminarían el jaque al rey.
    # Cada jugada se representa como ((r_from,c_from),(r_to,c_to)).
    # list_black_legal_escapes removed: its logic is now integrated into isCheckMate
    
    # Busca y devuelve la entrada de `state` correspondiente al identificador `piece`.
    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState


    def h(self, state):
        """Heurística mejorada:
        - kingDistance: distancia de Manhattan desde el rey blanco hasta la casilla objetivo (rey negro).
        - rookAlignDist: movimientos mínimos para que la torre se alinee en la misma fila o columna que el objetivo.
        Se devuelve el máximo de ambas medidas para mantener admisibilidad y dar una estimación
        más informativa que favorezca la coordinación rey+torre.
        """

        whiteKingPosition = state[0]
        whiteRookPosition = state[1]
        targetKingPosition = (0, 5)
        kingDistance = abs(whiteKingPosition[0] - targetKingPosition[0]) + abs(whiteKingPosition[1] - targetKingPosition[1])
        # distancia de alineación de la torre: movimientos para que la torre esté en la misma fila O columna que el objetivo
        rookAlignRow = abs(whiteRookPosition[0] - targetKingPosition[0])
        rookAlignCol = abs(whiteRookPosition[1] - targetKingPosition[1])
        rookAlignDist = min(rookAlignRow, rookAlignCol)
        # la heurística es el máximo de las dos distancias componentes
        return max(kingDistance, rookAlignDist)


    def AStarSearch(self, currentState):
        import heapq
        # AStarSearch(currentState)
        # Implementa búsqueda A* sobre estados normalizados (blancas: [rey, torre]).
        # Estrategia principal:
        # 1) Normalizar estado de inicio y usar `heapq` para la frontera con (f, estado, g).
        # 2) Al extraer un nodo, reconstruir `boardSim` a partir del estado blanco
        #    y las piezas negras iniciales para generar sucesores.
        # 3) Usar `isCheckMate` dinámico como condición objetivo y `dictPath`
        #    para reconstruir la ruta cuando se encuentre mate.
        
        # Normalizar el estado de inicio a (rey, torre)
        currentState = self.getWhiteState(currentState)
        # las entradas de la frontera son tuplas (f_score, estado, g_cost)
        frontier = []
        start_key = str(currentState)
        heapq.heappush(frontier, (self.h(currentState), currentState, 0))

        # coste hasta ahora (g) y mapa de padres para reconstrucción (almacenar estados normalizados)
        cost_so_far = {start_key: 0}
        self.dictPath[start_key] = (None, 0)
        found = False

        # Guarda lista de nodos visitados
        self.listVisitedStates = []
        
        
        # mientras expandimos nodos, reconstruir boardSim desde el nodo blanco normalizado
        # más las piezas negras originales para evitar sincronizaciones complejas de movimientos
        while frontier:
            f, node, g = heapq.heappop(frontier)
            # debug
            print(f"Exploring state: {node}")
            self.listVisitedStates.append(node)   # ← estado normalizado [rey, torre]

            # reconstruir boardSim para que la generación de sucesores sea precisa
            try:
                # node es el estado blanco normalizado (lista de estados de piezas blancas)
                # combinar con el estado negro original desde self.chess.board.currentStateB
                full_state = []
                # piezas blancas del nodo
                for w in node:
                    full_state.append(w)
                # piezas negras del tablero inicial (sin cambios durante la búsqueda A*)
                for b in self.chess.board.currentStateB:
                    full_state.append(b)
                # reconstruir tablero de simulación
                self.newBoardSim(full_state)
            except Exception as e:
                print(f"Failed to rebuild boardSim for node {node}: {e}")
                continue

            # comprobación de objetivo (ahora devuelve escapes si se solicita)
            is_mate_result = self.isCheckMate(node, return_escapes=True)
            if isinstance(is_mate_result, tuple):
                is_mate, escapes = is_mate_result
            else:
                is_mate = is_mate_result
                escapes = []
            if is_mate:
                print("Checkmate found!")
                # reconstruir ruta usando dictPath; node está normalizado
                self.reconstructPath(node, g)
                # diagnóstico: mostrar posibles escapes si los hubiera
                if len(escapes) == 0:
                    print("No legal black escapes found by diagnostic (confirmed mate)")
                else:
                    print("Diagnostic found black escape moves (not mate):")
                    for mv in escapes:
                        print(mv)
                found = True
                break

            # expandir sucesores; asegurar que el nodo pasado al generador de sucesores está normalizado
            for son in self.getListNextStatesW(node):
                # normalizar sucesor a forma canónica (rey, torre)
                son_norm = self.getWhiteState(son)
                son_key = str(son_norm)
                tentative_g = g + 1
                # si ya tenemos un camino mejor hacia son, saltar
                if son_key in cost_so_far and tentative_g >= cost_so_far[son_key]:
                    continue
                cost_so_far[son_key] = tentative_g
                # puntero al padre y profundidad (almacenar padre como estado normalizado)
                self.dictPath[son_key] = (node, tentative_g)
                heapq.heappush(frontier, (tentative_g + self.h(son_norm), son_norm, tentative_g))

        if found:
            print("Minimal depth to checkmate:", len(self.pathToTarget) - 1)
        else:
            print("No checkmate found within search limits.")



if __name__ == "__main__":
    # Initialize an empty 8x8 chess board
    TA = np.zeros((8, 8))
    # Load initial positions of the pieces
    # White pieces
    TA[7][0] = 2  
    TA[7][5] = 6   

    # REY NEGRO
    TA[6][7] = 12  
    print("Starting AI chess...")
    aichess = Aichess(TA, True)
    # Print initial board
    print("Printing board:")
    aichess.chess.boardSim.print_board()
    # Normalizar el estado blanco (rey en índice 0, torre en índice 1)
    currentState = aichess.getWhiteState(aichess.getCurrentState())
    print("Current State:", currentState, "\n")

    # Run A* search
    aichess.AStarSearch(currentState)
    # Mostrar nodos visitados por A*
    print("\n#A* visited nodes:")
    for idx, node in enumerate(aichess.listVisitedStates):
        print(f"{idx}: {node}")
    print("#A* move sequence:", aichess.pathToTarget)
    print("A* End\n")
    print("Printing final board after A*:")
    aichess.chess.boardSim.print_board()
