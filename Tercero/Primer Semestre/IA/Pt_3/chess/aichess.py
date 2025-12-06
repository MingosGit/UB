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
        # - Siempre hay 3 piezas: rey blanco (color True), torre blanca (color True), rey negro (color False).
        # - La posición del rey negro se considera fija para la generación de movimientos por parte del usuario,
        #   pero aún debemos comprobar si el rey negro tendría algún movimiento legal para escapar del jaque.
        # La función devuelve un booleano o (es_mate, escapes) cuando return_escapes es True.

        board_sim = self.chess.boardSim

        # buscar rey negro
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
            # no hay rey negro -> no es mate
            if return_escapes:
                return (False, [])
            return False

        # auxiliar: ¿está la casilla atacada por alguna pieza blanca (rey o torre)?
        def square_attacked(r, c):
            b = board_sim.board
            # comprobar ataques de torre (líneas rectas)
            # mirar en las cuatro direcciones
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for di, dj in dirs:
                x, y = r + di, c + dj
                while 0 <= x < 8 and 0 <= y < 8:
                    p = b[x][y]
                    if p is None:
                        x += di; y += dj
                        continue
                    # si la pieza es torre blanca, ataca
                    if p.color and p.name == 'R':
                        return True
                    # cualquier otra pieza bloquea
                    break

            # comprobar adyacencia del rey blanco
            for x in range(max(0, r - 1), min(7, r + 1) + 1):
                for y in range(max(0, c - 1), min(7, c + 1) + 1):
                    if x == r and y == c:
                        continue
                    p = b[x][y]
                    if p is not None and p.color and p.name == 'K':
                        return True
            return False

        # si el rey negro no está actualmente en jaque -> no es mate
        if not square_attacked(bk_pos[0], bk_pos[1]):
            if return_escapes:
                return (False, [])
            return False

        # generar movimientos del rey negro (8 casillas circundantes) y comprobar si alguno es legal y no está atacado
        escapes = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = bk_pos[0] + di
                nj = bk_pos[1] + dj
                if not (0 <= ni < 8 and 0 <= nj < 8):
                    continue
                dest = board_sim.board[ni][nj]
                # no puede moverse a una casilla ocupada por otra pieza negra (no existen en este escenario)
                if dest is not None and dest.color == False:
                    continue
                # no puede moverse a una casilla adyacente al rey blanco
                # pero confiaremos en square_attacked para incluir ataques del rey

                # simular movimiento: colocar rey negro en (ni, nj), vaciar la casilla original
                orig_from = board_sim.board[bk_pos[0]][bk_pos[1]]
                orig_to = board_sim.board[ni][nj]
                board_sim.board[bk_pos[0]][bk_pos[1]] = None
                board_sim.board[ni][nj] = orig_from

                attacked = square_attacked(ni, nj)

                # revertir
                board_sim.board[bk_pos[0]][bk_pos[1]] = orig_from
                board_sim.board[ni][nj] = orig_to

                if not attacked:
                    escapes.append(((bk_pos[0], bk_pos[1]), (ni, nj)))
                    if not return_escapes:
                        return False

        is_mate = len(escapes) == 0
        if return_escapes:
            return (is_mate, escapes)
        return is_mate



    
    # Busca y devuelve la entrada de `state` correspondiente al identificador `piece`.
    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState


    def h(self, state):
        """Heurística admisible para K+R vs K.

        Estrategia:
        - Usar la posición real del rey negro (desde self.chess.board.currentStateB) en lugar de una constante.
        - kingDistance: mínima distancia de Manhattan desde el rey blanco hasta cualquier casilla adyacente
          al rey negro (el rey blanco debe acercarse para ayudar al mate).
        - rookAlignDist: lower bound para que la torre quede en la misma fila o columna que el rey negro: 0 si
          ya está alineada, 1 en caso contrario (la torre puede alinearse en al menos una jugada si no está alineada).
        La heurística devuelve el máximo de ambas cantidades (cota inferior admisible del número de jugadas necesarias).
        """

        # estado: state[0] = [wr, wc, code] for white king; state[1] = [rr, rc, code] for white rook
        whiteKingPosition = state[0]
        whiteRookPosition = state[1]

        # obtener posición del rey negro desde el tablero inicial/current (A* mantiene piezas negras constantes)
        bk_list = self.chess.board.currentStateB
        if len(bk_list) == 0:
            # sin información del rey negro, caer de vuelta a una cota simple
            targetKingPosition = (0, 5)
        else:
            bk = bk_list[0]
            targetKingPosition = (bk[0], bk[1])

        # calcular distancia mínima del rey blanco a cualquier casilla adyacente al rey negro
        bk_r, bk_c = targetKingPosition
        min_king_dist = math.inf
        for ar in range(max(0, bk_r - 1), min(7, bk_r + 1) + 1):
            for ac in range(max(0, bk_c - 1), min(7, bk_c + 1) + 1):
                if ar == bk_r and ac == bk_c:
                    continue
                dist = abs(whiteKingPosition[0] - ar) + abs(whiteKingPosition[1] - ac)
                if dist < min_king_dist:
                    min_king_dist = dist

        if min_king_dist == math.inf:
            min_king_dist = 0

        # rook alignment lower bound: 0 if same row or same column as black king, else 1
        if whiteRookPosition[0] == bk_r or whiteRookPosition[1] == bk_c:
            rookAlignDist = 0
        else:
            rookAlignDist = 1

        # la heurística es el máximo de ambas componentes (cota admisible)
        return int(max(min_king_dist, rookAlignDist))


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
                # evitar estados ilegales donde los reyes quedan adyacentes (no permitido)
                bk_list_local = self.chess.board.currentStateB
                if len(bk_list_local) > 0:
                    bk_local = bk_list_local[0]
                    bk_r_local, bk_c_local = bk_local[0], bk_local[1]
                    wk_local = son_norm[0]
                    if abs(wk_local[0] - bk_r_local) <= 1 and abs(wk_local[1] - bk_c_local) <= 1:
                        # descartamos este sucesor
                        continue
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
    TA[1][0] = 12  
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
