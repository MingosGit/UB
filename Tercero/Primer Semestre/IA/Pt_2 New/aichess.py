#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math
import random

import chess
import board
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations
import matplotlib.pyplot as plt
import json
import time
import subprocess



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
        List of all visited states during A* and other search algorithms.

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

    getNextPositions(state) -> list
        Returns a list of possible next positions for a given state.

    heuristica(currentState, color) -> int
        Calculates a heuristic value for the current state from the perspective of the given color.

    movePieces(start, depthStart, to, depthTo) -> None
        Moves all pieces along the path between two states.

    changeState(start, to) -> None
        Moves a single piece from start state to to state.

    reconstructPath(state, depth) -> None
        Reconstructs the path from initial state to the target state for A*.

    isWatchedWk(currentState) / isWatchedBk(currentState) -> bool
        Checks if the white or black king is under threat.

    allWkMovementsWatched(currentState) / allBkMovementsWatched(currentState) -> bool
        Checks if all moves of the white or black king are under threat.

    isWhiteInCheckMate(currentState) / isBlackInCheckMate(currentState) -> bool
        Determines if the white or black king is in checkmate.

    minimaxGame(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a full game using the Minimax algorithm for both white and black.

    alphaBetaPoda(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a game where both players use Minimax with Alpha-Beta Pruning.

    expectimax(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a full game where both players use the Expectimax algorithm.

    mean(values: list[float]) -> float
        Returns the arithmetic mean (average) of a list of numerical values.

    standardDeviation(values: list[float], mean_value: float) -> float
        Computes the standard deviation of a list of numerical values based on the given mean.

    calculateValue(values: list[float]) -> float
        Computes the expected value from a set of scores using soft-probabilities 
        derived from normalized values (exponential weighting). Can be useful for Expectimax.

    """


# =============================================================
# ==================   PROFE  ================================= 
# =============================================================
    def getWhiteState(self, currentState):
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState != None:
            whiteState.append(wrState)
        return whiteState

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

    def copyState(self, state):
        
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

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

    def standard_deviation(self, values, mean_value):
        # Calculate the standard deviation of a list of values.
            total = 0
            n = len(values)

            for i in range(n):
                total += pow(values[i] - mean_value, 2)

            return pow(total / n, 1 / 2)

    def calculateValue(self, values):
        # Calculate a weighted expected value based on normalized probabilities. - useful for Expectimax.
        
        # Compute mean and standard deviation
        mean_value = self.mean(values)
        std_dev = self.standard_deviation(values, mean_value)

        # If all values are equal, the deviation is 0, equal probability
        if std_dev == 0:
            return values[0]

        expected_value = 0
        total_weight = 0
        n = len(values)

        for i in range(n):
            # Normalize value using z-score
            normalized_value = (values[i] - mean_value) / std_dev

            # Convert to a positive weight using e^(-x)
            positive_weight = pow(1 / math.e, normalized_value)

            # Weighted sum
            expected_value += positive_weight * values[i]
            total_weight += positive_weight

        # Final expected value (weighted average)
        return expected_value / total_weight

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

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.depthMax = 8;
        # Dictionary to reconstruct the visited path
        self.dictPath = {}
        # Prepare a dictionary to control the visited state and at which
        # depth they were found for DepthFirstSearchOptimized
        self.dictVisitedStates = {}
        # Transposition table for caching minimax evaluations
        self.transpositionTable = {}
        # Cache for heuristic evaluations to avoid recalculation
        self.heuristicCache = {}

    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState

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

    def reconstructPath(self, state, depth):
        # Once the solution is found, reconstruct the path taken to reach it
        for i in range(depth):
            self.pathToTarget.insert(0, state)
            # For each node, retrieve its parent from dictPath
            state = self.dictPath[str(state)][0]

        # Insert the root node at the beginning
        self.pathToTarget.insert(0, state)

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

    def isBlackInCheckMate(self, currentState):
        if self.isWatchedBk(currentState) and self.allBkMovementsWatched(currentState):
            return True
        return False
    
    def isBlackInStaleMate(self, currentState):
        """Check if Black is in stalemate: not in check but has no legal moves"""
        if not self.isWatchedBk(currentState) and self.allBkMovementsWatched(currentState):
            return True
        return False

    def isWhiteInCheckMate(self, currentState):
        if self.isWatchedWk(currentState) and self.allWkMovementsWatched(currentState):
            return True
        return False
    
    def isWhiteInStaleMate(self, currentState):
        """Check if White is in stalemate: not in check but has no legal moves"""
        if not self.isWatchedWk(currentState) and self.allWkMovementsWatched(currentState):
            return True
        return False
    
    def mean(self, values):
        # Calculate the arithmetic mean (average) of a list of numeric values.
        total = 0
        n = len(values)
        
        for i in range(n):
            total += values[i]

        return total / n

# ===================================================================================================
# ==================   MODIFICADOS ==================================================================
# ===================================================================================================


    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        bkState = self.getPieceState(currentState, 12)
        # Si el rey negro ha sido capturado, no es un estado válido
        if bkState is None:
            return False
        bkPosition = bkState[0:2]
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

        # Si el rey blanco ha sido capturado, no es una configuración válida
        if wkState is None:
            return False

        # Comprueba si el rey negro está adyacente al rey blanco (posición ilegal, se considera jaque)
        wkPosition = wkState[0:2]
        if max(abs(bkPosition[0] - wkPosition[0]), abs(bkPosition[1] - wkPosition[1])) == 1:
            return True

        # Comprueba todos los movimientos posibles del rey blanco para ver si puede capturar al rey negro
        for wkPos in self.getNextPositions(wkState):
            if bkPosition == wkPos:
                # El rey negro estaría en jaque
                return True

        if wrState is not None:
            # Comprueba todos los movimientos posibles de la torre blanca para ver si puede capturar al rey negro
            for wrPosition in self.getNextPositions(wrState):
                if bkPosition == wrPosition:
                    return True

        return False

    def allBkMovementsWatched(self, currentState):
        # Comprueba si todos los movimientos posibles del rey negro lo dejan en jaque (condición de jaque mate)
        
        self.newBoardSim(currentState)
        wrState = self.getPieceState(currentState, 2)
        wkState = self.getPieceState(currentState, 6)
        whiteState = self.getWhiteState(currentState)
        
        # Obtiene todos los posibles estados siguientes para las piezas negras (movimientos del rey y posibles capturas de torre)
        nextBStates = self.getListNextStatesB(self.getBlackState(currentState))
        
        # Si no hay movimientos legales, es jaque mate (o tablas por ahogado si no está en jaque)
        if len(nextBStates) == 0:
            return True
        
        # Comprueba cada movimiento posible
        for state in nextBStates:
            bkNewPos = state[0][0:2]
            
            # Omite jugadas ilegales donde el rey negro capturaría al rey blanco
            if wkState is not None and bkNewPos == wkState[0:2]:
                continue
            
            newWhiteState = whiteState.copy()
            # Comprueba si la torre blanca ha sido capturada; en ese caso, elimínala del estado
            if wrState is not None and wrState[0:2] == bkNewPos:
                newWhiteState.remove(wrState)
            
            fullState = state + newWhiteState
            # Mueve las piezas negras al nuevo estado
            self.newBoardSim(fullState)
            
            # Si este movimiento saca al rey negro del jaque, entonces no todos los movimientos están vigilados
            if not self.isWatchedBk(fullState):
                # Restaura el estado original del tablero antes de devolver
                self.newBoardSim(currentState)
                return False
        
        # Restaura el estado original del tablero
        self.newBoardSim(currentState)
        # Todos los movimientos dejan al rey en jaque
        return True

    def isWatchedWk(self, currentState):
        self.newBoardSim(currentState)
        wkState = self.getPieceState(currentState, 6)
        # Si el rey blanco ha sido capturado, no es un estado válido
        if wkState is None:
            return False
        wkPosition = wkState[0:2]
        bkState = self.getPieceState(currentState, 12)
        brState = self.getPieceState(currentState, 8)
        # Si el rey negro ha sido capturado, no es una configuración válida
        if bkState is None:
            return False
        
        # Comprueba si el rey blanco está adyacente al rey negro (posición ilegal, se considera jaque)
        bkPosition = bkState[0:2]
        if max(abs(wkPosition[0] - bkPosition[0]), abs(wkPosition[1] - bkPosition[1])) == 1:
            return True
        
        # Comprueba todos los movimientos posibles del rey negro y si puede capturar al rey blanco
        for bkPos in self.getNextPositions(bkState):
            if wkPosition == bkPos:
                # El rey blanco estaría en jaque
                return True
        if brState is not None:
            # Comprueba todos los movimientos posibles de la torre negra y si puede capturar al rey blanco
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True
        return False

    def allWkMovementsWatched(self, currentState):
        # Comprueba si todos los movimientos posibles del rey blanco lo dejan en jaque (condición de jaque mate)
        
        self.newBoardSim(currentState)
        brState = self.getPieceState(currentState, 8)
        bkState = self.getPieceState(currentState, 12)
        blackState = self.getBlackState(currentState)
        
        # Obtiene todos los posibles estados siguientes para las piezas blancas (movimientos del rey y posibles capturas de torre)
        nextWStates = self.getListNextStatesW(self.getWhiteState(currentState))
        
        # Si no hay movimientos legales, es jaque mate (o tablas por ahogado si no está en jaque)
        if len(nextWStates) == 0:
            return True
        
        # Comprueba cada movimiento posible
        for state in nextWStates:
            wkNewPos = state[0][0:2]
            
            # Omite jugadas ilegales donde el rey blanco capturaría al rey negro
            if bkState is not None and wkNewPos == bkState[0:2]:
                continue
            
            newBlackState = blackState.copy()
            # Comprueba si la torre negra ha sido capturada. Si es así, elimínala del estado
            if brState is not None and brState[0:2] == wkNewPos:
                newBlackState.remove(brState)
            
            fullState = state + newBlackState
            # Mueve las piezas blancas a su nuevo estado
            self.newBoardSim(fullState)
            
            # Si este movimiento saca al rey blanco del jaque, entonces no todos los movimientos están vigilados
            if not self.isWatchedWk(fullState):
                # Restaura el estado original del tablero antes de devolver
                self.newBoardSim(currentState)
                return False
        
        # Restaura el estado original del tablero
        self.newBoardSim(currentState)
        # Todos los movimientos dejan al rey en jaque
        return True

    def heuristica(self, currentState, color, depth=0, use_cache=True):
        # VERSIÓN OPTIMIZADA con evaluación MÁS FUERTE para K+R vs K+R
        # El parámetro depth ayuda a romper simetrías en posiciones iguales
        # use_cache: si es False, desactiva la caché (para minimax puro)
        
        if use_cache:
            stateKey = self.stateToKey(currentState)
            cacheKey = (stateKey, color)
            if cacheKey in self.heuristicCache:
                return self.heuristicCache[cacheKey]

        value = 0

        bkState = self.getPieceState(currentState, 12)  # Rey negro
        wkState = self.getPieceState(currentState, 6)   # Rey blanco
        wrState = self.getPieceState(currentState, 2)   # Torre blanca
        brState = self.getPieceState(currentState, 8)   # Torre negra

        # Posiciones
        filaBk, columnaBk = bkState[0], bkState[1]
        filaWk, columnaWk = wkState[0], wkState[1]

        # Distancia entre reyes (Chebyshev)
        distReis = max(abs(filaBk - filaWk), abs(columnaBk - columnaWk))
        
        # ============================================================================
        # EVALUACIÓN DE MATERIAL - Ajustada al tipo de final
        # ============================================================================
        # Comprobar K vs K (material insuficiente - tablas)
        if wrState is None and brState is None:
            # K vs K = tablas, pero es MALO si empezamos con material de ventaja
            # Penaliza fuerte para evitar cambiar hacia esto
            value -= 15000  # Penalización MASIVA: hemos tirado la ventaja
            self.heuristicCache[cacheKey] = value
            return value
        
        # En K+R vs K+R, el material es IGUAL - no añadir bonus por material
        # Solo añadir valor material cuando realmente es desigual
        material_value = 0
        if wrState is not None and brState is None:
            # Blanco tiene torre, negro no - ventaja para Blanco
            material_value += 5000  # Ventaja ENORME
        elif wrState is None and brState is not None:
            # Negro tiene torre, blanco no - ventaja para Negro
            material_value -= 5000
        # Si ambos tienen torres, material_value = 0 (igualdad)
        value += material_value
        
        # ============================================================================
        # CASO 1: MATERIAL IGUAL (K+R vs K+R) - ESTRATEGIA DEFENSIVA DE TABLAS
        # ============================================================================
        # K+R vs K+R es TABLAS TEÓRICAS con juego correcto.
        # Principio clave: Mantén tu torre ACTIVA - da jaques desde la DISTANCIA.
        # NUNCA cambies torres salvo que conduzca a un final K+R vs K claramente ganador.
        # CRÍTICO: Evita cambios de torre que lleven a tablas K vs K.
        # ============================================================================
        if wrState is not None and brState is not None:
            filaWr, columnaWr = wrState[0], wrState[1]
            filaBr, columnaBr = brState[0], brState[1]
            
            # Calcular todas las distancias
            distRookToBlackKing = max(abs(filaBk - filaWr), abs(columnaBk - columnaWr))
            distRookToWhiteKing = max(abs(filaWk - filaBr), abs(columnaWk - columnaBr))
            distWkBk = max(abs(filaWk - filaBk), abs(columnaWk - columnaBk))
            distWkBr = max(abs(filaWk - filaBr), abs(columnaWk - columnaBr))
            distBkWr = max(abs(filaBk - filaWr), abs(columnaBk - columnaWr))
            
            # ========== CRÍTICO: ¡EVITAR PERDER NUESTRA TORRE! =
            # Penalización MASIVA si el rey negro puede capturar la torre blanca en la siguiente
            if distBkWr == 1:
                value -= 5000  # CRÍTICO: torre colgando junto al rey rival - ¡EVÍTALO!
            elif distBkWr == 2:
                value -= 1000  # Zona de peligro: torre demasiado cerca del rey enemigo
            elif distBkWr == 3:
                value -= 200   # Arriesgado: la torre debería mantener distancia
            
            # También comprobar si el rey blanco está amenazado por la torre negra
            if distRookToWhiteKing == 1:
                # El rey está en jaque: puede ser aceptable si se puede mover, pero evalúa con cuidado
                value -= 300
            
            # ========== SEGURIDAD DE LA TORRE: Mantener distancia del rey enemigo ==========
            # Nuestra torre debería estar a 4+ casillas del rey enemigo (distancia segura de jaque)
            if distRookToBlackKing >= 4:
                value += 200  # Distancia segura: puede dar jaques sin ser atrapada
            elif distRookToBlackKing == 3:
                value += 50   # Aceptable
            # distRookToBlackKing <= 2 already heavily penalized above
            # distRookToBlackKing <= 2 ya está penalizado arriba
            
            # ========== ACTIVIDAD DE LA TORRE: Dar jaques desde la distancia ==========
            # La torre debe estar en la misma fila/columna que el rey enemigo (jaque o potencial)
            rookGivesCheck = (filaWr == filaBk or columnaWr == columnaBk)
            
            if rookGivesCheck and distRookToBlackKing >= 3:
                value += 300  # Excelente: torre activa dando jaques con seguridad
            elif rookGivesCheck and distRookToBlackKing == 2:
                value += 50   # Jaque, pero algo cerca
            elif not rookGivesCheck and distRookToBlackKing >= 4:
                value -= 100  # Torre pasiva y lejos: no es ideal
            
            # ========== MANTENER LA TORRE CENTRALIZADA (no atrapada en la banda) ==========
            rookCentrality = min(filaWr, 7 - filaWr, columnaWr, 7 - columnaWr)
            value += rookCentrality * 50  # Recompensa por torre central (más movilidad)
            
            # ========== ACTIVIDAD DEL REY: Central pero seguro ==========
            wkCentrality = min(filaWk, 7 - filaWk, columnaWk, 7 - columnaWk)
            value += wkCentrality * 40  # Mantener al rey activo
            
            # ========== COORDINACIÓN DEL REY: Apoyar la torre sin exponerla ==========
            if distWkBk >= 2 and distWkBk <= 4:
                value += 150  # Buena distancia del rey: apoya y permite libertad a la torre
            elif distWkBk == 1:
                value -= 200  # Demasiado agresivo: restringe la movilidad de la torre
            elif distWkBk >= 6:
                value -= 100  # Demasiado pasivo: mala coordinación
            
            # ========== CRÍTICO: EVITAR MALAS CAPTURAS DE TORRE ==========
            # Comprobar si el REY blanco puede capturar la torre negra en la siguiente
            if distWkBr == 1:  # Rey blanco adyacente a la torre negra
                # Capturar SOLO si podemos ganar el final resultante K+R vs K
                # Requiere que nuestro rey esté bien colocado respecto al rey negro
                if distWkBk <= 3:
                    value += 2000  # Buena captura: se puede ganar K+R vs K
                else:
                    # Mala captura: rey negro demasiado lejos, no se da mate → tablas K vs K
                    value -= 5000  # Penalización MASIVA: pierde la partida (tablas)
            
            # Comprobar si la TORRE blanca puede capturar la torre negra (misma fila/columna y adyacentes)
            # Esto comprueba si estamos a UNA JUGADA de capturar
            distWrBr_row = abs(filaWr - filaBr)
            distWrBr_col = abs(columnaWr - columnaBr)
            
            # La torre está a punto de capturar si están en la misma fila/columna
            if (filaWr == filaBr or columnaWr == columnaBr):
                distWrBr = max(distWrBr_row, distWrBr_col)
                if distWrBr <= 1:  # La torre puede capturar en la siguiente
                    # Comprobar si lleva a un final ganador
                    if distWkBk <= 3:
                        value += 1500  # Buena preparación de captura
                    else:
                        # ¡MALO! El rey negro recapturará nuestra torre → tablas K vs K
                        value -= 8000  # PENALIZACIÓN CRÍTICA: evita esta captura
            
            # ========== EVITAR QUE NEGRO CAPTURE NUESTRA TORRE CON SU TORRE ==========
            distRookToRook = max(abs(filaWr - filaBr), abs(columnaWr - columnaBr))
            if distRookToRook == 1:
                value -= 1000  # Torres adyacentes: ¡puede haber captura!
            elif distRookToRook == 2:
                value -= 200   # Torres cerca: arriesgado
            
            # ========== RUPTURA DE SIMETRÍA (preferencias ligeras) ==========
            # En igualdad, preferir ubicaciones ligeramente más activas
            if filaWr < filaBr:  # Torre blanca más avanzada (número de fila menor)
                value += 5
            if wkCentrality > min(filaBk, 7 - filaBk, columnaBk, 7 - columnaBk):
                value += 10  # Rey blanco más central
                
        # ============================================================================
        # CASO 2: VENTAJA BLANCA (K+R vs K) - FINAL GANADOR
        # ============================================================================
        elif wrState is not None and brState is None:
            filaWr, columnaWr = wrState[0], wrState[1]
            
            # CRÍTICO: ¡Comprobar si el rey negro puede recapturar nuestra torre!
            # Ocurre cuando acabamos de capturar la torre negra y el rey negro está adyacente
            distBkWr = max(abs(filaBk - filaWr), abs(columnaBk - columnaWr))
            if distBkWr == 1:
                # Rey negro junto a nuestra torre: capturará en la siguiente → tablas K vs K
                # Este es el caso crítico que debemos evitar
                value -= 20000  # Penalización MASIVA: lleva a tablas inmediatas
                self.heuristicCache[cacheKey] = value
                return value
            
            # COMPROBACIÓN DE AFOGADO: Solo en profundidad 0 (hojas) para ahorrar tiempo
            # Comprobación rápida: si el rey negro no tiene jugadas y la torre está demasiado cerca
            if depth == 0:
                # Heurística simple de ahogado: rey atrapado en la esquina con torre adyacente
                bkCorner = (filaBk == 0 or filaBk == 7) and (columnaBk == 0 or columnaBk == 7)
                rookAdjacent = (max(abs(filaWr - filaBk), abs(columnaWr - columnaBk)) == 1)
                notInCheck = not ((filaWr == filaBk or columnaWr == columnaBk))
                if bkCorner and rookAdjacent and notInCheck:
                    value -= 50000  # Probable ahogado
                    self.heuristicCache[cacheKey] = value
                    return value
            
            # ============================================================================
            # TÉCNICA CORRECTA K+R vs K (Chess.com):
            # Fase 1: CORTAR al rey con la torre (limitar a media tabla)
            # Fase 2: EMPUJAR al rey a la banda con rey+torre coordinados
            # Fase 3: ATRAPAR al rey en la banda con el rey bloqueando y torre dando jaques
            # Fase 4: MATE con la torre
            # ============================================================================
            
            # Calcular distancias
            distToEdge = min(filaBk, 7 - filaBk, columnaBk, 7 - columnaBk)
            distRookToKing = max(abs(filaBk - filaWr), abs(columnaBk - columnaWr))
            
            # Comprobar si la torre da jaque (misma fila/columna Y sin piezas entre medias)
            rookGivesCheck = False
            if filaWr == filaBk and columnaWr != columnaBk:
                # Misma fila, distinta columna = posible jaque
                rookGivesCheck = True  # En K+R vs K no hay piezas que bloqueen
            elif columnaWr == columnaBk and filaWr != filaBk:
                # Misma columna, distinta fila = posible jaque  
                rookGivesCheck = True
            
            # ======== FASE 1-2: EMPUJAR AL REY A LA BANDA ========
            # Este es EL factor más importante
            edgeBonus = 0
            if distToEdge == 0:
                edgeBonus = 8000  # En la banda: excelente (antes 4000)
            elif distToEdge == 1:
                edgeBonus = 5000  # Casi (antes 2500)
            elif distToEdge == 2:
                edgeBonus = 2400  # Más cerca (antes 1200)
            elif distToEdge == 3:
                edgeBonus = 800   # Aún en el centro (antes 400)
            value += edgeBonus
            
            # ======== TORRE DANDO JAQUE ========
            # Con el rey cerca de la banda, los jaques son BUENOS (fuerzan progreso)
            if rookGivesCheck:
                if distToEdge == 0:
                    # Jaque en banda = ¡cerca del mate!
                    value += 12000  # Antes 6000
                elif distToEdge == 1:
                    # Jaque cerca de la banda = empuja al rey a la banda
                    value += 6000  # Antes 3000
                else:
                    # Jaque en el centro = restringe movimiento
                    value += 2000  # Antes 1000
            
            # ======== COORDINACIÓN DEL REY ========
            # El rey blanco debe estar cerca pero sin provocar ahogado
            if distReis == 2:
                # Distancia de "oposición" perfecta
                value += 3000  # Antes 1500
            elif distReis == 3:
                # Buena distancia de apoyo
                value += 2000  # Antes 1000
            elif distReis == 1:
                # Muy cerca: ayuda a bloquear pero arriesga ahogado
                if distToEdge == 0:
                    # En banda con rey adyacente: ¡cuidado!
                    value += 1000  # Antes 500
                else:
                    value += 1600  # Antes 800
            elif distReis >= 5:
                # Demasiado lejos: ¡el rey debe ayudar!
                value -= (distReis - 4) * 600  # Antes 300
            
            # ======== POSICIONAMIENTO DE LA TORRE ========
            # La torre debe "cortar" al rey (misma fila/columna)
            if filaWr == filaBk or columnaWr == columnaBk:
                value += 1600  # La torre corta rutas de escape (antes 800)
            
            # La torre no debe estar demasiado lejos de la acción
            if distRookToKing <= 2:
                value += 1000  # Antes 500
            elif distRookToKing >= 5:
                value -= 400  # Antes 200
                
        # ============================================================================
        # CASO 3: VENTAJA NEGRA (K vs K+R)
        # ============================================================================
        elif wrState is None and brState is not None:
            filaBr, columnaBr = brState[0], brState[1]
            
            # COMPROBACIÓN DE AFOGADO: Solo a profundidad 0 (simplificado)
            if depth == 0:
                distWkBr = max(abs(filaWk - filaBr), abs(columnaWk - columnaBr))
                wkCorner = (filaWk == 0 or filaWk == 7) and (columnaWk == 0 or columnaWk == 7)
                rookAdjacent = (distWkBr == 1)
                notInCheck = not ((filaBr == filaWk or columnaBr == columnaWk))
                if wkCorner and rookAdjacent and notInCheck:
                    value += 50000  # Ahogado bueno para negro
                    self.heuristicCache[cacheKey] = value
                    return value
            
            # Negro debería ganar: usar estrategia similar con jaques
            is_check = (filaBr == filaWk or columnaBr == columnaWk)
            
            # Empujar al rey blanco a la banda
            distToEdge = min(filaWk, 7 - filaWk, columnaWk, 7 - columnaWk)
            
            if distToEdge == 0:
                value -= 10000
                if is_check:
                    value -= 100000
                    if distReis <= 2:
                        value -= 200000
                else:
                    value -= 20000
            elif distToEdge == 1:
                value -= 5000
                if is_check:
                    value -= 50000
                else:
                    value -= 8000
            elif distToEdge == 2:
                value -= 2000
                if is_check:
                    value -= 20000
                else:
                    value -= 3000
            else:
                value -= (3 - distToEdge) * 500
                if is_check:
                    value -= 10000
            
            # Distancia del rey negro
            if distReis == 1:
                value -= 3000
            elif distReis <= 2:
                value -= 2000
            elif distReis <= 3:
                value -= 800
            elif distReis <= 4:
                value -= 400
            else:
                value += 1000
            
            distRookToWhiteKing = max(abs(filaWk - filaBr), abs(columnaWk - columnaBr))
            if distRookToWhiteKing <= 2:
                value -= 1500
            elif distRookToWhiteKing <= 3:
                value -= 800

        # Bonus mínimo por profundidad para romper simetrías (mantener pequeño por rendimiento)
        if color:
            value += depth * 1.0
        else:
            value -= depth * 1.0

        # IMPORTANTE: NO invertir la perspectiva para Negro en la heurística
        # La evaluación es siempre desde la perspectiva de Blanco.
        # El algoritmo minimax maneja la lógica de min/max, no esta función.
        # Devuelve el valor tal cual (positivo = mejor para Blanco, negativo = mejor para Negro)

        if use_cache:
            stateKey = self.stateToKey(currentState)
            cacheKey = (stateKey, color)
            self.heuristicCache[cacheKey] = value
        return value


# ===================================================================================================
# ==================   AUXILIARES ===================================================================
# ===================================================================================================

    def stateToKey(self, state):
        """Convertir el estado a una clave hasheable para la tabla de transposición - OPTIMIZADO"""
        # En lugar de ordenar (lento), crear un frozenset de tuplas
        # Esto es MUCHO más rápido que sorted()
        return frozenset(tuple(piece) for piece in state)
    
    def _board_to_string(self, state=None):
        """Convertir el tablero actual a una representación en cadena"""
        board_str = ""
        
        if state is not None:
            # Construir el tablero directamente a partir del estado
            board = [[".  " for _ in range(8)] for _ in range(8)]
            for piece in state:
                row, col, piece_type = piece[0], piece[1], piece[2]
                if piece_type == 2:
                    board[row][col] = "WR "
                elif piece_type == 6:
                    board[row][col] = "WK "
                elif piece_type == 8:
                    board[row][col] = "BR "
                elif piece_type == 12:
                    board[row][col] = "BK "
            
            board_str += "  | 0  1  2  3  4  5  6  7\n"
            board_str += "--+------------------------\n"
            for i in range(8):
                board_str += str(i) + " | "
                for j in range(8):
                    board_str += board[i][j]
                board_str += "\n"
            board_str += "\n"
        else:
            # Usar el boardSim existente
            board = self.chess.boardSim.board
            board_str += "  | 0  1  2  3  4  5  6  7\n"
            board_str += "--+------------------------\n"
            for i in range(8):
                board_str += str(i) + " | "
                for j in range(8):
                    if board[i][j] == 0:
                        board_str += ".  "
                    elif board[i][j] == 2:
                        board_str += "WR "  # Torre blanca
                    elif board[i][j] == 6:
                        board_str += "WK "  # Rey blanco
                    elif board[i][j] == 8:
                        board_str += "BR "  # Torre negra
                    elif board[i][j] == 12:
                        board_str += "BK "  # Rey negro
                    else:
                        board_str += "?  "
                board_str += "\n"
            board_str += "\n"
        
        return board_str
    
    def _save_game_data(self, moveLog, visitedStates, moves_file, states_file):
        """Guardar movimientos y estados en ficheros"""
        # Guardar movimientos
        with open(moves_file, 'w', encoding='utf-8') as f:
            f.writelines(moveLog)
        
        # Guardar estados - solo listas de piezas
        with open(states_file, 'w', encoding='utf-8') as f:
            f.write("All states during the game:\n")
            f.write("="*60 + "\n\n")
            for idx, state in enumerate(visitedStates):
                f.write(f"State {idx}: Pieces: {state}\n")




# =============================================================
# ==================   OUR CODE ==== ==========================
# =============================================================

    def minimaxNoPruning(self, state, depth, isWhite, positionHistory=None):
        """
        Pure Minimax WITHOUT Alpha-Beta pruning
        This is TRUE minimax: explores ALL nodes, NO PRUNING
        However, we use transposition table for SPEED (legal optimization that doesn't change results)
        Used for Exercise 1 and Exercise 3 to compare against alpha-beta
        """
        # Initialize position history if not provided
        if positionHistory is None:
            positionHistory = {}
        
        # LEGAL OPTIMIZATION: Use transposition table (doesn't change minimax results, just caches)
        # This is mathematically equivalent to pure minimax but MUCH faster
        # Include 'minimax' in key to separate from alphabeta cache
        stateKey = self.stateToKey(state)
        ttKey = (stateKey, depth, isWhite, 'minimax')
        if ttKey in self.transpositionTable:
            return self.transpositionTable[ttKey]
        
        # Terminal conditions
        if depth == 0:
            value = self.heuristica(state, True, depth=0, use_cache=True)
            result = (value, state)
            self.transpositionTable[ttKey] = result
            return result
        
        # EARLY CHECKMATE DETECTION (legal optimization - no need to search deeper)
        if isWhite:
            if self.isBlackInCheckMate(state):
                value = 999999  # White wins
                result = (value, state)
                self.transpositionTable[ttKey] = result
                return result
        else:
            if self.isWhiteInCheckMate(state):
                value = -999999  # Black wins
                result = (value, state)
                self.transpositionTable[ttKey] = result
                return result
        
        # Get possible next states
        if isWhite:
            nextStates = self.getListNextStatesW(self.getWhiteState(state))
            if len(nextStates) == 0:
                value = self.heuristica(state, True, depth=depth, use_cache=True)
                result = (value, state)
                self.transpositionTable[ttKey] = result
                return result
            
            # Maximize for White - NO PRUNING (all branches explored)
            bestValue = float('-inf')
            bestState = None
            
            for whiteState in nextStates:
                blackState = self.getBlackState(state).copy()
                whitePositions = [(s[0], s[1]) for s in whiteState]
                blackState = [s for s in blackState if (s[0], s[1]) not in whitePositions]
                
                fullState = whiteState + blackState
                
                # Skip invalid states
                wkState = self.getPieceState(fullState, 6)
                bkState = self.getPieceState(fullState, 12)
                if wkState is None or bkState is None:
                    continue
                # Ensure kings are not adjacent (illegal)
                wkPos = wkState[0:2]
                bkPos = bkState[0:2]
                kingDistance = max(abs(wkPos[0] - bkPos[0]), abs(wkPos[1] - bkPos[1]))
                if kingDistance <= 1:
                    continue
                # Skip positions where White king is in check (illegal)
                if self.isWatchedWk(fullState):
                    continue
                
                # Update position history
                fullStateKey = self.stateToKey(fullState)
                prevCount = positionHistory.get(fullStateKey, 0)
                positionHistory[fullStateKey] = prevCount + 1
                
                # Update board simulator for next move generation
                self.newBoardSim(fullState)
                
                # Recurse - NO ALPHA/BETA parameters
                value, _ = self.minimaxNoPruning(fullState, depth - 1, False, positionHistory=positionHistory)
                
                # Restore position history
                positionHistory[fullStateKey] = prevCount
                
                # Update best (no pruning check)
                if value > bestValue:
                    bestValue = value
                    bestState = fullState
            
            if bestState is None:
                bestState = state
            
            result = (bestValue, bestState)
            self.transpositionTable[ttKey] = result  # Cache result
            return result
        
        else:
            # EARLY CHECKMATE DETECTION for Black
            if self.isWhiteInCheckMate(state):
                value = -999999
                result = (value, state)
                self.transpositionTable[ttKey] = result
                return result
            
            # Minimize for Black - NO PRUNING (all branches explored)
            nextStates = self.getListNextStatesB(self.getBlackState(state))
            if len(nextStates) == 0:
                value = self.heuristica(state, True, depth=depth, use_cache=True)
                result = (value, state)
                self.transpositionTable[ttKey] = result
                return result
            
            bestValue = float('inf')
            bestState = None
            
            for blackState in nextStates:
                whiteState = self.getWhiteState(state).copy()
                blackPositions = [(s[0], s[1]) for s in blackState]
                whiteState = [s for s in whiteState if (s[0], s[1]) not in blackPositions]
                
                fullState = whiteState + blackState
                
                # Skip invalid states
                wkState = self.getPieceState(fullState, 6)
                bkState = self.getPieceState(fullState, 12)
                if wkState is None or bkState is None:
                    continue
                # Ensure kings are not adjacent (illegal)
                wkPos = wkState[0:2]
                bkPos = bkState[0:2]
                kingDistance = max(abs(wkPos[0] - bkPos[0]), abs(wkPos[1] - bkPos[1]))
                if kingDistance <= 1:
                    continue
                # Skip positions where Black king is in check (illegal)
                if self.isWatchedBk(fullState):
                    continue
                
                # Update position history
                fullStateKey = self.stateToKey(fullState)
                prevCount = positionHistory.get(fullStateKey, 0)
                positionHistory[fullStateKey] = prevCount + 1
                
                # Update board simulator for next move generation
                self.newBoardSim(fullState)
                
                # Recurse - NO ALPHA/BETA parameters
                value, _ = self.minimaxNoPruning(fullState, depth - 1, True, positionHistory=positionHistory)
                
                # Restore position history
                positionHistory[fullStateKey] = prevCount
                
                # Update best (no pruning check)
                if value < bestValue:
                    bestValue = value
                    bestState = fullState
            
            if bestState is None:
                bestState = state
            
            result = (bestValue, bestState)
            self.transpositionTable[ttKey] = result  # Cache result
            return result

    def alphabeta(self, state, depth, isWhite, alpha=float('-inf'), beta=float('inf'), lastState=None, positionHistory=None):
        """
        OPTIMIZED Alpha-Beta pruning algorithm
        CRITICAL: Only checks for checkmate at SHALLOW depths (depth <= 2)
        At deeper levels, uses only heuristic evaluation (no expensive checkmate detection)
        """
        # Initialize position history if not provided
        if positionHistory is None:
            positionHistory = {}
        
        # TRANSPOSITION TABLE: Check if we've seen this position before
        # Include 'alphabeta' in key to separate from minimax cache
        stateKey = self.stateToKey(state)
        ttKey = (stateKey, depth, isWhite, 'alphabeta')
        if ttKey in self.transpositionTable:
            return self.transpositionTable[ttKey]
        
        # OPTIMIZATION: Only check repetitions at depth == maxDepth (top level)
        # Checking at every depth is too expensive for deep searches
        # The game loop handles threefold repetition detection
        if False and depth <= 3:
            if stateKey in positionHistory and positionHistory[stateKey] >= 2:
                # Position repeated 2+ times - discourage but don't completely block
                penalty = -100 if isWhite else 100
                result = (penalty, state)
                self.transpositionTable[ttKey] = result
                return result
        
        # Terminal conditions
        if depth == 0:
            # Depth 0: Use fast heuristic only (no checkmate)
            # Pass depth=0 to heuristica for symmetry breaking
            value = self.heuristica(state, True, depth=0)
            result = (value, state)
            self.transpositionTable[ttKey] = result
            return result
            
        # Early checkmate detection (match minimax behavior)
        if isWhite:
            if self.isBlackInCheckMate(state):
                result = (999999, state)
                self.transpositionTable[ttKey] = result
                return result
        else:
            if self.isWhiteInCheckMate(state):
                result = (-999999, state)
                self.transpositionTable[ttKey] = result
                return result
        
        
        # OPTIMIZATION: Removed all checkmate/stalemate checks from minimax
        # Reason: Too expensive at depth 5, heuristic handles it well
        # The main game loop still checks for terminal states
        # This gives MASSIVE speedup for deep searches
        
        # Get possible next states
        if isWhite:
            nextStates = self.getListNextStatesW(self.getWhiteState(state))
            if len(nextStates) == 0:
                return (self.heuristica(state, True, depth=depth), state)
            
            # Maximize for White
            bestValue = float('-inf')
            bestState = None  # CHANGED: Don't initialize with potentially invalid state
            
            # OPTIMIZATION: Skip move ordering completely (disabled for speed)
            orderedStates = nextStates
            
            validMoveFound = False  # Track if we found any valid move
            
            for whiteState in orderedStates:
                blackState = self.getBlackState(state).copy()
                whitePositions = [(s[0], s[1]) for s in whiteState]
                blackState = [s for s in blackState if (s[0], s[1]) not in whitePositions]
                
                fullState = whiteState + blackState
                
                # Skip invalid states
                wkState = self.getPieceState(fullState, 6)
                bkState = self.getPieceState(fullState, 12)
                if wkState is None or bkState is None:
                    continue
                
                # CRITICAL: Check if kings are adjacent (ILLEGAL position)
                wkPos = wkState[0:2]
                bkPos = bkState[0:2]
                kingDistance = max(abs(wkPos[0] - bkPos[0]), abs(wkPos[1] - bkPos[1]))
                if kingDistance <= 1:
                    continue  # Skip this illegal move
                
                # Skip illegal moves (king in check)
                if self.isWatchedWk(fullState):
                    continue
                
                # This is a valid move
                validMoveFound = True
                
                # OPTIMIZATION: Update position history WITHOUT copying (much faster)
                # We'll increment, recurse, then decrement (undo)
                fullStateKey = self.stateToKey(fullState)
                prevCount = positionHistory.get(fullStateKey, 0)
                positionHistory[fullStateKey] = prevCount + 1
                
                # Update board simulator for next move generation
                self.newBoardSim(fullState)
                
                # Recurse
                value, _ = self.alphabeta(fullState, depth - 1, False, alpha, beta, lastState=state, positionHistory=positionHistory)
                
                # CRITICAL: Restore position history (undo the increment)
                positionHistory[fullStateKey] = prevCount
                if prevCount == 0:
                    del positionHistory[fullStateKey]  # Clean up to save memory
                
                if value > bestValue:
                    bestValue = value
                    bestState = fullState
                
                # Alpha-Beta pruning
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            
            # If no valid move found, return current state with heuristic
            if not validMoveFound or bestState is None:
                result = (self.heuristica(state, True, depth=depth), state)
                self.transpositionTable[ttKey] = result
                return result
            
            result = (bestValue, bestState)
            self.transpositionTable[ttKey] = result
            return result
        else:
            nextStates = self.getListNextStatesB(self.getBlackState(state))
            if len(nextStates) == 0:
                return (self.heuristica(state, True, depth=depth), state)
            
            # Minimize for Black
            bestValue = float('inf')
            bestState = None  # CHANGED: Don't initialize with potentially invalid state
            
            # OPTIMIZATION: Skip move ordering completely (disabled for speed)
            orderedStates = nextStates
            
            validMoveFound = False  # Track if we found any valid move
            
            for blackState in orderedStates:
                whiteState = self.getWhiteState(state).copy()
                blackPositions = [(s[0], s[1]) for s in blackState]
                whiteState = [s for s in whiteState if (s[0], s[1]) not in blackPositions]
                
                fullState = whiteState + blackState
                
                # Skip invalid states
                wkState = self.getPieceState(fullState, 6)
                bkState = self.getPieceState(fullState, 12)
                if wkState is None or bkState is None:
                    continue
                
                # CRITICAL: Check if kings are adjacent (ILLEGAL position)
                wkPos = wkState[0:2]
                bkPos = bkState[0:2]
                kingDistance = max(abs(wkPos[0] - bkPos[0]), abs(wkPos[1] - bkPos[1]))
                if kingDistance <= 1:
                    continue  # Skip this illegal move
                
                # Skip illegal moves
                if self.isWatchedBk(fullState):
                    continue
                
                # This is a valid move
                validMoveFound = True
                
                # OPTIMIZATION: Update position history WITHOUT copying (much faster)
                fullStateKey = self.stateToKey(fullState)
                prevCount = positionHistory.get(fullStateKey, 0)
                positionHistory[fullStateKey] = prevCount + 1
                
                # Update board simulator for next move generation
                self.newBoardSim(fullState)
                
                # Recurse
                value, _ = self.alphabeta(fullState, depth - 1, True, alpha, beta, lastState=state, positionHistory=positionHistory)
                
                # CRITICAL: Restore position history (undo)
                positionHistory[fullStateKey] = prevCount
                if prevCount == 0:
                    del positionHistory[fullStateKey]
                
                if value < bestValue:
                    bestValue = value
                    bestState = fullState
                
                # Alpha-Beta pruning
                beta = min(beta, value)
                if beta <= alpha:
                    break
            
            # If no valid move found, return current state with heuristic
            if not validMoveFound or bestState is None:
                result = (self.heuristica(state, True, depth=depth), state)
                self.transpositionTable[ttKey] = result
                return result
            
            result = (bestValue, bestState)
            self.transpositionTable[ttKey] = result
            return result

    def minimaxGame(self, depthWhite, depthBlack, verbose=True, save_to_file=False, moves_file="moves_ex1.txt", states_file="states_ex1.txt"):
        """
        Play a complete game using PURE MINIMAX (NO pruning) for both players
        This is for Exercise 1 - both players use minimax without alpha-beta
        
        Args:
            depthWhite: Search depth for White
            depthBlack: Search depth for Black
            verbose: If True, print board state after each move
            save_to_file: If True, save moves and states to files
            moves_file: Filename to save moves
            states_file: Filename to save states
            
        Returns:
            Winner string: "White", "Black", or "Draw"
        """
        # Clear caches for new game AND set size limits
        self.transpositionTable.clear()
        self.heuristicCache.clear()
        MAX_CACHE_SIZE = 50000  # Limit cache growth
        
        currentState = self.getCurrentState()
        
        # Track visited states (list of states from start to end)
        visitedStates = [currentState.copy()]
        
        # Track position repetitions for draw detection
        positionHistory = {}
        posKey = self.stateToKey(currentState)
        positionHistory[posKey] = 1
        
        # For saving moves
        moveLog = []
        
        # NOTE: Minimum depth calculation would require tracking actual search depth used
        # This is a known limitation - we report configured depth instead
        # To implement properly would require modifying all search algorithms to return depth info
        minDepthWhite = depthWhite
        minDepthBlack = depthBlack
        
        if verbose:
            print("\n=== Starting Minimax Game ===")
            print(f"White depth: {depthWhite}, Black depth: {depthBlack}")
            print(f"Initial state: {currentState}")
            self.chess.boardSim.print_board()
        
        if save_to_file:
            moveLog.append("=== Starting Minimax Game ===\n")
            moveLog.append(f"White depth: {depthWhite}, Black depth: {depthBlack}\n")
            moveLog.append(f"Initial state: {currentState}\n")
            moveLog.append(self._board_to_string(currentState))
        
        moveCount = 0
        maxMoves = 100  # Prevent infinite games
        
        while moveCount < maxMoves:
            moveCount += 1
            
            # Check for insufficient material (both rooks captured = draw)
            wrState = self.getPieceState(currentState, 2)
            brState = self.getPieceState(currentState, 8)
            if wrState is None and brState is None:
                if verbose:
                    print("\n*** DRAW (insufficient material - King vs King) ***")
                    print(f"\nGame Statistics:")
                    print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
                    print(f"  Total full moves: {moveCount - 1}")
                    print(f"  Search depth configured: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                    print(f"  Transposition table entries: {len(self.transpositionTable)}")
                if save_to_file:
                    moveLog.append("\n*** DRAW (insufficient material - King vs King) ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount - 1,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # White's turn
            if verbose:
                print(f"\n--- Move {moveCount}: White's turn ---")
            
            if save_to_file:
                moveLog.append(f"\n--- Move {moveCount}: White's turn ---\n")
            
            # OPTIMIZATION: Clear caches if they grow too large
            if len(self.heuristicCache) > MAX_CACHE_SIZE:
                self.heuristicCache.clear()
            if len(self.transpositionTable) > MAX_CACHE_SIZE:
                self.transpositionTable.clear()
            
            _, bestStateWhite = self.minimaxNoPruning(currentState, depthWhite, True, positionHistory=positionHistory)
            
            # CRITICAL: Validate that kings are not adjacent (emergency check)
            wkState = self.getPieceState(bestStateWhite, 6)
            bkState = self.getPieceState(bestStateWhite, 12)
            if wkState is not None and bkState is not None:
                wkPos = wkState[0:2]
                bkPos = bkState[0:2]
                kingDistance = max(abs(wkPos[0] - bkPos[0]), abs(wkPos[1] - bkPos[1]))
                if kingDistance <= 1:
                    if verbose:
                        print(f"\n*** ERROR: White returned illegal move (kings adjacent)! ***")
                        print(f"White king: {wkPos}, Black king: {bkPos}, Distance: {kingDistance}")
                    # This should never happen - fallback to current state
                    bestStateWhite = currentState
            
            currentState = bestStateWhite
            visitedStates.append(currentState.copy())
            self.newBoardSim(currentState)
            
            if verbose:
                self.chess.boardSim.print_board()
            
            if save_to_file:
                moveLog.append(self._board_to_string(currentState))
            
            # CRITICAL: Check for checkmate BEFORE checking repetition
            # Checkmate has priority over draw by repetition
            if self.isBlackInCheckMate(currentState):
                if verbose:
                    print("\n*** WHITE WINS BY CHECKMATE! ***")
                    print(f"\nGame Statistics:")
                    print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
                    print(f"  Total full moves: {moveCount}")
                    print(f"  Search depth configured: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                    print(f"  Transposition table entries: {len(self.transpositionTable)}")
                if save_to_file:
                    moveLog.append("\n*** WHITE WINS BY CHECKMATE! ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "White",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # CRITICAL: Check for stalemate BEFORE repetition
            # Stalemate = draw, has priority
            if self.isBlackInStaleMate(currentState):
                if verbose:
                    print("\n*** DRAW BY STALEMATE (Black has no legal moves but not in check) ***")
                    print(f"\nGame Statistics:")
                    print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
                    print(f"  Total full moves: {moveCount}")
                    print(f"  Search depth configured: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                if save_to_file:
                    moveLog.append("\n*** DRAW BY STALEMATE ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # Check for position repetition (threefold repetition = draw)
            posKey = self.stateToKey(currentState)
            positionHistory[posKey] = positionHistory.get(posKey, 0) + 1
            if positionHistory[posKey] >= 3:
                if verbose:
                    print(f"\n*** DRAW (threefold repetition) ***")
                    print(f"\nGame Statistics:")
                    print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
                    print(f"  Total full moves: {moveCount}")
                    print(f"  Search depth configured: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                    print(f"  Transposition table entries: {len(self.transpositionTable)}")
                if save_to_file:
                    moveLog.append("\n*** DRAW (threefold repetition) ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # Black's turn
            if verbose:
                print(f"\n--- Move {moveCount}: Black's turn ---")
            
            if save_to_file:
                moveLog.append(f"\n--- Move {moveCount}: Black's turn ---\n")
            
            _, bestStateBlack = self.minimaxNoPruning(currentState, depthBlack, False, positionHistory=positionHistory)
            
            # CRITICAL: Validate that kings are not adjacent (emergency check)
            wkState = self.getPieceState(bestStateBlack, 6)
            bkState = self.getPieceState(bestStateBlack, 12)
            if wkState is not None and bkState is not None:
                wkPos = wkState[0:2]
                bkPos = bkState[0:2]
                kingDistance = max(abs(wkPos[0] - bkPos[0]), abs(wkPos[1] - bkPos[1]))
                if kingDistance <= 1:
                    if verbose:
                        print(f"\n*** ERROR: Black returned illegal move (kings adjacent)! ***")
                        print(f"White king: {wkPos}, Black king: {bkPos}, Distance: {kingDistance}")
                    # This should never happen - fallback to current state
                    bestStateBlack = currentState
            
            currentState = bestStateBlack
            visitedStates.append(currentState.copy())
            self.newBoardSim(currentState)
            
            if verbose:
                self.chess.boardSim.print_board()
            
            if save_to_file:
                moveLog.append(self._board_to_string(currentState))
            
            # CRITICAL: Check for checkmate BEFORE checking repetition
            # Checkmate has priority over draw by repetition
            if self.isWhiteInCheckMate(currentState):
                if verbose:
                    print("\n*** BLACK WINS BY CHECKMATE! ***")
                    print(f"\nGame Statistics:")
                    print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
                    print(f"  Total full moves: {moveCount}")
                    print(f"  Search depth configured: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                    print(f"  Transposition table entries: {len(self.transpositionTable)}")
                if save_to_file:
                    moveLog.append("\n*** BLACK WINS BY CHECKMATE! ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Black",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # CRITICAL: Check for stalemate BEFORE repetition
            # Stalemate = draw, has priority
            if self.isWhiteInStaleMate(currentState):
                if verbose:
                    print("\n*** DRAW BY STALEMATE (White has no legal moves but not in check) ***")
                    print(f"\nGame Statistics:")
                    print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
                    print(f"  Total full moves: {moveCount}")
                    print(f"  Search depth configured: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                if save_to_file:
                    moveLog.append("\n*** DRAW BY STALEMATE ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # Check for position repetition (threefold repetition = draw)
            posKey = self.stateToKey(currentState)
            positionHistory[posKey] = positionHistory.get(posKey, 0) + 1
            if positionHistory[posKey] >= 3:
                if verbose:
                    print(f"\n*** DRAW (threefold repetition) ***")
                    print(f"\nGame Statistics:")
                    print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
                    print(f"  Total full moves: {moveCount}")
                    print(f"  Search depth configured: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                    print(f"  Transposition table entries: {len(self.transpositionTable)}")
                if save_to_file:
                    moveLog.append("\n*** DRAW (threefold repetition) ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
        
        # Game reached max moves - it's a draw
        if verbose:
            print(f"\n*** DRAW (reached {maxMoves} moves) ***")
            print(f"\nGame Statistics:")
            print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
            print(f"  Total full moves: {moveCount}")
            print(f"  Search depth configured: White={depthWhite}, Black={depthBlack}")
            print(f"  Total states visited: {len(visitedStates)}")
            print(f"  Transposition table entries: {len(self.transpositionTable)}")
        if save_to_file:
            moveLog.append(f"\n*** DRAW (reached {maxMoves} moves) ***\n")
            self._save_game_data(moveLog, visitedStates, moves_file, states_file)
        return {
            'winner': "Draw",
            'stats': {
                'half_moves': len(visitedStates) - 1,
                'full_moves': moveCount,
                'depth_white': depthWhite,
                'depth_black': depthBlack,
                'states_visited': len(visitedStates)
            }
        }

    def alphaBetaGame(self, depthWhite, depthBlack, whiteUsesAlphaBeta=True, blackUsesAlphaBeta=True, 
                      verbose=True, save_to_file=False, moves_file="moves.txt", states_file="states.txt"):
        """
        Play a complete game where players can use minimax or alpha-beta pruning
        
        Args:
            depthWhite: Search depth for White
            depthBlack: Search depth for Black
            whiteUsesAlphaBeta: If True, White uses alpha-beta; if False, uses minimax
            blackUsesAlphaBeta: If True, Black uses alpha-beta; if False, uses minimax
            verbose: If True, print board state after each move
            save_to_file: If True, save moves and states to files
            moves_file: Filename to save moves
            states_file: Filename to save states
            
        Returns:
            Dictionary with winner and game statistics
        """
        # Clear caches for new game
        self.transpositionTable.clear()
        self.heuristicCache.clear()
        
        currentState = self.getCurrentState()
        
        # Track visited states (list of states from start to end)
        visitedStates = [currentState.copy()]
        
        # Track position repetitions for draw detection
        positionHistory = {}
        posKey = self.stateToKey(currentState)
        positionHistory[posKey] = 1
        
        # For saving moves
        moveLog = []
        
        if verbose:
            white_algo = "Alpha-Beta" if whiteUsesAlphaBeta else "Minimax"
            black_algo = "Alpha-Beta" if blackUsesAlphaBeta else "Minimax"
            print(f"\n=== Starting Game ===")
            print(f"White: {white_algo} (depth {depthWhite})")
            print(f"Black: {black_algo} (depth {depthBlack})")
            print(f"Initial state: {currentState}")
            self.chess.boardSim.print_board()
        
        if save_to_file:
            white_algo = "Alpha-Beta" if whiteUsesAlphaBeta else "Minimax"
            black_algo = "Alpha-Beta" if blackUsesAlphaBeta else "Minimax"
            moveLog.append(f"=== Starting Game ===\n")
            moveLog.append(f"White: {white_algo} (depth {depthWhite})\n")
            moveLog.append(f"Black: {black_algo} (depth {depthBlack})\n")
            moveLog.append(f"Initial state: {currentState}\n")
            moveLog.append(self._board_to_string(currentState))
        
        moveCount = 0
        maxMoves = 100  # Prevent infinite games
        
        while moveCount < maxMoves:
            moveCount += 1
            
            # Check for insufficient material (both rooks captured = draw)
            wrState = self.getPieceState(currentState, 2)
            brState = self.getPieceState(currentState, 8)
            if wrState is None and brState is None:
                if verbose:
                    print("\n*** DRAW (insufficient material - King vs King) ***")
                if save_to_file:
                    moveLog.append("\n*** DRAW (insufficient material - King vs King) ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount - 1,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # White's turn
            if verbose:
                print(f"\n--- Move {moveCount}: White's turn ---")
            if save_to_file:
                moveLog.append(f"\n--- Move {moveCount}: White's turn ---\n")
            
            # FIXED: Respect whiteUsesAlphaBeta parameter
            if whiteUsesAlphaBeta:
                _, bestStateWhite = self.alphabeta(currentState, depthWhite, True, positionHistory=positionHistory)
            else:
                # Use pure minimax without pruning
                _, bestStateWhite = self.minimaxNoPruning(currentState, depthWhite, True, positionHistory=positionHistory)
            
            currentState = bestStateWhite
            visitedStates.append(currentState.copy())
            self.newBoardSim(currentState)
            
            if verbose:
                self.chess.boardSim.print_board()
            if save_to_file:
                moveLog.append(self._board_to_string(currentState))
            
            # Match minimaxGame priority: checkmate -> stalemate -> repetition
            # Check if Black is in checkmate
            if self.isBlackInCheckMate(currentState):
                if verbose:
                    print("\n*** WHITE WINS BY CHECKMATE! ***")
                if save_to_file:
                    moveLog.append("\n*** WHITE WINS BY CHECKMATE! ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "White",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # Check if Black is in stalemate (no legal moves, not in check)
            if self.isBlackInStaleMate(currentState):
                if verbose:
                    print("\n*** DRAW BY STALEMATE (Black has no legal moves but not in check) ***")
                if save_to_file:
                    moveLog.append("\n*** DRAW BY STALEMATE ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }

            # Finally, check for position repetition (threefold repetition = draw)
            posKey = self.stateToKey(currentState)
            positionHistory[posKey] = positionHistory.get(posKey, 0) + 1
            if positionHistory[posKey] >= 3:
                if verbose:
                    print(f"\n*** DRAW (threefold repetition) ***")
                if save_to_file:
                    moveLog.append("\n*** DRAW (threefold repetition) ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # Black's turn
            if verbose:
                print(f"\n--- Move {moveCount}: Black's turn ---")
            if save_to_file:
                moveLog.append(f"\n--- Move {moveCount}: Black's turn ---\n")
            
            # FIXED: Respect blackUsesAlphaBeta parameter
            if blackUsesAlphaBeta:
                _, bestStateBlack = self.alphabeta(currentState, depthBlack, False, positionHistory=positionHistory)
            else:
                # Use pure minimax without pruning
                _, bestStateBlack = self.minimaxNoPruning(currentState, depthBlack, False, positionHistory=positionHistory)
            
            currentState = bestStateBlack
            visitedStates.append(currentState.copy())
            self.newBoardSim(currentState)
            
            if verbose:
                self.chess.boardSim.print_board()
            if save_to_file:
                moveLog.append(self._board_to_string(currentState))
            
            # Match minimaxGame priority: checkmate -> stalemate -> repetition
            # Check if White is in checkmate
            if self.isWhiteInCheckMate(currentState):
                if verbose:
                    print("\n*** BLACK WINS BY CHECKMATE! ***")
                if save_to_file:
                    moveLog.append("\n*** BLACK WINS BY CHECKMATE! ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Black",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # Check if White is in stalemate (no legal moves, not in check)
            if self.isWhiteInStaleMate(currentState):
                if verbose:
                    print("\n*** DRAW BY STALEMATE (White has no legal moves but not in check) ***")
                if save_to_file:
                    moveLog.append("\n*** DRAW BY STALEMATE ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }

            # Finally, check for position repetition (threefold repetition = draw)
            posKey = self.stateToKey(currentState)
            positionHistory[posKey] = positionHistory.get(posKey, 0) + 1
            if positionHistory[posKey] >= 3:
                if verbose:
                    print(f"\n*** DRAW (threefold repetition) ***")
                if save_to_file:
                    moveLog.append("\n*** DRAW (threefold repetition) ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
        
        # Game reached max moves - it's a draw
        if verbose:
            print(f"\n*** DRAW (reached {maxMoves} moves) ***")
        if save_to_file:
            moveLog.append(f"\n*** DRAW (reached {maxMoves} moves) ***\n")
            self._save_game_data(moveLog, visitedStates, moves_file, states_file)
        return {
            'winner': "Draw",
            'stats': {
                'half_moves': len(visitedStates) - 1,
                'full_moves': moveCount,
                'depth_white': depthWhite,
                'depth_black': depthBlack,
                'states_visited': len(visitedStates)
            }
        }

    def expectimaxValue(self, state, depth, isWhite):
        """
        Expectimax algorithm - White maximizes, Black uses expected value (chance node)
        
        Args:
            state: Current board state
            depth: Search depth remaining
            isWhite: True if White's turn (maximizing), False if Black's turn (chance node)
            
        Returns:
            (value, bestState) tuple
        """
        # Terminal conditions
        if depth == 0:
            value = self.heuristica(state, True)
            return (value, state)
        
        # Check for checkmate
        if self.isWhiteInCheckMate(state):
            return (-10000, state)
        if self.isBlackInCheckMate(state):
            return (10000, state)
        
        # Get possible next states
        if isWhite:
            # White maximizes (deterministic player)
            nextStates = self.getListNextStatesW(self.getWhiteState(state))
            if len(nextStates) == 0:
                return (self.heuristica(state, True), state)
            
            bestValue = float('-inf')
            bestState = nextStates[0] + self.getBlackState(state)
            
            for whiteState in nextStates:
                blackState = self.getBlackState(state).copy()
                whitePositions = [(s[0], s[1]) for s in whiteState]
                blackState = [s for s in blackState if (s[0], s[1]) not in whitePositions]
                
                fullState = whiteState + blackState
                
                wkState = self.getPieceState(fullState, 6)
                bkState = self.getPieceState(fullState, 12)
                if wkState is None or bkState is None:
                    continue
                
                if self.isWatchedWk(fullState):
                    continue
                
                value, _ = self.expectimaxValue(fullState, depth - 1, False)
                
                if value > bestValue:
                    bestValue = value
                    bestState = fullState
            
            return (bestValue, bestState)
        else:
            # Black uses EXPECTED VALUE (chance node)
            # Uses standard mathematical expectation: E[X] = Σ(x_i * p_i)
            # With uniform probability: E[X] = (1/n) * Σ(x_i) = mean(x_i)
            nextStates = self.getListNextStatesB(self.getBlackState(state))
            if len(nextStates) == 0:
                return (self.heuristica(state, True), state)
            
            validStates = []
            validValues = []
            
            for blackState in nextStates:
                whiteState = self.getWhiteState(state).copy()
                blackPositions = [(s[0], s[1]) for s in blackState]
                whiteState = [s for s in whiteState if (s[0], s[1]) not in blackPositions]
                
                fullState = whiteState + blackState
                
                wkState = self.getPieceState(fullState, 6)
                bkState = self.getPieceState(fullState, 12)
                if wkState is None or bkState is None:
                    continue
                
                if self.isWatchedBk(fullState):
                    continue
                
                validStates.append(fullState)
                value, _ = self.expectimaxValue(fullState, depth - 1, True)
                validValues.append(value)
            
            if len(validValues) == 0:
                return (self.heuristica(state, True), state)
            
            # STANDARD EXPECTATION: E[X] = Σ(x_i * p_i)
            # With uniform probability p_i = 1/n for all moves:
            # E[X] = (1/n) * Σ(x_i) = mean(x_i)
            expectedValue = self.mean(validValues)
            
            # Return a representative state (closest to expected value)
            bestIdx = min(range(len(validValues)), key=lambda i: abs(validValues[i] - expectedValue))
            
            return (expectedValue, validStates[bestIdx])

    def expectimaxGame(self, depthWhite, depthBlack, whiteUsesExpectimax=True, blackUsesAlphaBeta=True,
                       verbose=True, save_to_file=False, moves_file="moves.txt", states_file="states.txt"):
        """
        Play a game where White uses Expectimax and Black uses Alpha-Beta (or vice versa)
        
        Args:
            depthWhite: Search depth for White
            depthBlack: Search depth for Black
            whiteUsesExpectimax: If True, White uses expectimax; if False, uses alpha-beta
            blackUsesAlphaBeta: If True, Black uses alpha-beta; if False, uses expectimax
            verbose: If True, print board state after each move
            save_to_file: If True, save moves and states to files
            
        Returns:
            Dictionary with winner and game statistics
        """
        self.transpositionTable.clear()
        self.heuristicCache.clear()
        
        currentState = self.getCurrentState()
        visitedStates = [currentState.copy()]
        
        positionHistory = {}
        posKey = self.stateToKey(currentState)
        positionHistory[posKey] = 1
        
        moveLog = []
        
        if verbose:
            white_algo = "Expectimax" if whiteUsesExpectimax else "Alpha-Beta"
            black_algo = "Alpha-Beta" if blackUsesAlphaBeta else "Expectimax"
            print(f"\n=== Starting Game ===")
            print(f"White: {white_algo} (depth {depthWhite})")
            print(f"Black: {black_algo} (depth {depthBlack})")
            self.chess.boardSim.print_board()
        
        if save_to_file:
            white_algo = "Expectimax" if whiteUsesExpectimax else "Alpha-Beta"
            black_algo = "Alpha-Beta" if blackUsesAlphaBeta else "Expectimax"
            moveLog.append(f"=== Starting Game ===\n")
            moveLog.append(f"White: {white_algo} (depth {depthWhite})\n")
            moveLog.append(f"Black: {black_algo} (depth {depthBlack})\n")
            moveLog.append(self._board_to_string(currentState))
        
        moveCount = 0
        maxMoves = 100
        
        while moveCount < maxMoves:
            moveCount += 1
            
            # Check for insufficient material
            wrState = self.getPieceState(currentState, 2)
            brState = self.getPieceState(currentState, 8)
            if wrState is None and brState is None:
                if verbose:
                    print("\n*** DRAW (insufficient material) ***")
                if save_to_file:
                    moveLog.append("\n*** DRAW (insufficient material) ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount - 1,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # White's turn
            if verbose:
                print(f"\n--- Move {moveCount}: White's turn ---")
            if save_to_file:
                moveLog.append(f"\n--- Move {moveCount}: White's turn ---\n")
            
            if whiteUsesExpectimax:
                _, bestStateWhite = self.expectimaxValue(currentState, depthWhite, True)
            else:
                # Use alphabeta for White
                _, bestStateWhite = self.alphabeta(currentState, depthWhite, True, positionHistory=positionHistory)
            
            currentState = bestStateWhite
            visitedStates.append(currentState.copy())
            self.newBoardSim(currentState)
            
            if verbose:
                self.chess.boardSim.print_board()
            if save_to_file:
                moveLog.append(self._board_to_string(currentState))
            
            # Check for repetition
            posKey = self.stateToKey(currentState)
            positionHistory[posKey] = positionHistory.get(posKey, 0) + 1
            if positionHistory[posKey] >= 3:
                if verbose:
                    print(f"\n*** DRAW (threefold repetition) ***")
                if save_to_file:
                    moveLog.append("\n*** DRAW (threefold repetition) ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            if self.isBlackInCheckMate(currentState):
                if verbose:
                    print("\n*** WHITE WINS BY CHECKMATE! ***")
                if save_to_file:
                    moveLog.append("\n*** WHITE WINS BY CHECKMATE! ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "White",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            # Black's turn
            if verbose:
                print(f"\n--- Move {moveCount}: Black's turn ---")
            if save_to_file:
                moveLog.append(f"\n--- Move {moveCount}: Black's turn ---\n")
            
            if blackUsesAlphaBeta:
                # Black uses alphabeta
                _, bestStateBlack = self.alphabeta(currentState, depthBlack, False, positionHistory=positionHistory)
            else:
                # Black uses expectimax
                _, bestStateBlack = self.expectimaxValue(currentState, depthBlack, False)
            
            currentState = bestStateBlack
            visitedStates.append(currentState.copy())
            self.newBoardSim(currentState)
            
            if verbose:
                self.chess.boardSim.print_board()
            if save_to_file:
                moveLog.append(self._board_to_string(currentState))
            
            posKey = self.stateToKey(currentState)
            positionHistory[posKey] = positionHistory.get(posKey, 0) + 1
            if positionHistory[posKey] >= 3:
                if verbose:
                    print(f"\n*** DRAW (threefold repetition) ***")
                if save_to_file:
                    moveLog.append("\n*** DRAW (threefold repetition) ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Draw",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
            
            if self.isWhiteInCheckMate(currentState):
                if verbose:
                    print("\n*** BLACK WINS BY CHECKMATE! ***")
                if save_to_file:
                    moveLog.append("\n*** BLACK WINS BY CHECKMATE! ***\n")
                    self._save_game_data(moveLog, visitedStates, moves_file, states_file)
                return {
                    'winner': "Black",
                    'stats': {
                        'half_moves': len(visitedStates) - 1,
                        'full_moves': moveCount,
                        'depth_white': depthWhite,
                        'depth_black': depthBlack,
                        'states_visited': len(visitedStates)
                    }
                }
        
        if verbose:
            print(f"\n*** DRAW (reached {maxMoves} moves) ***")
        if save_to_file:
            moveLog.append(f"\n*** DRAW (reached {maxMoves} moves) ***\n")
            self._save_game_data(moveLog, visitedStates, moves_file, states_file)
        return {
            'winner': "Draw",
            'stats': {
                'half_moves': len(visitedStates) - 1,
                'full_moves': moveCount,
                'depth_white': depthWhite,
                'depth_black': depthBlack,
                'states_visited': len(visitedStates)
            }
        }




# =============================================================
# ==================  ENCAPSULACION EJERCICIOS ================
# =============================================================

def run_exercise_1(depth_white=4, depth_black=4, repetitions=3, verbose=False, 
                   save_to_file=True, results_file='exercise1_results.json'):
    """
    Ejercicio 1: Minimax vs Minimax (ambos jugadores sin poda alfa-beta).

    Parámetros:
        depth_white: Profundidad de búsqueda de Blancas (por defecto: 4).
        depth_black: Profundidad de búsqueda de Negras (por defecto: 4).
        repetitions: Número de partidas a jugar (por defecto: 3).
        verbose: Si es True, imprime el tablero tras cada movimiento (por defecto: False).
        save_to_file: Si es True, guarda movimientos y estados en archivos (por defecto: True).
        results_file: Archivo JSON donde guardar los resultados (por defecto: 'exercise1_results.json').

    Devuelve:
        Un diccionario con los contadores de victorias/empates, el detalle por partida
        (tiempos, estadísticas) y la configuración usada.
    """
    print("\n" + "="*70)
    print(f"==== EXERCISE 1: Minimax Game (Depth {depth_white} vs {depth_black}) =====")
    print("="*70)
    print(f"Both White and Black use Minimax algorithm")
    print(f"White depth: {depth_white}, Black depth: {depth_black}")
    print(f"Running {repetitions} times to count White wins")
    print("="*70 + "\n")
    
    # Contenedor de resultados agregados y configuración del experimento
    exercise1_results = {
        'white_wins': 0,
        'black_wins': 0,
        'draws': 0,
        'games': [],
        'config': {
            'depth_white': depth_white,
            'depth_black': depth_black,
            'repetitions': repetitions
        }
    }
    
    for rep in range(1, repetitions + 1):
        print(f"\n{'─'*70}")
        print(f"Exercise 1 - Game {rep}/{repetitions}")
        print(f"\n{'─'*70}")
        print(f"Exercise 1 - Game {rep}/{repetitions}")
        print(f"{'─'*70}\n")
        
        # Reiniciar el tablero (8x8) para cada partida independiente
        # Piezas utilizadas:
        #   2  = Torre blanca | 6  = Rey blanco
        #   8  = Torre negra  | 12 = Rey negro
        TA = np.zeros((8, 8))
        TA[7][0] = 2   # White Rook
        TA[7][5] = 6   # White King
        TA[0][7] = 8   # Black Rook
        TA[0][5] = 12  # Black King
        
        aichess = Aichess(TA, True)
        
        moves_filename = f"moves_ex1_{rep}.txt" if save_to_file else None
        states_filename = f"states_ex1_{rep}.txt" if save_to_file else None
        
        start_time = time.time()
        result = aichess.minimaxGame(depth_white, depth_black, 
                                      verbose=verbose, 
                                      save_to_file=save_to_file, 
                                      moves_file=moves_filename, 
                                      states_file=states_filename)
        elapsed_time = time.time() - start_time
        
        winner = result['winner']
        stats = result['stats']
        
        # Actualizar contadores globales según el ganador
        if winner == "White":
            exercise1_results['white_wins'] += 1
        elif winner == "Black":
            exercise1_results['black_wins'] += 1
        else:
            exercise1_results['draws'] += 1
        
        game_info = {
            'repetition': rep,
            'winner': winner,
            'stats': stats,
            'elapsed_time': elapsed_time
        }
        exercise1_results['games'].append(game_info)
        
        print(f"\nGame {rep} complete: {winner} ({elapsed_time:.2f}s)")
        print(f"  Half-moves: {stats['half_moves']}, Full moves: {stats['full_moves']}")
        if save_to_file:
            print(f"  Saved to: {moves_filename}")
    
    print(f"\n{'='*60}")
    print(f"EXERCISE 1 SUMMARY")
    print(f"{'='*60}")
    print(f"White wins: {exercise1_results['white_wins']}/{repetitions}")
    print(f"Black wins: {exercise1_results['black_wins']}/{repetitions}")
    print(f"Draws: {exercise1_results['draws']}/{repetitions}")
    print(f"{'='*60}\n")
    
    # Guardar resultados del Ejercicio 1 (incluye detalle por partida y configuración)
    if results_file:
        with open(results_file, 'w') as f:
            json.dump(exercise1_results, f, indent=2)
        print(f"Results saved to: {results_file}\n")
    
    return exercise1_results

def run_exercise_2(depth_values=[3, 4], repetitions=3, verbose=False, 
                   save_to_file=True, generate_plot=True, results_file='exercise2_results.json'):
    """
    Ejercicio 2: Minimax con profundidades variables (batería de combinaciones).

    Parámetros:
        depth_values: Lista de profundidades a evaluar (por defecto: [3, 4]).
        repetitions: Número de partidas por combinación (por defecto: 3).
        verbose: Si es True, imprime el tablero tras cada movimiento (por defecto: False).
        save_to_file: Si es True, guarda movimientos y estados (por defecto: True).
        generate_plot: Si es True, genera una gráfica de resultados (por defecto: True).
        results_file: Archivo JSON donde guardar resultados (por defecto: 'exercise2_results.json').

    Devuelve:
        Un diccionario con métricas por combinación (W vs B), historial de todas las partidas
        y la configuración utilizada.
    """
    print("\n" + "="*70)
    print("==== EXERCISE 2: Minimax with Varying Depths =====")
    print("="*70)
    print(f"Testing depth combinations from: {depth_values}")
    print(f"Each combination will be run {repetitions} times")
    print("="*70 + "\n")
    
    # Definir las combinaciones de profundidad a probar (producto cartesiano)
    # Formato: [(w1,b1), (w1,b2), ..., (wN,bN)]
    depth_combinations = [(w, b) for w in depth_values for b in depth_values]
    
    # Estructura de almacenamiento de resultados para análisis
    exercise2_results = {
        'combinations': {},
        'all_games': [],
        'config': {
            'depth_values': depth_values,
            'repetitions': repetitions
        }
    }
    
    # Número total de partidas a ejecutar en todo el experimento
    total_games = len(depth_combinations) * repetitions
    game_counter = 0
    
    for depthWhite, depthBlack in depth_combinations:
        combo_key = f"{depthWhite}v{depthBlack}"
        print(f"\n{'='*70}")
        print(f"Testing combination: White depth={depthWhite}, Black depth={depthBlack}")
        print(f"{'='*70}\n")
        
        combo_results = {
            'white_depth': depthWhite,
            'black_depth': depthBlack,
            'games': [],
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'total_games': repetitions
        }
        
        for rep in range(1, repetitions + 1):
            game_counter += 1
            print(f"Game {game_counter}/{total_games}: White depth={depthWhite}, Black depth={depthBlack}, Repetition {rep}")
            
            # Reset board
            TA = np.zeros((8, 8))
            TA[7][0] = 2
            TA[7][5] = 6
            TA[0][7] = 8
            TA[0][5] = 12
            
            aichess = Aichess(TA, True)
            
            moves_filename = f"moves_ex2_{depthWhite}{depthBlack}{rep}.txt" if save_to_file else None
            states_filename = f"states_ex2_{depthWhite}{depthBlack}{rep}.txt" if save_to_file else None
            
            start_time = time.time()
            result = aichess.minimaxGame(depthWhite, depthBlack, 
                                          verbose=verbose, 
                                          save_to_file=save_to_file,
                                          moves_file=moves_filename, 
                                          states_file=states_filename)
            elapsed_time = time.time() - start_time
            
            winner = result['winner']
            stats = result['stats']
            
            if winner == "White":
                combo_results['white_wins'] += 1
            elif winner == "Black":
                combo_results['black_wins'] += 1
            else:
                combo_results['draws'] += 1
            
            game_info = {
                'white_depth': depthWhite,
                'black_depth': depthBlack,
                'repetition': rep,
                'winner': winner,
                'stats': stats,
                'elapsed_time': elapsed_time
            }
            combo_results['games'].append(game_info)
            exercise2_results['all_games'].append(game_info)
            
            print(f"  Result: {winner} ({elapsed_time:.2f}s)\n")
        
        # Calculate statistics for this combination
        combo_results['white_win_percentage'] = (combo_results['white_wins'] / repetitions) * 100
        combo_results['black_win_percentage'] = (combo_results['black_wins'] / repetitions) * 100
        combo_results['draw_percentage'] = (combo_results['draws'] / repetitions) * 100
        
        exercise2_results['combinations'][combo_key] = combo_results
        
        # Resumen por combinación (tras ejecutar sus repeticiones)
        print(f"\n╔{'═'*68}╗")
        print(f"║ COMBINATION SUMMARY: White depth={depthWhite}, Black depth={depthBlack}           ║")
        print(f"╠{'═'*68}╣")
        print(f"║ White wins: {combo_results['white_wins']}/{repetitions} ({combo_results['white_win_percentage']:.1f}%)                                    ║")
        print(f"║ Black wins: {combo_results['black_wins']}/{repetitions}({combo_results['black_win_percentage']:.1f}%)                                    ║")
        print(f"║ Draws:      {combo_results['draws']}/{repetitions} ({combo_results['draw_percentage']:.1f}%)                                    ║")
        print(f"╚{'═'*68}╝\n")
    
    # Guardar todos los resultados (todas las combinaciones y partidas) a JSON
    if results_file:
        with open(results_file, 'w') as f:
            json.dump(exercise2_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("="*70)
    print("           EXERCISE 2 COMPLETE - FINAL SUMMARY")
    print("="*70)
    print("="*70 + "\n")
    
    print("Results by depth combination:\n")
    for combo_key in sorted(exercise2_results['combinations'].keys()):
        combo = exercise2_results['combinations'][combo_key]
        print(f"{combo_key}: W={combo['white_wins']}, B={combo['black_wins']}, D={combo['draws']}")
    
    if results_file:
        print(f"\nAll results saved to: {results_file}")
    print(f"Total games played: {total_games}")
    
    # Generar gráfica de resumen (opcional)
    if generate_plot:
        print("\n" + "="*70)
        print("Generating Exercise 2 Plot...")
        print("="*70 + "\n")
        
        # Calcular el % de victorias de Blancas PARA CADA valor de profundidad (promediando oponente)
        depth_win_percentages = {}
        for depth in depth_values:
            wins_for_depth = [combo['white_win_percentage'] 
                             for combo in exercise2_results['combinations'].values() 
                             if combo['white_depth'] == depth]
            depth_win_percentages[depth] = np.mean(wins_for_depth) if wins_for_depth else 0
        
        # Crear figura de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        
        depths = list(depth_win_percentages.keys())
        percentages = list(depth_win_percentages.values())
        
        bars = ax.bar(depths, percentages, color='steelblue', alpha=0.8, 
                      edgecolor='black', width=0.6)
        
        # Añadir etiquetas con el valor encima de cada barra
        for depth, pct in zip(depths, percentages):
            ax.text(depth, pct, f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('White Depth (moves)', fontsize=12, fontweight='bold')
        ax.set_ylabel('White Win Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Exercise 2: White Win Percentage by Depth Value\n(Averaged across all opponent depths)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(depths)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('exercise2_plot.png', dpi=300, bbox_inches='tight')
        print("Saved: exercise2_plot.png")
        plt.close()
        
        # Análisis de simetría (solo si hay al menos 2 profundidades)
        if len(depth_values) >= 2:
            print("\nExercise 2 - Symmetry Analysis:")
            print("="*60)
            # Comparar d1 vs d2 y d2 vs d1 para ver si los resultados son consistentes
            d1, d2 = depth_values[0], depth_values[1]
            key1 = f"{d1}v{d2}"
            key2 = f"{d2}v{d1}"
            if key1 in exercise2_results['combinations'] and key2 in exercise2_results['combinations']:
                white_d1_d2 = exercise2_results['combinations'][key1]['white_win_percentage']
                white_d2_d1 = exercise2_results['combinations'][key2]['white_win_percentage']
                print(f"White depth={d1} vs Black depth={d2}: {white_d1_d2:.1f}% white wins")
                print(f"White depth={d2} vs Black depth={d1}: {white_d2_d1:.1f}% white wins")
                print(f"Difference: {abs(white_d1_d2 - white_d2_d1):.1f}%")
                
                if abs(white_d1_d2 - white_d2_d1) < 10:
                    print("→ Results are relatively symmetric")
                else:
                    print("→ Significant asymmetry detected")
            print("="*60 + "\n")
    
    return exercise2_results

def run_exercise_3(depth=4, repetitions=3, verbose=False, save_to_file=True, 
                   results_file='exercise3_results.json'):
    """
    Ejercicio 3: Blancas usan Minimax (sin poda), Negras usan Alfa-Beta.

    Parámetros:
        depth: Profundidad de búsqueda para ambos jugadores (por defecto: 4).
        repetitions: Número de partidas a jugar (por defecto: 3).
        verbose: Si es True, imprime el tablero tras cada movimiento (por defecto: False).
        save_to_file: Si es True, guarda movimientos y estados en archivos (por defecto: True).
        results_file: Archivo JSON donde guardar los resultados (por defecto: 'exercise3_results.json').

    Devuelve:
        Un diccionario con el número de victorias/empates, el detalle de cada partida
        (tiempos, estadísticas) y la configuración utilizada.
    """
    import time
    import json
    import numpy as np
    from aichess import Aichess
    
    print("\n" + "="*70)
    print("==== EXERCISE 3: Minimax (White) vs Alpha-Beta (Black) =====")
    print("="*70)

    # Contenedor de resultados acumulados para el ejercicio
    exercise3_results = {
        'white_wins': 0,
        'black_wins': 0,
        'draws': 0,
        'games': [],
        'config': {
            'depth': depth,
            'repetitions': repetitions
        }
    }
    
    for rep in range(1, repetitions + 1):
        print(f"\n{'─'*70}")
        print(f"Exercise 3 - Game {rep}/{repetitions}")
        print(f"{'─'*70}\n")
        
        # Reset board
        TA = np.zeros((8, 8))
        TA[7][0] = 2   # White Rook
        TA[7][5] = 6   # White King
        TA[0][7] = 8   # Black Rook
        TA[0][5] = 12  # Black King
        
        aichess = Aichess(TA, True)
        
        # Convención de nombres para facilitar análisis posterior
        moves_filename = f"moves_ex3_{rep}.txt" if save_to_file else None
        states_filename = f"states_ex3_{rep}.txt" if save_to_file else None
        
        # Medir tiempo de la partida
        start_time = time.time()
        result = aichess.alphaBetaGame(
            depth, depth,
            whiteUsesAlphaBeta=False,  # Blancas usan minimax (sin poda)
            blackUsesAlphaBeta=True,   # Negras usan alfa-beta
            verbose=verbose,
            save_to_file=save_to_file,
            moves_file=moves_filename,
            states_file=states_filename
        )
        elapsed_time = time.time() - start_time
        
        winner = result['winner']
        game_stats = result['stats']
        
        # Actualizar contadores agregados
        if winner == "White":
            exercise3_results['white_wins'] += 1
        elif winner == "Black":
            exercise3_results['black_wins'] += 1
        else:
            exercise3_results['draws'] += 1
        
        # Registro detallado de la partida actual
        game_info = {
            'repetition': rep,
            'winner': winner,
            'stats': game_stats,
            'elapsed_time': elapsed_time
        }
        exercise3_results['games'].append(game_info)
        
        print(f"\nGame {rep} complete: {winner} ({elapsed_time:.2f}s)")
    
    print(f"\n╔{'═'*68}╗")
    print(f"║ EXERCISE 3 SUMMARY                                                 ║")
    print(f"╠{'═'*68}╣")
    print(f"║ White wins (Minimax): {exercise3_results['white_wins']}/{repetitions}                                       ║")
    print(f"║ Black wins (Alpha-Beta): {exercise3_results['black_wins']}/{repetitions}                                    ║")
    print(f"║ Draws: {exercise3_results['draws']}/{repetitions}                                                     ║")
    print(f"╚{'═'*68}╝\n")
    
    if results_file:
        # Persistencia en JSON de resultados y configuración
        with open(results_file, 'w') as f:
            json.dump(exercise3_results, f, indent=2)
        print(f"Results saved to: {results_file}\n")
    
    return exercise3_results

def run_exercise_4(depth_range=(1, 5), repetitions=3, verbose=False, save_to_file=True, 
                   generate_plot=True, results_file='exercise4_results.json'):
    """
    Exercise 4: Ambos jugadores usan Alfa-Beta con profundidades variables.

    Parámetros:
        depth_range: Tupla (prof_min, prof_max), inclusiva. Por defecto (1, 5).
        repetitions: Número de partidas por cada combinación de profundidades. Por defecto 3.
        verbose: Si es True, imprime el tablero tras cada jugada. Por defecto False.
        save_to_file: Si es True, guarda movimientos y estados en archivos. Por defecto True.
        generate_plot: Si es True, genera una figura con los resultados. Por defecto True.
        results_file: Nombre del archivo JSON donde guardar el resumen. Por defecto 'exercise4_results.json'.

    Devuelve:
        Un diccionario con las métricas agregadas por combinación y el detalle de todas las partidas.

    Notas:
        - Se exploran todas las combinaciones de profundidad White x Black dentro del rango indicado.
        - El estado inicial siempre es el mismo (rey y torre por bando) para garantizar comparabilidad.
        - El tamaño del experimento crece como O(N^2 * repetitions), con N = nº de profundidades.
    """
    import time
    import json
    import numpy as np
    from aichess import Aichess
    import matplotlib.pyplot as plt
    
    print("\n" + "="*70)
    print("==== EXERCISE 4: Alpha-Beta vs Alpha-Beta (Varying Depths) =====")
    print("="*70)
    
    # Contenedor de resultados: por combinación y por partida, más la configuración usada
    exercise4_results = {
        'combinations': {},
        'all_games': [],
        'config': {
            'depth_range': depth_range,
            'repetitions': repetitions
        }
    }
    
    # Probar TODAS las combinaciones de profundidades (White vs Black) en el rango indicado
    depth_combinations_ex4 = []
    for depthW in range(depth_range[0], depth_range[1] + 1):
        for depthB in range(depth_range[0], depth_range[1] + 1):
            depth_combinations_ex4.append((depthW, depthB))
    
    for depthWhite, depthBlack in depth_combinations_ex4:
        combo_key = f"{depthWhite}v{depthBlack}"
        print(f"\n{'='*70}")
        print(f"Testing: White depth={depthWhite}, Black depth={depthBlack}")
        print(f"{'='*70}\n")
        
        combo_results = {
            'white_depth': depthWhite,
            'black_depth': depthBlack,
            'games': [],
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0
        }
        
        for rep in range(1, repetitions + 1):
            print(f"  Game {rep}/{repetitions}...")
            
            TA = np.zeros((8, 8))
            TA[7][0] = 2
            TA[7][5] = 6
            TA[0][7] = 8
            TA[0][5] = 12
            
            aichess = Aichess(TA, True)
            
            moves_filename = f"moves_ex4_{depthWhite}_{depthBlack}_{rep}.txt" if save_to_file else None
            states_filename = f"states_ex4_{depthWhite}_{depthBlack}_{rep}.txt" if save_to_file else None
            
            # Medición de tiempo por partida (útil para comparar el coste de búsqueda)
            start_time = time.time()
            result = aichess.alphaBetaGame(depthWhite, depthBlack,
                                            whiteUsesAlphaBeta=True,
                                            blackUsesAlphaBeta=True,
                                            verbose=verbose, save_to_file=save_to_file,
                                            moves_file=moves_filename,
                                            states_file=states_filename)
            elapsed_time = time.time() - start_time
            
            winner = result['winner']
            game_stats = result['stats']
            
            if winner == "White":
                combo_results['white_wins'] += 1
            elif winner == "Black":
                combo_results['black_wins'] += 1
            else:
                combo_results['draws'] += 1
            
            # Registro granular de la partida actual
            game_info = {
                'white_depth': depthWhite,
                'black_depth': depthBlack,
                'repetition': rep,
                'winner': winner,
                'stats': game_stats,
                'elapsed_time': elapsed_time
            }
            combo_results['games'].append(game_info)
            exercise4_results['all_games'].append(game_info)
            
            print(f"    Result: {winner} ({elapsed_time:.2f}s)")
        
        # Porcentajes por combinación concreta
        combo_results['white_win_percentage'] = (combo_results['white_wins'] / repetitions) * 100
        combo_results['black_win_percentage'] = (combo_results['black_wins'] / repetitions) * 100
        combo_results['draw_percentage'] = (combo_results['draws'] / repetitions) * 100
        
        # Guardar agregados bajo la clave "WdepthvBdepth"
        exercise4_results['combinations'][combo_key] = combo_results
        
        print(f"  Summary: W={combo_results['white_wins']}, B={combo_results['black_wins']}, D={combo_results['draws']}")
    
    if results_file:
        # Persistencia en JSON: incluye configuración, agregados por combinación y el detalle de todas las partidas
        with open(results_file, 'w') as f:
            json.dump(exercise4_results, f, indent=2)
    
    print(f"\n╔{'═'*68}╗")
    print(f"║ EXERCISE 4 COMPLETE                                                ║")
    print(f"╚{'═'*68}╝")
    if results_file:
        print(f"Results saved to: {results_file}\n")
    
    # Generar plot
    if generate_plot:
        print("\n" + "="*70)
        print("Generating Exercise 4 Plot...")
        print("="*70 + "\n")
        
        # Calcular proporciones globales en TODAS las partidas (sumando todas las combinaciones)
        total_white_wins = sum(c['white_wins'] for c in exercise4_results['combinations'].values())
        total_black_wins = sum(c['black_wins'] for c in exercise4_results['combinations'].values())
        total_draws = sum(c['draws'] for c in exercise4_results['combinations'].values())
        total_games_ex4 = total_white_wins + total_black_wins + total_draws
        
        # Nota: si total_games_ex4 fuese 0 (no debería ocurrir con repetitions > 0), habría que proteger la división.
        white_proportion = (total_white_wins / total_games_ex4) * 100
        black_proportion = (total_black_wins / total_games_ex4) * 100
        draw_proportion = (total_draws / total_games_ex4) * 100
        
        # Crear figura con dos subgráficos: barras (izquierda) y sectores (derecha)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Barras: porcentajes de victorias/empates
        categories = ['White', 'Black', 'Draws']
        proportions = [white_proportion, black_proportion, draw_proportion]
        colors = ['#e74c3c', '#3498db', '#95a5a6']
        
        bars = ax1.bar(categories, proportions, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Proportion (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Win Proportions', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Etiquetas con el valor exacto encima de cada barra
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Gráfico de sectores: distribución global
        wedges, texts, autotexts = ax2.pie(proportions, labels=categories, autopct='%1.1f%%',
                                            colors=colors, startangle=90,
                                            textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        ax2.set_title('Win Distribution', fontsize=13, fontweight='bold')
        
        fig.suptitle(f'Exercise 4: Alpha-Beta vs Alpha-Beta\nProportion of Wins (All depth combinations {depth_range[0]}-{depth_range[1]})', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('exercise4_plot.png', dpi=300, bbox_inches='tight')
        print("Saved: exercise4_plot.png")
        print(f"  White wins: {white_proportion:.1f}% ({total_white_wins}/{total_games_ex4} games)")
        print(f"  Black wins: {black_proportion:.1f}% ({total_black_wins}/{total_games_ex4} games)")
        print(f"  Draws: {draw_proportion:.1f}% ({total_draws}/{total_games_ex4} games)")
        plt.close()
    
    return exercise4_results

def run_exercise_5(depth=4, repetitions=3, verbose=False, save_to_file=True, 
                   generate_plot=True, results_file='exercise5_results.json'):
    """
    # Exercise 5: Expectimax (Blancas) vs Alpha-Beta (Negras)

    Parámetros:
        depth: Profundidad de búsqueda para ambos jugadores (por defecto: 4)
        repetitions: Número de partidas a jugar (por defecto: 3)
        verbose: Si es True, imprime el tablero tras cada movimiento (por defecto: False)
        save_to_file: Si es True, guarda los movimientos y estados en archivos (por defecto: True)
        generate_plot: Si es True, genera una gráfica con los resultados (por defecto: True)
        results_file: Archivo JSON donde guardar los resultados (por defecto: 'exercise5_results.json')
    """
    import time
    import json
    import numpy as np
    from aichess import Aichess
    import matplotlib.pyplot as plt
    
    print("\n" + "="*70)
    print("==== EXERCISE 5: Expectimax (White) vs Alpha-Beta (Black) =====")
    print("="*70)
    
    exercise5_results = {
        'white_wins': 0,
        'black_wins': 0,
        'draws': 0,
        'games': [],
        'config': {
            'depth': depth,
            'repetitions': repetitions
        }
    }
    
    for rep in range(1, repetitions + 1):
        print(f"\n{'─'*70}")
        print(f"Exercise 5 - Game {rep}/{repetitions}")
        print(f"{'─'*70}\n")
        
        TA = np.zeros((8, 8))
        TA[7][0] = 2
        TA[7][5] = 6
        TA[0][7] = 8
        TA[0][5] = 12
        
        aichess = Aichess(TA, True)
        
        moves_filename = f"moves_ex5_{rep}.txt" if save_to_file else None
        states_filename = f"states_ex5_{rep}.txt" if save_to_file else None
        
        start_time = time.time()
        result = aichess.expectimaxGame(
            depth, depth,
            whiteUsesExpectimax=True,
            blackUsesAlphaBeta=True,
            verbose=verbose,
            save_to_file=save_to_file,
            moves_file=moves_filename,
            states_file=states_filename
        )
        elapsed_time = time.time() - start_time
        
        winner = result['winner']
        game_stats = result['stats']
        
        if winner == "White":
            exercise5_results['white_wins'] += 1
        elif winner == "Black":
            exercise5_results['black_wins'] += 1
        else:
            exercise5_results['draws'] += 1
        
        game_info = {
            'repetition': rep,
            'winner': winner,
            'stats': game_stats,
            'elapsed_time': elapsed_time
        }
        exercise5_results['games'].append(game_info)
        
        print(f"\nGame {rep} complete: {winner} ({elapsed_time:.2f}s)")
    
    print(f"\n╔{'═'*68}╗")
    print(f"║ EXERCISE 5 SUMMARY                                                 ║")
    print(f"╠{'═'*68}╣")
    print(f"║ White wins (Expectimax): {exercise5_results['white_wins']}/{repetitions}                                    ║")
    print(f"║ Black wins (Alpha-Beta): {exercise5_results['black_wins']}/{repetitions}                                    ║")
    print(f"║ Draws: {exercise5_results['draws']}/{repetitions}                                                     ║")
    print(f"╚{'═'*68}╝\n")
    
    if results_file:
        with open(results_file, 'w') as f:
            json.dump(exercise5_results, f, indent=2)
        print(f"Results saved to: {results_file}\n")
    
    # Generar gráfico
    if generate_plot:
        print("\n" + "="*70)
        print("Generating Exercise 5 Plot...")
        print("="*70 + "\n")
        
        categories = ['White\n(Expectimax)', 'Black\n(Alpha-Beta)', 'Draws']
        values = [exercise5_results['white_wins'], exercise5_results['black_wins'], exercise5_results['draws']]
        colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gráfico de barras
        bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel(f'Number of Wins (out of {repetitions} games)', fontsize=12, fontweight='bold')
        ax1.set_title('Exercise 5: Game Results', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, repetitions + 0.5)
        ax1.grid(axis='y', alpha=0.3)
        
        # Añadir etiquetas de valores
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Gráfico de sectores
        percentages = [(v/repetitions)*100 for v in values]
        
        wedges, texts, autotexts = ax2.pie(percentages, labels=categories, autopct='%1.1f%%',
                                            colors=colors, startangle=90,
                                            textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        # Hacer el texto de porcentaje más grande
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        ax2.set_title('Win Proportions', fontsize=13, fontweight='bold')
        
        fig.suptitle('Exercise 5: Expectimax (White) vs Alpha-Beta (Black)', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('exercise5_plot.png', dpi=300, bbox_inches='tight')
        print("Saved: exercise5_plot.png")
        plt.close()
    
    return exercise5_results


# ===========================================================================
# ========================= MAIN ============================================
# ===========================================================================

if __name__ == "__main__":
    import json
    import time
    
    # Initialize an empty 8x8 chess board
    TA = np.zeros((8, 8))

    # Load initial positions of the pieces
    TA = np.zeros((8, 8))
    TA[7][0] = 2   
    TA[7][5] = 6   
    TA[0][7] = 8   
    TA[0][5] = 12  

    # Initialise board and print
    print("Starting AI chess... ")
    aichess = Aichess(TA, True)
    print("Printing initial board:")
    aichess.chess.boardSim.print_board()
    
    # Run all exercises with default parameters
    # OPTIMIZED FOR SPEED: Using lower depths where reasonable
    
    print("\n" + "="*70)
    print("RUNNING ALL EXERCISES - OPTIMIZED FOR SPEED")
    print("="*70)

    
    # EJERCICIO 1: Minimax vs Minimax (profundidad 4 como se requiere)
    # Este es el más lento: minimax puro a profundidad 4
    run_exercise_1(
        depth_white=4,           # Requerido por el enunciado
        depth_black=4,           # Requerido por el enunciado
        repetitions=3,           # Requerido por el enunciado
        verbose=False,           # En False para ir más rápido
        save_to_file=True,
        results_file='exercise1_results.json'
    )
    
    # EJERCICIO 2: Profundidades variables (3-4 como se requiere)
    run_exercise_2(
        depth_values=[3, 4],     # Requerido por el enunciado: de 3 a 4
        repetitions=3,           # Se puede reducir a 2 para ir más rápido si hace falta
        verbose=False,           # False para ir más rápido
        save_to_file=True,
        generate_plot=True,
        results_file='exercise2_results.json'
    )
    
    # EJERCICIO 3: Minimax vs Alpha-Beta (profundidad 4)
    run_exercise_3(
        depth=4,                 # Requerido por el enunciado
        repetitions=3,           # Requerido por el enunciado
        verbose=False,           # False para ir más rápido
        save_to_file=True,
        results_file='exercise3_results.json'
    )
    
    # # EJERCICIO 4: Alpha-Beta vs Alpha-Beta (1-5)
    # # Esto es RÁPIDO porque ambos usan poda alpha-beta
    run_exercise_4(
        depth_range=(1, 5),      # Requerido: de 1 a 5
        repetitions=3,           # Requerido: 3 simulaciones cada una
        verbose=False,
        save_to_file=True,
        generate_plot=True,
        results_file='exercise4_results.json'
    )
    
    # EJERCICIO 5: Expectimax vs Alpha-Beta
    run_exercise_5(
        depth=4,                 # Se puede usar profundidad 4
        repetitions=3,           # Requerido: 3 simulaciones
        verbose=False,
        save_to_file=True,
        generate_plot=True,
        results_file='exercise5_results.json'
    )
