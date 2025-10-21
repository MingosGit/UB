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




# =============================================================
# ==================   MODIFICADOS ============================ 
# =============================================================



  

    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        bkState = self.getPieceState(currentState, 12)
        # If the black king has been captured, this is not a valid state
        if bkState is None:
            return False
        bkPosition = bkState[0:2]
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

        # If the white king has been captured, this is not a valid configuration
        if wkState is None:
            return False

        # Check if the black king is adjacent to the white king (illegal position, treat as in check)
        wkPosition = wkState[0:2]
        if max(abs(bkPosition[0] - wkPosition[0]), abs(bkPosition[1] - wkPosition[1])) == 1:
            return True

        # Check all possible moves of the white king to see if it can capture the black king
        for wkPos in self.getNextPositions(wkState):
            if bkPosition == wkPos:
                # Black king would be in check
                return True

        if wrState is not None:
            # Check all possible moves of the white rook to see if it can capture the black king
            for wrPosition in self.getNextPositions(wrState):
                if bkPosition == wrPosition:
                    return True

        return False

    def allBkMovementsWatched(self, currentState):
        # Check if all possible moves for the black king leave it in check (checkmate condition)
        
        self.newBoardSim(currentState)
        wrState = self.getPieceState(currentState, 2)
        wkState = self.getPieceState(currentState, 6)
        whiteState = self.getWhiteState(currentState)
        
        # Get all possible next states for black pieces (king moves and potential rook captures)
        nextBStates = self.getListNextStatesB(self.getBlackState(currentState))
        
        # If there are no legal moves at all, it's checkmate (or stalemate if not in check)
        if len(nextBStates) == 0:
            return True
        
        # Check each possible move
        for state in nextBStates:
            bkNewPos = state[0][0:2]
            
            # Skip illegal moves where black king would capture white king
            if wkState is not None and bkNewPos == wkState[0:2]:
                continue
            
            newWhiteState = whiteState.copy()
            # Check if the white rook has been captured; if so, remove it from the state
            if wrState is not None and wrState[0:2] == bkNewPos:
                newWhiteState.remove(wrState)
            
            fullState = state + newWhiteState
            # Move the black pieces to the new state
            self.newBoardSim(fullState)
            
            # If this move gets the black king out of check, then not all moves are watched
            if not self.isWatchedBk(fullState):
                # Restore the original board state before returning
                self.newBoardSim(currentState)
                return False
        
        # Restore the original board state
        self.newBoardSim(currentState)
        # All moves leave the king in check
        return True

    def isBlackInCheckMate(self, currentState):
        if self.isWatchedBk(currentState) and self.allBkMovementsWatched(currentState):
            return True
        return False


    def isWatchedWk(self, currentState):
        self.newBoardSim(currentState)
        wkState = self.getPieceState(currentState, 6)
        # If the white king has been captured, this is not a valid state
        if wkState is None:
            return False
        wkPosition = wkState[0:2]
        bkState = self.getPieceState(currentState, 12)
        brState = self.getPieceState(currentState, 8)
        # If the black king has been captured, this is not a valid configuration
        if bkState is None:
            return False
        
        # Check if the white king is adjacent to the black king (illegal position, treat as in check)
        bkPosition = bkState[0:2]
        if max(abs(wkPosition[0] - bkPosition[0]), abs(wkPosition[1] - bkPosition[1])) == 1:
            return True
        
        # Check all possible moves for the black king and see if it can capture the white king
        for bkPos in self.getNextPositions(bkState):
            if wkPosition == bkPos:
                # White king would be in check
                return True
        if brState is not None:
            # Check all possible moves for the black rook and see if it can capture the white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True
        return False

    def allWkMovementsWatched(self, currentState):
        # Check if all possible moves for the white king leave it in check (checkmate condition)
        
        self.newBoardSim(currentState)
        brState = self.getPieceState(currentState, 8)
        bkState = self.getPieceState(currentState, 12)
        blackState = self.getBlackState(currentState)
        
        # Get all possible next states for white pieces (king moves and potential rook captures)
        nextWStates = self.getListNextStatesW(self.getWhiteState(currentState))
        
        # If there are no legal moves at all, it's checkmate (or stalemate if not in check)
        if len(nextWStates) == 0:
            return True
        
        # Check each possible move
        for state in nextWStates:
            wkNewPos = state[0][0:2]
            
            # Skip illegal moves where white king would capture black king
            if bkState is not None and wkNewPos == bkState[0:2]:
                continue
            
            newBlackState = blackState.copy()
            # Check if the black rook has been captured. If so, remove it from the state
            if brState is not None and brState[0:2] == wkNewPos:
                newBlackState.remove(brState)
            
            fullState = state + newBlackState
            # Move the white pieces to their new state
            self.newBoardSim(fullState)
            
            # If this move gets the white king out of check, then not all moves are watched
            if not self.isWatchedWk(fullState):
                # Restore the original board state before returning
                self.newBoardSim(currentState)
                return False
        
        # Restore the original board state
        self.newBoardSim(currentState)
        # All moves leave the king in check
        return True


    def isWhiteInCheckMate(self, currentState):
        if self.isWatchedWk(currentState) and self.allWkMovementsWatched(currentState):
            return True
        return False
    

    def heuristica(self, currentState, color):
        # This method calculates the heuristic value for the current state.
        # The value is initially computed from White's perspective.
        # If the 'color' parameter indicates Black, the final value is multiplied by -1.

        value = 0

        bkState = self.getPieceState(currentState, 12)  # Black King
        wkState = self.getPieceState(currentState, 6)   # White King
        wrState = self.getPieceState(currentState, 2)   # White Rook
        brState = self.getPieceState(currentState, 8)   # Black Rook

        # Check for checkmate states first (highest priority)
        if self.isBlackInCheckMate(currentState):
            return 10000 if color else -10000
        if self.isWhiteInCheckMate(currentState):
            return -10000 if color else 10000

        filaBk, columnaBk = bkState[0], bkState[1]
        filaWk, columnaWk = wkState[0], wkState[1]

        if wrState is not None:
            filaWr, columnaWr = wrState[0], wrState[1]
        if brState is not None:
            filaBr, columnaBr = brState[0], brState[1]

        # Calculate king-to-king distance (Chebyshev distance)
        fila = abs(filaBk - filaWk)
        columna = abs(columnaWk - columnaBk)
        distReis = max(fila, columna)

        # If the black rook has been captured
        if brState is None:
            value += 500  # Significant material advantage
            
            # CRITICAL: VERY Strong incentive for White King to approach Black King
            # This is THE MOST IMPORTANT factor for delivering checkmate!
            value += (7 - distReis) * 30  # DOUBLED from 15 to 30 - Kings MUST get closer!

            # If white rook exists, encourage coordination with king
            if wrState is not None:
                filaR = abs(filaBk - filaWr)
                columnaR = abs(columnaWr - columnaBk)
                distRookToBlackKing = max(filaR, columnaR)
                
                # Strong bonus for rook controlling same rank or file as black king (cutting off escape)
                if filaWr == filaBk or columnaWr == columnaBk:
                    value += 40  # Increased further
                    
                # Additional bonus if rook is close but not adjacent (ideal mating distance)
                if 1 < distRookToBlackKing <= 3:
                    value += 30  # Increased
                
                # NEW: Heavily reward restricting Black King's mobility
                # Count how many squares the black king can move to
                self.newBoardSim(currentState)
                blackKingMoves = len(self.getNextPositions(bkState))
                # The fewer moves Black has, the better for White (closer to checkmate)
                value += (8 - blackKingMoves) * 40  # CRITICAL: Restrict opponent mobility!

            # If the black king is on the edge, push toward corners (CHECKMATE POSITION)
            if bkState[0] in (0, 7) or bkState[1] in (0, 7):
                # Distance from nearest corner
                cornerDist = min(
                    abs(filaBk - 0) + abs(columnaBk - 0),
                    abs(filaBk - 0) + abs(columnaBk - 7),
                    abs(filaBk - 7) + abs(columnaBk - 0),
                    abs(filaBk - 7) + abs(columnaBk - 7)
                )
                value += (14 - cornerDist) * 25  # Increased from 20
                
                # Extra bonus if king is in corner or near corner (mating net)
                if cornerDist <= 2:
                    value += 60
            else:
                # Push toward edges first (Manhattan distance to nearest edge)
                distToEdge = min(filaBk, 7 - filaBk, columnaBk, 7 - columnaBk)
                value += (3 - distToEdge) * 30  # Increased from 25

        # If the white rook has been captured
        if wrState is None:
            value -= 500  # Significant material disadvantage
            
            # Encourage black king to approach white king
            value -= (7 - distReis) * 10

            # If black rook exists, encourage coordination
            if brState is not None:
                filaR = abs(filaWk - filaBr)
                columnaR = abs(columnaBr - columnaWk)
                distRookToWhiteKing = max(filaR, columnaR)
                if distRookToWhiteKing <= 3:
                    value -= 15
                # Bonus for rook controlling same rank or file as white king
                if filaBr == filaWk or columnaBr == columnaWk:
                    value -= 20

            # If the white king is on the edge, it's vulnerable
            if wkState[0] in (0, 7) or wkState[1] in (0, 7):
                cornerDist = min(
                    abs(filaWk - 0) + abs(columnaWk - 0),
                    abs(filaWk - 0) + abs(columnaWk - 7),
                    abs(filaWk - 7) + abs(columnaWk - 0),
                    abs(filaWk - 7) + abs(columnaWk - 7)
                )
                value -= (14 - cornerDist) * 15
            else:
                distToEdge = min(filaWk, 7 - filaWk, columnaWk, 7 - columnaWk)
                value -= (3 - distToEdge) * 20

        # Both rooks still on board - evaluate positional play
        if brState is not None and wrState is not None:
            # King activity: kings closer to center are more active
            whiteKingCentrality = 3.5 - max(abs(filaWk - 3.5), abs(columnaWk - 3.5))
            blackKingCentrality = 3.5 - max(abs(filaBk - 3.5), abs(columnaWk - 3.5))
            value += (whiteKingCentrality - blackKingCentrality) * 2

            # Rook activity: prefer 7th/2nd rank and central files
            if wrState is not None:
                # Reward white rook on 1st or 2nd rank (attacking position)
                if filaWr <= 1:
                    value += 5
                # Central files are valuable
                wrCentrality = 3.5 - abs(columnaWr - 3.5)
                value += wrCentrality * 2

            if brState is not None:
                # Reward black rook on 6th or 7th rank
                if filaBr >= 6:
                    value -= 5
                # Central files are valuable
                brCentrality = 3.5 - abs(columnaBr - 3.5)
                value -= brCentrality * 2

            # Penalty if kings are too close (risk of perpetual check)
            if distReis <= 2:
                value -= 3

        # If the black king is in check, reward this state.
        if self.isWatchedBk(currentState):
            value += 30

        # If the white king is in check, penalize this state.
        if self.isWatchedWk(currentState):
            value -= 30

        # Add small random tiebreaker to avoid loops when multiple moves have same value
        # This is crucial to prevent perpetual repetition
        value += random.uniform(-0.5, 0.5)

        # If the current player is Black, invert the heuristic value.
        if not color:
            value *= -1

        return value

    def mean(self, values):
        # Calculate the arithmetic mean (average) of a list of numeric values.
        total = 0
        n = len(values)
        
        for i in range(n):
            total += values[i]

        return total / n

    def stateToKey(self, state):
        """Convert state to hashable key for transposition table"""
        # Sort state to handle piece order variations
        return tuple(sorted(tuple(piece) for piece in state))

    def orderMoves(self, moveStates, currentState, isWhite):
        """
        Order moves to improve alpha-beta pruning efficiency.
        Prioritize: 1) Captures, 2) Checks, 3) Other moves
        """
        captures = []
        checks = []
        others = []
        
        if isWhite:
            blackState = self.getBlackState(currentState)
            blackPositions = set((s[0], s[1]) for s in blackState)
            
            for moveState in moveStates:
                whitePositions = set((s[0], s[1]) for s in moveState)
                
                # Check if it's a capture
                if whitePositions & blackPositions:
                    captures.append(moveState)
                else:
                    # Quick check if it gives check (simplified)
                    blackStateCopy = [s for s in blackState if (s[0], s[1]) not in whitePositions]
                    fullState = moveState + blackStateCopy
                    
                    # Only check for valid states
                    if self.getPieceState(fullState, 6) and self.getPieceState(fullState, 12):
                        if self.isWatchedBk(fullState):
                            checks.append(moveState)
                        else:
                            others.append(moveState)
                    else:
                        others.append(moveState)
        else:
            whiteState = self.getWhiteState(currentState)
            whitePositions = set((s[0], s[1]) for s in whiteState)
            
            for moveState in moveStates:
                blackPositions = set((s[0], s[1]) for s in moveState)
                
                # Check if it's a capture
                if blackPositions & whitePositions:
                    captures.append(moveState)
                else:
                    # Quick check if it gives check (simplified)
                    whiteStateCopy = [s for s in whiteState if (s[0], s[1]) not in blackPositions]
                    fullState = whiteStateCopy + moveState
                    
                    # Only check for valid states
                    if self.getPieceState(fullState, 6) and self.getPieceState(fullState, 12):
                        if self.isWatchedWk(fullState):
                            checks.append(moveState)
                        else:
                            others.append(moveState)
                    else:
                        others.append(moveState)
        
        # Return in priority order: captures first, then checks, then others
        return captures + checks + others

    def minimax(self, state, depth, isWhite, alpha=float('-inf'), beta=float('inf')):
        """
        Minimax algorithm with Alpha-Beta pruning and transposition table
        
        Args:
            state: Current board state
            depth: Search depth remaining
            isWhite: True if maximizing for White, False if minimizing for Black
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            (value, bestState) tuple
        """
        # Check transposition table (DISABLED to prevent repetition loops)
        # The transposition table causes deterministic behavior that leads to repetition
        # With random tiebreaker in heuristic, we need fresh evaluations each time
        stateKey = self.stateToKey(state)
        # if stateKey in self.transpositionTable and depth < 4:  # Only use cache for non-root positions
        #     cachedDepth, cachedValue, cachedState = self.transpositionTable[stateKey]
        #     if cachedDepth >= depth:
        #         return (cachedValue, cachedState)
        
        # Terminal conditions
        if depth == 0:
            # Always evaluate from White's perspective
            value = self.heuristica(state, True)
            return (value, state)
        
        # Check for checkmate (always return from White's perspective)
        if self.isWhiteInCheckMate(state):
            return (-10000, state)
        if self.isBlackInCheckMate(state):
            return (10000, state)
        
        # Get possible next states
        if isWhite:
            nextStates = self.getListNextStatesW(self.getWhiteState(state))
            if len(nextStates) == 0:
                # No moves available (stalemate or checkmate)
                return (self.heuristica(state, True), state)
            
            # Maximize for White
            bestValue = float('-inf')
            bestState = nextStates[0] + self.getBlackState(state)
            
            # Move ordering: prioritize captures and checks
            orderedStates = self.orderMoves(nextStates, state, True)
            
            for whiteState in orderedStates:
                # Build full state, removing any captured black pieces
                blackState = self.getBlackState(state).copy()
                
                # Get the original black king position (before white's move)
                origBkState = self.getPieceState(state, 12)
                origBkPos = (origBkState[0], origBkState[1]) if origBkState else None
                
                # Check if white king is trying to capture a piece
                wkNewState = self.getPieceState(whiteState, 6)
                wkNewPos = (wkNewState[0], wkNewState[1]) if wkNewState else None
                
                # Check if any white piece occupies a black piece's square (capture)
                whitePositions = [(s[0], s[1]) for s in whiteState]
                capturedBlack = [s for s in blackState if (s[0], s[1]) in whitePositions]
                blackState = [s for s in blackState if (s[0], s[1]) not in whitePositions]
                
                # If white king captured a black piece, check if it's protected by black king
                if wkNewPos and capturedBlack and origBkPos:
                    # Check if the captured piece's square is adjacent to black king (protected)
                    for captured in capturedBlack:
                        capturedPos = (captured[0], captured[1])
                        if max(abs(capturedPos[0] - origBkPos[0]), abs(capturedPos[1] - origBkPos[1])) == 1:
                            # White king is trying to capture a piece protected by black king - ILLEGAL!
                            continue
                
                fullState = whiteState + blackState
                
                # Skip invalid states where king is captured
                wkState = self.getPieceState(fullState, 6)
                bkState = self.getPieceState(fullState, 12)
                if wkState is None or bkState is None:
                    continue
                
                # Skip moves that leave White King in check (illegal move)
                if self.isWatchedWk(fullState):
                    continue
                
                # Recurse with alpha-beta
                value, _ = self.minimax(fullState, depth - 1, False, alpha, beta)
                
                if value > bestValue:
                    bestValue = value
                    bestState = fullState
                
                # Alpha-Beta pruning
                alpha = max(alpha, value)
                if beta <= alpha:
                    break  # Beta cutoff
            
            # Store in transposition table (DISABLED to prevent repetition)
            # self.transpositionTable[stateKey] = (depth, bestValue, bestState)
            return (bestValue, bestState)
        else:
            nextStates = self.getListNextStatesB(self.getBlackState(state))
            if len(nextStates) == 0:
                # No moves available (stalemate or checkmate)
                return (self.heuristica(state, True), state)
            
            # Minimize for Black
            bestValue = float('inf')
            bestState = self.getWhiteState(state) + nextStates[0]
            
            # Move ordering: prioritize captures and checks
            orderedStates = self.orderMoves(nextStates, state, False)
            
            for blackState in orderedStates:
                # Build full state, removing any captured white pieces
                whiteState = self.getWhiteState(state).copy()
                
                # Get the original white king position (before black's move)
                origWkState = self.getPieceState(state, 6)
                origWkPos = (origWkState[0], origWkState[1]) if origWkState else None
                
                # Check if black king is trying to capture a piece
                bkNewState = self.getPieceState(blackState, 12)
                bkNewPos = (bkNewState[0], bkNewState[1]) if bkNewState else None
                
                # Check if any black piece occupies a white piece's square (capture)
                blackPositions = [(s[0], s[1]) for s in blackState]
                capturedWhite = [s for s in whiteState if (s[0], s[1]) in blackPositions]
                whiteState = [s for s in whiteState if (s[0], s[1]) not in blackPositions]
                
                # If black king captured a white piece, check if it's protected by white king
                if bkNewPos and capturedWhite and origWkPos:
                    # Check if the captured piece's square is adjacent to white king (protected)
                    for captured in capturedWhite:
                        capturedPos = (captured[0], captured[1])
                        if max(abs(capturedPos[0] - origWkPos[0]), abs(capturedPos[1] - origWkPos[1])) == 1:
                            # Black king is trying to capture a piece protected by white king - ILLEGAL!
                            continue
                
                fullState = whiteState + blackState
                
                # Skip invalid states where king is captured
                wkState = self.getPieceState(fullState, 6)
                bkState = self.getPieceState(fullState, 12)
                if wkState is None or bkState is None:
                    continue
                
                # Skip moves that leave Black King in check (illegal move)
                if self.isWatchedBk(fullState):
                    continue
                
                # Recurse with alpha-beta
                value, _ = self.minimax(fullState, depth - 1, True, alpha, beta)
                
                if value < bestValue:
                    bestValue = value
                    bestState = fullState
                
                # Alpha-Beta pruning
                beta = min(beta, value)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            # Store in transposition table (DISABLED to prevent repetition)
            # self.transpositionTable[stateKey] = (depth, bestValue, bestState)
            return (bestValue, bestState)

    def minimaxGame(self, depthWhite, depthBlack, verbose=True):
        """
        Play a complete game using minimax for both players
        
        Args:
            depthWhite: Search depth for White
            depthBlack: Search depth for Black
            verbose: If True, print board state after each move
            
        Returns:
            Winner string: "White", "Black", or "Draw"
        """
        # Clear transposition table for new game
        self.transpositionTable.clear()
        
        currentState = self.getCurrentState()
        
        # Track visited states (list of states from start to end)
        visitedStates = [currentState.copy()]
        
        # Track position repetitions for draw detection
        positionHistory = {}
        posKey = self.stateToKey(currentState)
        positionHistory[posKey] = 1
        
        if verbose:
            print("\n=== Starting Minimax Game ===")
            print(f"White depth: {depthWhite}, Black depth: {depthBlack}")
            print(f"Initial state: {currentState}")
            self.chess.boardSim.print_board()
        
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
                    print(f"  Minimax depth used: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                    print(f"  Transposition table entries: {len(self.transpositionTable)}")
                return "Draw"
            
            # White's turn
            if verbose:
                print(f"\n--- Move {moveCount}: White's turn ---")
            
            _, bestStateWhite = self.minimax(currentState, depthWhite, True)
            currentState = bestStateWhite
            visitedStates.append(currentState.copy())
            self.newBoardSim(currentState)
            
            if verbose:
                self.chess.boardSim.print_board()
            
            # Check for position repetition (threefold repetition = draw)
            posKey = self.stateToKey(currentState)
            positionHistory[posKey] = positionHistory.get(posKey, 0) + 1
            if positionHistory[posKey] >= 3:
                if verbose:
                    print(f"\n*** DRAW (threefold repetition) ***")
                    print(f"\nGame Statistics:")
                    print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
                    print(f"  Total full moves: {moveCount}")
                    print(f"  Minimax depth used: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                    print(f"  Transposition table entries: {len(self.transpositionTable)}")
                return "Draw"
            
            # Check if Black is in checkmate
            if self.isBlackInCheckMate(currentState):
                if verbose:
                    print("\n*** WHITE WINS BY CHECKMATE! ***")
                    print(f"\nGame Statistics:")
                    print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
                    print(f"  Total full moves: {moveCount}")
                    print(f"  Minimax depth used: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                    print(f"  Transposition table entries: {len(self.transpositionTable)}")
                return "White"
            
            # Black's turn
            if verbose:
                print(f"\n--- Move {moveCount}: Black's turn ---")
            
            _, bestStateBlack = self.minimax(currentState, depthBlack, False)
            currentState = bestStateBlack
            visitedStates.append(currentState.copy())
            self.newBoardSim(currentState)
            
            if verbose:
                self.chess.boardSim.print_board()
            
            # Check for position repetition (threefold repetition = draw)
            posKey = self.stateToKey(currentState)
            positionHistory[posKey] = positionHistory.get(posKey, 0) + 1
            if positionHistory[posKey] >= 3:
                if verbose:
                    print(f"\n*** DRAW (threefold repetition) ***")
                    print(f"\nGame Statistics:")
                    print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
                    print(f"  Total full moves: {moveCount}")
                    print(f"  Minimax depth used: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                    print(f"  Transposition table entries: {len(self.transpositionTable)}")
                return "Draw"
            
            # Check if White is in checkmate
            if self.isWhiteInCheckMate(currentState):
                if verbose:
                    print("\n*** BLACK WINS BY CHECKMATE! ***")
                    print(f"\nGame Statistics:")
                    print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
                    print(f"  Total full moves: {moveCount}")
                    print(f"  Minimax depth used: White={depthWhite}, Black={depthBlack}")
                    print(f"  Total states visited: {len(visitedStates)}")
                    print(f"  Transposition table entries: {len(self.transpositionTable)}")
                return "Black"
        
        # Game reached max moves - it's a draw
        if verbose:
            print(f"\n*** DRAW (reached {maxMoves} moves) ***")
            print(f"\nGame Statistics:")
            print(f"  Total moves (half-moves): {len(visitedStates) - 1}")
            print(f"  Total full moves: {moveCount}")
            print(f"  Minimax depth used: White={depthWhite}, Black={depthBlack}")
            print(f"  Total states visited: {len(visitedStates)}")
            print(f"  Transposition table entries: {len(self.transpositionTable)}")
        return "Draw"


# =============================================================
# ==================   OUR CODE ==== ==========================
# =============================================================

    def minimaxGame_OLD(self, depthWhite,depthBlack):
        
        currentState = self.getCurrentState()        
        # Your code here


    def alphaBetaPoda(self, depthWhite,depthBlack):
        
        currentState = self.getCurrentState()
        # Your code here  
        
    def expectimax(self, depthWhite, depthBlack):
        
        currentState = self.getCurrentState()
        # Your code here       
        

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit(usage())

    # Initialize an empty 8x8 chess board
    TA = np.zeros((8, 8))


    # Load initial positions of the pieces
    TA = np.zeros((8, 8))
    TA[7][0] = 2   
    TA[7][5] = 6   
    TA[0][7] = 8   
    TA[0][5] = 12  

    # Initialise board and print
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    print("printing board")
    aichess.chess.boardSim.print_board()
    
    # Run exercise 1
    print("\n==== Exercise 1: Minimax Game (Depth 4 vs 4) ===== \n")
    print("Both White and Black use Minimax algorithm with depth 4")
    print("White moves first (as per chess rules)")
    print("\nStarting game...\n")
    
    winner = aichess.minimaxGame(4, 4, verbose=True)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {winner} wins!")
    print(f"{'='*60}")
    # Add code to save results and continue with other exercises
