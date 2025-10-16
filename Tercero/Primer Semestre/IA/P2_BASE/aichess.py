#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
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

    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        bkPosition = self.getPieceState(currentState, 12)[0:2]
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

        # If the white king has been captured, this is not a valid configuration
        if wkState is None:
            return False

        # Check all possible moves of the white king to see if it can capture the black king
        for wkPosition in self.getNextPositions(wkState):
            if bkPosition == wkPosition:
                # Black king would be in check
                return True

        if wrState is not None:
            # Check all possible moves of the white rook to see if it can capture the black king
            for wrPosition in self.getNextPositions(wrState):
                if bkPosition == wrPosition:
                    return True

        return False

    def allBkMovementsWatched(self, currentState):
        # In this method, we check if the black king is threatened by the white pieces

        self.newBoardSim(currentState)
        # Get the current state of the black king
        bkState = self.getPieceState(currentState, 12)
        allWatched = False

        # If the black king is on the edge of the board, all its moves might be under threat
        if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
            wrState = self.getPieceState(currentState, 2)
            whiteState = self.getWhiteState(currentState)
            allWatched = True
            # Get the future states of the black pieces
            nextBStates = self.getListNextStatesB(self.getBlackState(currentState))

            for state in nextBStates:
                newWhiteState = whiteState.copy()
                # Check if the white rook has been captured; if so, remove it from the state
                if wrState is not None and wrState[0:2] == state[0][0:2]:
                    newWhiteState.remove(wrState)
                state = state + newWhiteState
                # Move the black pieces to the new state
                self.newBoardSim(state)

                # Check if in this position the black king is not threatened; 
                # if so, not all its moves are under threat
                if not self.isWatchedBk(state):
                    allWatched = False
                    break

        # Restore the original board state
        self.newBoardSim(currentState)
        return allWatched

    def isBlackInCheckMate(self, currentState):
        if self.isWatchedBk(currentState) and self.allBkMovementsWatched(currentState):
            return True

        return False


    def isWatchedWk(self, currentState):
        self.newBoardSim(currentState)

        wkPosition = self.getPieceState(currentState, 6)[0:2]
        bkState = self.getPieceState(currentState, 12)
        brState = self.getPieceState(currentState, 8)

        # If the black king has been captured, this is not a valid configuration
        if bkState is None:
            return False

        # Check all possible moves for the black king and see if it can capture the white king
        for bkPosition in self.getNextPositions(bkState):
            if wkPosition == bkPosition:
                # White king would be in check
                return True

        if brState is not None:
            # Check all possible moves for the black rook and see if it can capture the white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True

        return False

    def allWkMovementsWatched(self, currentState):

        self.newBoardSim(currentState)
        # In this method, we check if the white king is threatened by black pieces
        # Get the current state of the white king
        wkState = self.getPieceState(currentState, 6)
        allWatched = False

        # If the white king is on the edge of the board, it may be more vulnerable
        if wkState[0] == 0 or wkState[0] == 7 or wkState[1] == 0 or wkState[1] == 7:
            # Get the state of the black pieces
            brState = self.getPieceState(currentState, 8)
            blackState = self.getBlackState(currentState)
            allWatched = True

            # Get the possible future states for the white pieces
            nextWStates = self.getListNextStatesW(self.getWhiteState(currentState))
            for state in nextWStates:
                newBlackState = blackState.copy()
                # Check if the black rook has been captured. If so, remove it from the state
                if brState is not None and brState[0:2] == state[0][0:2]:
                    newBlackState.remove(brState)
                state = state + newBlackState
                # Move the white pieces to their new state
                self.newBoardSim(state)
                # Check if the white king is not threatened in this position,
                # which implies that not all of its possible moves are under threat
                if not self.isWatchedWk(state):
                    allWatched = False
                    break

        # Restore the original board state
        self.newBoardSim(currentState)
        return allWatched


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

        # If any king is captured, return terminal-like scores
        if bkState is None and wkState is None:
            # Extremely rare/impossible, treat as draw
            return 0
        if bkState is None:
            # White wins
            return 10000 if color else -10000
        if wkState is None:
            # Black wins
            return -10000 if color else 10000

        filaBk, columnaBk = bkState[0], bkState[1]
        filaWk, columnaWk = wkState[0], wkState[1]

        if wrState is not None:
            filaWr, columnaWr = wrState[0], wrState[1]
        if brState is not None:
            filaBr, columnaBr = brState[0], brState[1]

        # If the black rook has been captured
        if brState is None:
            value += 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and wrState is not None:
                filaR = abs(filaBk - filaWr)
                columnaR = abs(columnaWr - columnaBk)
                value += (min(filaR, columnaR) + abs(filaR - columnaR)) / 10

            # For White: the closer our king is to the opponent’s king, the better.
            # Subtract 7 from the king-to-king distance since 7 is the maximum distance possible on the board.
            value += (7 - distReis)

            # If the black king is against a wall, prioritize pushing him into a corner (ideal for checkmate).
            if bkState[0] in (0, 7) or bkState[1] in (0, 7):
                value += (abs(filaBk - 3.5) + abs(columnaBk - 3.5)) * 10
            # Otherwise, encourage moving the black king closer to the wall.
            else:
                value += (max(abs(filaBk - 3.5), abs(columnaBk - 3.5))) * 10

        # If the white rook has been captured.
        # The logic is similar to the previous section but with reversed (negative) values.
        if wrState is None:
            value -= 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and brState is not None:
                filaR = abs(filaWk - filaBr)
                columnaR = abs(columnaBr - columnaWk)
                value -= (min(filaR, columnaR) + abs(filaR - columnaR)) / 10

            # For White: being closer to the opposing king is better.
            # Subtract 7 from the distance since that’s the maximum possible distance.
            value += (-7 + distReis)

            # If the white king is against a wall, penalize that position.
            if wkState[0] in (0, 7) or wkState[1] in (0, 7):
                value -= (abs(filaWk - 3.5) + abs(columnaWk - 3.5)) * 10
            # Otherwise, encourage the king to stay away from the wall.
            else:
                value -= (max(abs(filaWk - 3.5), abs(columnaWk - 3.5))) * 10

        # Note: avoid expensive board reconstructions here for speed.
        # If desired, check conditions can be incorporated with fast checks below.

        # If the current player is Black, invert the heuristic value.
        if not color:
            value *= -1

        return value

    # --------- Fast threat detection for this reduced piece set (K+R vs K+R) ---------
    def _extract_positions(self, state):
        wk = self.getPieceState(state, 6)
        wr = self.getPieceState(state, 2)
        bk = self.getPieceState(state, 12)
        br = self.getPieceState(state, 8)
        return wk, wr, bk, br

    def _rook_attacks(self, from_pos, to_pos, occupied):
        if from_pos is None or to_pos is None:
            return False
        r1, c1 = from_pos
        r2, c2 = to_pos
        if r1 != r2 and c1 != c2:
            return False
        # collect positions strictly between
        if r1 == r2:
            rng = range(min(c1, c2) + 1, max(c1, c2))
            for y in rng:
                if (r1, y) in occupied:
                    return False
            return True
        else:
            rng = range(min(r1, r2) + 1, max(r1, r2))
            for x in rng:
                if (x, c1) in occupied:
                    return False
            return True

    def _king_adjacent(self, k1, k2):
        if k1 is None or k2 is None:
            return False
        return max(abs(k1[0] - k2[0]), abs(k1[1] - k2[1])) == 1

    def isWatchedBk_fast(self, state):
        wk, wr, bk, br = self._extract_positions(state)
        if bk is None:
            return False
        occ = set()
        for p in state:
            occ.add((p[0], p[1]))
        if wk is not None and self._king_adjacent((wk[0], wk[1]), (bk[0], bk[1])):
            return True
        if wr is not None and self._rook_attacks((wr[0], wr[1]), (bk[0], bk[1]), occ - {(wr[0], wr[1]), (bk[0], bk[1])}):
            return True
        return False

    def isWatchedWk_fast(self, state):
        wk, wr, bk, br = self._extract_positions(state)
        if wk is None:
            return False
        occ = set()
        for p in state:
            occ.add((p[0], p[1]))
        if bk is not None and self._king_adjacent((bk[0], bk[1]), (wk[0], wk[1])):
            return True
        if br is not None and self._rook_attacks((br[0], br[1]), (wk[0], wk[1]), occ - {(br[0], br[1]), (wk[0], wk[1])}):
            return True
        return False
    
    def mean(self, values):
        # Calculate the arithmetic mean (average) of a list of numeric values.
        total = 0
        n = len(values)
        
        for i in range(n):
            total += values[i]

        return total / n


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

    def minimaxGame(self, depthWhite,depthBlack, verbose=True):
        # Play a full game where White and Black both use Minimax with the given depths.
        # Whites always move first.

        # Helpers
        WIN_SCORE = 10_000
        LOSE_SCORE = -10_000

        def is_terminal(state):
            # King captured ends immediately
            wk = self.getPieceState(state, 6)
            bk = self.getPieceState(state, 12)
            if wk is None or bk is None:
                return True
            # Detect checkmates using provided methods (slower but more accurate)
            if self.isBlackInCheckMate(state) or self.isWhiteInCheckMate(state):
                return True
            return False

        def terminal_value(state, perspective_white):
            # perspective_white: True if evaluating for White to move
            wk = self.getPieceState(state, 6)
            bk = self.getPieceState(state, 12)
            if bk is None:
                # Black king captured -> White wins
                return WIN_SCORE if perspective_white else LOSE_SCORE
            if wk is None:
                # White king captured -> Black wins
                return LOSE_SCORE if perspective_white else WIN_SCORE
            if self.isBlackInCheckMate(state):
                return WIN_SCORE if perspective_white else LOSE_SCORE
            if self.isWhiteInCheckMate(state):
                return LOSE_SCORE if perspective_white else WIN_SCORE
            # Not terminal
            return None

        def combine_with_opponent(mover_only_state, mover_is_white, opponent_state):
            # Remove captured opponent piece if moved piece lands on it
            moved_pos = mover_only_state[0][0:2]
            new_opp = []
            for p in opponent_state:
                if [p[0], p[1]] == moved_pos:
                    # captured
                    continue
                new_opp.append(p)
            if mover_is_white:
                return mover_only_state + new_opp
            else:
                return new_opp + mover_only_state

        succ_cache = {}
        check_cache = {}
        def successors(state, white_to_move):
            # Build next full states combining moving side's states with opponent pieces, handling captures
            key = (tuple(sorted((p[0], p[1], p[2]) for p in state)), white_to_move)
            if key in succ_cache:
                return succ_cache[key]
            self.newBoardSim(state)  # ensure generators see the right board
            if white_to_move:
                w_state = self.getWhiteState(state)
                b_state = self.getBlackState(state)
                next_w_only = self.getListNextStatesW(w_state)
                combined = [combine_with_opponent(s, True, b_state) for s in next_w_only]
                # filter moves that leave white in check
                res = []
                for ns in combined:
                    kt2 = ('wk', tuple(sorted((p[0], p[1], p[2]) for p in ns)))
                    if kt2 in check_cache:
                        wcheck = check_cache[kt2]
                    else:
                        wcheck = self.isWatchedWk_fast(ns)
                        check_cache[kt2] = wcheck
                    if not wcheck:
                        res.append(ns)
                succ_cache[key] = res
                return res
            else:
                w_state = self.getWhiteState(state)
                b_state = self.getBlackState(state)
                next_b_only = self.getListNextStatesB(b_state)
                combined = [combine_with_opponent(s, False, w_state) for s in next_b_only]
                # filter moves that leave black in check
                res = []
                for ns in combined:
                    kt2 = ('bk', tuple(sorted((p[0], p[1], p[2]) for p in ns)))
                    if kt2 in check_cache:
                        bcheck = check_cache[kt2]
                    else:
                        bcheck = self.isWatchedBk_fast(ns)
                        check_cache[kt2] = bcheck
                    if not bcheck:
                        res.append(ns)
                succ_cache[key] = res
                return res

        # Transposition table: key -> (value, best_state)
        tt = {}

        def state_key(state, depth, white_to_move):
            # Sort pieces by id to normalize ordering
            tup = tuple(sorted((p[0], p[1], p[2]) for p in state))
            return (tup, depth, white_to_move)

        def minimax(state, depth, white_to_move, alpha=-math.inf, beta=math.inf):
            # Returns (value, best_state)
            term = terminal_value(state, white_to_move)
            if term is not None:
                return term, state
            if depth == 0:
                # Heuristic from the current player's perspective
                return self.heuristica(state, color=white_to_move), state

            key = state_key(state, depth, white_to_move)
            if key in tt:
                return tt[key]

            next_states = successors(state, white_to_move)
            if not next_states:
                # No legal moves -> treat as very bad for player to move
                val = (LOSE_SCORE if white_to_move else WIN_SCORE)
                res = (val, state)
                tt[key] = res
                return res

            # basic move ordering: evaluate heuristic shallowly to order (with caching)
            scored = []
            h_cache = {}
            for ns in next_states:
                kt = tuple(sorted((p[0], p[1], p[2]) for p in ns))
                if kt in h_cache:
                    h = h_cache[kt]
                else:
                    h = self.heuristica(ns, color=white_to_move)
                    h_cache[kt] = h
                scored.append((h, ns))
            scored.sort(key=lambda x: x[0], reverse=white_to_move)
            ordered_states = [ns for _, ns in scored]

            if white_to_move:
                best_val = -math.inf
                best_state = ordered_states[0]
                for ns in ordered_states:
                    val, _ = minimax(ns, depth - 1, False, alpha, beta)
                    if val > best_val:
                        best_val = val
                        best_state = ns
                    alpha = max(alpha, best_val)
                    if beta <= alpha:
                        break
                res = (best_val, best_state)
                tt[key] = res
                return res
            else:
                best_val = math.inf
                best_state = ordered_states[0]
                for ns in ordered_states:
                    val, _ = minimax(ns, depth - 1, True, alpha, beta)
                    if val < best_val:
                        best_val = val
                        best_state = ns
                    beta = min(beta, best_val)
                    if beta <= alpha:
                        break
                res = (best_val, best_state)
                tt[key] = res
                return res

        # Initialize from the simulation board (not the main board), then play
        currentState = []
        for i in self.chess.boardSim.currentStateW:
            currentState.append(i)
        for j in self.chess.boardSim.currentStateB:
            currentState.append(j)

        white_turn = True
        visited_path = [self.copyState(currentState)]
        max_moves = 200
        moves_done = 0

        while not is_terminal(currentState) and moves_done < max_moves:
            self.newBoardSim(currentState)
            depth = depthWhite if white_turn else depthBlack
            _, best_state = minimax(currentState, depth, white_turn)
            currentState = self.copyState(best_state)
            visited_path.append(self.copyState(currentState))
            self.newBoardSim(currentState)
            moves_done += 1
            white_turn = not white_turn

        # Report
        winner = None
        if self.getPieceState(currentState, 12) is None or self.isBlackInCheckMate(currentState):
            winner = 'white'
        elif self.getPieceState(currentState, 6) is None or self.isWhiteInCheckMate(currentState):
            winner = 'black'
        else:
            winner = 'draw'

        if verbose:
            print("Game finished. Winner:", winner)
            print("Minimal depth (plies) to reach target:", len(visited_path) - 1)
            print("Visited states from origin to target (sequence):")
            for idx, st in enumerate(visited_path):
                print(f"Step {idx}: {st}")
            # Show final board
            self.newBoardSim(currentState)
            self.chess.boardSim.print_board()

        return winner, visited_path

def run_depth_grid(TA: np.ndarray, repeats: int = 3, depth_values=(3,4), verbose: bool = False):
    # Run all combinations of white/black depths and compute white win rates. Plot if matplotlib is available.
    results = {}
    for dw in depth_values:
        for db in depth_values:
            white=black=draw=0
            for r in range(repeats):
                aich = Aichess(TA, True)
                winner, _ = aich.minimaxGame(dw, db, verbose=verbose)
                if winner == 'white':
                    white += 1
                elif winner == 'black':
                    black += 1
                else:
                    draw += 1
            rate = white / max(1, (white+black+draw))
            results[(dw,db)] = {
                'white': white,
                'black': black,
                'draw': draw,
                'white_win_rate': rate
            }

    # Build a matrix for plotting
    vals = list(depth_values)
    size = len(vals)
    heat = np.zeros((size,size), dtype=float)
    for i,dw in enumerate(vals):
        for j,db in enumerate(vals):
            heat[i,j] = results[(dw,db)]['white_win_rate']

    # Try plotting
    saved = None
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(heat, cmap='Blues', vmin=0.0, vmax=1.0)
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticklabels([str(d) for d in vals])
        ax.set_yticklabels([str(d) for d in vals])
        ax.set_xlabel('Black depth')
        ax.set_ylabel('White depth')
        ax.set_title('White win rate (Minimax)')
        for i in range(size):
            for j in range(size):
                ax.text(j, i, f"{heat[i,j]*100:.0f}%", ha='center', va='center', color='black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        saved = 'white_win_rate_heatmap.png'
        plt.tight_layout()
        plt.savefig(saved)
        # Optionally show if running interactively
        # plt.show()
    except Exception as e:
        # Fallback: no plotting available
        saved = None

    return results, heat, depth_values, saved


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

    # Single sanity run (optional)
    aichess = Aichess(TA, True)
    print("initial board")
    aichess.chess.boardSim.print_board()
    aichess.minimaxGame(4, 4, verbose=True)

    # Depth grid 3..4 for both colors, a few repeats each
    print("\nRunning depth grid (3..4) repeats=3 ...")
    results, heat, depths, saved = run_depth_grid(TA, repeats=3, depth_values=(3,4), verbose=False)
    print("Results (white_win_rate) by (white_depth, black_depth):")
    for dw in depths:
        for db in depths:
            r = results[(dw,db)]
            print(f" W{dw} vs B{db}: rate={r['white_win_rate']:.2f}  (W:{r['white']} B:{r['black']} D:{r['draw']})")
    if saved:
        print("Saved heatmap to:", saved)
