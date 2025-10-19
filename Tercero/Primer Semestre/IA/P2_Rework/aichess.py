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

        # Fast-lookup caches for visited checks (order-insensitive)
        self._visited_state_signatures = set()
        self._visited_situation_signatures = set()

    # ---------- Internal helpers ----------
    def _state_signature(self, state: RawStateType):
        # Use frozenset of (piece, row, col) to ignore list ordering
        return frozenset((p[2], p[0], p[1]) for p in state)

    def _situation_signature(self, color: bool, state: RawStateType):
        return (bool(color), self._state_signature(state))

    def copyState(self, state):
        
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState
        
    def isVisitedSituation(self, color, mystate):
        # Order-insensitive, O(N log N) using set signature
        sig = self._situation_signature(color, mystate)
        if sig in self._visited_situation_signatures:
            return True
        # Fallback for legacy entries stored in listVisitedSituations
        for c, s in self.listVisitedSituations:
            if c == color and self.isSameState(mystate, s):
                # cache to speed up next time
                self._visited_situation_signatures.add(sig)
                return True
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
        # Fast, order-insensitive equality
        return self._state_signature(a) == self._state_signature(b)

    def isVisited(self, mystate):
        # O(1) with signature caching, avoids factorial permutations
        sig = self._state_signature(mystate)
        if sig in self._visited_state_signatures:
            return True
        # Fallback: check legacy list and populate cache lazily
        for s in self.listVisitedStates:
            if self.isSameState(mystate, s):
                self._visited_state_signatures.add(sig)
                return True
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
        # Return all white pieces present in currentState (codes 1..6)
        whiteState = []
        for p in currentState:
            if 1 <= p[2] <= 6:
                whiteState.append(p)
        return whiteState

    def getBlackState(self, currentState):
        # Return all black pieces present in currentState (codes 7..12)
        blackState = []
        for p in currentState:
            if 7 <= p[2] <= 12:
                blackState.append(p)
        return blackState

    def getMovement(self, state, nextState):
        # Identify moved piece by comparing per-piece positions; handle captures.
        start_map = {p[2]: (p[0], p[1]) for p in state}
        to_map = {p[2]: (p[0], p[1]) for p in nextState}

        moved_code = None
        moved_from = None
        moved_to = None

        # Prefer pieces present in both; moved piece changes coordinates
        for code, pos in start_map.items():
            if code in to_map and pos != to_map[code]:
                moved_code = code
                moved_from = pos
                moved_to = to_map[code]
                break

        # Fallback: if none changed, there may be a piece that appears only in nextState (promotion-like),
        # or disappears only in start (capture of that piece by another already accounted). Not expected here.
        if moved_code is None:
            return [None, None]

        return [[moved_from[0], moved_from[1], moved_code], [moved_to[0], moved_to[1], moved_code]]

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
        # Robustly detect moved piece between two arbitrary-sized states and move it on boardSim
        # 'start' and 'to' are lists of piece states [row, col, code]
        moved_from_to = self.getMovement(start, to)
        if moved_from_to[0] is None or moved_from_to[1] is None:
            return
        start_pos = (moved_from_to[0][0], moved_from_to[0][1])
        end_pos = (moved_from_to[1][0], moved_from_to[1][1])
        self.chess.moveSim(start_pos, end_pos)

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
        # Check if ALL legal black moves still leave the black king in check.
        # If there exists at least one move where the black king is safe, return False.

        self.newBoardSim(currentState)

        whiteState = self.getWhiteState(currentState)
        wrState = self.getPieceState(currentState, 2)

        # Generate all legal next states for black (moving BK or BR)
        nextBStates = self.getListNextStatesB(self.getBlackState(currentState))

        for state in nextBStates:
            # Build full position combining moved black state with (possibly reduced) white state
            newWhiteState = whiteState.copy()
            # If black's move lands on the white rook square, remove the rook from the next position
            if wrState is not None and wrState[0:2] == state[0][0:2]:
                newWhiteState.remove(wrState)

            candidate = state + newWhiteState
            self.newBoardSim(candidate)

            # If candidate position leaves black king NOT in check, then not all moves are watched
            if not self.isWatchedBk(candidate):
                self.newBoardSim(currentState)
                return False

        # Restore and report all moves keep king in check
        self.newBoardSim(currentState)
        return True

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
        # Check if ALL legal white moves still leave the white king in check.
        # If there exists at least one move where the white king is safe, return False.

        self.newBoardSim(currentState)

        blackState = self.getBlackState(currentState)
        brState = self.getPieceState(currentState, 8)

        # Generate all legal next states for white (moving WK or WR)
        nextWStates = self.getListNextStatesW(self.getWhiteState(currentState))

        for state in nextWStates:
            # Build full position combining moved white state with (possibly reduced) black state
            newBlackState = blackState.copy()
            # If white's move lands on the black rook square, remove the rook from the next position
            if brState is not None and brState[0:2] == state[0][0:2]:
                newBlackState.remove(brState)

            candidate = state + newBlackState
            self.newBoardSim(candidate)

            # If candidate position leaves white king NOT in check, then not all moves are watched
            if not self.isWatchedWk(candidate):
                self.newBoardSim(currentState)
                return False

        # Restore and report all moves keep king in check
        self.newBoardSim(currentState)
        return True


    def isWhiteInCheckMate(self, currentState):
        if self.isWatchedWk(currentState) and self.allWkMovementsWatched(currentState):
            return True
        return False
    

    def heuristica(self, currentState, color):
        # Heuristic value from White's point of view. If color is Black, we invert at the end.

        # Helpers
        def chebyshev(a, b):
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

        def dist_to_edge_sq(pos):
            # 0..7 indexes; 3.5 is board center. Higher is closer to edge.
            return max(abs(pos[0] - 3.5), abs(pos[1] - 3.5))

        # Material and positions
        bkState = self.getPieceState(currentState, 12)  # Black King
        wkState = self.getPieceState(currentState, 6)   # White King
        wrState = self.getPieceState(currentState, 2)   # White Rook
        brState = self.getPieceState(currentState, 8)   # Black Rook

        value = 0.0

        # Immediate draw detection: Kings only (K vs K)
        if self.isKingsOnlyDraw(currentState):
            return 0.0 if color else -0.0

        # Material balance (only rooks exist besides kings in this variant)
        if wrState is not None:
            value += 100
        if brState is not None:
            value -= 100

        # King vs King+Rook or Rook+King vs King
        if wrState is not None and brState is None:
            # Push BK to edge/corner
            edge_pull = dist_to_edge_sq((bkState[0], bkState[1]))  # 0..3.5
            value += edge_pull * 12

            # Stronger bonus near corners when on edge
            if bkState[0] in (0, 7) or bkState[1] in (0, 7):
                corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
                min_corner = min(chebyshev((bkState[0], bkState[1]), c) for c in corners)
                value += (3 - min_corner) * 5

            # King opposition (closer WK to BK is generally better)
            kk = chebyshev((bkState[0], bkState[1]), (wkState[0], wkState[1]))
            value += (7 - kk) * 2

            # Rook alignment and safety
            if wrState is not None:
                if wrState[0] == bkState[0] or wrState[1] == bkState[1]:
                    value += 6  # cutting ranks/files
                # Keep rook at safe distance from BK
                rb = chebyshev((wrState[0], wrState[1]), (bkState[0], bkState[1]))
                if rb <= 1:
                    value -= 12
                elif rb >= 3:
                    value += 3
                # Prefer rook protected by king
                rw = chebyshev((wrState[0], wrState[1]), (wkState[0], wkState[1]))
                if rw <= 1:
                    value += 2

        elif wrState is None and brState is not None:
            # Symmetric when White lacks rook and Black has it
            edge_push_wk = dist_to_edge_sq((wkState[0], wkState[1]))
            value -= edge_push_wk * 12

            if wkState[0] in (0, 7) or wkState[1] in (0, 7):
                corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
                min_corner = min(chebyshev((wkState[0], wkState[1]), c) for c in corners)
                value -= (3 - min_corner) * 5

            kk = chebyshev((bkState[0], bkState[1]), (wkState[0], wkState[1]))
            value -= (7 - kk) * 2

            if brState is not None:
                if brState[0] == wkState[0] or brState[1] == wkState[1]:
                    value -= 6
                rw = chebyshev((brState[0], brState[1]), (wkState[0], wkState[1]))
                if rw <= 1:
                    value += 12  # good for White if enemy rook blunders near WK
                elif rw >= 3:
                    value -= 3
                rb = chebyshev((brState[0], brState[1]), (bkState[0], bkState[1]))
                if rb <= 1:
                    value += 2  # enemy rook near its king is slightly worse for pressure

        else:
            # Both rooks on board (KR vs KR) tends to be drawish.
            # Keep evaluation near 0 with small positional nudges.
            # Prefer rook safety (not adjacent to enemy king), and central kings.
            if wrState is not None:
                rb = chebyshev((wrState[0], wrState[1]), (bkState[0], bkState[1]))
                if rb <= 1:
                    value -= 6
            if brState is not None:
                rw = chebyshev((brState[0], brState[1]), (wkState[0], wkState[1]))
                if rw <= 1:
                    value += 6
            value += (3.5 - dist_to_edge_sq((wkState[0], wkState[1])))  # centralize WK
            value -= (3.5 - dist_to_edge_sq((bkState[0], bkState[1])))  # push BK away from center

        # Tactical bonuses/penalties
        if self.isWatchedBk(currentState):
            value += 20
        if self.isWatchedWk(currentState):
            value -= 20

        # Perspective
        if not color:
            value *= -1

        return value

    def isKingsOnlyDraw(self, currentState):
        """Return True if only both kings remain on the board (K vs K)."""
        wkState = self.getPieceState(currentState, 6)
        bkState = self.getPieceState(currentState, 12)
        wrState = self.getPieceState(currentState, 2)
        brState = self.getPieceState(currentState, 8)
        return wkState is not None and bkState is not None and wrState is None and brState is None
    
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
        # Softmax-weighted expected value (soft-max, not soft-min); numerically stable
        if not values:
            return 0.0
        if all(v == values[0] for v in values):
            return float(values[0])

        m = max(values)
        weights = [math.exp(v - m) for v in values]
        total_w = sum(weights)
        return sum(w * v for w, v in zip(weights, values)) / total_w

    def minimaxGame(self, depthWhite,depthBlack):
        
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
    aichess.minimaxGame(4,4)
    # Add code to save results and continue with other exercises
