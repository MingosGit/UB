#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        self.dictPath = {}
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
        for k in range(len(a)):
            if a[k] not in b:
                isSameState1 = False
        isSameState2 = True
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

    def isCheckMate(self, mystate):
        board_sim = self.chess.boardSim

        # pure attack detector (no side-effects) to avoid calling piece.is_valid_move
        def square_attacked_pure(r, c):
            b = board_sim.board
            # scan all white pieces and check if they attack (r,c)
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
                    # white pawn attacks one up-left or up-right (row-1)
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
                        # same row or same column
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
                        # diagonal
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

        # find black king position
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
            return False

        # if king not in check, not checkmate (use pure detector)
        if not square_attacked_pure(bk_pos[0], bk_pos[1]):
            return False

        # simulate every legal black move (including captures and king moves).
        for i in range(8):
            for j in range(8):
                bp = board_sim.board[i][j]
                if bp is None or bp.color:
                    continue
                # try all destination squares
                for r in range(8):
                    for c in range(8):
                        dest = board_sim.board[r][c]
                        # can't capture own piece
                        if dest is not None and dest.color == False:
                            continue
                        try:
                            if not bp.is_valid_move(board_sim, (i, j), (r, c)):
                                continue
                        except Exception:
                            continue

                        # simulate
                        orig_from = board_sim.board[i][j]
                        orig_to = board_sim.board[r][c]
                        board_sim.board[i][j] = None
                        board_sim.board[r][c] = bp

                        # find black king pos after move
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

                        # revert
                        board_sim.board[i][j] = orig_from
                        board_sim.board[r][c] = orig_to

                        if not attacked_after:
                            return False

        # no legal move removes check -> checkmate
        return True

    def list_black_legal_escapes(self):
        """Diagnostic: return a list of black moves that would result in the king not being in check.
        Each move is ((r_from,c_from),(r_to,c_to))."""
        board_sim = self.chess.boardSim
        escapes = []

        # reuse same pure detector as in isCheckMate
        def square_attacked(r, c):
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

        # find current black king pos
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
            return escapes

        # try all black moves
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

                        # simulate
                        orig_from = board_sim.board[i][j]
                        orig_to = board_sim.board[r][c]
                        board_sim.board[i][j] = None
                        board_sim.board[r][c] = bp

                        # find new black king pos
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
                            attacked_after = square_attacked(new_bk_pos[0], new_bk_pos[1])

                        # revert
                        board_sim.board[i][j] = orig_from
                        board_sim.board[r][c] = orig_to

                        if not attacked_after:
                            escapes.append(((i, j), (r, c)))
        return escapes

    def newBoardSim(self, listStates):
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
        moveList = []
        nodeTo = to
        nodeStart = start
        while(depthTo > depthStart):
            moveList.insert(0,to)
            nodeTo = self.dictPath[str(nodeTo)][0]
            depthTo-=1
        while(depthStart > depthTo):
            ancestreStart = self.dictPath[str(nodeStart)][0]
            self.changeState(nodeStart, ancestreStart)
            nodeStart = ancestreStart
            depthStart -= 1
        moveList.insert(0,nodeTo)
        while nodeStart != nodeTo:
            ancestreStart = self.dictPath[str(nodeStart)][0]
            self.changeState(nodeStart,ancestreStart)
            nodeTo = self.dictPath[str(nodeTo)][0]
            moveList.insert(0,nodeTo)
            nodeStart = ancestreStart
        for i in range(len(moveList)):
            if i < len(moveList) - 1:
                self.changeState(moveList[i],moveList[i+1])

    def reconstructPath(self, state, depth):
        for i in range(depth):
            self.pathToTarget.insert(0, state)
            state = self.dictPath[str(state)][0]
        self.pathToTarget.insert(0, state)

    def h(self, state):
        whiteKingPosition = state[0]
        whiteRookPosition = state[1]
        targetKingPosition = [0, 5]
        kingDistance = abs(whiteKingPosition[0] - targetKingPosition[0]) + abs(whiteKingPosition[1] - targetKingPosition[1])
        rookDistance = abs(whiteRookPosition[0] - targetKingPosition[0]) + abs(whiteRookPosition[1] - targetKingPosition[1])
        return kingDistance + rookDistance

    def changeState(self, start, to):
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
        self.chess.moveSim(start[movedPieceStart], to[movedPieceTo])       

    def DepthFirstSearch(self, currentState, depth):
        self.listVisitedStates.append(currentState)
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True
        if depth + 1 <= self.depthMax:
            for son in self.getListNextStatesW(currentState):
                if not self.isVisited(son):
                    if son[0][2] == currentState[0][2]:
                        movedPieceIndex = 0
                    else:
                        movedPieceIndex = 1
                    self.chess.moveSim(currentState[movedPieceIndex], son[0])
                    if self.DepthFirstSearch(son, depth + 1):
                        self.pathToTarget.insert(0, currentState)
                        return True
                    self.chess.moveSim(son[0], currentState[movedPieceIndex])
        self.listVisitedStates.remove(currentState)

    def worthExploring(self, state, depth):
        if depth > self.depthMax:
            return False
        visited = False
        for perm in list(permutations(state)):
            permStr = str(perm)
            if permStr in list(self.dictVisitedStates.keys()):
                visited = True
                if depth < self.dictVisitedStates[perm]:
                    self.dictVisitedStates[permStr] = depth
                    return True
        if not visited:
            permStr = str(state)
            self.dictVisitedStates[permStr] = depth
            return True

    def DepthFirstSearchOptimized(self, currentState, depth):
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True
        for son in self.getListNextStatesW(currentState):
            if self.worthExploring(son, depth + 1):
                if son[0][2] == currentState[0][2]:
                    movedPieceIndex = 0
                else:
                    movedPieceIndex = 1
                self.chess.moveSim(currentState[movedPieceIndex], son[0])
                if self.DepthFirstSearchOptimized(son, depth + 1):
                    self.pathToTarget.insert(0, currentState)
                    return True
                self.chess.moveSim(son[0], currentState[movedPieceIndex])

    def BreadthFirstSearch(self, currentState, depth):
        BFSQueue = queue.Queue()
        self.dictPath[str(currentState)] = (None, -1)
        depthCurrentState = 0
        BFSQueue.put(currentState)
        self.listVisitedStates.append(currentState)
        while BFSQueue.qsize() > 0:
            node = BFSQueue.get()
            depthNode = self.dictPath[str(node)][1] + 1
            if depthNode > self.depthMax:
                break
            if depthNode > 0:
                self.movePieces(currentState, depthCurrentState, node, depthNode)
            if self.isCheckMate(node):
                self.reconstructPath(node, depthNode)
                break
            for son in self.getListNextStatesW(node):
                if not self.isVisited(son):
                    self.listVisitedStates.append(son)
                    BFSQueue.put(son)
                    self.dictPath[str(son)] = (node, depthNode)
            currentState = node
            depthCurrentState = depthNode

    def AStarSearch(self, currentState):
        import heapq
        # Normalize the start state to (king, rook)
        currentState = self.getWhiteState(currentState)
        # frontier entries are tuples: (f_score, state, g_cost)
        frontier = []
        start_key = str(currentState)
        heapq.heappush(frontier, (self.h(currentState), currentState, 0))

        # cost-so-far (g) and parent map for reconstruction (store normalized states)
        cost_so_far = {start_key: 0}
        self.dictPath[start_key] = (None, 0)

        found = False
        # while expanding nodes, rebuild boardSim from the normalized white node
        # plus the original black pieces to avoid complex move synchronization
        while frontier:
            f, node, g = heapq.heappop(frontier)
            # debug
            print(f"Exploring state: {node}")

            # rebuild boardSim so successor generation is accurate
            try:
                # node is normalized white-state (list of white piece states)
                # combine with the original black state from self.chess.board.currentStateB
                full_state = []
                # white pieces from node
                for w in node:
                    full_state.append(w)
                # black pieces from initial board (unchanged during A* search)
                for b in self.chess.board.currentStateB:
                    full_state.append(b)
                # rebuild simulation board
                self.newBoardSim(full_state)
            except Exception as e:
                print(f"Failed to rebuild boardSim for node {node}: {e}")
                continue

            # goal check
            if self.isCheckMate(node):
                print("Checkmate found!")
                # reconstruct path using dictPath; node is normalized
                self.reconstructPath(node, g)
                # diagnostic: check for any legal black escapes in this boardSim
                escapes = self.list_black_legal_escapes()
                if len(escapes) == 0:
                    print("No legal black escapes found by diagnostic (confirmed mate)")
                else:
                    print("Diagnostic found black escape moves (not mate):")
                    for mv in escapes:
                        print(mv)
                found = True
                break

            # expand successors; ensure node passed to successor generator is in normalized form
            for son in self.getListNextStatesW(node):
                # normalize successor to canonical (king, rook)
                son_norm = self.getWhiteState(son)
                son_key = str(son_norm)
                tentative_g = g + 1
                # if we already have a better path to son, skip
                if son_key in cost_so_far and tentative_g >= cost_so_far[son_key]:
                    continue
                cost_so_far[son_key] = tentative_g
                # parent pointer and depth (store parent as normalized state)
                self.dictPath[son_key] = (node, tentative_g)
                heapq.heappush(frontier, (tentative_g + self.h(son_norm), son_norm, tentative_g))

        # diagnostic output
        if found:
            print("Minimal depth to checkmate:", len(self.pathToTarget) - 1)
        else:
            print("No checkmate found within search limits.")
if __name__ == "__main__":
    TA = np.zeros((8, 8))
    TA[7][0] = 2  
    TA[7][5] = 6   
    TA[0][4] = 12  
    print("Starting AI chess...")
    aichess = Aichess(TA, True)
    print("Printing board:")
    aichess.chess.boardSim.print_board()
    # Normalizar el estado blanco: asegurarse rey en índice 0, torre en índice 1
    currentState = aichess.getWhiteState(aichess.getCurrentState())
    print("Current State:", currentState, "\n")
    aichess.AStarSearch(currentState)
    print("#A* move sequence:", aichess.pathToTarget)
    print("A* End\n")
    print("Printing final board after A*:")
    aichess.chess.boardSim.print_board()
