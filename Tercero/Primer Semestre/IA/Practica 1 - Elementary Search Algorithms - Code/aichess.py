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
        listCheckMateStates = [[[0,0,2],[2,4,6]],[[0,1,2],[2,4,6]],[[0,2,2],[2,4,6]],[[0,6,2],[2,4,6]],[[0,7,2],[2,4,6]]]
        for permState in list(permutations(mystate)):
            if list(permState) in listCheckMateStates:
                return True
        return False   

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
        frontera = []
        heapq.heappush(frontera, (self.h(currentState), currentState, 0))
        visited = set()
        visited.add(str(currentState))
        while frontera:
            f, node, g = heapq.heappop(frontera)
            print(f"Exploring state: {node}")
            if self.isCheckMate(node):
                print("Checkmate found!")
                self.pathToTarget.append(node)
                self.reconstructPath(node, g)
                break
            for son in self.getListNextStatesW(node):
                if str(son) not in visited:
                    visited.add(str(son))
                    heapq.heappush(frontera, (g + 1 + self.h(son), son, g + 1))
        print("Visited states:", visited)
        print("Minimal depth to checkmate:", len(self.pathToTarget) - 1)

if __name__ == "__main__":
    TA = np.zeros((8, 8))
    TA[7][0] = 2  
    TA[7][5] = 6   
    TA[0][5] = 12  
    print("Starting AI chess...")
    aichess = Aichess(TA, True)
    print("Printing board:")
    aichess.chess.boardSim.print_board()
    currentState = aichess.chess.board.currentStateW.copy()
    print("Current State:", currentState, "\n")
    aichess.AStarSearch(currentState)
    print("#A* move sequence:", aichess.pathToTarget)
    print("A* End\n")
    print("Printing final board after A*:")
    aichess.chess.boardSim.print_board()
