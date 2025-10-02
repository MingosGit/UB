import piece
import numpy as np

class Board():

    def __init__(self, initState, xinit=True):

        self.listNames = ['P', 'R', 'H', 'B', 'Q', 'K', 'P', 'R', 'H', 'B', 'Q', 'K']
        self.listSuccessorStates = []
        self.listNextStates = []
        self.board = []
        self.currentStateW = []
        self.currentStateB = []
        self.listVisitedStates = []

        for i in range(8):
            self.board.append([None] * 8)

        if xinit:
            self.board[7][0] = piece.Rook(True)
            self.board[7][1] = piece.Knight(True)
            self.board[7][2] = piece.Bishop(True)
            self.board[7][3] = piece.Queen(True)
            self.board[7][4] = piece.King(True)
            self.board[7][5] = piece.Bishop(True)
            self.board[7][6] = piece.Knight(True)
            self.board[7][7] = piece.Rook(True)
            for i in range(8):
                self.board[6][i] = piece.Pawn(True)
            self.board[0][0] = piece.Rook(False)
            self.board[0][1] = piece.Knight(False)
            self.board[0][2] = piece.Bishop(False)
            self.board[0][3] = piece.Queen(False)
            self.board[0][4] = piece.King(False)
            self.board[0][5] = piece.Bishop(False)
            self.board[0][6] = piece.Knight(False)
            self.board[0][7] = piece.Rook(False)
            for i in range(8):
                self.board[1][i] = piece.Pawn(False)
        else:
            self.currentState = initState
            for i in range(8):
                for j in range(8):
                    if initState[i][j] == 1:
                        self.board[i][j] = piece.Pawn(True)
                    elif initState[i][j] == 2:
                        self.board[i][j] = piece.Rook(True)
                    elif initState[i][j] == 3:
                        self.board[i][j] = piece.Knight(True)
                    elif initState[i][j] == 4:
                        self.board[i][j] = piece.Bishop(True)
                    elif initState[i][j] == 5:
                        self.board[i][j] = piece.Queen(True)
                    elif initState[i][j] == 6:
                        self.board[i][j] = piece.King(True)
                    elif initState[i][j] == 7:
                        self.board[i][j] = piece.Pawn(False)
                    elif initState[i][j] == 8:
                        self.board[i][j] = piece.Rook(False)
                    elif initState[i][j] == 9:
                        self.board[i][j] = piece.Knight(False)
                    elif initState[i][j] == 10:
                        self.board[i][j] = piece.Bishop(False)
                    elif initState[i][j] == 11:
                        self.board[i][j] = piece.Queen(False)
                    elif initState[i][j] == 12:
                        self.board[i][j] = piece.King(False)
                    if 0 < initState[i][j] < 7:
                        self.currentStateW.append([i, j, int(initState[i][j])])
                    if 6 < initState[i][j] < 13:
                        self.currentStateB.append([i, j, int(initState[i][j])])

    def isSameState(self, a, b):
        isSameState1 = all(x in b for x in a)
        isSameState2 = all(x in a for x in b)
        return isSameState1 and isSameState2

    def getListNextStatesW(self, mypieces):
        self.listNextStates = []
        for j in range(len(mypieces)):
            self.listSuccessorStates = []
            mypiece = mypieces[j]
            listOtherPieces = mypieces.copy()
            listOtherPieces.remove(mypiece)
            listPotentialNextStates = []

            if str(self.board[mypiece[0]][mypiece[1]]) == 'K':
                listPotentialNextStates = [[mypiece[0] + 1, mypiece[1], 6],
                                           [mypiece[0] + 1, mypiece[1] - 1, 6],
                                           [mypiece[0], mypiece[1] - 1, 6],
                                           [mypiece[0] - 1, mypiece[1] - 1, 6],
                                           [mypiece[0] - 1, mypiece[1], 6],
                                           [mypiece[0] - 1, mypiece[1] + 1, 6],
                                           [mypiece[0], mypiece[1] + 1, 6],
                                           [mypiece[0] + 1, mypiece[1] + 1, 6]]
                for k in range(len(listPotentialNextStates)):
                    aa = listPotentialNextStates[k]
                    if 0 <= aa[0] < 8 and 0 <= aa[1] < 8 and aa not in listOtherPieces:
                        if self.board[aa[0]][aa[1]] is None or not self.board[aa[0]][aa[1]].color:
                            self.listSuccessorStates.append([aa[0], aa[1], aa[2]])

            elif str(self.board[mypiece[0]][mypiece[1]]) == 'P':
                listPotentialNextStates = [[mypiece[0], mypiece[1], 1], [mypiece[0] + 1, mypiece[1], 1]]
                for k in range(len(listPotentialNextStates)):
                    aa = listPotentialNextStates[k]
                    if 0 <= aa[0] < 8 and 0 <= aa[1] < 8 and aa not in listOtherPieces:
                        if self.board[aa[0]][aa[1]] is None:
                            self.listSuccessorStates.append([aa[0], aa[1], aa[2]])

            elif str(self.board[mypiece[0]][mypiece[1]]) == 'R':
                listPotentialNextStates = []
                ix, iy = mypiece[0], mypiece[1]
                while ix > 0:
                    ix -= 1
                    if self.board[ix][iy] is not None:
                        if not self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 2])
                        break
                    else:
                        listPotentialNextStates.append([ix, iy, 2])
                ix, iy = mypiece[0], mypiece[1]
                while ix < 7:
                    ix += 1
                    if self.board[ix][iy] is not None:
                        if not self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 2])
                        break
                    else:
                        listPotentialNextStates.append([ix, iy, 2])
                ix, iy = mypiece[0], mypiece[1]
                while iy > 0:
                    iy -= 1
                    if self.board[ix][iy] is not None:
                        if not self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 2])
                        break
                    else:
                        listPotentialNextStates.append([ix, iy, 2])
                ix, iy = mypiece[0], mypiece[1]
                while iy < 7:
                    iy += 1
                    if self.board[ix][iy] is not None:
                        if not self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 2])
                        break
                    else:
                        listPotentialNextStates.append([ix, iy, 2])
                for k in range(len(listPotentialNextStates)):
                    if listPotentialNextStates[k] not in listOtherPieces:
                        self.listSuccessorStates.append(listPotentialNextStates[k])

            elif str(self.board[mypiece[0]][mypiece[1]]) == 'H':
                listPotentialNextStates = []
                ix, iy = mypiece[0], mypiece[1]
                nextS = [[ix + 1, iy + 2, 3], [ix + 2, iy + 1, 3], [ix + 1, iy - 2, 3], [ix + 2, iy - 1, 3],
                         [ix - 2, iy - 1, 3], [ix - 1, iy - 2, 3], [ix - 1, iy + 2, 3], [ix - 2, iy + 1, 3]]
                for ns in nextS:
                    if 0 <= ns[0] < 8 and 0 <= ns[1] < 8:
                        listPotentialNextStates.append(ns)
                for k in range(len(listPotentialNextStates)):
                    if listPotentialNextStates[k] not in listOtherPieces:
                        self.listSuccessorStates.append(listPotentialNextStates[k])

            elif str(self.board[mypiece[0]][mypiece[1]]) == 'B':
                listPotentialNextStates = []
                ix, iy = mypiece[0], mypiece[1]
                dx, dy = [-1, 1, -1, 1], [-1, 1, 1, -1]
                for d in range(4):
                    nx, ny = ix, iy
                    while 0 <= nx + dx[d] < 8 and 0 <= ny + dy[d] < 8:
                        nx += dx[d]
                        ny += dy[d]
                        if self.board[nx][ny] is not None:
                            listPotentialNextStates.append([nx, ny, 4])
                            break
                        else:
                            listPotentialNextStates.append([nx, ny, 4])
                self.listSuccessorStates = listPotentialNextStates

            elif str(self.board[mypiece[0]][mypiece[1]]) == 'Q':
                listPotentialNextStates = []
                ix, iy = mypiece[0], mypiece[1]
                dx, dy = [-1, 1, -1, 1, -1, 1, 0, 0, 0, 0], [-1, 1, 1, -1, 0, 0, -1, 1, -1, 1]
                for d in range(len(dx)):
                    nx, ny = ix, iy
                    while 0 <= nx + dx[d] < 8 and 0 <= ny + dy[d] < 8:
                        nx += dx[d]
                        ny += dy[d]
                        if self.board[nx][ny] is not None:
                            listPotentialNextStates.append([nx, ny, 5])
                            break
                        else:
                            listPotentialNextStates.append([nx, ny, 5])
                for k in range(len(listPotentialNextStates)):
                    if listPotentialNextStates[k] not in listOtherPieces:
                        self.listSuccessorStates.append(listPotentialNextStates[k])

            for k in range(len(self.listSuccessorStates)):
                self.listNextStates.append([self.listSuccessorStates[k]] + listOtherPieces)

    def getListNextStatesB(self, mypieces):
        self.listNextStates = []
        for j in range(len(mypieces)):
            self.listSuccessorStates = []
            mypiece = mypieces[j]
            listOtherPieces = mypieces.copy()
            listOtherPieces.remove(mypiece)
            listPotentialNextStates = []

            if self.board[mypiece[0]][mypiece[1]].name == 'K':
                listPotentialNextStates = [[mypiece[0] + 1, mypiece[1], 12],
                                           [mypiece[0] + 1, mypiece[1] - 1, 12],
                                           [mypiece[0], mypiece[1] - 1, 12],
                                           [mypiece[0] - 1, mypiece[1] - 1, 12],
                                           [mypiece[0] - 1, mypiece[1], 12],
                                           [mypiece[0] - 1, mypiece[1] + 1, 12],
                                           [mypiece[0], mypiece[1] + 1, 12],
                                           [mypiece[0] + 1, mypiece[1] + 1, 12]]
                for k in range(len(listPotentialNextStates)):
                    aa = listPotentialNextStates[k]
                    if 0 <= aa[0] < 8 and 0 <= aa[1] < 8 and aa not in listOtherPieces:
                        if self.board[aa[0]][aa[1]] is None or self.board[aa[0]][aa[1]].color:
                            self.listSuccessorStates.append([aa[0], aa[1], aa[2]])

            elif self.board[mypiece[0]][mypiece[1]].name == 'P':
                listPotentialNextStates = [[mypiece[0], mypiece[1], 7], [mypiece[0] + 1, mypiece[1], 7]]
                for k in range(len(listPotentialNextStates)):
                    aa = listPotentialNextStates[k]
                    if 0 <= aa[0] < 8 and 0 <= aa[1] < 8 and aa not in listOtherPieces:
                        if self.board[aa[0]][aa[1]] is None:
                            self.listSuccessorStates.append([aa[0], aa[1], aa[2]])

            elif self.board[mypiece[0]][mypiece[1]].name == 'R':
                listPotentialNextStates = []
                ix, iy = mypiece[0], mypiece[1]
                while ix > 0:
                    ix -= 1
                    if self.board[ix][iy] is not None:
                        if self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 8])
                        break
                    else:
                        listPotentialNextStates.append([ix, iy, 8])
                ix, iy = mypiece[0], mypiece[1]
                while ix < 7:
                    ix += 1
                    if self.board[ix][iy] is not None:
                        if self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 8])
                        break
                    else:
                        listPotentialNextStates.append([ix, iy, 8])
                ix, iy = mypiece[0], mypiece[1]
                while iy > 0:
                    iy -= 1
                    if self.board[ix][iy] is not None:
                        if self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 8])
                        break
                    else:
                        listPotentialNextStates.append([ix, iy, 8])
                ix, iy = mypiece[0], mypiece[1]
                while iy < 7:
                    iy += 1
                    if self.board[ix][iy] is not None:
                        if self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 8])
                        break
                    else:
                        listPotentialNextStates.append([ix, iy, 8])
                for k in range(len(listPotentialNextStates)):
                    if listPotentialNextStates[k] not in listOtherPieces:
                        self.listSuccessorStates.append(listPotentialNextStates[k])

            elif self.board[mypiece[0]][mypiece[1]].name == 'H':
                listPotentialNextStates = []
                ix, iy = mypiece[0], mypiece[1]
                nextS = [[ix + 1, iy + 2, 9], [ix + 2, iy + 1, 9], [ix + 1, iy - 2, 9], [ix + 2, iy - 1, 9],
                         [ix - 2, iy - 1, 9], [ix - 1, iy - 2, 9], [ix - 1, iy + 2, 9], [ix - 2, iy + 1, 9]]
                for ns in nextS:
                    if 0 <= ns[0] < 8 and 0 <= ns[1] < 8:
                        listPotentialNextStates.append(ns)
                for k in range(len(listPotentialNextStates)):
                    if listPotentialNextStates[k] not in listOtherPieces:
                        self.listSuccessorStates.append(listPotentialNextStates[k])

            elif self.board[mypiece[0]][mypiece[1]].name == 'B':
                listPotentialNextStates = []
                ix, iy = mypiece[0], mypiece[1]
                dx, dy = [-1, 1, -1, 1], [-1, 1, 1, -1]
                for d in range(4):
                    nx, ny = ix, iy
                    while 0 <= nx + dx[d] < 8 and 0 <= ny + dy[d] < 8:
                        nx += dx[d]
                        ny += dy[d]
                        if self.board[nx][ny] is not None:
                            listPotentialNextStates.append([nx, ny, 10])
                            break
                        else:
                            listPotentialNextStates.append([nx, ny, 10])
                self.listSuccessorStates = listPotentialNextStates

            elif self.board[mypiece[0]][mypiece[1]].name == 'Q':
                listPotentialNextStates = []
                ix, iy = mypiece[0], mypiece[1]
                dx, dy = [-1, 1, -1, 1, -1, 1, 0, 0, 0, 0], [-1, 1, 1, -1, 0, 0, -1, 1, -1, 1]
                for d in range(len(dx)):
                    nx, ny = ix, iy
                    while 0 <= nx + dx[d] < 8 and 0 <= ny + dy[d] < 8:
                        nx += dx[d]
                        ny += dy[d]
                        if self.board[nx][ny] is not None:
                            listPotentialNextStates.append([nx, ny, 11])
                            break
                        else:
                            listPotentialNextStates.append([nx, ny, 11])
                for k in range(len(listPotentialNextStates)):
                    if listPotentialNextStates[k] not in listOtherPieces:
                        self.listSuccessorStates.append(listPotentialNextStates[k])

            for k in range(len(self.listSuccessorStates)):
                self.listNextStates.append([self.listSuccessorStates[k]] + listOtherPieces)

    def print_board(self):
        buffer = ""
        for i in range(33):
            buffer += "*"
        print(buffer)
        for i in range(len(self.board)):
            tmp_str = "|"
            for j in self.board[i]:
                if j is None or j.name == 'GP':
                    tmp_str += "   |"
                elif len(j.name) == 2:
                    tmp_str += (" " + str(j) + "|")
                else:
                    tmp_str += (" " + str(j) + " |")
            print(tmp_str)
        buffer = ""
        for i in range(33):
            buffer += "*"
        print(buffer)
