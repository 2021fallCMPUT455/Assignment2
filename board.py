"""
board.py

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
import operator
from board_util import (GoBoardUtil, BLACK, WHITE, EMPTY, BORDER, PASS,
                        is_black_white, is_black_white_empty, coord_to_point,
                        where1d, MAXSIZE, GO_POINT)
"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See GoBoardUtil.coord_to_point for explanations of the array encoding.
"""





class GoBoard(object):
    def __init__(self, size):
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)
        self.calculate_rows_cols_diags()
        self.tested_stone_list = []

    def calculate_rows_cols_diags(self):
        if self.size < 5:
            return
        # precalculate all rows, cols, and diags for 5-in-a-row detection
        self.rows = []
        self.cols = []
        for i in range(1, self.size + 1):
            current_row = []
            start = self.row_start(i)
            for pt in range(start, start + self.size):
                current_row.append(pt)
            self.rows.append(current_row)

            start = self.row_start(1) + i - 1
            current_col = []
            for pt in range(start, self.row_start(self.size) + i, self.NS):
                current_col.append(pt)
            self.cols.append(current_col)

        self.diags = []
        # diag towards SE, starting from first row (1,1) moving right to (1,n)
        start = self.row_start(1)
        for i in range(start, start + self.size):
            diag_SE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)
        # diag towards SE and NE, starting from (2,1) downwards to (n,1)
        for i in range(start + self.NS,
                       self.row_start(self.size) + 1, self.NS):
            diag_SE = []
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)
            if len(diag_NE) >= 5:
                self.diags.append(diag_NE)
        # diag towards NE, starting from (n,2) moving right to (n,n)
        start = self.row_start(self.size) + 1
        for i in range(start, start + self.size):
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_NE) >= 5:
                self.diags.append(diag_NE)
        assert len(self.rows) == self.size
        assert len(self.cols) == self.size
        assert len(self.diags) == (2 * (self.size - 5) + 1) * 2

    def reset(self, size):
        """
        Creates a start state, an empty board with given size.
        """
        self.size = size
        self.NS = size + 1
        self.WE = 1
        self.ko_recapture = None
        self.last_move = None
        self.last2_move = None
        self.current_player = BLACK
        self.maxpoint = size * size + 3 * (size + 1)
        self.board = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.calculate_rows_cols_diags()

    def copy(self):
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.ko_recapture = self.ko_recapture
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        return b

    def get_color(self, point):
        return self.board[point]

    def pt(self, row, col):
        return coord_to_point(row, col, self.size)

    def is_legal(self, point, color):
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        board_copy = self.copy()
        can_play_move = board_copy.play_move(point, color)
        return can_play_move

    def get_empty_points(self):
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def get_color_points(self, color):
        """
        Return:
            All points of color on the board
        """
        return where1d(self.board == color)

    def row_start(self, row):
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board):
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start = self.row_start(row)
            board[start:start + self.size] = EMPTY

    def is_eye(self, point, color):
        """
        Check if point is a simple eye for color
        """
        if not self._is_surrounded(point, color):
            return False
        # Eye-like shape. Check diagonals to detect false eye
        opp_color = GoBoardUtil.opponent(color)
        false_count = 0
        at_edge = 0
        for d in self._diag_neighbors(point):
            if self.board[d] == BORDER:
                at_edge = 1
            elif self.board[d] == opp_color:
                false_count += 1
        return false_count <= 1 - at_edge  # 0 at edge, 1 in center

    def _is_surrounded(self, point, color):
        """
        check whether empty point is surrounded by stones of color
        (or BORDER) neighbors
        """
        for nb in self._neighbors(point):
            nb_color = self.board[nb]
            if nb_color != BORDER and nb_color != color:
                return False
        return True

    def _has_liberty(self, block):
        """
        Check if the given block has any liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            empty_nbs = self.neighbors_of_color(stone, EMPTY)
            if empty_nbs:
                return True
        return False

    def _block_of(self, stone):
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block 
        """
        color = self.get_color(stone)
        assert is_black_white(color)
        return self.connected_component(stone)

    def connected_component(self, point):
        """
        Find the connected component of the given point.
        """
        marker = np.full(self.maxpoint, False, dtype=bool)
        pointstack = [point]
        color = self.get_color(point)
        assert is_black_white_empty(color)
        marker[point] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def _detect_and_process_capture(self, nb_point):
        """
        Check whether opponent block on nb_point is captured.
        If yes, remove the stones.
        Returns the stone if only a single stone was captured,
        and returns None otherwise.
        This result is used in play_move to check for possible ko
        """
        single_capture = None
        opp_block = self._block_of(nb_point)
        if not self._has_liberty(opp_block):
            captures = list(where1d(opp_block))
            self.board[captures] = EMPTY
            if len(captures) == 1:
                single_capture = nb_point
        return single_capture

    def play_move(self, point, color):
        """
        Play a move of color on point
        Returns boolean: whether move was legal
        """
        assert is_black_white(color)
        # Special cases
        if point == PASS:
            self.ko_recapture = None
            self.current_player = GoBoardUtil.opponent(color)
            self.last2_move = self.last_move
            self.last_move = point
            return True
        elif self.board[point] != EMPTY:
            return False
        # if point == self.ko_recapture:
        #     return False

        # General case: deal with captures, suicide, and next ko point
        # opp_color = GoBoardUtil.opponent(color)
        # in_enemy_eye = self._is_surrounded(point, opp_color)
        self.board[point] = color
        # single_captures = []
        # neighbors = self._neighbors(point)
        # for nb in neighbors:
        #     if self.board[nb] == opp_color:
        #         single_capture = self._detect_and_process_capture(nb)
        #         if single_capture != None:
        #             single_captures.append(single_capture)
        # block = self._block_of(point)
        # if not self._has_liberty(block):  # undo suicide move
        #     self.board[point] = EMPTY
        #     return False
        # self.ko_recapture = None
        # if in_enemy_eye and len(single_captures) == 1:
        #     self.ko_recapture = single_captures[0]
        self.current_player = GoBoardUtil.opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        return True

    def neighbors_of_color(self, point, color):
        """ List of neighbors of point of given color """
        nbc = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point):
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point):
        """ List of all four diagonal neighbors of point """
        return [
            point - self.NS - 1,
            point - self.NS + 1,
            point + self.NS - 1,
            point + self.NS + 1,
        ]

    def last_board_moves(self):
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not None, not PASS).
        """
        board_moves = []
        if self.last_move != None and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != None and self.last2_move != PASS:
            board_moves.append(self.last2_move)
            return

    def detect_five_in_a_row(self):
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        """
        for r in self.rows:
            result = self.has_five_in_list(r)
            if result != EMPTY:
                return result
        for c in self.cols:
            result = self.has_five_in_list(c)
            if result != EMPTY:
                return result
        for d in self.diags:
            result = self.has_five_in_list(d)
            if result != EMPTY:
                return result
        return EMPTY

    def has_five_in_list(self, list):
        """
        Returns BLACK or WHITE if any five in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = BORDER
        counter = 1
        for stone in list:
            if self.get_color(stone) == prev:
                counter += 1
            else:
                counter = 1
                prev = self.get_color(stone)
            if counter == 5 and prev != EMPTY:
                return prev
        return EMPTY

    #
    #
    #
    #
    #

    #
    #
    #
    #
    #
    #
    #
    def gomoku_neighbors(self, point):
        return (self._neighbors(point) + self._diag_neighbors(point))

    def gomoku_neighbors_of_color(self, point, color):
        """ List of neighbors of point of given color """
        nbc = []
        for nb in self.gomoku_neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def detect_straight_line_ver(self, point, color):
        """ Detect the straight line in four directions. """
        found_straight_line = False
        y = point % self.NS
        x = point // self.NS

        x_counter = 0

        for x_marker in range(x, self.NS):
            color_stone_line = self.get_color(self.pt(x_marker, y))
            if color_stone_line != color:
                break
            x_counter += 1
        for x_marker in range(x - 1, 0, -1):
            color_stone_line = self.get_color(self.pt(x_marker, y))
            if color_stone_line != color:
                break
            x_counter += 1
        if x_counter >= 5:
            #print('hor')
            print(str(color))
            return True
        else:
            return found_straight_line

    def detect_straight_line_hor(self, point, color):
        found_straight_line = False
        y = point % self.NS
        x = point // self.NS

        y_counter = 0

        for y_marker in range(y, self.NS):
            color_stone_line = self.get_color(self.pt(x, y_marker))
            if color_stone_line != color:
                break
            y_counter += 1
        for y_marker in range(y - 1, 0, -1):
            color_stone_line = self.get_color(self.pt(x, y_marker))
            if color_stone_line != color:
                break
            y_counter += 1
        if y_counter >= 5:
            #print('ver')
            return True
        else:
            return found_straight_line

    def detect_straight_line_left_diag(self, point, color):
        #print(f'\n check 00: {self.board(self.size + 1, self.size + 1)} \n')
        #found_straight_line = False
        y = point % self.NS
        x = point // self.NS

        coord_list = []

        counter = 0

        #print('start')
        #print(f'begin: {(x, y), self.get_color(self.pt(x, y))}')
        for x_marker, y_marker in zip(range(x, self.NS), range(y, self.NS)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line != color:
                #print(f'not black: {(x_marker, y_marker)}')
                break
            #print((x_marker, y_marker))
            counter += 1
            coord_list.append((x_marker, y_marker))

        for x_marker, y_marker in zip(range(x - 1, 0, -1), range(y - 1, 0,
                                                                 -1)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line != color:
                #print(f'not black: {(x_marker, y_marker)}')
                break
            #print((x_marker, y_marker))
            counter += 1
            coord_list.append((x_marker, y_marker))
        #print('end')
        if counter >= 5:
            #print('right')
            return True
        else:
            return False

    def detect_straight_line_right_diag(self, point, color):
        y = point % self.NS
        x = point // self.NS

        counter = 0
        for x_marker, y_marker in zip(range(x, 0, -1), range(y, self.NS)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line != color:
                break
            counter += 1

        for x_marker, y_marker in zip(range(x + 1, self.NS),
                                      range(y - 1, 0, -1)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line != color:
                break
            counter += 1

        if counter >= 5:
            #print('left')
            return True
        else:
            return False

    def working_on_detection(self, stone_list):

        for point in stone_list:
            color = self.get_color(point)
            neighbors_color = self.gomoku_neighbors_of_color(point, color)
            total_stone = (self.board == BLACK).sum() + (self.board
                                                         == WHITE).sum()
            if total_stone >= 5 and len(neighbors_color) != 0:
                if self.detect_straight_line_hor(point, color) == -1:

                    return True
                elif self.detect_straight_line_ver(point, color) == -1:

                    return True
                elif self.detect_straight_line_left_diag(point, color) == -1:

                    return True
                elif self.detect_straight_line_right_diag(point, color) == -1:

                    return True

        return False

    def trigger_detection(self):
        black_stone_list = [point for point in where1d(self.board == BLACK)]
        white_stone_list = [point for point in where1d(self.board == WHITE)]
        empty_point_list = where1d(self.board == EMPTY)
        two_pass_turn = (self.last_move == PASS and self.last2_move == PASS)

        print(black_stone_list)

        if self.working_on_detection(black_stone_list) == True:
            #print('bb')
            return BLACK
        elif self.working_on_detection(white_stone_list) == True:
            #print('ww')
            return WHITE

        elif len(empty_point_list) == 0 or two_pass_turn:
            return DRAW

        else:
            return

    def gomoku_play_move(self, point, color):
        assert is_black_white(color)
        # Special cases
        if point == PASS:
            #self.ko_recapture = None
            self.current_player = GoBoardUtil.opponent(color)
            self.last2_move = self.last_move
            self.last_move = point
            return True
        elif self.board[point] != EMPTY:
            return False

        opp_color = GoBoardUtil.opponent(color)
        #in_enemy_eye = self._is_surrounded(point, opp_color)
        self.board[point] = color
        #single_captures = []
        #neighbors = self._neighbors(point)
        #for nb in neighbors:
        #if self.board[nb] == color:
        #five_stone_line = self.trigger_detection(point)
        #if five_stone_line:
        #pass

        #if single_capture != None:
        #single_captures.append(single_capture)
        #block = self._block_of(point)
        #if not self._has_liberty(block):
        #self.board[point] = EMPTY
        #return False
        #self.ko_recapture = None
        #if in_enemy_eye and len(single_captures) == 1:
        #self.ko_recapture = single_captures[0]
        self.current_player = GoBoardUtil.opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        return True

    def find_legal_move(self):
        empty_point_list = np.argwhere(self.board == EMPTY)
        legal_move = []
        for empty_point in empty_point_list:
            '''
            valid_or_invalid = self.is_legal(
                coord_to_point(empty_point[0], empty_point[1], self.size),
                color)
            if valid_or_invalid:
            '''
            y = empty_point[0] % self.NS
            x = empty_point[0] // self.NS
            new_coord = chr(x + 96) + str(y)
            #print(new_coord)
            legal_move.append(new_coord)
        legal_move.sort(key=lambda x: x[0])
        legal_move = ' '.join([move for move in legal_move])
        return legal_move

    def detect_potential_win(self, point):
        #stone number in the row, stone location
        best_move = {'1': [], '2': [], '3': [], '4': []}

        color = self.get_color(point)
        neighbors_color = self.gomoku_neighbors_of_color(point, color)
        total_stone = (self.board == BLACK).sum() + (self.board == WHITE).sum()
        if total_stone >= 5 and len(neighbors_color) != 0:
            hor_detection_result = self.detect_straight_line_hor(point, color)
            if hor_detection_result == -1:

                return -100
            else:
                best_move[str(hor_detection_result)].append(point)
            ver_detection_result = self.detect_straight_line_ver(point, color)
            if self.detect_straight_line_ver(point, color) == -1:

                return -100
            else:
                best_move[str(ver_detection_result)].append(point)
            left_diag_result = self.detect_straight_line_left_diag(
                point, color)
            if left_diag_result == -1:

                return -100
            else:
                best_move[str(left_diag_result)].append(point)
            right_diag_result = self.detect_straight_line_right_diag(
                point, color)
            if right_diag_result == -1:

                return -100
            else:
                best_move[str(right_diag_result)].append(point)
        return best_move

    def using_tree_detection(self, color):
        # If the current player excutes this function, the program will try to place a stone in the opponent's neighbor.
        # The function will try to find if there is a way to stop the opponent from winning.

        opponent = WHITE + BLACK - color
        opponent_stone_list = [
            point for point in where1d(self.board == opponent)
        ]
        for opponent_stone in opponent_stone_list:
            for neighbor in self.get_neighbors(opponent_stone):
                if self.board[neighbor] == EMPTY:
                    self.tested_stone_list.append(neighbor)
                    self.board[neighbor] = color

    def minimax_or(self, color, current_depth):
        opponent = WHITE + BLACK - color
        current_move = None
        color_point = [point for point in where1d(self.board == color)]
        EMPTY_list = []
        if len(color_point) == 0:
            EMPTY_list = EMPTY_list + where1d(self.board == EMPTY).reshape(
                1, -1).tolist()
        else:
            for point in color_point:
                EMPTY_list = EMPTY_list + self.neighbors_of_color(point, EMPTY)
        if self.detect_five_in_a_row() == color:
            print('color:' + str(color))
            print(self.board)
            return current_move

        if len(EMPTY_list) == 0:
            # This is a draw situation
            return 0

        if current_depth <= self.depth:
            for location in EMPTY_list:
                current_move = location
                self.best_move_for_now[current_depth].append(location)
                self.board[location] = color
                is_win = self.minimax_and(opponent, current_depth)
                self.board[location] = EMPTY
                if is_win > 0:
                    return 1
            current_depth += 1
        return -1

    def minimax_and(self, color, current_depth):
        opponent = WHITE + BLACK - color
        #current_move = None
        color_point = [point for point in where1d(self.board == color)]
        EMPTY_list = []
        if len(color_point) == 0:
            EMPTY_list = EMPTY_list + where1d(self.board == EMPTY).reshape(
                1, -1).tolist()
        else:
            for point in color_point:
                EMPTY_list = EMPTY_list + self.neighbors_of_color(point, EMPTY)
        if self.detect_five_in_a_row() == color:
            print(self.board)
            return 1

        if len(EMPTY_list) == 0:
            # This is a draw situation
            return 0
        current_depth += 1
        if current_depth <= self.depth:
            for location in EMPTY_list:
                #current_move = location
                self.board[location] = color
                or_node_output = self.minimax_or(opponent, current_depth)
                print(or_node_output)
                is_loss = -1 * or_node_output
                self.board[location] = EMPTY
                if is_loss < 0:
                    return -1
            current_depth += 1
        return 1

    def negamax(self, color, current_depth):
        opponent = WHITE + BLACK - color
        color_point = [point for point in where1d(self.board == color)]
        EMPTY_list = []
        if len(color_point) == 0:
            EMPTY_list = EMPTY_list + where1d(self.board == EMPTY).reshape(
                1, -1).tolist()
        else:
            for point in color_point:
                EMPTY_list = EMPTY_list + self.neighbors_of_color(point, EMPTY)
        if self.detect_five_in_a_row() == color:
            return 1
        if current_depth == self.depth:
            return 0
        if current_depth <= self.depth:
            current_depth += 1
            for location in EMPTY_list:
                self.board[location] = color
                is_win = -1 * self.negamax(opponent, current_depth)
                self.board[location] = EMPTY
                if is_win == 1:
                    return 1
        return -1

    def alphabeta(self, color, alpha, beta, current_depth):
        play_dict, opponent_dict = mapping_all_heuristic
        if len(list(play_dict.keys())) == 0:
            player_list = [point for point in where1d(self.board == EMPTY)]
        else:
            player_best_move = max(player_dict, key=lambda k : player_dict[k])
        if len(list(opponent_dict.keys())) == 0:
            opponent_list = [point for point in where1d(self.board == EMPTY)]
        else:
            opponent_best_move = max(opponent_dict, key=lambda k : opponent_dict[k])
        color_point = []
        find_small_list = lambda a, b: min(a, b)
        for i in range(find_small_list):
            if player_list[]
        opponent = WHITE + BLACK - color
        color_point = [point for point in where1d(self.board == color)]
        EMPTY_list = []
        self.winning_move = None
        
        if len(color_point) == 0:
            EMPTY_list = where1d(self.board == EMPTY).reshape(1, -1)[0]
        else:
            for point in color_point:
                EMPTY_list = EMPTY_list + self.neighbors_of_color(point, EMPTY)
        
        if self.detect_five_in_a_row() == color:
            self.winner = color
            return 1

        if current_depth == self.depth or len(
                where1d(self.board == EMPTY).reshape(1, -1)[0]) == 0:
            return 0
        print(where1d(self.board == EMPTY).reshape(1, -1)[0])
        for location in EMPTY_list:
            self.board[location] = color
            print('current_depth: ' + str(current_depth))
            if alpha != beta:
                value = -self.alphabeta(opponent, -beta, -alpha,
                                        current_depth + 1)
                print('right now: ' + str(color) + ' ' + str(location))
                if value == 1:
                    self.winning_move = location
                    print('winning move for color ' + str(color) + ': ' + str(self.winning_move))
                    
                if value > alpha:
                    alpha = value
            self.board[location] = EMPTY
            if value >= beta:
                return beta

        return alpha

    def decide_winner(self):
        if self.winner == 1:
            print('BLACK')
        elif self.winner == 2:
            print('WHITE')

    def build_tree(self):
        self.depth = 0
        current_depth = 1
        self.best_move_for_now = {}
        for i in range(self.depth):
            self.best_move_for_now[i + 1] = []

        #location = self.minimax_or(WHITE, current_depth)
        #location = self.negamax(WHITE, current_depth)
        location = self.alphabeta(color=BLACK,
                                  alpha=-1,
                                  beta=1,
                                  current_depth=0)
        
        if location > 0:
            # The location is a winning move
            self.decide_winner()
            print('The wining move is ' + str(self.winning_move))
            return location
        elif location == 0:
            # This is a draw or the program reachs the time limit.


            print('The best move for now is: ' + str(self.mapping_all_heuristic(BLACK)))
            print('draw')
            
            return self.best_move_for_now

        elif location < 0:
            # There is no wining move for player right now.
            print('losing')
            return 'lose'

    def heuristic_function(self):
        for point in stone_list:
            color = self.get_color(point)
            neighbors_color = self.gomoku_neighbors_of_color(point, color)
            total_stone = (self.board == BLACK).sum() + (self.board
                                                         == WHITE).sum()
            if len(neighbors_color) != 0:
                if self.detect_straight_line_hor(point, color) == -1:

                    return True
                elif self.detect_straight_line_ver(point, color) == -1:

                    return True
                elif self.detect_straight_line_left_diag(point, color) == -1:

                    return True
                elif self.detect_straight_line_right_diag(point, color) == -1:

                    return True

        return False

    def analyze_hor(self, point, color):

        y = point % self.NS
        x = point // self.NS

        y_counter = 0

        right_neighbor = -1
        left_neighbor = -1
        for y_marker in range(y, self.NS):
            color_stone_line = self.get_color(self.pt(x, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                right_neighbor = self.pt(x, y_marker)
                break
            else:
                break
            y_counter += 1
        for y_marker in range(y - 1, 0, -1):
            color_stone_line = self.get_color(self.pt(x, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                left_neighbor = self.pt(x, y_marker)
                break
            else:
                break
            y_counter += 1
        
        return left_neighbor, right_neighbor, y_counter
        
    def analyze_ver(self, point, color):
        y = point % self.NS
        x = point // self.NS

        x_counter = 0
        right_neighbor = -1
        left_neighbor = -1

        for x_marker in range(x, self.NS):
            color_stone_line = self.get_color(self.pt(x_marker, y))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                right_neighbor = self.pt(x_marker, y)
                break
            else:
                break
            x_counter += 1
        for x_marker in range(x - 1, 0, -1):
            color_stone_line = self.get_color(self.pt(x_marker, y))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                left_neighbor = self.pt(x_marker, y)
                break
            else:
                break
            x_counter += 1
        
        return left_neighbor, right_neighbor, x_counter

    def analyze_left_diag(self, point, color):
        y = point % self.NS
        x = point // self.NS

        
        counter = 0
        right_neighbor = -1
        left_neighbor = -1
        
        for x_marker, y_marker in zip(range(x, self.NS), range(y, self.NS)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                right_neighbor = self.pt(x_marker, y_marker)
                break
            else:
                break
            counter += 1
            
        for x_marker, y_marker in zip(range(x - 1, 0, -1), range(y - 1, 0, -1)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                left_neighbor = self.pt(x_marker, y_marker)
                break
            else:
                break
            counter += 1
                 
        
        return left_neighbor, right_neighbor, counter

    def analyze_right_diag(self, point, color):
        y = point % self.NS
        x = point // self.NS

        counter = 0
        right_neighbor = -1
        left_neighbor = -1
        for x_marker, y_marker in zip(range(x, 0, -1), range(y, self.NS)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                right_neighbor = self.pt(x_marker, y_marker)
                break
            else:
                break
            counter += 1
            

        for x_marker, y_marker in zip(range(x + 1, self.NS),
                                      range(y - 1, 0, -1)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                left_neighbor = self.pt(x_marker, y_marker)
                break
            else:
                break
            counter += 1

        
        return left_neighbor, right_neighbor, counter 

    def check_heuristic_dict(self, point, counter, dictionary):
        if point != -1:
            if point not in dictionary.keys():
                dictionary[point] = counter
            elif point in dictionary.keys():
                if counter > dictionary[point]:
                    dictionary[point] = counter

    def mapping_player_heuristic(self, color):
        
        player_stone_list = [point for point in where1d(self.board == color)]
        if len(player_stone_list) == 0:
            player_stone_list = [point for point in where1d(self.board == EMPTY)]

        potential_move_dict = {}
        print(player_stone_list)
        for point in player_stone_list:
            potential_moves = []

            left_neighbor, right_neighbor, counter = self.analyze_hor(point, color)
            
            potential_moves.append([left_neighbor, counter])
            potential_moves.append([right_neighbor, counter])
            left_neighbor, right_neighbor, counter = self.analyze_ver(point, color)
            potential_moves.append([left_neighbor, counter])
            potential_moves.append([right_neighbor, counter])
            left_neighbor, right_neighbor, counter = self.analyze_left_diag(point, color)
            potential_moves.append([left_neighbor, counter])
            potential_moves.append([right_neighbor, counter])
            left_neighbor, right_neighbor, counter = self.analyze_right_diag(point, color)
            potential_moves.append([left_neighbor, counter])
            potential_moves.append([right_neighbor, counter])
            
            for value_pair in potential_moves:
                if value_pair[0] not in potential_move_dict.keys() and value_pair[0] != -1:
                    potential_move_dict[value_pair[0]] = value_pair[1]
                elif value_pair[0] in potential_move_dict.keys():
                    if value_pair[1] > potential_move_dict[value_pair[0]]:
                        potential_move_dict[value_pair[0]] = value_pair[1]

        print(potential_move_dict)
        #return max(potential_move_dict.items(), key=lambda k : k[1])
        return potential_move_dict

    def mapping_all_heuristic(self, color):
        opponent = WHITE + BLACK - color

        player_dict = self.mapping_player_heuristic(color)
        opponent_dict = self.mapping_player_heuristic(opponent)
        #player_best_move = max(player_dict, key=lambda k : player_dict[k])
        #opponent_best_move = max(opponent_dict, key=lambda k : opponent_dict[k])
        
        player_dict = dict(sorted(player_dict.items(), key=lambda item: item[1]))
        opponent_dict = dict(sorted(opponent_dict.items(), key=lambda item: item[1]))
            
        

        return player_dict, opponent_dict
        '''
        if player_dict[player_best_move] < opponent_dict[opponent_best_move]:
            return opponent_best_move, opponent_dict[opponent_best_move]
        else:
            return player_best_move, player_dict[player_best_move]
        '''
        

