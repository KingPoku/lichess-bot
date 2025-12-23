"""
Smart Chess Bot using Minimax Algorithm with Alpha-Beta Pruning

File: minmax_bot.py
Author: Aduse-Poku Kingsford
Course: Artificial Intelligence

OVERVIEW
--------
This chess engine implements the Minimax algorithm enhanced with:

1. Alpha-Beta Pruning
   - Avoids searching branches that cannot influence the final decision
   - Dramatically reduces the number of explored positions

2. Move Ordering
   - Searches promising moves first (captures, checks, promotions)
   - Improves pruning effectiveness

3. Transposition Table
   - Caches previously evaluated positions
   - Prevents re-evaluating the same board state multiple times

4. Perspective-Correct Evaluation
   - Evaluation is always returned from the point of view of the side to move
   - Prevents incorrect decisions for Black

The engine is designed to integrate with the lichess-bot framework
using the MinimalEngine interface.
"""

import chess
from chess.engine import PlayResult
import logging

import json
import os


from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import HOMEMADE_ARGS_TYPE

# Logger used for debugging and performance statistics
logger = logging.getLogger(__name__)


class SmartBot(MinimalEngine):
    """
    Chess engine based on Minimax with Alpha-Beta pruning.

    The engine assumes:
    - Both players play optimally
    - White tries to maximize the evaluation score
    - Black tries to minimize the evaluation score
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the engine.

        Attributes:
        ----------
        max_depth : int
            Maximum depth of the Minimax search tree.
            (4 plies = 2 full moves)

        nodes_searched : int
            Counter to track how many positions were evaluated.

        transposition_table : dict
            Cache mapping (position, depth, player) to evaluation score.
            Used to avoid recomputing identical positions.
        """
        super().__init__(*args, **kwargs)

        self.max_depth = 4
        self.nodes_searched = 0
        self.transposition_table = {}

        # Default evaluation weights
        self.weights = {
            "material": 1.0,
            "position": 1.0,
            "mobility": 0.1
        }

        # Try to load trained weights if available
        weights_path = os.path.join("training", "trained_weights.json")
        if os.path.exists(weights_path):
            try:
                with open(weights_path, "r", encoding="utf-8") as f:
                    trained = json.load(f)
                    self.weights.update(trained)
                logger.info("Loaded trained evaluation weights.")
            except Exception as e:
                logger.warning(f"Failed to load trained weights: {e}")
        else:
            logger.info("No trained weights found, using defaults.")


            logger.info(f"Using evaluation weights: {self.weights}")



            


    # ------------------------------------------------------------
    # ENTRY POINT (called by lichess-bot)
    # ------------------------------------------------------------

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """
        Main search function called by the lichess-bot framework.

        This function:
        1. Iterates over all legal moves
        2. Uses Minimax + Alpha-Beta to evaluate each move
        3. Selects the best move found

        Parameters:
        -----------
        board : chess.Board
            Current board position.

        Returns:
        --------
        PlayResult
            Object containing the chosen move.
        """
        self.nodes_searched = 0
        self.transposition_table.clear()

        best_move = None
        best_value = float("-inf")

        # Alpha and beta values for pruning
        alpha = float("-inf")
        beta = float("inf")

        # Try all legal moves from the current position
        for move in self.order_moves(board, self.max_depth):
            board.push(move)

            # After our move, opponent plays next -> minimizing
            value = self.minimax_ab(
                board,
                self.max_depth - 1,
                alpha,
                beta,
                maximizing=False
            )

            board.pop()

            # Keep track of the best move found
            if value > best_value:
                best_value = value
                best_move = move

            # Update alpha for pruning
            alpha = max(alpha, value)

        logger.info(
            f"Searched {self.nodes_searched} nodes, "
            f"best evaluation: {best_value:.2f}"
        )

        return PlayResult(best_move, None)

    # ------------------------------------------------------------
    # MINIMAX WITH ALPHA-BETA PRUNING
    # ------------------------------------------------------------

    def minimax_ab(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool
    ) -> float:
        """
        Minimax algorithm with Alpha-Beta pruning.

        Parameters:
        -----------
        board : chess.Board
            Current position.

        depth : int
            Remaining search depth.

        alpha : float
            Best value that the maximizing player can guarantee.

        beta : float
            Best value that the minimizing player can guarantee.

        maximizing : bool
            True if the current player is maximizing,
            False if minimizing.

        Returns:
        --------
        float
            Evaluation score for the position.
        """
        self.nodes_searched += 1

        # --- TRANSPOSITION TABLE LOOKUP ---
        # Same position + depth + role => same result
        key = (board.fen(), depth, maximizing)
        if key in self.transposition_table:
            return self.transposition_table[key]

        # --- BASE CASE ---
        if depth == 0 or board.is_game_over():
            value = self.evaluate(board)
            self.transposition_table[key] = value
            return value

        # --- MAXIMIZING PLAYER ---
        if maximizing:
            max_eval = float("-inf")

            for move in self.order_moves(board, depth):
                board.push(move)
                eval_score = self.minimax_ab(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=False
                )
                board.pop()

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                # Beta cutoff (pruning)
                if beta <= alpha:
                    break

            self.transposition_table[key] = max_eval
            return max_eval

        # --- MINIMIZING PLAYER ---
        else:
            min_eval = float("inf")

            for move in self.order_moves(board, depth):
                board.push(move)
                eval_score = self.minimax_ab(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=True
                )
                board.pop()

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                # Alpha cutoff (pruning)
                if beta <= alpha:
                    break

            self.transposition_table[key] = min_eval
            return min_eval

    # ------------------------------------------------------------
    # MOVE ORDERING
    # ------------------------------------------------------------

    def order_moves(self, board: chess.Board, depth: int = None) -> list:
        """
        Order legal moves to improve Alpha-Beta pruning efficiency.

        Move priority:
        1. Captures (MVV-LVA heuristic)
        2. Checks
        3. Promotions
        4. Other moves

        At very low depth, ordering is skipped for speed.
        """
        if depth is not None and depth <= 1:
            return list(board.legal_moves)

        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

        def move_priority(move: chess.Move) -> float:
            score = 0

            # Captures (MVV-LVA)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += (
                        piece_values[victim.piece_type] * 10
                        - piece_values[attacker.piece_type]
                    )

            # Checks
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()

            # Promotions
            if move.promotion:
                score += 100

            return score

        return sorted(board.legal_moves, key=move_priority, reverse=True)

    # ------------------------------------------------------------
    # EVALUATION FUNCTIONS
    # ------------------------------------------------------------

    def evaluate(self, board: chess.Board) -> float:
        """
        Perspective-correct evaluation.

        The internal evaluation is White-centric.
        If Black is to move, the score is inverted.
        """
        score = self.evaluate_white(board)
        return score if board.turn == chess.WHITE else -score

    def evaluate_white(self, board: chess.Board) -> float:
        """
        Evaluate a position from White's perspective.

        Positive score  -> advantage for White
        Negative score  -> advantage for Black
        """
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0.0
        score += self.weights["material"] * self.evaluate_material(board)
        score += self.weights["position"] * self.evaluate_position(board)
        score += self.weights["mobility"] * self.evaluate_mobility(board)


        return score

    def evaluate_material(self, board: chess.Board) -> float:
        """
        Material evaluation using standard piece values.
        """
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.2,
            chess.BISHOP: 3.3,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0
        }

        score = 0.0
        for piece_type, value in piece_values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value

        return score

    def evaluate_position(self, board: chess.Board) -> float:
        """
        Positional evaluation using piece-square tables (PST).
        """
        pawn_table = [
            0,0,0,0,0,0,0,0,
            5,10,10,-20,-20,10,10,5,
            5,-5,-10,0,0,-10,-5,5,
            0,0,0,20,20,0,0,0,
            5,5,10,25,25,10,5,5,
            10,10,20,30,30,20,10,10,
            50,50,50,50,50,50,50,50,
            0,0,0,0,0,0,0,0
        ]

        knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,0,5,5,0,-20,-40,
            -30,5,10,15,15,10,5,-30,
            -30,0,15,20,20,15,0,-30,
            -30,5,15,20,20,15,5,-30,
            -30,0,10,15,15,10,0,-30,
            -40,-20,0,0,0,0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]

        score = 0.0
        for sq in board.pieces(chess.PAWN, chess.WHITE):
            score += pawn_table[sq] * 0.01
        for sq in board.pieces(chess.PAWN, chess.BLACK):
            score -= pawn_table[63 - sq] * 0.01
        for sq in board.pieces(chess.KNIGHT, chess.WHITE):
            score += knight_table[sq] * 0.01
        for sq in board.pieces(chess.KNIGHT, chess.BLACK):
            score -= knight_table[63 - sq] * 0.01

        return score

    def evaluate_mobility(self, board: chess.Board) -> float:
        """
        Mobility evaluation: number of legal moves.
        """
        current = board.legal_moves.count()
        board.push(chess.Move.null())
        opponent = board.legal_moves.count()
        board.pop()

        diff = current - opponent
        return diff * 0.1 if board.turn == chess.WHITE else -diff * 0.1
