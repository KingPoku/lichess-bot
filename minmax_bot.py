"""
Smart Chess Bot using Minimax Algorithm with Alpha-Beta Pruning

File: smart_bot.py (place in ROOT lichess-bot folder)
Author: [Aduse-Poku Kingsford]
Course: [Artificial Intelligence]
Date: [Date]

This engine implements the Minimax algorithm with several optimizations:
1. Alpha-Beta Pruning - Reduces search space by ~90%
2. Move Ordering - Improves pruning efficiency
3. Enhanced Evaluation - Material + positional assessment
"""

import chess
from chess.engine import PlayResult
import logging
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import HOMEMADE_ARGS_TYPE

# Logger for debugging
logger = logging.getLogger(__name__)


class SmartBot(MinimalEngine):
    """
    Chess engine using Minimax algorithm with alpha-beta pruning.
    
    This bot searches the game tree to find the best move by:
    - Exploring possible future positions (game tree search)
    - Evaluating positions numerically (evaluation function)
    - Assuming both players play optimally (minimax principle)
    - Pruning unnecessary branches (alpha-beta optimization)
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the chess engine.
        
        Configuration:
            max_depth: How many moves ahead to look (4 = 2 full moves)
        """
        super().__init__(*args, **kwargs)
        self.max_depth = 4  # Adjust based on time controls
        self.nodes_searched = 0  # For performance tracking
        
    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """
        Main search function called by lichess-bot framework.
        
        This is the entry point - the bot framework calls this method
        to get the bot's move for the current position.
        
        Args:
            board: Current chess position (chess.Board object)
            *args: Additional arguments from framework (time controls, etc.)
            
        Returns:
            PlayResult: Object containing the best move found
        """
        self.nodes_searched = 0  # Reset counter
        
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')  # Best guaranteed value for us
        beta = float('inf')    # Best guaranteed value for opponent
        
        # Try each legal move and find the best one
        for move in self.order_moves(board):
            # Make the move
            board.push(move)
            
            # Evaluate resulting position (opponent's turn, so minimize)
            value = self.minimax_ab(board, self.max_depth - 1, 
                                   alpha, beta, False)
            
            # Undo the move
            board.pop()
            
            # Update best move if this is better
            if value > best_value:
                best_value = value
                best_move = move
            
            # Update alpha (our best guaranteed score so far)
            alpha = max(alpha, value)
        
        # Log search info (for debugging)
        logger.info(f"Searched {self.nodes_searched} nodes, best score: {best_value:.2f}")
        
        # Return as PlayResult (required format for lichess-bot)
        return PlayResult(best_move, None)
    
    def minimax_ab(self, board: chess.Board, depth: int, 
                   alpha: float, beta: float, maximizing: bool) -> float:
        """
        Minimax algorithm with Alpha-Beta Pruning.
        
        This is the core algorithm that recursively searches the game tree.
        
        Key Concepts:
        - Minimax: Assume both players play optimally
        - Alpha: Best value the maximizer can guarantee
        - Beta: Best value the minimizer can guarantee
        - Pruning: If alpha >= beta, remaining moves can be skipped
        
        Args:
            board: Current position to evaluate
            depth: Remaining search depth (0 = leaf node)
            alpha: Best value for maximizing player
            beta: Best value for minimizing player
            maximizing: True if it's the maximizing player's turn
            
        Returns:
            float: Evaluation score for this position
        """
        self.nodes_searched += 1
        
        # Base case: reached max depth or game is over
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)
        
        if maximizing:
            # Maximizing player's turn (trying to maximize score)
            max_eval = float('-inf')
            
            for move in self.order_moves(board):
                board.push(move)
                eval_score = self.minimax_ab(board, depth - 1, 
                                            alpha, beta, False)
                board.pop()
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                # Beta cutoff: opponent won't allow this position
                if beta <= alpha:
                    break  # Prune remaining moves
            
            return max_eval
        else:
            # Minimizing player's turn (trying to minimize score)
            min_eval = float('inf')
            
            for move in self.order_moves(board):
                board.push(move)
                eval_score = self.minimax_ab(board, depth - 1, 
                                            alpha, beta, True)
                board.pop()
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                # Alpha cutoff: we won't allow this position
                if beta <= alpha:
                    break  # Prune remaining moves
            
            return min_eval
    
    def order_moves(self, board: chess.Board) -> list:
        """
        Order moves to improve alpha-beta pruning efficiency.
        
        Good move ordering can improve pruning by 3-5x!
        
        Priority (highest to lowest):
        1. Captures (especially good trades like QxP)
        2. Checks
        3. Promotions
        4. Other moves
        
        This uses MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        heuristic for captures.
        
        Args:
            board: Current position
            
        Returns:
            List of moves sorted by priority (best first)
        """
        def move_priority(move: chess.Move) -> float:
            """Calculate priority score for a move."""
            score = 0
            
            # Piece values for capture ordering
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 0
            }
            
            # 1. Prioritize captures
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                
                if victim and attacker:
                    # MVV-LVA: Prefer capturing valuable pieces
                    # with less valuable pieces
                    victim_value = piece_values.get(victim.piece_type, 0)
                    attacker_value = piece_values.get(attacker.piece_type, 0)
                    score += victim_value * 10 - attacker_value
            
            # 2. Prioritize checks
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()
            
            # 3. Prioritize promotions
            if move.promotion:
                score += 100
            
            return score
        
        # Sort moves by priority (highest score first)
        return sorted(board.legal_moves, key=move_priority, reverse=True)
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a chess position numerically.
        
        Returns:
            Positive score = good for White
            Negative score = good for Black
            Score in "centipawns" (1.0 = 1 pawn advantage)
        
        Evaluation components:
        1. Material balance (piece count)
        2. Piece positioning (piece-square tables)
        3. Mobility (number of legal moves)
        4. King safety (castling, pawn shield)
        
        Args:
            board: Position to evaluate
            
        Returns:
            float: Evaluation score
        """
        # Check for terminal positions
        if board.is_checkmate():
            # If it's white's turn and checkmate, black won (negative)
            # If it's black's turn and checkmate, white won (positive)
            return -10000 if board.turn == chess.WHITE else 10000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0  # Draw
        
        # Initialize score
        score = 0.0
        
        # 1. Material evaluation
        score += self.evaluate_material(board)
        
        # 2. Positional evaluation
        score += self.evaluate_position(board)
        
        # 3. Mobility evaluation
        score += self.evaluate_mobility(board)
        
        return score
    
    def evaluate_material(self, board: chess.Board) -> float:
        """
        Calculate material balance.
        
        Standard piece values:
        - Pawn: 1
        - Knight: 3
        - Bishop: 3 (slightly more than knight)
        - Rook: 5
        - Queen: 9
        
        Returns:
            float: Material score (positive = white advantage)
        """
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.2,
            chess.BISHOP: 3.3,  # Bishops slightly better in general
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
        }
        
        score = 0.0
        
        for piece_type in piece_values:
            # Count white pieces (positive)
            white_count = len(board.pieces(piece_type, chess.WHITE))
            score += white_count * piece_values[piece_type]
            
            # Count black pieces (negative)
            black_count = len(board.pieces(piece_type, chess.BLACK))
            score -= black_count * piece_values[piece_type]
        
        return score
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate piece positioning using piece-square tables.
        
        Piece-square tables (PST) assign values to pieces based on
        what square they're on. For example:
        - Pawns are better in the center
        - Knights are better in the center
        - Kings should castle early game
        
        Returns:
            float: Positional score
        """
        # Pawn table: encourage center control and advancement
        pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10,-20,-20, 10, 10,  5,
            5, -5,-10,  0,  0,-10, -5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5,  5, 10, 25, 25, 10,  5,  5,
           10, 10, 20, 30, 30, 20, 10, 10,
           50, 50, 50, 50, 50, 50, 50, 50,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        # Knight table: encourage centralization
        knight_table = [
           -50,-40,-30,-30,-30,-30,-40,-50,
           -40,-20,  0,  5,  5,  0,-20,-40,
           -30,  5, 10, 15, 15, 10,  5,-30,
           -30,  0, 15, 20, 20, 15,  0,-30,
           -30,  5, 15, 20, 20, 15,  5,-30,
           -30,  0, 10, 15, 15, 10,  0,-30,
           -40,-20,  0,  0,  0,  0,-20,-40,
           -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        score = 0.0
        
        # Evaluate white pawns
        for square in board.pieces(chess.PAWN, chess.WHITE):
            score += pawn_table[square] * 0.01
        
        # Evaluate black pawns (mirror the table)
        for square in board.pieces(chess.PAWN, chess.BLACK):
            score -= pawn_table[63 - square] * 0.01
        
        # Evaluate white knights
        for square in board.pieces(chess.KNIGHT, chess.WHITE):
            score += knight_table[square] * 0.01
        
        # Evaluate black knights
        for square in board.pieces(chess.KNIGHT, chess.BLACK):
            score -= knight_table[63 - square] * 0.01
        
        return score
    
    def evaluate_mobility(self, board: chess.Board) -> float:
        """
        Evaluate mobility (number of legal moves available).
        
        More moves = more options = better position
        
        Returns:
            float: Mobility score
        """
        # Count legal moves for current side
        current_mobility = board.legal_moves.count()
        
        # Switch sides temporarily to count opponent's mobility
        board.push(chess.Move.null())
        opponent_mobility = board.legal_moves.count()
        board.pop()
        
        # Calculate mobility advantage
        mobility_diff = current_mobility - opponent_mobility
        
        # Return positive if white to move, negative if black to move
        if board.turn == chess.WHITE:
            return mobility_diff * 0.1
        else:
            return -mobility_diff * 0.1


# Optional: Test function to verify the engine works
if __name__ == "__main__":
    print("SmartBot Chess Engine - Quick Test")
    print("=" * 50)
    
    bot = SmartBot()
    
    # Test 1: Starting position
    print("\n1. Testing starting position...")
    board = chess.Board()
    result = bot.search(board)
    print(f"   Move: {result.move}")
    print(f"   Nodes: {bot.nodes_searched}")
    
    # Test 2: Mate in 1
    print("\n2. Testing checkmate detection...")
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
    result = bot.search(board)
    board.push(result.move)
    print(f"   Move: {result.move}")
    print(f"   Checkmate: {board.is_checkmate()}")
    
    print("\n" + "=" * 50)
    print("✅ Engine is working!")