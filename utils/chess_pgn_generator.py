"""
A class responsible for validating chess moves, maintaining the game state, 
and generating Portable Game Notation (PGN) records.
"""
import chess
import chess.pgn
import re
from datetime import datetime
from typing import List, Optional

class ChessPGNGenerator:
    """
    A class responsible for validating moves using chess rules and generating PGN records.

    This class acts as a bridge between the computer vision pipeline and the official 
    chess rules (via the `python-chess` library). It handles:
    1. Validating raw moves (e.g., 'e2' -> 'e4') against the current board state.
    2. Maintaining the game state (handling captures, castling, en passant).
    3. Constructing the PGN game tree (Standard Algebraic Notation).
    4. Supporting mid-game initialization via FEN strings.
    """
    def __init__(self, event_name:str='Chess Move Tracking'):
        """
        Initializes the ChessPGNGenerator.

        Attributes:
            board (chess.Board): The internal representation of the chess board state.
            game (chess.pgn.Game): The root object of the PGN game tree.
            node (chess.pgn.GameNode): A pointer to the current move in the game tree.
            event_name (str): The name of the tournament or match.
        """
        # 1. Initialize an empty standard board
        self.board = chess.Board()
        
        # 2. Create a PGN Game object to store headers and moves
        self.game = chess.pgn.Game()
        
        # 3. Set standard headers
        self.game.headers['Event'] = event_name
        self.game.headers['Date'] = datetime.now().strftime('%Y.%m.%d')

        # 4. Initialize the 'node' pointer to the root of the game
        # (This pointer will move forward as we add moves)
        self.node = self.game
        
    def set_board_from_fen(self, fen_str: str) -> bool:
        """
        (Main Method) Initializes the board from a specific FEN string (for mid-game scenarios).

        Args:
            fen_str (str): The Forsyth-Edwards Notation string representing the board state.

        Returns:
            bool: True if the FEN is valid and set successfully, False otherwise.
        """
        try:
            # 1. Update the internal board state
            self.board = chess.Board(fen_str)
            
            # 2. Tell the PGN object that this game starts from a specific position
            # (This adds the 'FEN' and 'SetUp' headers automatically)
            self.game.setup(self.board)
            
            # 3. Reset the node pointer to the root (clearing previous history if any)
            self.node = self.game
            
            return True
        
        except ValueError:
            print(f'❌ Invalid FEN string provided: {fen_str}')
            return False
        

    def _is_promotion(self, move: chess.Move) -> bool:
        """
        (Internal method) Checks if a proposed move results in a pawn promotion.

        Since computer vision detects 'from' and 'to' squares but not the promotion 
        piece choice, we need to detect this scenario to auto-promote (usually to Queen).

        Args:
            move (chess.Move): The move object to check.

        Returns:
            bool: True if the move is a pawn advancing to the last rank (rank 1 or 8),
                  False otherwise.
        """
        # Check if the piece is a PAWN
        piece = self.board.piece_at(move.from_square)
        
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(move.to_square)
            
            # Rank 0 is White's 1st rank (promotion for Black)
            # Rank 7 is White's 8th rank (promotion for White)
            if rank == 0 or rank == 7:
                return True
        return False
    
    def push_move(self, from_sq: str, to_sq: str, ignore_rules: bool=False) -> Optional[str]:
        """
        Validates and applies a move to the board, returning the SAN string.

        Args:
            from_sq (str): Source square (e.g., 'e2').
            to_sq (str): Destination square (e.g., 'e4').
            ignor_rules (bool): whether to ignore legality for chess moving. If True, ignor the rules.
                                If False, otherwise.

        Returns:
            san_move (str): The move in Standard Algebraic Notation (e.g., 'Nf3', 'exd5') if valid. 
                            Returns None if 'ignore_rule' = False and the move is illegal.
        """
        try:
            # 1. Convert raw strings to Move object (UCI format)
            uci_str = f'{from_sq}{to_sq}'
            move = chess.Move.from_uci(uci_str)
            
            # 2. Handle Auto-Promotion
            # If it's a promotion, we MUST specify the promotion piece (defaulting to Queen)
            # otherwise, legal_moves check will fail.
            if self._is_promotion(move):
                move.promotion = chess.QUEEN
        
            if not ignore_rules:
                # 3. Check Legality (The Rule Engine)
                if move in self.board.legal_moves:
                    # 4. Convert to SAN (e.g., 'e4', 'O-O')
                    san_move = self.board.san(move)
                    # 5. Add the move to the PGN tree .add_variation() creates a new child node and returns it
                    self.node = self.node.add_variation(move)
                    # 6. Update the internal board state
                    self.board.push(move)
                    return san_move
                else:
                    # Move logic is impossible (e.g., Knight jumping incorrectly)
                    return None
            else:
                san_move = self.board.san(move)
                self.node = self.node.add_variation(move)
                self.board.push(move)
                return san_move

        except Exception as e:
            print(f'🔴 Error processing move {from_sq}->{to_sq}: {e}')
            return None

    def get_pgn_string(self, headers: bool=True, clean_format: bool=False) -> str:
        """
        Exports the complete game history as a PGN formatted string.
        
        Args:
            headers (bool): Wheter or not to keep the headers in output. Default to True
            clean_format (bool): If true, removes '*' and converts '1...' to '1.'

        Returns:
            str: The full PGN string including headers and move list.
        """
        # StringExporter is the standard way to format PGNs in python-chess
        exporter = chess.pgn.StringExporter(headers=headers, variations=True, comments=False)
        pgn_raw = self.game.accept(exporter)
        
        if clean_format:
            # delete * at the end (if any)
            pgn_clean = re.sub(r"\s+(1-0|0-1|1\/2-1\/2|\*)$", "", pgn_raw)
            # replace ... with .
            pgn_clean = pgn_clean.replace('...', '.')
            # delete \n
            pgn_clean = pgn_clean.replace('\n', ' ')
            # delete double spaces
            pgn_clean = re.sub(r"\s+", " ", pgn_clean)
            
            return pgn_clean.strip()
        
        return pgn_raw
    
    def save_pgn_file(self, filename: str='game.pgn'):
        """
        Saves the PGN string to a text file.
        
        Args:
            filename (str): PGN filename.
            
        """
        pgn_content = self.get_pgn_string()
        
        try:
            with open(filename, 'w') as f:
                f.write(pgn_content)
            print(f"💾 PGN successfully saved to: {filename}")
        except IOError as e:
            print(f'❌ Could not write to file: {e}')   