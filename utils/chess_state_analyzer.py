"""
A class for analyzing the temporal sequence of the chessboard states to detect and confirm chess move.
"""
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple

class ChessStateAnalyzer:
    """
    This  class acts as a filter and a state machine. It takes raw object detections from 
    each video frame, convert them into a logical board representation (dictionary), 
    and use temporal stability analysis to filter out noise (e.g., hand occlusions,
    flickering detections). It identifies when a move has settled and determines the
    'from' to 'to' squares.
    
    Attributes:
        - square_size (float): The pixel width/height of a single square on the warped board.
        - stability_thresh (int): The number of consecutive identical frames required to confirm a new board state.
        - files (list): Mapping of column indices (0-7) to file letter ('a'-'h').
        - ranks (list): Mapping of row indices (0-7) to rank numbers ('8'-'1').
        - prev_stable_board (dict): The last confirmed board state (before the current move).
        - candidate_board_str (str): The string representation of the board state currently being verified for stability.
        - stability_counter (int): A counter tracking how many consecutive frames the candidate state has remained unchanged.
        - initial_board_state (dict): The initial board state before the fist move occure.
        - game_initialized (bool): A boolean tells whether the first move has already occured  or not.
    """
    def __init__(self, board_size: int=640, stability_thresh: int=10):
        
        """
        Initialize the ChessStateAnalyzer.
        
        Args: 
            board_size (int): The dimension (width/height) of the warped chessboard image in pixels.Defaults to 640.
            stability_thresh (int): The number of consecutive frames a state must remain constant to be accepted as stable. Default to 10.
            
        """        
        self.board_size = board_size
        self.square_size = board_size / 8
        self.stability_thresh = stability_thresh
        
        # Mapping for converting grid coordinates to algebraic notation.
        # Note: In image coordinates, y=0 is the top (Rank 8), y=max is the bottom (Rank 1).
        self.files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',]
        self.ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        
        # State Machine Memory
        self.prev_stable_board_state = None
        self.candidate_board_str = None
        self.stability_counter = 0
        
        # Initial Board State Memory
        self.initial_board_state = None
        self.game_initialized = False

    def _get_square_from_xy(self, x: float, y: float) -> str:
        
        """
        (internal method) Convert pixels coordinates (x, y) into a chess square notation (e.g., 'e4')
        
        Args:
        x (float): The x-coordinate on the warped board image.
        y (float): The y-coordinate on the warped board image.

        Returns:
            str: The algebraric notation of the square (e.g., 'a1', 'h8').
        """
        col = int(x // self.square_size)
        row = int(y // self.square_size)
        
        # Clamp values to 0-7 to prevent crashes if a detection is slightly out of bounds
        col = max(0, min(7, col))
        row = max(0, min(7, row))
        
        return self.files[col] + self.ranks[row]
    
    def detections_to_board(self, detections: List) -> Dict[str, str] | None:
        """
        (main mathod) Convert a list of YOLO detections into a logical board dictionary.
        
        This method maps each detected bounding box to a specific square on chessboard.
        It uses the 'top-center' point of the bounding box to detemine the square,
        which is more robust to perspective distortion than the center point.
        It also handles conflicts (multiple pieces on one square) by keeping the one with higher confidence.
        
        Args:
            detections (list): A list of detections, where each detection is a list:
            [x1, y1, x2, y2, confidence, class_id, class_name].

        Returns:
            dict: A dictionary representing the board state, where keys are square names (e.g., 'e4')
                  and values are pieces classes (e.g., 'wP').
            None: If a 'Hand' is detected, indicating the frame is unstable/occluded.
        
        """    
        board_state = {}
        conf_map = {}
        
        for box in detections:
            x1, y1, x2, y2 = box[:4]
            conf = box[4]
            cls_name = box[6]
            
            # Stability Check: If a hand is visible, the board state is unreliable.
            if cls_name == 'Hand':
                return None

            # Logic: Calculate the reference point (top-center) of the box.
            # We use y1 (top edge) and move slightly down (25%)
            ref_x = (x1 + x2) / 2
            ref_y = y1 + 0.25 * (y2 - y1)
            
            square = self._get_square_from_xy(ref_x, ref_y)
            
            # Conflict Resolution:
            # If a square is already occupied by another detection (e.g., overlapping boxes),
            # keep the detection with the higher confidence score.
            if square in board_state:
                if conf > conf_map[square]:
                    board_state[square] = cls_name
                    conf_map[square] = conf
            else:
                board_state[square] = cls_name
                conf_map[square] = conf
                
        return board_state
    
    
    def _detect_move_diff(self, 
                          old_board: Dict[str, str], 
                          new_board: Dict[str, str]) -> Tuple[str] | None:
        """
        (internal method) Compares two stable board states to deduce the move that occurred.
        
        It identifies squares that have changed (pieces disappeared, appeared, or changed type)
        and infers the source and destination squares.

        Args:
            old_board (Dict): The board state before the move.
            new_board (Dict): The board state afer the move.

        Returns:
            tuple: A tuple (from_square, to_square) representing the detected move
                    (e.g., ('e2', 'e4')).
            None: If the difference logic cannot determine a valid single move.
        """
        # Identify all squares where the state differs between old and new boards
        all_squares = set(old_board.keys()) | set(new_board.keys())
        diff_squares = set()
        
        for sq in all_squares:
            piece_old = old_board.get(sq)
            piece_new = new_board.get(sq)
            if piece_old != piece_new:
                diff_squares.add(sq)
                
        from_sq, to_sq = None, None
        
        # Infer Move Logic
        for sq in diff_squares:
            piece_old = old_board.get(sq)
            piece_new = new_board.get(sq)
            
            # Case 1: Source Square
            # A piece existed here in the old board state but is gone in the new board state.
            if piece_old and not piece_new:
                from_sq = sq
            
            # Case 2: Destination Square (Move or Capture)
            # A piece exists here in the new board (it was empty before OR occupied by an opponent).
            elif piece_new:
                to_sq = sq
                
        if from_sq and to_sq: 
            return (from_sq, to_sq)

        # If diff_squares set have < 2 different squares
        return None
    
    
    def process_frame(self, detections: List) -> Dict | None:
        """
        (main method) The main processing loop to be called on every video frame.

        This method handles the temporal stability logic. It waits for the board state 
        to remain consistent for `stability_thresh` frames before confirming a change. 
        Once confirmed, if the state differs from the previous stable state, it triggers 
        move detection.
        
        The method operates in two main phases:
        1. Initialization Phase: It waits for the first valid move to occur to 
           determine the active color (who moved first) and establish the initial FEN.
        2. Game Loop Phase: Once initialized, it tracks standard moves by comparing 
           the current stable state with the previous one.
        

        Args:
            detections (List): Raw YOLO detections for the current frame. 
                               where each detection is a list:
                               [x1, y1, x2, y2, confidence, class_id, class_name].

        Returns:
            dict | None: 
            - Returns **None** if the board is unstable, occluded (hand detected), 
              or if no new move has been confirmed yet.
            - Returns an **Event Dictionary** if a significant event occurs:
                > Type 'first_move': When the first move is detected.
                  Structure: `{'type': 'first_move', 'initial_board': dict, 'active_color': str, 'move': tuple}`
                > Type 'move': For subsequent moves.
                  Structure: `{'type': 'move', 'move': (from_sq, to_sq)}`
        """
        # 1. Convert raw detections to a logical board state
        current_board_state = self.detections_to_board(detections)
        
        # 2. Reset if unstable (Hand detected or empty/invalid board)
        if current_board_state is None or len(current_board_state) < 2:
            self.stability_counter = 0
            self.candidate_board_str = None
            return None
        
        # Serialize the board to string for easy comparison 
        # (e.g., "[('a1', 'wR'), ('a2', 'wP')...]")
        current_board_str = str(sorted(current_board_state.items()))
        
        # 3. Check for Stability
        if current_board_str == self.candidate_board_str:
            self.stability_counter += 1
        else:
            # State changed (flickering or piece moved) -> Reset counter and start tracking new candidate
            self.candidate_board_str = current_board_str
            self.stability_counter = 0
        
        # 4. Check Threshold
        if self.stability_counter >= self.stability_thresh:

            
            # ---- 1. Initialization Phase ----
            if not self.game_initialized:
            
                # A: Initial Board State is Not in Memory
                if self.initial_board_state is None:
                    self.initial_board_state = current_board_state
                    print('⏳ Waiting for the first move to determine active color...')
                    return None
                
                # B: Initial Board State is already in Memory -> wait until first the move occure
                prev_board_str = str(sorted(self.initial_board_state.items()))
                
                if current_board_str != prev_board_str: 
                    # (first move) Analyze the difference to find the move
                    move = self._detect_move_diff(self.initial_board_state, current_board_state)
                    if move:
                        from_sq, to_sq = move
                        # Get the color of the moving piece
                        moving_piece = self.initial_board_state.get(from_sq)
                        
                        if moving_piece:
                            # 'wP' -> 'w', 'bK' -> 'b'
                            active_color = moving_piece[0]
                            
                            # Update memory to the new state
                            self.game_initialized = True
                            self.prev_stable_board_state = current_board_state
                            self.stability_counter = 0
                            
                            print(f'✅ First Move Detected: {active_color} moves {from_sq}->{to_sq}')
                            
                            # Return Event 'first_move'
                            return {
                                'type': 'first_move',
                                'initial_board_state': self.initial_board_state,
                                'active_color': active_color,
                                'move': move # e.g., ('e2', 'e4')
                            }
            
            # ---- 2. Game Loop Phase -----             
            else:
                prev_board_str = str(sorted(self.prev_stable_board_state.items()))
                
                if current_board_str != prev_board_str:
                    # (subsequence move) Analyze the difference to find the move
                    move = self._detect_move_diff(self.prev_stable_board_state, current_board_state)
                    # Update memory to the new state
                    if move:
                        self.prev_stable_board_state = current_board_state
                        self.stability_counter = 0
                        
                        # Return Event 'move'
                        return {
                            'type': 'move',
                            'move': move 
                        }
            
        return None