from typing import Dict


def generate_fen_from_dict(board_state: Dict[str, str], active_color: str) -> str:
    """
    Converts a dictionary representation of the board into a FEN string.

    This function generates the Forsyth-Edwards Notation (FEN) string based on the 
    visual detection of pieces. It includes intelligent logic to handle both 
    the standard starting position and arbitrary mid-game positions.    
    
    Args:
        board_state (Dict): A dictionary mapping square coordinates to piece codes 
                            (e.g., {'e4': 'wP', 'a1': 'wR'}).
                            
        active_color (str): The color of the player to move next ('w' for White, 'b' for Black).
    
    Returns:
        str: A valid FEN string representing the board state 
             (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").
    """
    # ---- 1. Construct Position String for Piece Placement -----
    piece_rows = []
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',]
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    
    for rank in ranks:
        empty_count = 0
        row_str = ''
        for file in files:
            square = file + rank
            if square in board_state:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                    
                piece_code = board_state[square]
                color = piece_code[0]
                role = piece_code[1]
                symbol = role.upper() if color == 'w' else role.lower()
                row_str += symbol
            else:
                empty_count += 1
                
        if empty_count > 0:
            row_str += str(empty_count)
        piece_rows.append(row_str)
        
    piece_placement = '/'.join(piece_rows)
    
    # ----- 2. Check for Standard Starting Position ----
    STANDARD_START_PLACEMENT = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
    
    if piece_placement == STANDARD_START_PLACEMENT and active_color == 'w':
        # Exact match for the standard start: return full rights immediately
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    
    # ----- 3. Mid-Game Logic (Dynamic Castling Rights) ----
    # If positions don't match the start, calculate castling rights based on 
    # the presence of Kings and Rooks on their starting squares.
    
    castling_rights = ''
    
    # Check White Rights
    if board_state.get('e1') == 'wK':
        if board_state.get('h1') == 'wR':
            castling_rights += 'K'
        if board_state.get('a1') == 'wR':
            castling_rights += 'Q'
    
    # Check Black Roght
    if board_state.get('e8') == 'bK':
        if board_state.get('h8') == 'bR':
            castling_rights += 'k'
        if board_state.get('a8') == 'bR':
            castling_rights += 'q'
    
    if castling_rights == '':
        castling_rights = '-'
        
    # Defaults for undetermined states
    # En Pasant (-), Halfmove (0), Fullmove(1)
    return f'{piece_placement} {active_color} {castling_rights} - 0 1'