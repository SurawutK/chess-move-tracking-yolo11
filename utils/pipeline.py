import cv2
import torch
import sys
from pathlib import Path
from ultralytics import YOLO
from board_localizer import BoardLocalizer
from chess_state_analyzer import ChessStateAnalyzer
from chess_pgn_generator import ChessPGNGenerator
from fen_utils import generate_fen_from_dict

# ---- CONFIG ----
POSE_MODEL_PATH = Path('chess_detection/runs/chessboard_localization/training/yolo11s_pose_run_6/weights/best.pt')
PIECE_MODEL_PATH = Path('chess_detection/runs/chess-pieces-detection/training/yolo11m_run_3/weights/best.pt')

# Input Video
VIDEO_PATH = Path(sys.argv[1])

# PGN Saving Path
SAVING_DIR = Path('chess_detection/output')
PGN_NAME = f'{VIDEO_PATH.stem}.pgn'

# Device-agnostic Code
device = '0' if torch.cuda.is_available() else 'cpu'


def run_pipeline(video_path: str):
    print(f'⏳ Processing: {VIDEO_PATH.name}...')
    
    # ---- Initialize ----
    # 1. Chessboard Localizer
    localizer = BoardLocalizer(model_path=POSE_MODEL_PATH, target_size=640, conf_thresh=0.5)
    # 2. Piece Detector
    print(f'🔄 Loading YOLO model from {PIECE_MODEL_PATH}')
    piece_model = YOLO(PIECE_MODEL_PATH)
    # 3. Chess State Analyzer
    state_analyzer = ChessStateAnalyzer(board_size=640, stability_thresh=1.5*30)
    # 4. PGN Engine
    pgn_engine = ChessPGNGenerator(event_name=f'{VIDEO_PATH.stem}')
    
    # ---- PIPELINE START ----
    cap = cv2.VideoCapture(VIDEO_PATH)
        
    if not cap.isOpened():
        print('❌ Error opening video.')
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Phase 1: Board Localization
        warped_board = localizer.get_warped_frame(frame)
        
        if warped_board is not None:
            # Phase 2: Piece Detection
            results = piece_model.predict(warped_board,
                                          imgsz=640,
                                          conf=0.25,
                                          device=device,
                                          verbose=False)
            
            # Format detections
            detections = []
            if results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    conf = box.conf.cpu().item()
                    cls_id = int(box.cls.cpu().item())
                    cls_name = piece_model.names[cls_id]
                    detections.append([x1, y1, x2, y2, conf, cls_id, cls_name])
                    
            # Phase 3: State Recognition & Temporal Analysis
            event = state_analyzer.process_frame(detections)
            
            if event:
                # Phase 4: PGN Generation & Rules Validation
                if event['type'] == 'first_move':
                    # ----- First Move Founded: Setting the Board -----
                    initial_board_state = event['initial_board_state']
                    active_color = event['active_color']
                    from_sq, to_sq = event['move']
                    
                    
                    # 1. Create FEN String from Initial Board State
                    start_fen = generate_fen_from_dict(board_state=initial_board_state, active_color=active_color)
                    print(f'🏁 Game Start Detected! Active Color: {active_color}')
                    print(f'📝 Initial FEN String: {start_fen}\n')
                    
                    # 2. Init PGN Engine
                    pgn_engine.set_board_from_fen(start_fen)
                    
                    # 3. Push the First Move
                    san = pgn_engine.push_move(from_sq, to_sq, ignore_rules=True)
                    if san:
                        print(f'♟️  First Move: {san}')
            
                elif event['type'] == 'move' :
                    # ---- Subsequence moves ----
                    from_sq, to_sq = event['move']
                    san = pgn_engine.push_move(from_sq, to_sq)
                    if san:
                        print(f'♟️  Move: {san}')
    
    cap.release()
    
    # ---- OUTPUT RESULT ----
    final_pgn = pgn_engine.get_pgn_string(headers=False, clean_format=True)
    print(f'\n✨ Final PGN Output: {final_pgn}') 
    
    # ---- SAVING PGN -----
    SAVING_PATH = SAVING_DIR / PGN_NAME
    print(f'🔄️ Saving PGN to: {SAVING_PATH}')
    pgn_engine.save_pgn_file(SAVING_PATH)

if __name__ == '__main__':
    run_pipeline(VIDEO_PATH)  