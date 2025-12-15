import numpy as np
import cv2
from ultralytics import YOLO
from typing import Tuple, List

class BoardLocalizer():
    """
    A class to localize and warp the chessboard in an image frame.
    
    Attributes:
        model_path: Path to the best.pt (board localizer YOLO-pose model).
        target_size: The desired output size (width, height) of the warped chessboard image.
        conf_thresh: Confidence threshold (0.0-1.0)
        
    Methods:
        - is_calibrated(): Check if the localizer is calibrated.
        - calibrate(): Calibrate the localizer using the input frame.
        - warp(): Warp the input frame to get a top-down view of the chessboard.
        - reset(): Reset the calibration.
    """
    def __init__(self,
                 model_path: str,
                 target_size: int=640,
                 conf_thresh: float=0.5
                 ):
        print(f'🔄 Loading YOLO-Pose model from {model_path}')
        
        # Attributes
        self.model = YOLO(model_path)
        self.target_size = target_size
        self.conf_thresh = conf_thresh
        
        # Memories
        self.M = None # Perspective Transform Matrix (3 x 3)
        self.is_locked = False
        self.locked_points = None
        
        # Destination points for for the perspective transform
        self.dst_points = np.array([
            [0, target_size],                  # a1: x=0, y=640
            [target_size, target_size],        # h1: x=640, y=640
            [0, 0],                            # a8: x=0, y=0
            [target_size, 0]                   # h8: x=640, y=0
        ], dtype=np.float32)
    
    def reset(self):
        """
        (Main Method) Reset the stored memory if cammera position have changed.
        """
        print('🔃 Board Localizer Reset!')
        self.M = None
        self.is_locked = False
        self.locked_points = None
    
    def _attempt_calibration(self, frame: np.ndarray) -> bool:
        """
        (internal method) Attempt to find 4 corner points of the chessboard 
        and calculate the perspective transform matrix M
        
        Args:
            frame: Input image frame.
        Returns:
            bool: True if calibration is successful, False otherwise.
        """
        # 1. Run inference on frame using trained YOLO-pose model.
        results = self.model(frame, conf=self.conf_thresh, verbose=False)
        
        # 2. Check Detections
        if len(results) == 0 or len(results[0].keypoints) == 0:
            return False # chessboard not found
        
        # 3. Extract keypoints of the first board detected
        # shape: (4, 2) -> a1_corner, h1_corner, a8_corner, h8_corner respectively
        kpts = results[0].keypoints.xy[0].cpu().numpy() 
        confs = results[0].keypoints.conf[0].cpu().numpy()
        
        # 4. Valiadate logic
        # condition A: a valid corner point must have x, y coordinate >= 0. 
        # In this case we check if all 4 keypoints is valid or not.
        if np.any(kpts <= 0):
            return False
        
        # condition B: valid corners points must have average confidence > conf threshold
        if np.mean(confs) < self.conf_thresh:
            return False
        
        # 5. Calculate matrix M and lock
        self.M = cv2.getPerspectiveTransform(kpts.astype(np.float32), self.dst_points)
        self.is_locked = True
        self.locked_points = kpts.astype(np.float32)
        
        print(f'✅ Board Localizer Calibration Locked! (Average Conf: {np.mean(confs):.2f})')
        return True
    
    def get_locked_points(self):
        """
        (Main method) get the locked 4 corner points
        """
        return self.locked_points
    
    def get_warped_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """
        (Main method) Warp the input frame to get a top-down view of the chessboard of size (640, 640).
        which the layout is
            - top left: a8_corner
            - top right: h8_corner
            - bottom left: a1_corner
            - bottom right: h1_corner
    
        Args:
            frame (np.ndarray): Input image frame
        Returns:
            np.ndarray or None: warped board from top-down view of size (640, 640).
            or return None if M is not memorized yet.
        """
        # case 1: If not locked -> attempting calibrate on this frame
        if not self.is_locked:
            success = self._attempt_calibration(frame)
            if not success:
                return None
        
        # If locked: Create a warped board on this frame 
        warped = cv2.warpPerspective(frame,
                                     M=self.M, 
                                     dsize=(self.target_size, self.target_size)
                                     )
        
        return warped