# ♟️ Chess Move Tracking: End-to-End Computer Vision Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO11](https://img.shields.io/badge/Model-YOLOv11-green)
![Library](https://img.shields.io/badge/Library-Ultralytics%20%7C%20OpenCV-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An automated system that converts raw chess gameplay videos into standard PGN (Portable Game Notation) records using YOLO11 and Logic-based State Analysis.**

---

## 📖 Overview
This project addresses the challenge of digitizing physical chess games in real-time. Unlike traditional solutions that require electronic boards or overhead cameras, this pipeline works with **angled camera views** and handles **occlusions** (hands blocking the board).

The system utilizes a hybrid approach combining **YOLO11** for visual recognition and a **Rule-based State Analysis** for logic validation, achieving an average **PGN Edit Distance Score of 0.93 on the Testing Dataset**.

---

## System Architecture (The 4-Phase Pipeline)

The processing pipeline consists of four sequential phases:

### Phase 1: Board Localization & Perspective Warping 📐
We use a pretrained **YOLO11s-Pose** model as the baseline model and then fine-tune it on our custom chessboard dataset to detect 4 semantic keypoints (corners) on the chessboard: `a1`, `h1`, `a8`, and `h8`. Using these points, we calculate a Homography Matrix to un-warp the image, correcting camera angle and rotation.

* **Key Components:** Fine-tuned YOLO11s-Pose and Homography Transformation.
* **Result:** Warps the angled video frame into a standardized **640x640 Top-down view**.
* **Robustness:** Trained with rotation augmentation (0°, 90°, 180°) and occluded corners.

### Phase 2: Piece Detection ♟️
We use a pretrained **YOLO11m** as a baseline model and fine-tuned it on custom chess pieces dataset then the model is run on the 640x640 warped image. The model is trained to recognize 13 classes: 12 chess pieces (wP, bK, etc.) and a Hand class.

* **Classes:** 13 Classes (`wP`, `wR`, `wN`, `wB`, `wQ`, `wK`, `bP`, `bR`, `bN`, `bB`, `bQ`, `bK`, `Hand`).
* **Occlusion Handling:** The `Hand` class acts as a trigger to pause stability analysis when a player is moving a piece.

### Phase 3: State Analysis (Logic Core) 🧠
To translate raw bounding boxes into logical board states and detect move events.

**Methodology:**
1.  **Coordinate Mapping:** Bounding boxes are mapped to the 8x8 grid using their *top-center* coordinates.
2.  **Temporal Stability:** A board state must remain unchanged for `N` frames to be confirmed, filtering out flicker and hand movements.
3.  **Active Color Detection:** The system waits for the first move to determine who started (White or Black), enabling mid-game initialization.

### Phase 4: Rule Validation & PGN Generation 📜
Integrates with the **`python-chess`** library to validate moves against official chess rules.

**Methodology:**
We use the `python-chess` library as a referee.
1.  **FEN Generation:** A utility function converts the detected board dictionary into a FEN string. It includes logic to auto-detect the standard starting position or fallback to mid-game dynamic rights.
2.  **Validation:** Every move from Phase 3 is checked. Illegal moves (visual errors) are rejected.
3.  **PGN Export:** Valid moves are recorded and formatted cleanly.

---

## 📊 Performance & Evaluation

The models were evaluated on a validation dataset derived from gameplay videos.

| Model / Metric | Value | Description |
| :--- | :--- | :--- |
| **Pose Estimation** | **0.995** | mAP50-95 (Pixel-perfect corner detection) |
| **Object Detection** | **0.978** | mAP50 (High accuracy on warped images) |
| **End-to-End PGN** | **0.93** | Edit Distance Score (Accuracy of generated game record) |

---

## 📥 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/SurawutK/chess-move-tracking-yolo11.git](https://github.com/SurawutK/chess-move-tracking-yolo11.git)
    cd chess-move-tracking-yolo11
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Model Weights**
    Due to file size limits, the pre-trained YOLO11 models are hosted on Hugging Face.
    
    The script `pipeline.py` will auto-download models using `huggingface_hub`.
    
    | Model | Task | Download Link |
    | :--- | :--- | :--- |
    | `yolo11s_pose_chessboard.pt` | Board Localization | [Link](https://huggingface.co/surawut/chess-move-tracking-yolo11/resolve/main/models/yolo11s_pose_chessboard.pt) |
    | `yolo11m_pieces.pt` | Piece Detection | [Link](https://huggingface.co/surawut/chess-move-tracking-yolo11/resolve/main/models/yolo11m_pieces.pt) |

---

## 🚀 Usage

Run the main pipeline on a video file:

```bash
python utils/pipeline.py --source "path/to/input_video.mp4" --output path/to/output/game.pgn
```