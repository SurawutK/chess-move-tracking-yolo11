# ♟️ AI Chess Move Tracking: End-to-End Computer Vision Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-green)
![Library](https://img.shields.io/badge/Library-Ultralytics%20%7C%20OpenCV-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An automated system that converts raw chess gameplay videos into standard PGN (Portable Game Notation) records using Deep Learning and Logic-based State Analysis.**

![Demo Preview](assets/demo_preview.gif)
*(Please place a demo GIF here in your assets folder)*

---

## 📖 Overview
This project addresses the challenge of digitizing physical chess games in real-time. Unlike traditional solutions that require electronic boards or overhead cameras, this pipeline works with **angled camera views** and handles **occlusions** (hands blocking the board).

The system utilizes a hybrid approach combining **YOLO11** for visual recognition and a **Rule-based State Machine** for logic validation, achieving an average **PGN Edit Distance Score of 0.93 on the Testing Dataset**.

---

## 🏗️ System Architecture (The 4-Phase Pipeline)

The processing pipeline consists of four sequential phases:

### Phase 1: Board Localization & Perspective Warping 📐
We use a fine-tuned **YOLO11s-Pose** model to detect 4 semantic keypoints (`a1`, `h1`, `a8`, `h8`).
* **Technique:** Homography Transformation.
* **Result:** Warps the angled video frame into a standardized **640x640 Top-down view**.
* **Robustness:** Trained with rotation augmentation (0°, 90°, 180°) and occluded corners.

### Phase 2: Piece Detection ♟️
Object detection is performed on the warped image using **YOLO11m**.
* **Classes:** 13 Classes (12 Chess Pieces + `Hand`).
* **Occlusion Handling:** The `Hand` class acts as a trigger to pause stability analysis when a player is moving a piece.

### Phase 3: State Analysis (Logic Core) 🧠
Converts raw bounding boxes into a logical board state.
* **Grid Mapping:** Uses **Footpoint-based mapping** (top-center of the piece) to solve perspective distortion for tall pieces (e.g., King/Queen).
* **Temporal Stability:** Implements a "Wait-and-Init" strategy. The board must remain static for $N$ frames to confirm a move.
* **Mid-Game Support:** Automatically detects the active color and generates the initial FEN string from the first move.

### Phase 4: Rule Validation & PGN Generation 📜
Integrates with the **`python-chess`** library to validate moves against official chess rules.
* **Legal Check:** Illegal moves detected by vision are rejected.
* **Resync Logic:** If the visual state desynchronizes from the engine state, the system forces a reset to the engine's legal state.

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
    git clone [https://github.com/YourUsername/chess-move-tracking-yolo11.git](https://github.com/YourUsername/chess-move-tracking-yolo11.git)
    cd chess-move-tracking-yolo11
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Model Weights**
    Due to file size limits, the pre-trained YOLOv11 models are hosted on Hugging Face.
    
    * **Option A (Automatic):** The script will auto-download models using `huggingface_hub`.
    * **Option B (Manual):** Download `.pt` files and place them in the `models/` directory.
    
    | Model | Task | Download Link |
    | :--- | :--- | :--- |
    | `yolo11_pose_best.pt` | Board Localization | [Link](https://huggingface.co/surawut/chess-move-tracking-yolo11/resolve/main/models/yolo11s_pose_chessboard.pt) |
    | `yolo11_piece_best.pt` | Piece Detection | [Link](https://huggingface.co/surawut/chess-move-tracking-yolo11/resolve/main/models/yolo11m_pieces.pt) |

---

## 🚀 Usage

Run the main pipeline on a video file:

```bash
python pipeline.py "input_video.mp4"