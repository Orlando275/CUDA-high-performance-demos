<!-- Banner -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=100&section=header&text=CUDA-high-performance-demos&fontSize=40&animation=fadeIn" />
</p>

## 🎯 Project Overview
**CUDA-high-performance-demos** is a Python-based game where you can:
- Play **Player vs Player** Tic-Tac-Toe.
- **Save match history** for later analysis.
- **Train a custom AI model** to improve over time based on past games.

This project is designed to showcase **game logic implementation**, **data persistence**, and **basic AI training workflows** — all in a simple, interactive environment.

---

## 📥 Installation & Setup

### Clone this repository
```bash
git clone https://github.com/Orlando275/CUDA-high-performance-demos
```

### Activate virtual enviroment python 
```bash
# On Linux / Mac
source venv/bin/activate
# On Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### Start playing
```bash
cd game
python -m game.play_tic_tac_toe
```

---

## 🐳 Docker Image

You can pull and run the latest version of the TicTacToe AI from Docker Hub:

```bash
docker pull orlando2705/tictactoeai:latest
docker run -it --rm orlando2705/tictactoeai:latest
```

---

## ✨ Features
- 👫 **PvP Mode** – Play with a friend on the same machine.
- 💾 **Match History Saving** – Every game is stored for future analysis and training.
- 🧠 **AI Training Module** – The AI improves by learning from stored games.
- 🎯 **Replay & Analysis** – Inspect past games to understand AI decisions.
- 🔄 **Modular Design** – Easy to expand with new features or game variations.

---

## 📂 Project Structure
<pre>
CUDA-high-performance-demos/
├── Vectors/
│   ├── normalize_vector.cu
│   ├── SH_normalize_vector.cu
│   ├── SH_total_vector_sum.cu
|   └── sum_of_vectors.cu
├── Matrices/
│   ├── matrix_multiplication.cu
│   └── SH_matrix_multiplication.cu
├── .gitignore
├── Dockerfile
├── README.md
</pre>
---

## 🖼️ Screenshots

### Game in progress
![Game in progress](https://github.com/user-attachments/assets/8328278b-aadf-4180-bfe7-40198af31f34)

### Game end screen
![Game end screen](https://github.com/user-attachments/assets/e291aed6-b898-4ec7-becd-a5bccb8e4610)

### AI training output
![AI training output](https://github.com/user-attachments/assets/4f65e74b-eb56-4688-981a-85f42d4b03f3)

---

## 🎯 How It Works

- **Game Execution**: The game launches from `main.py` and prompts for Player 1 and Player 2 moves.
- **Data Storage**: Every move and match result is saved in `/data` as a structured file for training.
- **AI Training**: The saved game moves are used to train the AI, improving its decision-making abilities.
- **Real-Time Inference**: The game uses a previously trained model to make predictions and play in real time.

## 🛠 Technologies Used
- Python 3
- Framework Pytorch
- JSON – Data persistence
- Basic Machine Learning concepts – Custom AI training loop

## 🚀 Future Improvements
- Add AI vs Player mode in real-time
- Implement a Tkinter interface 
- Visualize AI decision-making with heatmaps
- Add Docker support for easy deployment

---
