# 🐍 Snake AI with Deep Q-Learning

This project implements the **classic Snake Game** using **Pygame** and trains an **AI agent** using **Deep Q-Learning (DQN)** with **PyTorch**.  
The agent learns to play Snake by trial and error, improving over time by maximizing its score.

---

## 📌 Features
- Snake game built using **Pygame** 🎮  
- AI agent using **Deep Q-Learning (DQN)** 🤖  
- Neural Network built with **PyTorch**  
- Experience Replay & Short/Long Memory Training  
- Live **matplotlib plots** of training progress 📈  
- Model saving & loading  

---

## 📂 Project Structure
.
├── snake_game.py # Snake game environment (Pygame)

├── agent.py # RL agent (Deep Q-Learning logic + training loop)

├── model.py # Neural network model & trainer

├── helper.py # Plotting functions

├── model/ # Saved trained models

└── README.md # Project documentation


---

## 🚀 Installation & Setup

### 1. Clone the repository

## Create a virtual environment

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate   

## Install dependencies

pip install -r requirements.txt

## How to Run

python agent.py

The game will run in the background
The AI will start training
Training progress will be shown in a plot

## Training Results

The AI starts with random moves and gradually learns to avoid collisions and eat food.
You’ll see:
Score per game
Mean score trend improving over time

<img width="1286" height="551" alt="image" src="https://github.com/user-attachments/assets/55141f7f-5d56-4e8e-9237-737b64edc9ad" />





