"""
Offline training script for SmartBot evaluation function.

This script:
- Loads labeled positions from training/training_data.json
- Evaluates them using SmartBot's evaluation logic
- Adjusts evaluation weights to reduce prediction error
- Saves trained weights to training/trained_weights.json

IMPORTANT:
SmartBot inherits from MinimalEngine, which cannot be instantiated
outside lichess-bot runtime. Therefore, we bypass __init__ safely
using __new__ and reuse only the evaluation logic.
"""

import sys
import os
import json
import chess

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from minmax_bot import SmartBot


# -----------------------------
# Training configuration
# -----------------------------
LEARNING_RATE = 0.01
EPOCHS = 5


# -----------------------------
# Lightweight evaluator
# -----------------------------
class TrainingEvaluator:
    """
    Training-only evaluator that reuses SmartBot's evaluation logic
    without requiring lichess-bot runtime dependencies.
    """

    def __init__(self):
        # Initial evaluation weights
        self.weights = {
            "material": 1.0,
            "position": 1.0,
            "mobility": 0.1
        }

        # Create SmartBot instance WITHOUT calling __init__()
        self.bot = SmartBot.__new__(SmartBot)

        # Inject weights manually
        self.bot.weights = self.weights

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a board position using SmartBot's evaluation.
        """
        return self.bot.evaluate(board)


# -----------------------------
# Utility functions
# -----------------------------
def load_training_data(path):
    """
    Load training dataset from JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Training loop
# -----------------------------
def train():
    evaluator = TrainingEvaluator()
    data = load_training_data("training/training_data.json")

    print(f"Loaded {len(data)} training samples")

    for epoch in range(EPOCHS):
        total_error = 0.0

        for entry in data:
            board = chess.Board(entry["fen"])
            target = entry["target"]

            # Model prediction
            prediction = evaluator.evaluate(board)

            # Error signal
            error = target - prediction
            total_error += abs(error)

            # Gradient-style update (simple & explainable)
            evaluator.weights["material"] += LEARNING_RATE * error
            evaluator.weights["position"] += LEARNING_RATE * error
            evaluator.weights["mobility"] += LEARNING_RATE * error

        avg_error = total_error / len(data)
        print(f"Epoch {epoch + 1}/{EPOCHS} — avg error: {avg_error:.4f}")

    # Save trained weights
    with open("training/trained_weights.json", "w", encoding="utf-8") as f:
        json.dump(evaluator.weights, f, indent=2)

    print("Training complete.")
    print("Trained weights saved to training/trained_weights.json")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    train()
