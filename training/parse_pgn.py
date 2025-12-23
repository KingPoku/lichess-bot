import chess.pgn

def load_positions_from_pgn(pgn_path, max_games=50):
    positions = []

    with open(pgn_path, "r", encoding="utf-8") as pgn:
        for _ in range(max_games):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)

                # Store middle-game positions only
                if board.fullmove_number > 10 and not board.is_game_over():
                    positions.append(board.fen())

    return positions



def build_training_data(pgn_path):
    data = []

    with open(pgn_path, "r", encoding="utf-8") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            # Determine game result
            result = game.headers.get("Result", "*")
            if result == "1-0":
                target = 1.0
            elif result == "0-1":
                target = -1.0
            else:
                target = 0.0

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)

                # Skip opening and finished games
                if board.fullmove_number > 10 and not board.is_game_over():
                    data.append({
                        "fen": board.fen(),
                        "target": target
                    })

    return data


import json

def save_training_data(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)











if __name__ == "__main__":
    data = build_training_data("training/masters_1.pgn")
    save_training_data(data, "training/training_data.json")

    print(f"Saved {len(data)} training samples to training/training_data.json")
