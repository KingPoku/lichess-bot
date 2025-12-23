# ♟️ SmartBot – Lichess Minimax Chess Engine

<div align="center">

**A classical AI chess engine built with Minimax and Alpha-Beta Pruning**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Lichess](https://img.shields.io/badge/Lichess-Bot%20API-green.svg)](https://lichess.org/api)
[![Algorithm](https://img.shields.io/badge/Algorithm-Minimax-orange.svg)]()

</div>

---

## 📖 Overview

SmartBot is a chess-playing AI built as a homemade engine on top of the [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) framework. The project focuses on **classical AI techniques** rather than machine learning approaches.

At its core, SmartBot uses the **Minimax algorithm with Alpha-Beta Pruning** to search the game tree and select strong moves under time constraints. The engine demonstrates how traditional search algorithms can produce competent chess play through careful optimization and heuristic evaluation.

### 🎮 Capabilities

The bot is fully integrated with the Lichess Bot API, allowing it to:

- ♟️ Play rated and unrated games on [lichess.org](https://lichess.org)
- 🤖 Compete against humans and other bots
- ⚡ Run continuously using configurable time controls
- 📊 Track performance metrics and search statistics

### 🎓 Academic Context

This project was developed as part of an **Artificial Intelligence course**, with emphasis on:

- ✅ Correct algorithmic implementation
- ⚡ Performance optimization through pruning
- 📝 Code clarity and explainability
- 🏗️ Clean software architecture
- 🧪 Testable and maintainable design

---

## 🎯 Project Goals

The main objectives of this project are:

| Goal | Status | Description |
|------|--------|-------------|
| **Minimax Implementation** | ✅ Complete | Implement a correct and efficient Minimax-based chess engine |
| **Alpha-Beta Pruning** | ✅ Complete | Optimize search using alpha-beta pruning and move ordering |
| **Lichess Integration** | ✅ Complete | Integrate the engine with a real-world platform |
| **Clean Codebase** | ✅ Complete | Maintain minimal, understandable, and well-documented code |
| **Future Extensibility** | 🔜 Planned | Prepare for evaluation tuning and advanced features |

---

## 🏗️ Architecture

The project is structured around two main components working together:

### System Diagram

```
┌─────────────────┐
│  Lichess.org    │  ← Online chess platform
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   lichess-bot Framework             │
│   • Handles API communication       │
│   • Manages games & challenges      │
│   • Provides engine interface       │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   SmartBot Engine (minmax_bot.py)  │
│   • Minimax with Alpha-Beta         │
│   • Position evaluation             │
│   • Move ordering                   │
└─────────────────────────────────────┘
```

### Component Overview

#### 1️⃣ **lichess-bot Framework**
- Handles all communication with lichess.org
- Manages game state, challenges, and API requests
- Provides a standard interface for chess engines
- Handles time management and connection stability

#### 2️⃣ **SmartBot Engine** (`minmax_bot.py`)
- Implements Minimax with Alpha-Beta pruning
- Evaluates positions using:
  - Material counting (piece values)
  - Positional heuristics (piece-square tables)
  - Mobility assessment (available moves)
- Decides the best move for each position
- Tracks performance metrics

### 🎨 Design Benefits

This separation ensures:

- 🔌 **Platform Independence**: Engine logic is decoupled from networking
- 🔧 **Easy Maintenance**: Improvements don't require touching platform code
- 📦 **Modularity**: Components can be tested and updated independently
- 🚀 **Scalability**: Easy to swap engines or add features

---

## 🛠️ Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core programming language | 3.12+ |
| **python-chess** | Board representation & move generation | Latest |
| **lichess-bot** | Lichess Bot API integration | Latest |
| **Minimax Algorithm** | Core game tree search | N/A |
| **Alpha-Beta Pruning** | Search optimization | N/A |

