# PokerVision

An intelligent poker assistant that uses computer vision and machine learning to recognize cards, analyze hands, and provide strategic advice in real-time for PokerStars Texas Hold'em games.

## Overview

PokerVision combines advanced computer vision techniques with artificial intelligence to create a comprehensive poker strategy assistant. The system automatically captures and analyzes the game state from your screen, identifies cards using either neural networks or template matching, calculates win probabilities through Monte Carlo simulation, and provides strategic recommendations. It also features an interactive NLP assistant that answers poker-related questions based on the current game context.

## Features

### Card Recognition
- Screen capture functionality for automatic game state detection
- Two recognition methods:
  - Neural Networks: Highly accurate (98%+) CNN-based classification
  - Template Matching: Fast, resource-efficient, no training required
- Support for all Texas Hold'em cards (2-10, J, Q, K, A of all suits)
- Empty position detection to track game progress

### Strategy Engine
- Real-time hand strength evaluation
- Win probability calculation via Monte Carlo simulation
- Strategic advice (Fold/Call/Raise) based on card strength
- Position-aware recommendations
- Game state tracking (Pre-flop, Flop, Turn, River)

### User Interface
- Clean, intuitive Pygame-based interface
- Live probability and hand strength displays
- Card visualization with suit and rank identification
- Interactive NLP assistant for poker questions and advice
- Session statistics tracking

## Prerequisites

- Python 3.7+
- OpenCV
- TensorFlow 2.x
- Pygame
- NumPy
- scikit-learn
- Matplotlib (for evaluation graphs)
- Seaborn (for visualization)
- Nltk
- PokerStars desktop client installed

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Adameelmadani/PokerVision.git
   cd PokerVision
   ```

2. Install virtual environment:

   ```bash
   pip install virtualenv
   ```

3. Create a virtual environment:

   ```bash
   python -m venv env
   ```

4. Activate the virtual environment:

   ```bash
   env\Scripts\activate.bat
   ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The dataset contains annotated card images organized as follows:

1. `/data/cards_numbers/` - Contains images of card ranks (2-9, 10, J, Q, K, A) - Name format: `r_2.png`, `r_3.png`, `b_10.png`, `r_J.png`, `b_Q.png`, `b_K.png`, `r_A.png`, etc.

2. `/data/cards_suits/` - Contains images of card suits - Name format: `Clubs_1.png`, `Clubs_2.png`, `Diamonds_1.png`, `Diamonds_2.png`, etc.

3. `/data/empty_positions/` - Contains images of empty card positions - Name format: `rank_pos1.png`, `suit_pos1.png`, etc.

## Training the Models

1. Train the models:

   ```bash
   python train_model.py
   ```

2. Test model performance:

   ```bash
   python test_models.py
   ```

## Usage

1. Install and launch the PokerStars client
   - Join a 6-player Texas Hold'em table
   - Position the client window consistently on your screen

2. Run the main application:

   ```bash
   python main.py
   ```

3. The app will start detecting cards in the specified screen regions
   - Player cards will be displayed
   - Table cards will be tracked as they appear
   - Hand analysis will be updated in real-time

4. Configure screen regions:
   - Adjust the card regions in `main.py` to match your PokerStars window size and position

## NLP Assistant

PokerVision includes an interactive natural language processing assistant that helps players with poker-related questions:

- Ask questions about poker rules, strategies, and concepts
- Get real-time advice based on your current hand and table state
- Receive context-aware responses that consider win probability and game position

### Features

- Text-based interface within the main application UI
- Pre-defined responses to common poker questions
- Dynamic responses based on current game state analysis
- TF-IDF vectorization with cosine similarity for query matching

### Example Queries

- General poker knowledge: "What are the poker hand rankings?"
- Strategy questions: "When should I fold?"
- Context-aware queries: "Is my hand good?" or "Should I raise?"
- System questions: "How accurate is the model?"

### Interaction

1. Type your question in the text input field
2. Click "Ask" or press Enter to submit
3. View the response in the answer display area

The assistant provides customized advice based on your current hand strength, win probability, and the strategic recommendation from the poker engine.

## Project Structure

- `main.py` - Main application with Pygame interface
- `screenshot.py` - Screen capture utilities
- `card_detector.py` - Card detection functions
- `card_recognizer.py` - Card recognition using trained models
- `train_model.py` - Model training scripts
- `test_models.py` - Model evaluation scripts
- `poker_evaluator.py` - Poker hand evaluation
- `/data/` - Training and test data
- `/models/` - Trained model files
- `/evaluation/` - Model performance metrics and visualizations

## Configuration

To adapt the system to your PokerStars client:

1. Adjust the card region coordinates in `main.py` to match your screen resolution
2. The default configuration is optimized for PokerStars 6-player Texas Hold'em tables
3. Take screenshots of your specific PokerStars theme's cards
4. Train the models using your specific card images
5. Fine-tune the detection parameters if necessary

## Troubleshooting

- **Models not detecting cards correctly?**
  - Ensure the region coordinates match your poker client
  - Collect more training data specific to your client
  - Run `test_models.py` to evaluate model performance

- **Application runs slowly?**
  - Adjust the detection frequency in `main.py`
  - Reduce the screen capture resolution
