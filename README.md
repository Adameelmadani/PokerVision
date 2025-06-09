# PokerVision

A computer vision system for recognizing poker cards from screen captures using machine learning, specifically designed for PokerStars Texas Hold'em 6-player tables.

## Overview

PokerVision uses computer vision and machine learning techniques to recognize poker cards directly from screen captures. The system can identify card ranks and suits in real-time, track table cards, analyze hand strength, and estimate win probability. It is optimized for PokerStars Texas Hold'em 6-player tables.

## Features

- Screen capture functionality to grab card regions
- Card detection using computer vision
- Card rank and suit recognition using trained neural networks
- Empty position detection to track game progress
- Real-time display interface built with Pygame
- Support for Texas Hold'em cards (2-9, 10, J, Q, K, A)
- Hand evaluation and win probability calculation
- Game state tracking (Pre-flop, Flop, Turn, River)
- Optimized for PokerStars UI and 6-player table layout

## Prerequisites

- Python 3.7+
- OpenCV
- TensorFlow 2.x
- Pygame
- NumPy
- scikit-learn
- Matplotlib (for evaluation graphs)
- Seaborn (for visualization)
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
   virtualenv -p python env
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

Create the following folders for training data:

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
