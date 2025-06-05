# PokerCV

A computer vision system for recognizing poker cards from screen captures.

## Features

- Screen capture functionality
- Card detection using computer vision
- Card rank and suit recognition using machine learning
- Real-time display interface
- Support for Texas Hold'em cards (2-9, J, Q, K, A)

## Installation

1. Clone the repository

## Dataset Structure

Create two separate folders for training data:

1. `/data/cards_numbers/` - Contains images of card numbers (2-9, J, Q, K, A)
   - Use PNG format with filenames like: 2.png, 3.png, A.png, etc.

2. `/data/cards_suits/` - Contains images of card suits (Clubs, Diamonds, Hearts, Spades)
   - Use PNG format with filenames like: Clubs.png, Hearts.png, etc.

## Create Env

1. Install virtualenv
pip install virtualenv

2. Create a virtualenv environment
virtualenv -p python2 env

3. Activate the virtualenv
env\Scripts\activate.bat
