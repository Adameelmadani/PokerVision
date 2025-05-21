# PokerVision
Recognition of all entities on the poker table( now only for Pokerstars) and added analytics on the basis of which you can make decisions about your moves 

NOTE: it only works with the theme that I have and for 6 players. It has been tested on ubuntu with python version 3.8


# Installing and Running

```
$ git clone https://github.com/Adameelmadani/PokerVision.git
$ cd PokerVision
$ apt-get update
$ apt install python3-virtualenv
$ virtualenv venv
$ source venv/bin/activate
$ pip3 install Cython
$ pip3 install -r requirements.txt
$ cd scripts
$ python3 grab_table.py
```