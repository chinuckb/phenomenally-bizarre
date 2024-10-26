## Introduction
- 

## Aim
- This is a hobby project aimed at building an FX trading tool which generates & trades all FX Majors.
- I am writing this in Python & Go.

## Data Sources
- OANDA v20 REST API - https://developer.oanda.com/rest-live-v20/introduction/
- You can open a Demo Account for free here & you will get a Paper Trading account, and API Key (It was a bit tricky for me so if you face issues please get in touch with me).

## Future Scope
- Expansion into all Minors & Exotics.
- Integrate Fundamental & Geopolitical Factors that influence FX movements.
- Training the famed Hidden Markov Model (and other Stochastic Processes) on historical data & trying to predict short-term price action.

## Note 
- I'm not trained formally in Stochastic Processes, Python, etc. I'm self-taught. 
- I have invested in Equities before based on Fundamental analysis & was successful. Also self-taught.
- I don't believe in Market Prediction, only in educated bets. 
- Some Mathematical Methods may help, this is a journey to discover which ones do.


## Timeline
   ### 26 October 2024
   Initial Prediction Code Uploaded. This code (train.py) can predict the prices based on the trained HMM. Next step is to predict the state we are in and color code (Up Trend - Green, Down Trend - Red) the Predicted Price line.
