# TS-PINN for Chemical Kinetics on Time Scales

This repository contains the implementation accompanying the paper

"Chemical Kinetics on Time Scales: Analysis and Physics-Informed Neural Network Approximation"

## Features

- Dynamic equations on time scales
- Hybrid time scale
  T = [0,10] ∪ {11,12,...,50}
- Physics-Informed Neural Networks
- Conservation preservation
- Positivity preservation
- Continuous-discrete unified framework

## Dependencies

pip install -r requirements.txt

## Run

python src/tspinn_hybrid.py

## Outputs

- solution_comparison.png
- mass_conservation.png
- training_history.png
- error_metrics.csv
- performance_metrics.csv
- Author: Dr. Mahammad Khuddush
- Email: khuddush89@gmail.com
