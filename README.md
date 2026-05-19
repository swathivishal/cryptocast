# 🪙 CryptoCast - Multi-Horizon Bitcoin Price Forecasting

## Project Overview
Deep learning models to forecast Bitcoin prices for 1-day, 3-day, and 7-day horizons.

## Models Used
- 1D CNN
- RNN
- LSTM
- Transformer

## Tech Stack
- Python, Pandas, NumPy
- TensorFlow / Keras
- Matplotlib, Seaborn
- Scikit-learn

## Project Structure
```
CryptoCast/
├── data/
│   └── bitcoin.csv
├── notebook/
│   └── cryptocast.ipynb
├── models/
│   └── best_lstm_model.h5
├── requirements.txt
└── README.md
```

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

| Package | Version |
|---|---|
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| matplotlib | 3.9.0 |
| seaborn | 0.13.2 |
| scikit-learn | 1.5.0 |
| tensorflow | 2.21.0 |
| keras | 3.14.0 |
| jupyter | 1.0.0 |
| ipykernel | 6.29.4 |

## How to Run

1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/cryptocast.git
cd cryptocast
```

2. Install requirements
```bash
pip install -r requirements.txt
```

3. Open the notebook
```bash
jupyter notebook notebook/cryptocast.ipynb
```

4. Run all cells

## Results

| Model | 1D RMSE | 3D RMSE | 7D RMSE |
|-------|---------|---------|---------|
| CNN         |    |    |    |
| RNN         |    |    |    |
| LSTM        |    |    |    |
| Transformer |    |    |    |

## Conclusion
- 1-Day forecast is most accurate across all models
- 7-Day forecast has higher error — harder to predict long-term
- Transformer showed strong performance on longer horizons
- LSTM is recommended for real-world crypto forecasting use cases