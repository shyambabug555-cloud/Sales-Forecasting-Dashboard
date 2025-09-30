Sales Forecasting Dashboard

Sales Forecasting Dashboard is an AI-powered desktop application that predicts future sales using LSTM (Long Short-Term Memory) deep learning models. The app allows users to load historical sales data, train the model, visualize actual vs predicted sales, and forecast future sales with confidence intervals.

Features

Load CSV files containing historical sales data.

Generate 3 years of synthetic sales data for testing.

Configure LSTM model parameters: sequence length, epochs, batch size.

Train the LSTM model with early stopping to prevent overfitting.

Visualize actual vs predicted sales.

Forecast future sales for any number of days.

Shows metrics like MAE, RMSE for model evaluation.

Technologies Used

Python 3.x

Tkinter for GUI

Matplotlib for plotting

Pandas & NumPy for data manipulation

TensorFlow / Keras for LSTM deep learning model

Scikit-learn for preprocessing and metrics

Installation

Clone the repository:

git clone https://github.com/<your-username>/sales-forecasting-dashboard.git


Create a virtual environment and activate it:

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run the application:

python sales_forecasting_app.py
