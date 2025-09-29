# sales_forecasting_app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class SalesForecastingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Forecasting Dashboard")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 30
        self.is_trained = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # File loading section
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Load CSV File", 
                  command=self.load_csv).pack(fill=tk.X)
        
        ttk.Button(file_frame, text="Generate Sample Data", 
                  command=self.generate_sample_data).pack(fill=tk.X, pady=5)
        
        # Model configuration
        model_frame = ttk.LabelFrame(control_frame, text="Model Configuration", padding=5)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Sequence Length:").pack(anchor=tk.W)
        self.seq_length_var = tk.StringVar(value="30")
        ttk.Entry(model_frame, textvariable=self.seq_length_var).pack(fill=tk.X)
        
        ttk.Label(model_frame, text="Epochs:").pack(anchor=tk.W, pady=(5,0))
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(model_frame, textvariable=self.epochs_var).pack(fill=tk.X)
        
        ttk.Label(model_frame, text="Batch Size:").pack(anchor=tk.W, pady=(5,0))
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(model_frame, textvariable=self.batch_size_var).pack(fill=tk.X)
        
        # Training section
        train_frame = ttk.Frame(control_frame)
        train_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(train_frame, text="Train Model", 
                  command=self.train_model).pack(fill=tk.X)
        
        # Forecasting section
        forecast_frame = ttk.LabelFrame(control_frame, text="Forecasting", padding=5)
        forecast_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(forecast_frame, text="Days to Forecast:").pack(anchor=tk.W)
        self.forecast_days_var = tk.StringVar(value="30")
        ttk.Entry(forecast_frame, textvariable=self.forecast_days_var).pack(fill=tk.X)
        
        ttk.Button(forecast_frame, text="Generate Forecast", 
                  command=self.generate_forecast).pack(fill=tk.X, pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(control_frame, text="Results", padding=5)
        results_frame.pack(fill=tk.X, pady=5)
        
        self.results_text = tk.Text(results_frame, height=8, width=30)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                 command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel - Visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Sales Data & Forecast")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Embed matplotlib figure in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to load data...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                # Check required columns
                if 'Date' not in self.data.columns or 'Sales' not in self.data.columns:
                    messagebox.showerror("Error", "CSV must contain 'Date' and 'Sales' columns")
                    return
                
                # Convert and process date
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                self.data = self.data.sort_values('Date')
                self.data = self.data.dropna()
                self.data.set_index('Date', inplace=True)
                
                self.update_status(f"Data loaded: {len(self.data)} records")
                self.plot_data()
                self.log_message("Data loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def generate_sample_data(self):
        """Generate sample sales data for testing"""
        try:
            start_date = datetime(2020, 1, 1)
            dates = [start_date + timedelta(days=x) for x in range(365*3)]
            
            # Generate realistic sales data with trend and seasonality
            np.random.seed(42)
            trend = np.linspace(1000, 5000, len(dates))
            seasonal = 1000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
            noise = np.random.normal(0, 200, len(dates))
            
            sales = trend + seasonal + noise
            sales = np.maximum(sales, 100)  # Ensure positive values
            
            self.data = pd.DataFrame({
                'Date': dates,
                'Sales': sales
            })
            self.data.set_index('Date', inplace=True)
            
            self.update_status("Sample data generated")
            self.plot_data()
            self.log_message("Sample data generated with 3 years of synthetic sales data")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate sample data: {str(e)}")
    
    def create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def train_model(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        try:
            self.update_status("Training model...")
            
            # Get parameters
            self.sequence_length = int(self.seq_length_var.get())
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            
            # Prepare data
            scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1, 1))
            
            # Split data (80% train, 20% test)
            train_size = int(len(scaled_data) * 0.8)
            train_data = scaled_data[:train_size]
            test_data = scaled_data[train_size:]
            
            # Create sequences
            X_train, y_train = self.create_sequences(train_data, self.sequence_length)
            X_test, y_test = self.create_sequences(test_data, self.sequence_length)
            
            # Reshape for LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            # Build model
            self.model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            self.model.compile(optimizer='adam', loss='mse')
            
            # Train with early stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate model
            train_predict = self.model.predict(X_train)
            test_predict = self.model.predict(X_test)
            
            # Inverse transform predictions
            train_predict = self.scaler.inverse_transform(train_predict)
            test_predict = self.scaler.inverse_transform(test_predict)
            y_train_actual = self.scaler.inverse_transform(y_train.reshape(-1, 1))
            y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train_actual, train_predict)
            test_mae = mean_absolute_error(y_test_actual, test_predict)
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
            test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
            
            self.is_trained = True
            
            # Update UI
            self.plot_training_results(history, train_predict, test_predict, 
                                     y_train_actual, y_test_actual, train_size)
            
            self.log_message(f"Model training completed!")
            self.log_message(f"Train MAE: ${train_mae:.2f}, RMSE: ${train_rmse:.2f}")
            self.log_message(f"Test MAE: ${test_mae:.2f}, RMSE: ${test_rmse:.2f}")
            self.update_status("Model trained successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.update_status("Training failed")
    
    def generate_forecast(self):
        if not self.is_trained or self.model is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return
        
        try:
            forecast_days = int(self.forecast_days_var.get())
            self.update_status("Generating forecast...")
            
            # Use last sequence_length points to start forecasting
            scaled_data = self.scaler.transform(self.data.values.reshape(-1, 1))
            last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            forecasts = []
            current_sequence = last_sequence.copy()
            
            for _ in range(forecast_days):
                # Predict next point
                next_pred = self.model.predict(current_sequence, verbose=0)
                forecasts.append(next_pred[0, 0])
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred[0, 0]
            
            # Inverse transform forecasts
            forecasts = np.array(forecasts).reshape(-1, 1)
            forecast_values = self.scaler.inverse_transform(forecasts)
            
            # Create future dates
            last_date = self.data.index[-1]
            future_dates = [last_date + timedelta(days=x) for x in range(1, forecast_days + 1)]
            
            # Plot forecast
            self.plot_forecast(future_dates, forecast_values)
            
            # Log forecast summary
            self.log_message(f"\nForecast for next {forecast_days} days:")
            self.log_message(f"Average forecast: ${forecast_values.mean():.2f}")
            self.log_message(f"Max forecast: ${forecast_values.max():.2f}")
            self.log_message(f"Min forecast: ${forecast_values.min():.2f}")
            
            self.update_status("Forecast generated successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Forecast failed: {str(e)}")
            self.update_status("Forecast failed")
    
    def plot_data(self):
        self.ax.clear()
        self.ax.plot(self.data.index, self.data['Sales'], label='Actual Sales', linewidth=2)
        self.ax.set_title('Sales Data Overview', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Sales ($)')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.fig.autofmt_xdate()
        self.canvas.draw()
    
    def plot_training_results(self, history, train_predict, test_predict, 
                            y_train_actual, y_test_actual, train_size):
        self.ax.clear()
        
        # Plot actual data
        dates = self.data.index
        self.ax.plot(dates, self.data['Sales'], label='Actual Sales', alpha=0.7, linewidth=2)
        
        # Plot training predictions
        train_dates = dates[self.sequence_length:train_size]
        self.ax.plot(train_dates, train_predict, label='Training Predictions', alpha=0.8)
        
        # Plot test predictions
        test_dates = dates[train_size + self.sequence_length:]
        self.ax.plot(test_dates, test_predict, label='Test Predictions', alpha=0.8)
        
        self.ax.set_title('Model Predictions vs Actual Sales', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Sales ($)')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.fig.autofmt_xdate()
        self.canvas.draw()
    
    def plot_forecast(self, future_dates, forecast_values):
        self.ax.clear()
        
        # Plot historical data
        self.ax.plot(self.data.index, self.data['Sales'], 
                    label='Historical Sales', linewidth=2)
        
        # Plot forecast
        self.ax.plot(future_dates, forecast_values, 
                    label='Forecast', linewidth=2, color='red', linestyle='--')
        
        # Add confidence interval (simple version)
        confidence = forecast_values.std()
        self.ax.fill_between(future_dates, 
                           forecast_values.flatten() - confidence,
                           forecast_values.flatten() + confidence,
                           alpha=0.2, color='red', label='Confidence Interval')
        
        self.ax.set_title('Sales Forecast', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Sales ($)')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.fig.autofmt_xdate()
        self.canvas.draw()
    
    def log_message(self, message):
        self.results_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def update_status(self, message):
        self.status_var.set(message)
        self.root.update()

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    root = tk.Tk()
    app = SalesForecastingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()