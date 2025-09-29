# generate_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(filename='sales_data.csv'):
    """Generate realistic sample sales data"""
    np.random.seed(42)
    
    # Generate 3 years of daily data
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(365*3)]
    
    # Create realistic sales pattern
    # Base trend
    trend = np.linspace(1000, 5000, len(dates))
    
    # Seasonal components
    yearly_seasonal = 800 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    monthly_seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
    weekly_seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    
    # Random events (promotions, holidays)
    events = np.zeros(len(dates))
    event_days = np.random.choice(len(dates), 20, replace=False)
    events[event_days] = np.random.uniform(500, 2000, 20)
    
    # Noise
    noise = np.random.normal(0, 150, len(dates))
    
    # Combine all components
    sales = (trend + yearly_seasonal + monthly_seasonal + 
             weekly_seasonal + events + noise)
    
    # Ensure positive values
    sales = np.maximum(sales, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales.round(2)
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Sample data saved to {filename}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Sales stats: Mean=${df['Sales'].mean():.2f}, Max=${df['Sales'].max():.2f}")

if __name__ == "__main__":
    generate_sample_data()