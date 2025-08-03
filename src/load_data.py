from sklearn.datasets import fetch_california_housing
import pandas as pd

def save_data():
    # Load California housing data
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # Save to CSV
    df.to_csv(r"C:\Balaji\BITS\Semester 3\MLOps\mlops-housing\data\housing.csv", index=False)
    print("Dataset saved to data/housing.csv")

if __name__ == "__main__":
    save_data()
