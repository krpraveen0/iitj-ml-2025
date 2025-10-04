# g25ait1119.py
#
# Name: Praveen Kumar
# Roll No: G25AIT1119
#
# Assignment: Implementing Linear Regression from Scratch
# Resources Used: NumPy Documentation, California Housing Dataset from sklearn
# Assumptions: Target variable does not require standardization

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. DATA PREPARATION FUNCTIONS
# ==============================================================================

def load_and_split_data(test_size=0.2, random_state=42):
    """Loads California housing data and splits it into training/testing sets."""
    
    # 1. Load the Dataset (Required by Assignment Prompt)
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target.reshape(-1, 1)  # Ensure target is a column vector
    
    # 2. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def standardize_features(X_train, X_test):
    """Standardizes features based on training data statistics."""
    
    # Calculate mean (mu) and standard deviation (sigma) ONLY from the training data
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    
    # Apply standardization to both Training and Test sets
    # Add a small epsilon (1e-8) to sigma to prevent division by zero
    X_train_scaled = (X_train - mu) / (sigma + 1e-8)
    X_test_scaled = (X_test - mu) / (sigma + 1e-8)
    
    return X_train_scaled, X_test_scaled

# ==============================================================================
# 2. LINEAR REGRESSION CLASS (AS REQUIRED BY ASSIGNMENT)
# ==============================================================================

class LinearRegression:
    """
    Linear Regression implementation using Gradient Descent.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_param=0.0):
        """
        Initialize the Linear Regression model.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param
        
        # Initialize parameters (will be set during training)
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train the model using Gradient Descent.
        """
        m, n = X.shape  # m = number of samples, n = number of features
        
        # Initialize weights and bias
        self.weights = np.zeros((n, 1))
        self.bias = 0
        self.cost_history = []
        
        # Gradient Descent Loop
        for i in range(self.n_iterations):
            # 1. Calculate Predictions: y_hat = X * w + b
            y_predicted = self.predict(X)
            
            # 2. Calculate Error (Residuals)
            error = y_predicted - y
            
            # 3. Calculate Gradients
            # Gradient for Weights: dw = (1/m) * X.T * error + (lambda/m) * w
            dw = (1/m) * X.T.dot(error) + (self.lambda_param / m) * self.weights
            
            # Gradient for Bias: db = (1/m) * sum(error)
            db = (1/m) * np.sum(error)
            
            # 4. Update Parameters (Gradient Descent Step)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 5. Calculate and Store Cost (MSE as required by assignment)
            mse_cost = np.mean(error**2)
            regularization_penalty = (self.lambda_param / m) * np.sum(self.weights**2)
            total_cost = mse_cost + regularization_penalty
            
            self.cost_history.append(total_cost)
    
    def predict(self, X):
        """
        Make predictions on new data using learned weights and bias.
        """
        return X.dot(self.weights) + self.bias

# ==============================================================================
# 3. EVALUATION FUNCTIONS
# ==============================================================================

def calculate_mse(y_actual, y_predicted):
    """Calculates Mean Squared Error."""
    return np.mean((y_actual - y_predicted)**2)

def calculate_r2(y_actual, y_predicted):
    """Calculates R-squared (R2) Score."""
    ss_residual = np.sum((y_actual - y_predicted)**2)
    ss_total = np.sum((y_actual - np.mean(y_actual))**2)
    # Handle SS_total = 0 case
    if ss_total == 0:
        return 0.0
    return 1 - (ss_residual / ss_total)

# ==============================================================================
# 4. VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_learning_curve(cost_history, title, filepath):
    """Plots the cost history vs. iterations (MSE on y-axis as required)."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history, color='blue', linewidth=2)
    plt.title(title)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost (MSE)')  # Changed from MSE/2 to MSE as per assignment
    plt.grid(True, alpha=0.3)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curve saved: {filepath}")

def plot_actual_vs_predicted(y_actual, y_predicted, title, filepath):
    """Creates a scatter plot of actual vs. predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_actual, y_predicted, alpha=0.6, color='blue', label='Predicted Points')
    
    # Plot the 45-degree perfect prediction line
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Perfect Fit Line')
    
    plt.title(title)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Actual vs Predicted plot saved: {filepath}")

def plot_actual_vs_predicted_with_ideal(y_actual, y_predicted, title, filepath):
    """Creates a comprehensive scatter plot of actual vs. predicted values with ideal line analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Standard Actual vs Predicted
    ax1.scatter(y_actual, y_predicted, alpha=0.6, color='blue', label='Predicted Points')
    
    # Plot the 45-degree perfect prediction line
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Perfect Fit Line (y=x)')
    
    ax1.set_title('Actual vs Predicted Values')
    ax1.set_xlabel('Actual Prices')
    ax1.set_ylabel('Predicted Prices')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Residuals with respect to ideal line
    residuals = y_predicted.flatten() - y_actual.flatten()
    ax2.scatter(y_actual, residuals, alpha=0.6, color='green', label='Residuals')
    
    # Plot zero line (ideal case where residuals = 0)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Ideal Line (Residual=0)')
    
    # Add statistical lines
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax2.axhline(y=mean_residual, color='orange', linestyle='-', alpha=0.7, label=f'Mean Residual ({mean_residual:.3f})')
    ax2.axhline(y=mean_residual + std_residual, color='orange', linestyle=':', alpha=0.5, label=f'±1 Std ({std_residual:.3f})')
    ax2.axhline(y=mean_residual - std_residual, color='orange', linestyle=':', alpha=0.5)
    
    ax2.set_title('Residual Analysis (Predicted - Actual)')
    ax2.set_xlabel('Actual Prices')
    ax2.set_ylabel('Residuals (Predicted - Actual)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Actual vs Predicted with Ideal analysis saved: {filepath}")

def plot_ideal_performance_analysis(y_actual, y_predicted, title, filepath):
    """Creates a detailed analysis of model performance against ideal predictions."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted with Perfect Line
    ax1.scatter(y_actual, y_predicted, alpha=0.6, color='blue', s=20)
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Actual vs Predicted Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals Plot
    residuals = y_predicted.flatten() - y_actual.flatten()
    ax2.scatter(y_actual, residuals, alpha=0.6, color='green', s=20)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Ideal (Residual=0)')
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram of Residuals
    ax3.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Ideal (Residual=0)')
    ax3.axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2, 
                label=f'Mean ({np.mean(residuals):.3f})')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Metrics Comparison
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(mse)
    
    metrics = ['MSE', 'MAE', 'RMSE', 'Mean Residual', 'Std Residual']
    values = [mse, mae, rmse, np.mean(residuals), np.std(residuals)]
    ideal_values = [0, 0, 0, 0, 0]  # Ideal case: all metrics should be 0
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x_pos - width/2, values, width, label='Actual Model', color='skyblue', alpha=0.8)
    ax4.bar(x_pos + width/2, ideal_values, width, label='Ideal Model', color='red', alpha=0.8)
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Values')
    ax4.set_title('Model Performance vs Ideal')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ideal performance analysis saved: {filepath}")

# ==============================================================================
# 5. MAIN EXECUTION BLOCK / DRIVER CODE
# ==============================================================================

if __name__ == '__main__':
    # Define hyperparameters
    LEARNING_RATE = 0.01
    N_ITERATIONS = 1000
    ROLL_NO = "G25AIT1119"

    print("=== LINEAR REGRESSION IMPLEMENTATION (Assignment) ===")
    print(f"Student: Praveen Kumar | Roll No: {ROLL_NO}")
    print("="*60)

    # ==================================================================
    # PART (A): IMPLEMENTATION OF LINEAR REGRESSION MODEL
    # ==================================================================
    
    print("\nPART (A): Data Loading and Preprocessing")
    
    # 1. Dataset Preparation
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Dataset loaded: {X_train.shape[0]} training, {X_test.shape[0]} testing samples")
    
    # 2. Data Preprocessing (Standardization)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    print(f"Features standardized: {X_train_scaled.shape[1]} features")
    print(f"Training set mean: {np.mean(X_train_scaled):.6f}, std: {np.std(X_train_scaled):.6f}")
    
    # 3. Model Implementation and Training
    print("\nPART (A): Training Linear Regression Model")
    model = LinearRegression(learning_rate=LEARNING_RATE, n_iterations=N_ITERATIONS)
    model.fit(X_train_scaled, y_train)
    print(f"Model trained with α={LEARNING_RATE}, iterations={N_ITERATIONS}")
    
    # ==================================================================
    # PART (B): EVALUATION AND VISUALIZATION 
    # ==================================================================
    
    print("\nPART (B): Model Evaluation")
    
    # 1. Model Evaluation
    y_pred = model.predict(X_test_scaled)
    mse = calculate_mse(y_test, y_pred)
    r2 = calculate_r2(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²) Score: {r2:.4f}")
    
    # 2. Visualization
    print("\nPART (B): Generating Visualizations")
    
    # Learning Curve
    plot_learning_curve(
        model.cost_history,
        'Learning Curve: Linear Regression Convergence',
        f'{ROLL_NO}_learning_curve.png'
    )
    
    # Actual vs Predicted Plot
    plot_actual_vs_predicted(
        y_test, y_pred,
        f'Actual vs Predicted Values | MSE={mse:.4f}, R²={r2:.4f}',
        f'{ROLL_NO}_actual_vs_predicted.png'
    )
    
    # Actual vs Predicted with Ideal Line Analysis
    plot_actual_vs_predicted_with_ideal(
        y_test, y_pred,
        f'Model Performance Analysis | MSE={mse:.4f}, R²={r2:.4f}',
        f'{ROLL_NO}_actual_vs_predicted_with_ideal.png'
    )
    
    # Comprehensive Ideal Performance Analysis
    plot_ideal_performance_analysis(
        y_test, y_pred,
        f'Comprehensive Performance Analysis vs Ideal Model | Roll No: {ROLL_NO}',
        f'{ROLL_NO}_ideal_performance_analysis.png'
    )
    
    # ==================================================================
    # BONUS CHALLENGE: L2 REGULARIZATION & LEARNING RATE EXPERIMENTS
    # ==================================================================
    
    print("\nBONUS: L2 Regularization (Ridge Regression)")
    
    # Ridge Regression Implementation
    ridge_model = LinearRegression(
        learning_rate=LEARNING_RATE, 
        n_iterations=N_ITERATIONS, 
        lambda_param=1.0
    )
    ridge_model.fit(X_train_scaled, y_train)
    
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    r2_ridge = calculate_r2(y_test, y_pred_ridge)
    
    print(f"Ridge Regression R² Score: {r2_ridge:.4f}")
    print(f"Standard weights norm: {np.linalg.norm(model.weights):.4f}")
    print(f"Ridge weights norm: {np.linalg.norm(ridge_model.weights):.4f}")
    
    print("\nBONUS: Learning Rate Experiments")
    
    # Different Learning Rates
    lr_high = LinearRegression(learning_rate=1.0, n_iterations=100)  # Too high
    lr_low = LinearRegression(learning_rate=0.0001, n_iterations=N_ITERATIONS)  # Too low
    
    lr_high.fit(X_train_scaled, y_train)
    lr_low.fit(X_train_scaled, y_train)
    
    # Learning Rate Comparison Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(len(lr_high.cost_history)), lr_high.cost_history, 
             'r-', label='Too High (α=1.0)', linewidth=2)
    plt.title('Learning Rate: Too High (Divergence)')
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(len(lr_low.cost_history)), lr_low.cost_history, 
             'orange', label='Too Low (α=0.0001)', linewidth=2)
    plt.plot(range(len(model.cost_history)), model.cost_history, 
             'blue', label=f'Optimal (α={LEARNING_RATE})', linewidth=2)
    plt.title('Learning Rate Comparison: Too Low vs Optimal')
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ROLL_NO}_learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning rate comparison saved: {ROLL_NO}_learning_rate_comparison.png")
    
    # Final Summary
    print("\n" + "="*60)
    print("ASSIGNMENT COMPLETION SUMMARY")
    print("="*60)
    print("Part (A): LinearRegression class implemented with required methods")
    print("Part (B): Model evaluation and visualization completed")
    print("BONUS: L2 regularization and learning rate experiments included")
    print("NEW: Enhanced visualizations with ideal performance analysis")
    print("Files generated:")
    print(f"  - {ROLL_NO}_learning_curve.png")
    print(f"  - {ROLL_NO}_actual_vs_predicted.png")
    print(f"  - {ROLL_NO}_actual_vs_predicted_with_ideal.png")
    print(f"  - {ROLL_NO}_ideal_performance_analysis.png")
    print(f"  - {ROLL_NO}_learning_rate_comparison.png")
    print(f"Final Model Performance: MSE={mse:.4f}, R²={r2:.4f}")
    print("="*60)
    

