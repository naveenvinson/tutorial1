import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the data
df = pd.read_csv('Advertising.csv', index_col=0)

# Prepare data
X = df[['TV', 'Radio', 'Newspaper']]
X = sm.add_constant(X)  # Add constant for intercept
y = df['Sales']

# Fit the model
model = sm.OLS(y, X).fit()

# Extract R-squared and F-statistic
r_squared = model.rsquared
f_statistic = model.fvalue
f_pvalue = model.f_pvalue

# Print results
print(f"R-squared: {r_squared:.4f}")
print(f"F-statistic: {f_statistic:.4f}")