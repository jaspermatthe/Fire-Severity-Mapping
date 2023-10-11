import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file into a DataFrame
file_path = 'cbi_dnbr_dnbr_plus_vector_table.xlsx'
df = pd.read_excel(file_path)

# Extract the 'Cbi' and 'dnbr_plus_LS71' columns and remove rows with NaN values
df = df.dropna(subset=['Cbi', 'dnbr_plus_LS71'])
Cbi = df['Cbi']
dnbr_plus_LS71 = df['dnbr_plus_LS71']

# Check if there are enough data points after removing NaN values
if len(Cbi) < 2:
    print("Insufficient data points for regression.")
else:
    # Perform cubic regression
    model = np.poly1d(np.polyfit(Cbi, dnbr_plus_LS71, 3))
    polyline = np.linspace(0, 3, 50)
    plt.scatter(Cbi, dnbr_plus_LS71, alpha=0.5, color='b', label='Data')
    plt.plot(polyline, model(polyline), color='r')
    coefficients = model.c
    print(coefficients)

    # Calculate R-squared
    yhat = model(Cbi)
    ybar = np.sum(dnbr_plus_LS71) / len(dnbr_plus_LS71)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((dnbr_plus_LS71 - ybar) ** 2)
    rsquared = ssreg / sstot
    print(rsquared)

    # Create a scatter plot
    # plt.figure(figsize=(8, 6))

    plt.xlabel('CBI', fontsize=12)
    plt.ylabel('dNBR+', fontsize=12)

    # Add the cubic regression equation and R-squared to the title with LaTeX-style formatting
    equation_text = fr'$dNBR+ = {coefficients[3]:.3f}\cdot CBI^3 + {coefficients[2]:.3f}\cdot CBI^2 + {coefficients[1]:.3f}\cdot CBI + {coefficients[0]:.3f}$' + '\n' + fr'$R^2 = {rsquared:.2f}$'
    plt.title(equation_text, fontsize=14, loc='center')

    # Save the plot as a PNG file
    plt.savefig('cubic_regression_plot_dNBR+_CBI.png')

    # Show the plot (optional)
    # plt.show()
