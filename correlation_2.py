import matplotlib.pyplot as plt
import numpy as np

# Correlation values for visualization
variables = [
    "Dataset Size", "Stop-Word Removal", "TF-IDF", "N-grams",
    "Ensemble Methods", "Cross-Validation"
]
correlations = [0.85, -0.50, 0.88, 0.83, 0.90, 0.92]

# Create a bar graph to visualize the correlations
plt.figure(figsize=(10, 6))
bars = plt.bar(variables, correlations, color=['skyblue', 'orange', 'green', 'purple', 'red', 'blue'])

# Annotate bars with their values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}",
             ha='center', va='bottom', fontsize=10)

# Graph formatting
plt.title("Correlation Analysis of Independent and Dependent Variables", fontsize=14)
plt.ylabel("Correlation Coefficient (r)", fontsize=12)
plt.xlabel("Independent Variables", fontsize=12)
plt.ylim(-0.6, 1.0)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xticks(rotation=20, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the graph
plt.show()