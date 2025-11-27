import matplotlib.pyplot as plt
import numpy as np

nodes = np.array([100, 200, 300, 450, 600, 900, 1200, 1500])
time = np.array([844.5, 911.59, 926.33, 1332.72, 3324.97, 4526.14, 5527.31, 7587.59])
epsilon = np.array([0.6284, 0.5773, 0.5861, 0.5872, 0.5921, 0.5927, 0.5963, 0.5989])
error_pct = np.array([4.73, 3.78, 2.32, 2.13, 1.32, 1.22, 0.62, 0.18])

plt.figure(figsize=(6, 4))
plt.plot(nodes, error_pct, 'o-', color='tab:blue', label="Error (%)")
plt.xlabel("Number of Nodes")
plt.ylabel("Error in ε (%)")
plt.title("Epsilon Estimation Error vs Node Count")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()