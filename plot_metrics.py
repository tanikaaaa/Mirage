import pandas as pd
import matplotlib.pyplot as plt

# Read your CSV
df = pd.read_csv("metrics.csv")

# ---- GRAPH 1: Accuracy ----
plt.figure()
plt.plot(df["Iteration"], df["Accuracy"])
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Iteration")
plt.grid()
plt.savefig("accuracy.pdf")   # saves as PDF
plt.show()

# ---- GRAPH 2: ASR ----
plt.figure()
plt.plot(df["Iteration"], df["Avg_ASR"])
plt.xlabel("Iteration")
plt.ylabel("ASR (%)")
plt.title("ASR vs Iteration")
plt.grid()
plt.savefig("asr.pdf")   # saves as PDF
plt.show()
plt.show()
print("GRAPH DONE")