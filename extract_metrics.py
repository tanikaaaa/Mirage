import re
import csv

# ✅ correct path (based on your screenshot)
log_file = "saved_logs/_Apr.26_12.13.49/log.txt"
output_file = "metrics.csv"

iteration = None
accuracy = None
asr_list = []

data = []

with open(log_file, "r") as f:
    for line in f:

        # Get iteration
        if "Test Global model in iteration" in line:
            match = re.search(r"iteration (\d+)", line)
            if match:
                iteration = int(match.group(1))

        # Get accuracy
        if "Acc" in line:
            match = re.search(r"Acc ([0-9.]+)%", line)
            if match:
                accuracy = float(match.group(1))

        # Get ASR
        if "ASR" in line:
            match = re.search(r"ASR ([0-9.]+)%", line)
            if match:
                asr_list.append(float(match.group(1)))

        # You have 3 attackers → collect 3 ASR values
        if len(asr_list) == 3:
            avg_asr = sum(asr_list) / 3
            data.append([iteration, accuracy, avg_asr])
            asr_list = []

# Save to CSV
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Accuracy", "Avg_ASR"])
    writer.writerows(data)

print("DONE → metrics.csv created")