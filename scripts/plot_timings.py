import matplotlib.pyplot as plt

file_path = "/tmp/timing_log.txt"

time_wbc = []
time_total = []

with open(file_path, "r") as f:
    next(f)  # skip header

    for line in f:
        line = line.strip()
        if not line:
            continue

        wbc, total = map(float, line.split(","))
        time_wbc.append(wbc)
        time_total.append(total)

cycles = range(len(time_wbc))

avg_wbc = sum(time_wbc) / len(time_wbc)
avg_total = sum(time_total) / len(time_total)

plt.figure(figsize=(10, 5))

plt.plot(cycles, time_wbc, label=f"WBC (avg = {avg_wbc:.1f} µs)")
plt.plot(cycles, time_total, label=f"Total (avg = {avg_total:.1f} µs)")

plt.axhline(avg_wbc, linestyle="--")
plt.axhline(avg_total, linestyle="--")

plt.xlabel("Control cycle")
plt.ylabel("Execution time [µs]")
plt.title("WBC and Total Execution Time")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()