import time
import numpy as np

def match_timestamps(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    matching = np.zeros(len(timestamps1), dtype=int)
    j = 0  # Pointer for timestamps2

    for i in range(len(timestamps1)):
        while j < len(timestamps2) - 1 and abs(timestamps2[j + 1] - timestamps1[i]) < abs(timestamps2[j] - timestamps1[i]):
            j += 1
        matching[i] = j

    return matching

def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    timestamps = np.linspace(st_ts, fn_ts, int((fn_ts - st_ts) * fps))
    timestamps += np.random.randn(len(timestamps))
    timestamps = np.unique(np.sort(timestamps))
    return timestamps

def main():
    timestamps1 = make_timestamps(30, time.time() - 100, time.time() + 3600 * 2)
    timestamps2 = make_timestamps(60, time.time() + 200, time.time() + 3600 * 2.5)
    matching = match_timestamps(timestamps1, timestamps2)
    print("Matching indices:", matching)

if __name__ == '__main__':
    main()