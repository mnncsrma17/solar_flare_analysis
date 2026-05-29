import pandas as pd
from ordpy import complexity_entropy, maximum_complexity_entropy, minimum_complexity_entropy
import numpy as np
import matplotlib.pyplot as plt

file_1 = "go1520140329_timeseries.csv"

df_1 = pd.read_csv(file_1)

signal_norm_1 = df_1["flux_norm"].values
time_1 = df_1["time"].values

dx = 3
tau = 1

dt = np.mean(np.diff(time_1))
window_list = []
for n in range(1,37):
    window_list.append(int(1440//(10*n*dt)))

# =====================================================
# 2. SEGMENT FUNCTION
# =====================================================

def segment_ch(series, window, dx=3, tau=1):

    pe_values = []
    sc_values = []

    n_segments = len(series) // window

    for i in range(n_segments):

        start = i * window
        end   = start + window

        segment = series[start:end]

        pe, sc = complexity_entropy(segment, dx=dx, taux=tau, tauy=tau)

        pe_values.append(pe)
        sc_values.append(sc)

    return np.array(pe_values), np.array(sc_values)

# =====================================================
# 3. COMPUTE PE & SC FOR ALL WINDOWS
# =====================================================

results = {}

for window_pe in window_list:

    pe_seg, sc_seg = segment_ch(signal_norm_1, window_pe, dx, tau)

    time_pe = (np.arange(len(pe_seg)) + 0.5) * window_pe * dt

    results[window_pe] = (time_pe, pe_seg, sc_seg)

# =====================================================
# 4. AUTOMATIC DETECTION REGION
# =====================================================

H_noise_list = []
H_peak_list = []
delta_H_list = []
C_noise_list = []
C_peak_list = []
delta_C_list = []

for window_pe in window_list:

    time_pe, pe_seg, sc_seg = results[window_pe]

    # =====================================================
    # FIND SEGMENT CLOSEST TO FLARE PEAK
    # =====================================================

    peak_index = np.argmin(np.abs(time_pe - 696.645))

    # entropy and complexity at flare peak
    H_peak = pe_seg[peak_index]
    C_peak = sc_seg[peak_index]

    # indices excluding the peak
    noise_indices = np.delete(np.arange(len(pe_seg)), peak_index)

    # =====================================================
    # NOISE ESTIMATES
    # =====================================================

    H_noise = np.mean(pe_seg[noise_indices])
    H_noise_list.append(H_noise)

    C_noise = np.mean(sc_seg[noise_indices])
    C_noise_list.append(C_noise)

    # =====================================================
    # PEAK VALUES
    # =====================================================

    H_peak_list.append(H_peak)
    C_peak_list.append(C_peak)


# =====================================================
# 5. SCALING ANALYSIS
# =====================================================

delta_t = np.array(window_list) * dt

fig, ax = plt.subplots(2, 1, figsize=(12,10), dpi=600, sharex=True)

# --- H_min ---
ax[0].plot(delta_t, H_peak_list, 'o-', lw=3, color='k')
ax[0].set_ylabel(r"$H_{\mathrm{peak}}$", fontsize=18)
ax[0].tick_params(axis='both', labelsize=16, direction='in')

# --- C_min ---
ax[1].plot(delta_t, C_peak_list, 'o-', lw=3, color='k')
ax[1].set_xlabel(r"$\Delta t$", fontsize=18)
ax[1].set_ylabel(r"$C_{\mathrm{peak}}$", fontsize=18)
ax[1].tick_params(axis='both', labelsize=16, direction='in')

plt.tight_layout()
plt.show()

# =====================================================
# CH PLANE
# =====================================================

plt.figure(figsize=(12,8), dpi=600)

hmax, cmax = maximum_complexity_entropy(dx=dx).T
hmin, cmin = minimum_complexity_entropy(dx=dx).T

plt.plot(hmax, cmax, linestyle='--', color='gray', lw=2.5)
plt.plot(hmin, cmin, linestyle='-.', color='gray', lw=2.5)

for i,window_pe in enumerate(window_list):

    time_pe, pe_seg, sc_seg = results[window_pe]

    plt.scatter(pe_seg, sc_seg,
                color='red',
                marker='o',
                s=300,
                alpha=0.2)

plt.xlabel(r"$H_{\mathrm{peak}}$", fontsize=18)
plt.ylabel(r"$C_{\mathrm{peak}}$", fontsize=18)

plt.tick_params(axis='both', labelsize=16, direction='in', pad=10)

plt.show()

