# Adaptive Physics-Informed System Modeling with Control (APSMC)

**Code and Data for the Supporting Paper**  
**Title**: *Adaptive Physics-Informed System Modeling with Control for Nonlinear Structural System Estimation*  
📌 This repository provides the official Python implementation and datasets supporting the APSMC framework proposed in our paper.  
📎 Paper: [https://doi.org/10.48550/arXiv.2505.06525](https://doi.org/10.48550/arXiv.2505.06525)  
🔗 GitHub Author Page: [https://github.com/Chen861368](https://github.com/Chen861368)

---

## 📄 Paper Summary

This paper introduces the **Adaptive Physics-Informed System Modeling with Control (APSMC)** framework, which integrates stochastic subspace theory, Kalman-based optimal estimation, and physics-constrained proximal gradient optimization. APSMC formulates nonlinear dynamics into time-varying state-space models, allowing real-time system identification from sparse and noisy measurements.

The framework is validated across:
- A **Duffing oscillator** under periodic forcing
- A **multi-degree-of-freedom frame** subjected to seismic excitation
- A **scaled bridge model** tested experimentally

It achieves low error (minimum NMSE of 0.398%) in predicting long-horizon responses from short initial segments, making it suitable for digital twin modeling, structural health monitoring, and real-time control.

---

## 📁 Repository Overview

<p align="center">
  <img src="04ed28aa-7e34-4bf3-a471-442589daf6ce.png" alt="APSMC Framework Overview" width="650"/>
</p>

### 📂 File Descriptions

| File Name            | Description |
|----------------------|-------------|
| `APSMC_Duffing.py`   | APSMC implementation for Duffing oscillator under harmonic load |
| `Duffing_system.py`  | Defines nonlinear Duffing dynamics and simulation setup |
| `APSMC_seismic.py`   | APSMC implementation for seismic response estimation |
| `absAccel.xlsx`      | Simulated absolute acceleration response of the frame structure |
| `ground_motion.xlsx` | Northridge ground motion excitation data (used in seismic test) |
| `load_data.npy`      | External load input (e.g., forcing function in seismic case) |
| `state_data.npy`     | Ground truth state trajectories for validation |

---

## ▶️ How to Run

1. **Ensure Python 3.8+** is installed with required packages.
2. To reproduce Duffing oscillator results, run:
   ```bash
   python APSMC_Duffing.py
