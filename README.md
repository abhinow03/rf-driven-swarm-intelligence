# LLM-Driven Semantic RF Analysis for Detection and Visualisation of UAV Swarm Communication

 
> **Team:** Aadhya S Shetty, Abhinav Waddinavar, Alisha Prakash, Sharva Chiradoni  
> **Guide:** Dr. Ashok Kumar Patil  


---

## Overview

Traditional RF-ML pipelines can classify signal types or localize individual drones — but they fail to interpret **swarm-level intent**. They can tell you *what* signals exist, not *what the swarm is doing*.

This project bridges that gap. We combine distributed RF sensing, multi-drone localization, graph-based swarm modeling, and an LLM reasoning layer to move from raw I/Q signal data to actionable, human-readable tactical assessments in real time.

**Example LLM output:**
```
"V-formation with increasing velocity and stable centroid suggests 
leader-follower attack pattern. Recommend alerting operator — confidence: HIGH."
```

---

## System Architecture

```
UAV RF Signals
     │
     ▼
[1] On-Drone DL Model (RF Fingerprinting)
     │  Extracts: AoA, TDoA, RSSI, Doppler, Bandwidth
     │  Output: 128-dim hardware fingerprint per drone
     ▼
[2] Base Station — Multilateration & Swarm Modeling
     │  Fuses features across drones → 3D position + velocity
     │  Swarm graph: Nodes = drones, Edges = spatial proximity
     │  GATv2 (spatial) + Transformer (temporal) → formation type,
     │  stability, approach rate, role differentiation
     ▼
[3] LLM Interpretation Layer
     │  Input: structured JSON from ML models
     │  Output: threat level, intent, recommended action, explanation
     ▼
[4] Visualisation (3D / AR Dashboard)
     │  Real-time emitter positions, swarm formation, behavior trends
```

---

## The Three-Layer Reasoning Model

| Layer | Question it answers | Features used |
|---|---|---|
| **Perception** | "What signals and objects exist?" | TDoA, AoA, Bandwidth, RSSI, Center freq, Doppler |
| **Kinematics** | "How are drones moving relative to each other?" | Position (x,y,z), velocity, acceleration, climb rate, timestamp |
| **Intent (LLM)** | "What does this collective motion *mean*?" | Structured JSON: formation type, stability, approach rate, rf_coordination_score, behavior_trend |

---

## Component Details

### 1. RF Fingerprinting Model (V4)

Extracts unique hardware fingerprints from raw I/Q signals to individually identify drones — even unseen ones.

- **Architecture:** Dual-branch
  - Time-domain: 1D ResNet
  - Frequency-domain: STFT + 2D ResNet
  - Fused via cross-attention
- **Output:** 128-dim L2-normalized embedding (not class probabilities)
- **Loss:** Supervised Contrastive (SupCon) — pulls same-device signals together, pushes others apart
- **Inference:** Cosine similarity on embeddings
- **Specs:** ~3.69M params training / ~1.48M inference · 4096-sample window

**Key results:**
- 74.3% assignment purity on simulated edge deployment (streaming unknown I/Q windows)
- Successfully detected and registered previously unseen emitters from held-out data
- Zero-shot distance accuracy: cosine similarity of **0.9856** on entirely unseen distances (20ft & 26ft) — trained only on 8ft & 14ft
- Peak separation margin (Sim Gap): **1.3440** — 159% improvement over baseline
- Within-device similarity: **0.7870** (above target threshold)

> **Hardware limit discovered:** Nanometer-level manufacturing variations within a single batch form a continuous "ring manifold" rather than discrete clusters. Two of four held-out USRP X310s were falsely merged — operating at the absolute boundary of physical detectability.

### 2. Multilateration & Tracking

- Measurements: Range, Azimuth, Elevation, Doppler, RSSI, TDOA
- Sensor fusion via **Extended Kalman Filter (EKF)**
- Outputs: 3D position estimates + velocity vectors

### 3. Swarm Behavior Model

- **Spatial modeling:** GATv2 (Graph Attention Network v2)
- **Temporal modeling:** Transformer encoder
- **Dataset:** ~19,600 sequences (augmented) · 7 formation types + transition cases · varying noise, speed, spread
- **Outputs:** Formation type, velocity, stability, approach rate

**Supported formations and their tactical meaning:**

| Formation | Tactical meaning |
|---|---|
| V-shape | Leader-follower coordinated movement |
| Encirclement / Converging | High threat — potential attack |
| Shield | Electromagnetic protection bubble |
| Diamond mesh | Resilience against anti-jamming |
| Column / Trail | Penetration pattern |
| Dispersed | Surveillance / area scanning |

### 4. LLM Interpretation Layer

Takes structured ML outputs and produces human-readable tactical intelligence.

- **Input:** JSON with formation type, stability, velocity, approach rate, rf_coordination_score, role_differentiation, behavior_trend
- **Pipeline:** Rule-based context building → prompt generation → LLM inference
- **Output (JSON):** threat level · intent · recommended action · explanation
- Low-temperature setup for deterministic responses
- Currently: prompt-engineered system with Groq API
- **In progress:** Fine-tuning Mistral 7B Instruct v0.3 with QLoRA (4-bit NF4 + LoRA adapters, r=16, α=32) on ~500 synthetic samples for domain adaptation, lower latency, and reduced prompt dependency

**Example input → output:**

```json
// Input to LLM
{
  "num_drones": 6,
  "formation_type": "encirclement",
  "formation_stability": 0.82,
  "centroid_velocity": 4.3,
  "approach_rate": -1.2,
  "rf_coordination_score": 0.76,
  "role_differentiation": true,
  "behavior_trend": "converging"
}
```

```
// LLM output
Threat Level: HIGH
Intent: APPROACH / ENCIRCLEMENT
Explanation: Increased RF coordination combined with adaptive encirclement
formation and base-oriented approach suggests coordinated surveillance or
attack intent rather than random flight.
Recommended action: ALERT OPERATOR — monitor next observation window.
```

---

## Datasets

### RF Fingerprinting — Iteration History

| Dataset | Purpose | Outcome |
|---|---|---|
| CS-SEI (5 UAVs) | Initial closed-set fingerprinting | 100% accuracy — but model memorized waveforms, not hardware fingerprints. Discarded. |
| DroneRF (5 brands) | Brand-level discrimination | Underfitting — high intra-class variance from distance/interference. Discarded. |
| **ORACLE / KRI-16** (Northeastern Univ.) | **Open-set hardware fingerprinting** | ✅ Current — 123 identical-model Wi-Fi radios, designed for hardware-imperfection-based SEI, supports open-set validation on unseen devices |

> ORACLE dataset: supplementary material for *"ORACLE: Optimized Radio Classification through Convolutional Neural Networks"* (IEEE INFOCOM 2019). [DOI: 10.1109/INFOCOM.2019.8737463](https://doi.org/10.1109/INFOCOM.2019.8737463)

### Swarm Behavior Model

- ~19,600 augmented sequences simulated in MATLAB
- 7 formation types + formation transition cases
- Varying noise, speed, spread, and drone count

---

## Tech Stack

```
Python · PyTorch · TensorFlow · MATLAB
Groq API / LLM APIs
ONNX (edge deployment target)
GATv2 · Transformer encoder · 1D ResNet · STFT + 2D ResNet
Extended Kalman Filter (EKF)
Supervised Contrastive Loss · QLoRA · PEFT · HuggingFace Transformers
```

---

## Current Progress

- [x] MATLAB simulation of full pipeline
- [x] RF fingerprinting model (V4) — dual-branch architecture with SupCon loss
- [x] Multilateration and EKF tracking
- [x] Swarm behavior model — GATv2 + Transformer on 19.6k sequences
- [x] Formation classification across 7 formation types
- [x] LLM interpretation layer (prompt-engineered, Groq API)
- [x] Structured JSON pipeline from ML models → LLM
- [x] Synthetic dataset generation pipeline (~500 samples for fine-tuning)
- [ ] Mistral 7B fine-tuning with QLoRA — **in progress**
- [ ] Dataset enhancement and model optimization
- [ ] Full system integration
- [ ] 3D / AR visualisation dashboard

---

## Roadmap (Phase 3 & 4)

- **Domain-adapted LLM** — fine-tuned Mistral 7B for RF/swarm domain, improved reasoning consistency, reduced token usage
- **Dataset enhancement** — broader formation diversity, real-world RF noise
- **Model optimization** — ONNX export for edge deployment, inference latency reduction
- **Full system integration** — end-to-end pipeline from raw I/Q to operator dashboard
- **3D / AR visualisation** — real-time emitter positions, swarm formation overlay, behavior trend display

---

