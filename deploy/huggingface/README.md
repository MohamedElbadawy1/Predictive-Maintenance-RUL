---
title: Predictive Maintenance RUL
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Predictive Maintenance — Remaining Useful Life (RUL)

Predicts how many operating cycles remain before a jet engine fails, from
NASA C-MAPSS FD004 sensor readings, served by an LSTM+GRU ensemble.

Upload a CSV of raw engine cycle readings (`unit_number`, `time_in_cycles`,
`operational_setting_1-3`, `sensor_1-21` — full history per engine, not just
the latest cycle) and get each engine's predicted remaining useful life.

Full source, training pipeline, and 22 sprints of documented experiments:
https://github.com/MohamedElbadawy1/Predictive-Maintenance-RUL

This Space's Dockerfile serves a pre-trained LSTM+GRU ensemble baked into
the image at build time (a free Space's storage is ephemeral, so training
happens before deploying, not inside the running Space) — see that repo's
`README.md` → "Deploying to Hugging Face Spaces" for exactly how this
Space's files were produced from it.
