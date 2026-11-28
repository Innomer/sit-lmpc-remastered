# First car run: bicycle, 2 m/s cap (Levine, 2026-11-28)

`configs/car/levine-bicycle-slow.yaml` on the Jetson (SICK TiM, RealSense D435i IMU, PF on the GPU, bag recorded).

- 5 pure-pursuit warm-up laps, 32.2-32.5 s at 2.0 m/s; stop after every lap, training while standing, restart on go: all worked; 0 watchdog stops. IMU levelled 1.4 deg on the floor.
- Attempt 1 slowed from 2.0 to 0.9 m/s at s = 13 m while the command stayed 2.0 (the mux or motor did not follow; likely the deadman).
- Learned laps (attempts 7, 8) failed within 0.7 s: planner steps took 0.6-0.75 s with the full stack running (204 ms in jetson_check without it), so the car drove 0.4-0.5 m per decision on stale steering; heading error grew to 0.45-0.55 rad, then a "stopped" failure and a LiDAR obstacle stop.
- `bag_check.json`, v11 vs the bicycle model on these laps (349 starts), position error median / p90: 0.5 s 0.14 / 0.25 vs 0.15 / 0.27 m; 1.0 s 0.26 / 0.49 vs 0.38 / 0.71 m; 1.5 s 0.49 / 0.96 vs 0.74 / 1.57 m. Yaw rate: v11 better at 0.5 s (0.45 vs 0.74 rad/s), equal later. Speed: bicycle better (0.01 vs 0.05 m/s).

The bag and checkpoints stay on the Jetson.
