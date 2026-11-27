# Car dry runs (ROS sim, car backend, real time)

The car configs run against the f1tenth_gym_ros bridge on the Levine sim map (`configs/car/dryrun-levine-*.yaml`,
bridge `configs/car/ros-bridge-dryrun-levine.yaml`): same code path as on the car, simulated PF pose, wheel speed,
IMU and LiDAR. Only small files are kept here (events, summaries, configs, logs); bags and models stay local.

| Run | What it showed |
|---|---|
| `dry-bicycle-oldstop` | an off-track stop ended the whole run (`speed_too_low`); fixed: a stop after moving is a failed attempt and the car waits for go |
| `dry-bicycle` | launch, warm-up laps, off-track stop, operator restart, Ctrl-C all work; the restart launched at 7 m/s (the failed lap's line speed); fixed: `retry_speed` 1.5 |
| `dry-v11-stall` | the lap checkpoint waited for value training: control loop frozen 2.8 s at 7 m/s, car into the wall; a stale obstacle flag stopped the next launch |
| `dry-v11-ckclash` | delayed checkpoint clashed with a failure checkpoint (`FileExistsError`); 0.34 s pause during the first training |
| `dry-v11-firstcompile` | first learned step compiled the planner for 26 s while driving |
| `dry-v11-switch` | with the planner precompiled at startup: still a 0.64 s pause at the first learned step; the watchdog's stop braked hard and spun the car |
| `dry-v11-compilelog` | JAX compile log; a 0.26 s pause during training made the warm-up steering hit full lock at 7 m/s |
| `dry-v11` | final: stop after every lap, train and save while standing, go. 5 warm-up + learned lap clean, 0 watchdog stops, planning max 62 ms, off-track stop and restart work; `bag_check.json` from its bag |

`bag_check.json` (dry-v11): v11 0.18 m / bicycle 0.10 m median position error at 0.5 s. In the sim the bicycle model is
the simulator's own model, so it is exact here; the comparison means something only on a real car bag.
