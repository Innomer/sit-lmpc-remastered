#!/bin/bash
source /opt/ros/jazzy/setup.bash && source ~/sim_ws/install/setup.bash && source ~/sim_ws/.venv/bin/activate
export ROS_DOMAIN_ID=77
cd ~/sim_ws/src/sit-lmpc-remastered
for name in known-ppwarm bnn-ppwarm bnn-nochance-ppwarm; do
  rm -rf results/Spielberg/${name}25
  timeout 14400 sit-racecar --config configs/f1tenth-ros-${name}.yaml --seed 1 --max-laps 25 --save-dir results/Spielberg/${name}25 > results/logs/${name}25.log 2>&1
  echo "EXIT=$?" >> results/logs/${name}25.log
  echo "[series] ${name} finished"
done
echo "[series] all done"
