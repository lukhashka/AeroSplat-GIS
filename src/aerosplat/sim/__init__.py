"""Software-in-the-loop simulation with ArduPilot SITL (docs/SIMULATION_AND_DELIVERY.md §4).

`sitl` launches the pinned ArduCopter SITL build (WSL2 or Linux); `vehicle` is the pymavlink
client tests use to fly it and to read the simulator's true state next to the EKF estimate.
pymavlink is optional: `pip install -e .[sim]`.
"""
