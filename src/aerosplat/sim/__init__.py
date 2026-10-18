"""Software-in-the-loop simulation with ArduPilot SITL (docs/SIMULATION_AND_DELIVERY.md §4).

`sitl` launches the pinned ArduCopter SITL build (WSL2 or Linux); `vehicle` is the pymavlink
client tests use to fly it and to read the simulator's true state next to the EKF estimate.
pymavlink is optional: `pip install -e .[sim]`.

The L2 runner (SIM-1, §4.10) flies missions exported by `aerosplat.mission`:

    scenario     the scenario file (§4.5)
    params       the pinned build's parameter table (configs/ardupilot/<tag>.params.json)
    faults       abstract faults -> SIM_* parameters; what the .parm makes of each trigger
    mavmission   mission / fence / rally upload over the mission protocol, read back
    runner       preflight -> SITL -> upload -> flight -> oracles -> report
    record       truth, estimate and events on the simulation clock
    oracles      §4.6 on the TRUE trajectory against the twin
    report       report.html / report.json next to the export
"""
