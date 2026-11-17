#!/usr/bin/env bash
# Quest 2 teleop launcher: check adb, keep the headset awake, run quest_teleop.py.
#
# Usage:
#   ./run_quest_teleop.sh                 # position + rotation, scale 2.0, recording on, controller
#                                         #   joint_impedance -> pick_place/datasets/panda_pick_place_joint_impedance
#   ./run_quest_teleop.sh --controller joint_position   # other arm controller (see ../controllers/)
#                                         #   -> pick_place/datasets/panda_pick_place_joint_position
#   ./run_quest_teleop.sh --scale 1.0     # override the scale
#   ./run_quest_teleop.sh --no-record     # teleop only, nothing saved
#   ./run_quest_teleop.sh --dataset test  # other dataset base name -> pick_place/datasets/test_joint_impedance
#   ./run_quest_teleop.sh --pos-only      # any other option is passed to quest_teleop.py
#
# Controls:
#   Right controller
#     Grip (RG, hold)      move the robot (clutch: release to reposition your hand)
#     Trigger (hold)       close the gripper, release to open
#     A                    --record only: save the current episode and reset the scene
#     B                    reset the scene (cube to a random position);
#                          --record: also discards the current episode
#   Left controller
#     Y, twice (2 s)       --record only: delete the last saved episode (undo a wrong A)
#     X (hold) + push      calibrate "forward": hold X, push the left controller ~20 cm
#                          toward the monitor, release X (saved to quest_calib.json)
#   Mouse (viewer)         rotate the view; control stays screen-relative
#                          (the view angle is read when the grip is pressed)
#
#   --record: recording starts at the first grip press after a reset.

set -euo pipefail

PYTHON="$HOME/anaconda3/envs/mujo/bin/python"  # conda env "mujo" (mujoco, mink, oculus_reader)
DEFAULT_ARGS=(--scale 2.0 --record --controller joint_impedance --dataset panda_pick_place)  # given first, so options on the command line override them
cd "$(dirname "$(realpath "$0")")"

# 1. adb connection
state=$(adb devices | awk 'NR > 1 && NF >= 2 {print $2; exit}')
case "$state" in
  device)
    ;;
  unauthorized)
    echo "Quest is 'unauthorized'. Put on the headset and press 'Always allow' on the USB debugging prompt."
    echo "No prompt? Check that Developer settings -> Link auto connection is OFF, then replug the cable."
    exit 1 ;;
  *)
    echo "No Quest found by adb. Check the USB cable, that the headset is on, and that"
    echo "Developer settings -> Link auto connection is OFF."
    exit 1 ;;
esac
echo "Quest connected."

# 2. keep the headset awake while it is not worn (resets on headset reboot)
adb shell am broadcast -a com.oculus.vrpowermanager.prox_close > /dev/null
echo "Proximity sensor disabled (headset stays awake)."

# 3. teleop
exec "$PYTHON" quest_teleop.py "${DEFAULT_ARGS[@]}" "$@"
