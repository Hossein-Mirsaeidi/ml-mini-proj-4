#!/usr/bin/env bash
# Source (do not execute) this after the venv is created to enable the TensorFlow GPU:
#   source venv/bin/activate && source setup_gpu_env.sh
# It puts the nvidia-*-cu12 pip-wheel libraries on LD_LIBRARY_PATH (TF 2.21 needs this)
# and clears the PYTHONPATH that ROS injects into interactive shells.
SP="$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)"
if [ -n "$SP" ] && [ -d "$SP/nvidia" ]; then
  NVLIBS="$(find "$SP/nvidia" -name '*.so*' -printf '%h\n' | sort -u | tr '\n' ':')"
  export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=1
echo "GPU env ready (LD_LIBRARY_PATH set, PYTHONPATH cleared)."
