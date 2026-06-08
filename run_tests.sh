#!/usr/bin/env bash
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

# Unset toàn bộ ROS2 environment để tránh plugin conflict
unset ROS_DISTRO
unset ROS_VERSION  
unset ROS_PYTHON_VERSION
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH

# Xóa ROS2 khỏi PYTHONPATH
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v '/opt/ros' | tr '\n' ':' | sed 's/:$//')

# Chạy pytest
exec pdm run pytest "$@"
