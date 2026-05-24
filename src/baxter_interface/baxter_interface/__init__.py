# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from .analog_io import AnalogIO as AnalogIO
from .base import BaxterInterface as BaxterInterface
from .base import BaxterNode as BaxterNode
from .camera import CameraController as CameraController
from .digital_io import DigitalIO as DigitalIO
from .gripper import Gripper as Gripper
from .head import Head as Head
from .limb import Limb as Limb
from .navigator import Navigator as Navigator
from .robot_enable import RobotEnable as RobotEnable
from .robust_controller import RobustController as RobustController
from .settings import (
    CHECK_VERSION as CHECK_VERSION,
)
from .settings import (
    HEAD_PAN_ANGLE_TOLERANCE as HEAD_PAN_ANGLE_TOLERANCE,
)
from .settings import (
    JOINT_ANGLE_TOLERANCE as JOINT_ANGLE_TOLERANCE,
)
from .settings import (
    SDK_VERSION as SDK_VERSION,
)
from .settings import (
    VERSIONS_SDK2GRIPPER as VERSIONS_SDK2GRIPPER,
)
from .settings import (
    VERSIONS_SDK2ROBOT as VERSIONS_SDK2ROBOT,
)
