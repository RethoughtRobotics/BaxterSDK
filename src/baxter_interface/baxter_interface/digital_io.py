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

import errno
import time

import baxter_dataflow
from baxter_core_msgs.msg import (
    DigitalIOState,
    DigitalOutputCommand,
)
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class DigitalIO(object):
    """
    Interface class for a simple Digital Input and/or Output on the
    Baxter robot

    Input
      - read input state
    Output
      - turn output On/Off
      - read current output state
    """

    def __init__(self, component_id: str, node: Node, timeout: float = 2.0, debounce: float = 0.0):
        """
        Constructor.

        @param component_id: unique id of the digital component
        @param timeout: seconds to wait for the first state; 0 does not wait, and state is None
            until it arrives (e.g. a simulator that publishes no IO)
        @param debounce: seconds a press must follow the last release to signal state_changed;
            Baxter's navigator buttons bounce on release (0.2 s is enough)
        """
        self._node = node
        self._id = component_id
        self._component_type = 'digital_io'
        self._is_output = False
        self._state = None
        self._debounce = float(debounce)
        self._released = -float('inf')  # when the state last went False
        self._pub_output = None
        self._callback_group = MutuallyExclusiveCallbackGroup()

        self.state_changed = baxter_dataflow.Signal()

        type_ns = '/robot/' + self._component_type
        topic_base = type_ns + '/' + self._id

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub_state = self._node.create_subscription(
            DigitalIOState,
            topic_base + '/state',
            self._on_io_state,
            qos,
            callback_group=self._callback_group,
        )

        self._command_topic, self._qos = type_ns + '/command', qos
        if timeout:
            baxter_dataflow.wait_for(
                self._node,
                lambda: self._state is not None,
                timeout=timeout,
                timeout_msg=f'Failed to get current digital_io state from {topic_base}',
            )

    def _on_io_state(self, msg):
        """
        Updates the internally stored state of the Digital Input/Output.
        """
        new_state = msg.state == DigitalIOState.PRESSED
        if self._state is None:
            self._is_output = not msg.is_input_only
        old_state = self._state
        self._state = new_state

        # trigger signal if changed, a press only once debounce has passed since the last release
        if old_state is not None and old_state != new_state:
            now = time.monotonic()
            if not new_state:
                self._released = now
            if new_state and now - self._released < self._debounce:
                return
            self.state_changed(new_state)

    @property
    def is_output(self):
        """
        Accessor to check if IO is capable of output.
        """
        return self._is_output

    @property
    def state(self):
        """
        Current state of the Digital Input/Output.
        """
        return self._state

    @state.setter
    def state(self, value):
        """
        Control the state of the Digital Output. (is_output must be True)

        @type value: bool
        @param value: new state to output {True, False}
        """
        self.set_output(value)

    def set_output(self, value, timeout=2.0):
        """
        Control the state of the Digital Output.

        Use this function for finer control over the wait_for timeout.

        @type value: bool
        @param value: new state {True, False} of the Output.
        @type timeout: float
        @param timeout: Seconds to wait for the io to reflect command.
                        If 0, just command once and return. [0]
        """
        if not self._is_output:
            raise IOError(errno.EACCES, f'Component is not an output [{self._component_type}: {self._id}]')
        if self._pub_output is None:
            self._pub_output = self._node.create_publisher(DigitalOutputCommand, self._command_topic, self._qos)
        cmd = DigitalOutputCommand()
        cmd.name = self._id
        cmd.value = value
        self._pub_output.publish(cmd)

        if not timeout == 0:
            baxter_dataflow.wait_for(
                self._node,
                test=lambda: self.state == value,
                timeout=timeout,
                rate=100,
                timeout_msg=(f'Failed to command digital io to: {value}'),
                body=lambda: self._pub_output.publish(cmd),
            )
