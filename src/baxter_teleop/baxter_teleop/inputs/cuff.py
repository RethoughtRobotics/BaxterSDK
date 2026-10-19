"""Baxter's wrist cuff: squeezing it puts the arm in zero-g for guiding by hand, and its two buttons
drive the gripper (dash closes, circle opens), as baxter_examples gripper_cuff_control does for
kinesthetic demonstrations."""

import baxter_interface


class Cuff:
    """One arm's cuff, read through baxter_interface.DigitalIO without waiting for the robot (the
    sim publishes no IO: the cuff then reads released and its buttons never press)."""

    def __init__(self, node, arm):
        self._squeeze = baxter_interface.DigitalIO(f'{arm}_lower_cuff', node, timeout=0)
        self._close = baxter_interface.DigitalIO(f'{arm}_upper_button', node, timeout=0)  # dash
        self._open = baxter_interface.DigitalIO(f'{arm}_lower_button', node, timeout=0)  # circle
        # Bound methods: Signal holds plain functions weakly.
        self._close.state_changed.connect(self._on_close)
        self._open.state_changed.connect(self._on_open)
        self._pressed = None  # 'close' or 'open', pressed since the last take_button()

    @property
    def squeezed(self):
        return bool(self._squeeze.state)

    def take_button(self):
        """The gripper button pressed since the last call ('close', 'open'), or None."""
        pressed, self._pressed = self._pressed, None
        return pressed

    def _on_close(self, state):
        if state:
            self._pressed = 'close'

    def _on_open(self, state):
        if state:
            self._pressed = 'open'
