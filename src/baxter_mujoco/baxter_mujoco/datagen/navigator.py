"""Press Baxter's navigator buttons as the recorder reads them (RethoughtAI recorder, digital_io trigger).

The sim has no navigator, so these topics carry only what this publishes: every button released at
20 Hz, and a press as a 0.2 s high. The recorder counts a 0 to 1 transition, at most one per 0.5 s.
"""

import threading
import time

from baxter_core_msgs.msg import DigitalIOState

BUTTONS = {
    'record': '/robot/digital_io/left_button_show/state',
    'save': '/robot/digital_io/left_button_ok/state',
    'discard': '/robot/digital_io/left_button_back/state',
}
PRESS, GAP = 0.2, 0.7  # s held down; s after release (the recorder's lockout is 0.5 s)


class Navigator:
    def __init__(self, node, buttons=BUTTONS):
        self._pubs = {name: node.create_publisher(DigitalIOState, topic, 10) for name, topic in buttons.items()}
        self._down = None
        threading.Thread(target=self._stream, daemon=True).start()

    def _stream(self):
        while True:
            for name, pub in self._pubs.items():
                pub.publish(DigitalIOState(state=int(name == self._down), is_input_only=True))
            time.sleep(0.05)

    def press(self, name):
        self._down = name
        time.sleep(PRESS)
        self._down = None
        time.sleep(GAP)
