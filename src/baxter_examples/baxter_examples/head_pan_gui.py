#!/usr/bin/env python3
"""Tkinter slider that pans Baxter's head."""

import threading
import tkinter as tk

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

import baxter_interface


class HeadPanGUI:
    def __init__(self, node: Node, executor: MultiThreadedExecutor):
        self._node = node
        self._executor = executor

        # Init RobotEnable and Head BEFORE spinning: their constructors
        # call spin_once internally via wait_for.
        self._rs = baxter_interface.RobotEnable(node)
        self._rs.enable(node)
        self._head = baxter_interface.Head(node)

        # Now start the executor so publish() flushes through Zenoh.
        self._spin_thread = threading.Thread(target=executor.spin, daemon=True)
        self._spin_thread.start()

        self._root = tk.Tk()
        self._root.title('Head Pan')

        self._slider = tk.Scale(
            self._root,
            from_=-1.5,
            to=1.5,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=400,
            label='pan (rad)',
        )
        self._slider.set(self._head.pan())
        self._slider.pack(padx=10, pady=10)

        self._root.after(10, self._tick)
        self._root.protocol('WM_DELETE_WINDOW', self._shutdown)

    def _tick(self):
        self._head.set_pan(float(self._slider.get()), speed=0.3, timeout=0)
        self._root.after(10, self._tick)

    def _shutdown(self):
        self._root.destroy()
        rclpy.try_shutdown()

    def run(self):
        self._root.mainloop()


def main():
    rclpy.init()
    node = rclpy.create_node('head_pan_gui')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    gui = HeadPanGUI(node, executor)
    print('Head pan GUI ready.')
    gui.run()


if __name__ == '__main__':
    main()
