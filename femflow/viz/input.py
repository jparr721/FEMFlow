import glfw
import numpy as np

from .camera import Camera


class Input(object):
    def __init__(self):
        self.current_mouse_pos = np.zeros(2)
        self.last_mouse_pos = np.zeros(2)
        self.mouse_pressed = False

    def scroll_event(self, window, xoffset: float, yoffset: float, camera: Camera):
        camera.zoom(-yoffset)

    def handle_mouse_move(self, window, xpos, ypos, camera: Camera):
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            camera.rotating = True
        else:
            camera.rotating = False
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            camera.panning = True
        else:
            camera.panning = False
        self.mouse_move(xpos, -ypos, camera)

    def handle_mouse(self, window, button: int, action: int, mods: int, camera: Camera):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                camera.rotating = True
            else:
                camera.rotating = False

        if button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                camera.panning = True
            else:
                camera.panning = False

    def mouse_move(self, x: float, y: float, camera: Camera):
        self.current_mouse_pos = np.array([x, y])
        dx, dy = self.last_mouse_pos - self.current_mouse_pos

        if camera.panning:
            camera.pan(dx, dy)

        if camera.rotating:
            camera.rotate(dx, dy)

        if camera.zooming:
            camera.zoom(-dy)

        self.last_mouse_pos = self.current_mouse_pos
