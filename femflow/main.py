import logging

import glfw
import imgui
import OpenGL
from imgui.integrations.glfw import GlfwRenderer

OpenGL.ERROR_LOGGING = True
OpenGL.FULL_LOGGING = True
from OpenGL.GL import *

from viz.visualizer import Visualizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        glClearColor(1.0, 1.0, 1.0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item("Quit", "Cmd+Q", False, True)

                if clicked_quit:
                    exit(1)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.begin("Custom window", True)
        imgui.text("Bar")
        imgui.text_ansi("B\033[31marA\033[mnsi ")
        imgui.text_ansi_colored("Eg\033[31mgAn\033[msi ", 0.2, 1.0, 0.0)
        imgui.extra.text_ansi_colored("Eggs", 0.2, 1.0, 0.0)
        imgui.end()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


def impl_glfw_init():
    width, height = 1280, 720
    window_name = "FEMFlow Viewer"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def flatten(matrix):
    if matrix.shape[1] == 1:
        return matrix
    return matrix.reshape(-1)


if __name__ == "__main__":
    # v, f, = igl.read_triangle_mesh("cuboid.obj")
    # t = wm.Tetrahedralizer(stop_quality=500)
    # t.set_mesh(v, f)
    # t.tetrahedralize()
    # vv, tt = t.get_tet_mesh()
    # print(igl.boundary_facets(tt))
    with Visualizer() as visualizer:
        visualizer.launch()
