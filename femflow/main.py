import contextlib
import os

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import *


def window():
    pass


def main():
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    compile_shaders()

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


def compile_shaders():
    frag_shader_source = ""
    vertex_shader_source = ""
    print(os.listdir("."))

    with open("core.frag.glsl", "r") as f:
        frag_shader_source = f.read()

    with open("core.vs.glsl", "r") as f:
        vertex_shader_source = f.read()

    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, 1, vertex_shader_source, None)


@contextlib.contextmanager
def create_vao():
    id = glGenVertexArrays(1)
    try:
        glBindVertexArray(id)
        yield
    finally:
        glDeleteVertexArrays(1, [id])


if __name__ == "__main__":
    main()
