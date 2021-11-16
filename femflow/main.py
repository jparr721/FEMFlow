# import glfw
# import imgui
# import OpenGL
# from imgui.integrations.glfw import GlfwRenderer
# from loguru import logger

# OpenGL.ERROR_LOGGING = True
# OpenGL.FULL_LOGGING = True
# from OpenGL.GL import *

# from viz.visualizer import Visualizer


# def main():
#     imgui.create_context()
#     window = impl_glfw_init()
#     impl = GlfwRenderer(window)

#     while not glfw.window_should_close(window):
#         glfw.poll_events()
#         impl.process_inputs()

#         glClearColor(1.0, 1.0, 1.0, 1)
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

#         imgui.new_frame()

#         if imgui.begin_main_menu_bar():
#             if imgui.begin_menu("File", True):

#                 clicked_quit, selected_quit = imgui.menu_item("Quit", "Cmd+Q", False, True)

#                 if clicked_quit:
#                     exit(1)

#                 imgui.end_menu()
#             imgui.end_main_menu_bar()

#         imgui.begin("Custom window", True)
#         imgui.text("Bar")
#         imgui.text_ansi("B\033[31marA\033[mnsi ")
#         imgui.text_ansi_colored("Eg\033[31mgAn\033[msi ", 0.2, 1.0, 0.0)
#         imgui.extra.text_ansi_colored("Eggs", 0.2, 1.0, 0.0)
#         imgui.end()

#         imgui.render()
#         impl.render(imgui.get_draw_data())
#         glfw.swap_buffers(window)

#     impl.shutdown()
#     glfw.terminate()


# def impl_glfw_init():
#     width, height = 1280, 720
#     window_name = "FEMFlow Viewer"

#     if not glfw.init():
#         print("Could not initialize OpenGL context")
#         exit(1)

#     glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
#     glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
#     glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

#     glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

#     # Create a windowed mode window and its OpenGL context
#     window = glfw.create_window(int(width), int(height), window_name, None, None)
#     glfw.make_context_current(window)

#     if not window:
#         glfw.terminate()
#         print("Could not initialize Window")
#         exit(1)

#     return window


# def flatten(matrix):
#     if matrix.shape[1] == 1:
#         return matrix
#     return matrix.reshape(-1)


# if __name__ == "__main__":
#     logger.info("Warming up...")
#     with Visualizer() as visualizer:
#         visualizer.launch()

import os

import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from OpenGL.GLU import *

from viz.camera import Camera
from viz.input import Input
from viz.shader_program import ShaderProgram

camera = Camera()
camera.resize(1280, 720)

vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
uniform mat4 mvp;
out vec3 v_color;
void main()
{
    gl_Position = mvp * vec4(a_position, 1.0);
    v_color = a_color;
}
"""

fragment_src = """
# version 330
in vec3 v_color;
out vec4 out_color;
void main()
{
    out_color = vec4(v_color, 1.0);
}
"""

input = Input()


def window_resize(window, width, height):
    glViewport(0, 0, width, height)


def scroll_callback(window, xoffset, yoffset):
    input.scroll_event(window, xoffset, yoffset, camera)


def mouse_move_callback(window, xpos, ypos):
    input.handle_mouse_move(window, xpos, ypos, camera)


# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(1280, 720, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 400, 200)

# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize)
glfw.set_scroll_callback(window, scroll_callback)
glfw.set_cursor_pos_callback(window, mouse_move_callback)

# make the context current
glfw.make_context_current(window)

vertices = [
    -0.5,
    -0.5,
    0.5,
    1.0,
    0.0,
    0.0,
    0.5,
    -0.5,
    0.5,
    0.0,
    1.0,
    0.0,
    0.5,
    0.5,
    0.5,
    0.0,
    0.0,
    1.0,
    -0.5,
    0.5,
    0.5,
    1.0,
    1.0,
    1.0,
    -0.5,
    -0.5,
    -0.5,
    1.0,
    0.0,
    0.0,
    0.5,
    -0.5,
    -0.5,
    0.0,
    1.0,
    0.0,
    0.5,
    0.5,
    -0.5,
    0.0,
    0.0,
    1.0,
    -0.5,
    0.5,
    -0.5,
    1.0,
    1.0,
    1.0,
]

indices = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 4, 5, 1, 1, 0, 4, 6, 7, 3, 3, 2, 6, 5, 6, 2, 2, 1, 5, 7, 4, 0, 0, 3, 7]

vertices = np.array(vertices, dtype=np.float32)
indices = np.array(indices, dtype=np.uint32)

FRAG_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "core.frag.glsl")
VERTEX_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "core.vs.glsl")
shader_program = ShaderProgram()
shader_program.add_shader(GL_VERTEX_SHADER, VERTEX_SHADER_PATH)
shader_program.add_shader(GL_FRAGMENT_SHADER, FRAG_SHADER_PATH)
shader_program.link()

# shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

# Vertex Buffer Object
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Element Buffer Object
EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

shader_program.bind()
glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)

mvp = shader_program.uniform_location("mvp")

# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
    # rot_y = pyrr.Matrix44.from_y_rotation(0.8 * glfw.get_time())

    # shader_program.set_matrix_uniform(mvp, pyrr.matrix44.multiply(rot_x, rot_y))
    shader_program.set_matrix_uniform(mvp, np.matmul(camera.projection_matrix, camera.view_matrix))
    # shader_program.set_matrix_uniform(mvp, camera.projection_matrix)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()
