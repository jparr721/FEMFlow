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


# import os

# import glfw
# import numpy as np
# import pyrr
# from OpenGL.GL import *
# from OpenGL.GLU import *

# from viz.camera import Camera
# from viz.input import Input
# from viz.shader_program import ShaderProgram

# camera = Camera()
# width = 1200
# height = 800
# camera.resize(width, height)

# input = Input()


# def window_resize(window, width, height):
#     print("Resizing")
#     glViewport(0, 0, width, height)
#     glMatrixMode(GL_PROJECTION)
#     glLoadIdentity()
#     camera.resize(width, height)
#     glMultMatrixf(camera.projection_matrix)
#     glMatrixMode(GL_MODELVIEW)
#     glLoadIdentity()


# def scroll_callback(window, xoffset, yoffset):
#     input.scroll_event(window, xoffset, -yoffset, camera)


# def mouse_move_callback(window, xpos, ypos):
#     input.handle_mouse_move(window, xpos, ypos, camera)


# # initializing glfw library
# if not glfw.init():
#     raise Exception("glfw can not be initialized!")

# # creating the window
# window = glfw.create_window(width, height, "My OpenGL window", None, None)

# # check if window was created
# if not window:
#     glfw.terminate()
#     raise Exception("glfw window can not be created!")

# # set window's position
# glfw.set_window_pos(window, 400, 200)

# # set the callback function for window resize
# glfw.set_window_size_callback(window, window_resize)
# glfw.set_scroll_callback(window, scroll_callback)
# glfw.set_cursor_pos_callback(window, mouse_move_callback)

# # make the context current
# glfw.make_context_current(window)

# vertices = [
#     -0.5,
#     -0.5,
#     0.5,
#     0.5,
#     -0.5,
#     0.5,
#     0.5,
#     0.5,
#     0.5,
#     -0.5,
#     0.5,
#     0.5,
#     -0.5,
#     -0.5,
#     -0.5,
#     0.5,
#     -0.5,
#     -0.5,
#     0.5,
#     0.5,
#     -0.5,
#     -0.5,
#     0.5,
#     -0.5,
# ]

# indices = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 4, 5, 1, 1, 0, 4, 6, 7, 3, 3, 2, 6, 5, 6, 2, 2, 1, 5, 7, 4, 0, 0, 3, 7]

# vertices = np.array(vertices, dtype=np.float32)
# indices = np.array(indices, dtype=np.uint32)

# FRAG_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "core.frag.glsl")
# VERTEX_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "core.vs.glsl")

# glEnable(GL_DEPTH_TEST)

# # the main application loop
# colors = np.tile(np.random.rand(3), (len(vertices) // 3, 1))
# window_resize(window, width, height)
# while not glfw.window_should_close(window):
#     glfw.poll_events()
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

#     glPushMatrix()
#     glMultMatrixf(camera.view_matrix)

#     glBegin(GL_TRIANGLES)
#     for index in indices:
#         i = index * 3
#         pos = vertices[i : i + 3]
#         glColor3f(*colors[index // 3])
#         glVertex3f(*pos)

#     glEnd()

#     glLineWidth(4)
#     glBegin(GL_LINES)

#     for index in indices:
#         i = index * 3
#         pos = vertices[i : i + 3]
#         glColor3f(1, 1, 1)
#         glVertex3f(*pos)

#     glColor3f(1.0, 0.0, 0.0)
#     glVertex3f(0.0, 0.0, 0.0)
#     glVertex3f(1.0, 0.0, 0.0)
#     glColor3f(0.0, 1.0, 0.0)
#     glVertex3f(0.0, 0.0, 0.0)
#     glVertex3f(0.0, 1.0, 0.0)
#     glColor3f(0.0, 0.0, 1.0)
#     glVertex3f(0.0, 0.0, 0.0)
#     glVertex3f(0.0, 0.0, 1.0)
#     glEnd()

#     glPopMatrix()
#     glFlush()

#     glfw.swap_buffers(window)

# # terminate glfw, free up allocated resources
# glfw.terminate()

import os

import glfw
import numpy as np
import pyrr
from loguru import logger
from OpenGL.GL import *
from OpenGL.GLU import *

from viz.camera import Camera
from viz.input import Input
from viz.shader_program import ShaderProgram

camera = Camera()
width = 1200
height = 800

input = Input()


def bind_vbo(id, name, buffer, stride, data):
    handle = glGetAttribLocation(id, name)
    logger.debug(f"Binding {name}, pos {handle}")
    glBindBuffer(GL_ARRAY_BUFFER, buffer)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(handle, stride, GL_FLOAT, GL_FALSE, data.itemsize * stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(handle)


def window_resize(window, width, height):
    print("Resizing")
    glViewport(0, 0, width, height)
    camera.resize(width, height)
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # camera.resize(width, height)
    # glMultMatrixf(camera.projection_matrix)
    # glMatrixMode(GL_MODELVIEW)
    # glLoadIdentity()


def scroll_callback(window, xoffset, yoffset):
    input.scroll_event(window, xoffset, -yoffset, camera)


def mouse_move_callback(window, xpos, ypos):
    input.handle_mouse_move(window, xpos, ypos, camera)


# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(width, height, "My OpenGL window", None, None)

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
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
]

colors = np.tile(np.array([1, 0, 0]), len(vertices) // 3).astype(np.float32)

indices = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 4, 5, 1, 1, 0, 4, 6, 7, 3, 3, 2, 6, 5, 6, 2, 2, 1, 5, 7, 4, 0, 0, 3, 7]

vertices = np.array(vertices, dtype=np.float32)
indices = np.array(indices, dtype=np.uint32)
print(colors.size, vertices.size)

FRAG_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "core.frag.glsl")
VERTEX_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "core.vs.glsl")

shader_program = ShaderProgram()
shader_program.add_shader(GL_VERTEX_SHADER, VERTEX_SHADER_PATH)
shader_program.add_shader(GL_FRAGMENT_SHADER, FRAG_SHADER_PATH)
shader_program.link()

vbo = glGenBuffers(1)

cvbo = glGenBuffers(1)

ebo = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

bind_vbo(shader_program.id, "position", vbo, 3, vertices)
bind_vbo(shader_program.id, "color", cvbo, 3, colors)

shader_program.bind()
glEnable(GL_DEPTH_TEST)

# the main application loop
window_resize(window, width, height)
while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    shader_program.set_matrix_uniform("projection", camera.projection_matrix)
    shader_program.set_matrix_uniform("view", camera.view_matrix)
    glDrawElements(GL_TRIANGLES, indices.size, GL_UNSIGNED_INT, None)

    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()
