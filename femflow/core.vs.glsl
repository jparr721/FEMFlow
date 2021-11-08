#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

out vec4 v_color;

uniform mat4 mvp;

void main() {
  v_color = color;
  gl_Position = mvp * vec4(position, 1.0);
}