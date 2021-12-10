#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texture_coordinates;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 color;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 light;

out vec2 texture_coordinatesi;
out vec3 normali;
out vec4 colori;

void main() {
  normali = (light * vec4(normal, 0.0f)).xyz;
  gl_Position = projection * view * vec4(position, 1.0);
  texture_coordinatesi = texture_coordinates;
  colori = vec4(color, 1.0f);
}