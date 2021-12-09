#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texture_coordinates;
layout(location = 2) in vec3 normal;

uniform mat4 view;
uniform mat4 projection;

uniform mat4 light;

out vec2 texture_coordinatesi;
out vec3 frag_normal;

void main() {
  frag_normal = (light * vec4(normal, 0.0f)).xyz;
  gl_Position = projection * view * vec4(position, 1.0);
  texture_coordinatesi = texture_coordinates;
}