#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 light;
uniform mat4 normal_matrix;

out vec3 surface_normal;
out vec4 ambient_color;
out vec3 light_dir;

void main() {
  vec4 vertex_pos = vec4(position, 1.0);

  light_dir = normalize(light - position);
  surface_normal = normalize(normal_matrix * vec4(normal, 1.0f)).xyz;

  gl_Position = projection * view * vertex_pos;

  ambient_color = vec4(color, 1.0f);
}
