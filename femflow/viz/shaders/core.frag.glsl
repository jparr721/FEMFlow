#version 330

in vec2 texture_coordinatesi;
in vec3 normali;
in vec4 colori;

out vec4 frag_color;

uniform sampler2D sampler_texture;

void main() {
  vec3 ambient_light_intensity = vec3(0.3f, 0.2f, 0.4f);
  vec3 sun_light_intensity = vec3(0.9f, 0.9f, 0.9f);
  vec3 sun_light_direction = normalize(vec3(-2.0f, -2.0f, 0.0f));

  vec4 texel = texture(sampler_texture, texture_coordinatesi);

  vec3 light_intensity =
      ambient_light_intensity +
      sun_light_intensity * max(dot(normali, sun_light_direction), 0.0f);

  frag_color = vec4(colori.rgb * light_intensity, colori.a);
}