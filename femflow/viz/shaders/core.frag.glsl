#version 330

in vec2 texture_coordinatesi;
in vec3 surface_normal;
in vec4 ambient_color;
in vec3 light_dir;

out vec4 frag_color;

uniform sampler2D sampler_texture;
uniform sampler2D diffuse_texture;
uniform sampler2D normal_texture;
uniform sampler2D specular_texture;

void main() {
  frag_color = ambient_color;

	vec4 specular_color = vec4(1.0f, 1.0f, 1.0f, 0.5f);
  vec4 diffuse_color = vec4(ambient_color.rgb, 0.5f);

  // Make this a var later.
  float shininess = 20.0f;

  float diffuse = max(0.0f, dot(normalize(surface_normal), normalize(light_dir)));
  vec4 diffuse_light = diffuse * diffuse_color;
  frag_color += diffuse_light;

  if (diffuse != 0.0f) {
    vec3 reflection = normalize(reflect(-normalize(light_dir), normalize(surface_normal)));

    float reflection_angle = max(0.0f, dot(normalize(surface_normal), reflection));

    float specular_exp = pow(reflection_angle, shininess);
    vec4 specular_light = specular_color * specular_exp;
    frag_color += specular_light;
  }
}
