#version 400
out vec4 frag_colour;
uniform float R_val;
uniform float G_val;
uniform float B_val;
uniform float A_val;

void main () {
	frag_colour = vec4(R_val, G_val, B_val, A_val);
}
