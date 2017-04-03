#version 400
out vec4 frag_colour;
uniform float val;

void main () {
	frag_colour = vec4(1.0f, val, 0.0f, 1.0f);
}
