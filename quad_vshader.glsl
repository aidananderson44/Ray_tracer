
#version 330 core

in vec3 vposition;


out vec2 uv;


void main() {
    gl_Position = vec4(vposition, 1.0);
    uv = (vposition.xy + 1)/2.0f;
}

