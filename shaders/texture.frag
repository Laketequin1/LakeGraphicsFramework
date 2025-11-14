#version 330 core

in vec2 fragmentTexCoord;
in vec3 fragmentPosition;
in vec3 fragmentNormal;

uniform sampler2D imageTexture;

out vec4 color;

void main() {
    color = vec4(texture(imageTexture, fragmentTexCoord).rgba);
}