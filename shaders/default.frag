#version 330 core

uniform vec4 objectColor;

out vec4 color;

void main()
{
    color = vec4(objectColor);
}