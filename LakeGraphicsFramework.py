### Imports ###
import glfw
import numpy as np
import pyrr
import OpenGL.GL as gl
import OpenGL.GL.shaders as gls
from variable_type_validation import *
from MessageLogger import MessageLogger as log
from threading import Lock
from typing import Callable
import copy
from safe_file_readlines import safe_file_readlines


### Type hints ###
ColorRGBA = Tuple[float, float, float, float]
ToDecide = Any # TODO


### Constants ###
VSYNC_VALUE = 1

SHADERS_PATH = "shaders/"
MODELS_PATH = "models/"

DEFAULT_FRAG_SHADER = "default.frag"
DEFAULT_VERT_SHADER = "default.vert"

DEFAULT_UNIFORM_NAMES = {
    "projection": "projection",
    "model": "model",
    "view": "view",
    "color": "objectColor",
}

DEFAULT_FOVY = 100
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 200
DEFAULT_SKYBOX_COLOR = (0, 0, 0, 1)


### Functions ###
def terminate_glfw():
    glfw.terminate()


### Classes ###
class Window:
    def __init__(self, size: Size = (0, 0), caption: str = "", fullscreen: bool = False, windowed: bool = True, vsync: bool = True, max_fps: int = 0, raw_mouse_input: bool = True, center_cursor: bool = True, hide_cursor: bool = True, fovy: float = DEFAULT_FOVY, near: float = DEFAULT_NEAR, far: float = DEFAULT_FAR, skybox_color: ColorRGBA = DEFAULT_SKYBOX_COLOR) -> None:
        """
        Initializes a window, input system, and graphics engine with the specified parameters.
        
        Parameters:
            size (Size): The initial size of the window (width, height).
            caption (str): The window's title.
            fullscreen (bool): Whether the window is fullscreen. Overrides size if True.
            windowed (bool): Whether the window is in fullscreen or windowed mode. Overrides fullscreen to True if windowed is False.
            vsync (bool): Whether vertical sync is enabled. Overrides max_fps if enabled.
            max_fps (int): The maximum frames per second. 0 means uncapped.

            raw_mouse_input (bool): Whether raw mouse input is used.
            center_cursor (bool): Whether the cursor is centered in the window.
            hide_cursor (bool): Whether the cursor is hidden.

            fovy (float): The view FOV.
            near (float): The closest distance to the camera which will be rendered. Should be a low number, greater than 0.
            far (float): The furthers distance to the camera which will be rendered. Must be a higher than 'near'. A number too large may hinder performance.
            skybox_color (float): Color of the background in normalised rgba. Color channels range from 0 to 1.
        """
        # Validate parameters
        validate_types([('size', size, Size),
                        ('caption', caption, str),
                        ('fullscreen', fullscreen, bool),
                        ('windowed', windowed, bool),
                        ('vsync', vsync, bool),
                        ('max_fps', max_fps, int),
                        ('raw_mouse_input', raw_mouse_input, bool),
                        ('center_cursor', center_cursor, bool),
                        ('hide_cursor', hide_cursor, bool),
                        ('fovy', fovy, Real),
                        ('near', near, Real),
                        ('far', far, Real),
                        ('skybox_color', skybox_color, ColorRGBA)])
        
        if not far > near:
            raise ValueError(f"The value for far '{far}' is not greater than near '{near}'.")
        
        # Set parameters
        self.size = size
        self.caption = str(caption)
        self.fullscreen = fullscreen or not windowed
        self.windowed = windowed
        self.vsync = vsync
        self.max_fps = int(max_fps)
        self.raw_mouse_input = raw_mouse_input
        self.center_cursor = center_cursor
        self.hide_cursor = hide_cursor

        if self.fullscreen != fullscreen:
            log.warn(f"Window was requested in non-windowed mode. Fullscreen forced to from {fullscreen} to {self.fullscreen}.")

        log.info(f"Window variable validation passed. Variables: {self.__dict__}")

        # Display init
        self.monitor, self.video_mode = self._init_display()
        self.display_size = self.video_mode.size

        # Update screen size if fullscreen
        if self.fullscreen:
            log.info("Window fullscreen.")
            self.size = self.display_size
        
        if self.size[1] != 0:
            self.aspect_ratio = self.size[0] / self.size[1]
        else:
            self.aspect_ratio = 1

        # Window init
        self.window = self._init_window(self.monitor, self.size, self.caption, self.windowed, self.vsync, self.max_fps)

        # Input mode init
        self._init_input(self.window, self.raw_mouse_input, self.hide_cursor)

        log.info(f"Window creation passed. Variables: {self.__dict__}")

        # Set variables
        self.active = True
        self.lock = Lock()
        self.graphics_engine = GraphicsEngine(fovy, self.aspect_ratio, near, far, skybox_color)
    
    def main(self):
        while self.active:
            gl.glFlush() # Wait for pipeline

            self._gl_check_error()
            glfw.poll_events()

            self.graphics_engine._render()
            glfw.swap_buffers(self.window)

    ### Init helpers ####
    @staticmethod
    def _init_display() -> tuple[glfw._GLFWmonitor, glfw._GLFWvidmode]:
        """
        [Private]
        Initializes the GLFW display by setting up the primary monitor and video mode.
        It ensures that GLFW can be initialized, the primary monitor can be found, and a valid video mode is available. An exception is raised if anything fails.
        
        Returns:
            tuple[glfw._GLFWmonitor, glfw._GLFWvidmode]: The primary monitor and its video mode.
        """
        # Init GLFW and display
        if not glfw.init():
            raise Exception("GLFW can't be initialized")
        
        monitor = glfw.get_primary_monitor()
        if not monitor:
            raise Exception("GLFW can't find primary monitor")
        
        video_mode = glfw.get_video_mode(monitor)
        if not video_mode:
            raise Exception("GLFW can't get video mode")
        
        return monitor, video_mode

    @staticmethod
    def _init_window(monitor: glfw._GLFWmonitor, size: Size, caption: str, windowed: bool, vsync: bool, max_fps: int) -> glfw._GLFWwindow:
        """
        [Private]
        Initializes the GLFW window associates it with the primary monitor.
        Sets the frame rate limit based on the 'vsync' and 'max_fps' parameters.
        If the window creation fails, GLFW is terminated and an exception is raised.
        
        Parameters:
            monitor (glfw._GLFWmonitor): The primary monitor to use for fullscreen.
            size (Size): The desired size of the window (width, height).
            caption (str): The title of the window.
            vsync (bool): Whether vertical synchronization should be enabled.
            max_fps (int): The maximum frames per second. 0 means uncapped.

        Returns:
            glfw._GLFWwindow: The created window.
        """
        screen_monitor = monitor if not windowed else None

        window = glfw.create_window(*size, caption, screen_monitor, None)
        if not window:
            glfw.terminate()
            raise Exception("GLFW window can't be created")

        glfw.make_context_current(window)

        # Set vsync if enabled
        fps_value = VSYNC_VALUE if vsync else max_fps
        glfw.swap_interval(fps_value)

        return window
    
    @staticmethod
    def _init_input(window: glfw._GLFWwindow, raw_mouse_input: bool, hide_cursor: bool) -> None:
        """
        [Private]
        Initializes input settings for the window. Configures the cursor visibility and raw mouse input mode.
        
        Parameters:
            window (glfw._GLFWwindow): The window for which input settings should be configured.
            raw_mouse_input (bool): Whether to enable raw mouse input.
            hide_cursor (bool): Whether to hide the cursor in the window.
        """
        if hide_cursor:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

        if raw_mouse_input:
            if glfw.raw_mouse_motion_supported():
                glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)
            else:
                log.warn("Raw mouse motion unsupported.")
    
    ### Private ###
    def _gl_check_error(self):
        error = gl.glGetError()

        if error != gl.GL_NO_ERROR:
            log.warn(f"OpenGL error: {error}")


class Shader:
    def __init__(self, name: str, vertex_path: str, fragment_path: str, fovy: float, aspect: float, near: float, far: float, model_name: str = DEFAULT_UNIFORM_NAMES["model"], view_name: str = DEFAULT_UNIFORM_NAMES["view"], projection_name: str = DEFAULT_UNIFORM_NAMES["projection"], texture_name: str = None, color_name: str = DEFAULT_UNIFORM_NAMES["color"], custom_uniform_names: dict = {}, compile_time_config: dict = {}):
        self.name = name
        log.info(f"Creating shader {name}")

        # Create shader
        self.shader = self._create_shader(vertex_path, fragment_path, compile_time_config)

        # Get shader handles
        self.uniform_handles = self._get_uniform_handles(self.shader, model_name, view_name, projection_name, texture_name, color_name)
        self.custom_uniform_handles = self._get_custom_uniform_handles(self.shader, custom_uniform_names)

        self.use()

        # Initilize shader handles
        self._init_handles(self.uniform_handles)
        self._init_perspective_projection(self.uniform_handles["projection"], fovy, aspect, near, far)

        log.info(f"Shader creation passed for {name}. Variables: {self.__dict__}")

    def _create_shader(self, vertex_path, fragment_path, compile_time_config) -> gls.ShaderProgram:
        vertex_src = safe_file_readlines(vertex_path)
        fragment_src = safe_file_readlines(fragment_path)

        vertex_program = self._update_file_config(vertex_src, compile_time_config)
        fragment_program = self._update_file_config(fragment_src, compile_time_config)
        
        shader = gls.compileProgram(
            gls.compileShader(vertex_program, gl.GL_VERTEX_SHADER),
            gls.compileShader(fragment_program, gl.GL_FRAGMENT_SHADER)
        )
        
        return shader
    
    @staticmethod
    def _update_file_config(file_src: str, compile_time_config: dict) -> str:
        for placeholder, value, in compile_time_config:
            file_src = file_src.replace(placeholder, value)

        return file_src
    
    @staticmethod
    def _get_uniform_handles(shader: gls.ShaderProgram, model_name: str, view_name: str, projection_name: str, texture_name: str = None, color_name: str = None) -> dict:
        uniforms = {
            "model": model_name,
            "view": view_name,
            "projection": projection_name
        }

        if texture_name:
            uniforms["texture"] = texture_name
        if color_name:
            uniforms["color"] = color_name

        uniform_handles = {uniform_handle_name: gl.glGetUniformLocation(shader, uniform_name) for uniform_handle_name, uniform_name in uniforms.items()}

        return uniform_handles
    
    @staticmethod
    def _get_custom_uniform_handles(shader: gls.ShaderProgram, uniforms: dict) -> dict:
        custom_uniform_handles = {uniform_handle_name: gl.glGetUniformLocation(shader, uniform_name) for uniform_handle_name, uniform_name in uniforms.items()}

        return custom_uniform_handles
    
    def _init_handles(self, handles):
        if "texture" in handles:
            gl.glUniform1i(handles["texture"], 0)
        else:
            log.info(f"No texture handle for the shader {self.name}.")

    @staticmethod
    def _init_perspective_projection(projection_handle, fovy: float, aspect: float, near: float, far: float):
        projection_transform = pyrr.matrix44.create_perspective_projection(fovy = fovy, aspect = aspect, near = near, far = far, dtype = np.float32)
        gl.glUniformMatrix4fv(projection_handle, 1, gl.GL_FALSE, projection_transform)
    
    def use(self):
        gl.glUseProgram(self.shader)

    def set_view(self, view_transform):
        gl.glUniformMatrix4fv(self.uniform_handles["view"], 1, gl.GL_FALSE, view_transform)


class GraphicsEngine:
    def __init__(self, fovy: float, aspect: float, near: float, far: float, skybox_color: ColorRGBA) -> None:
        log.info("Setting up Graphics Engine")
        self.fovy = fovy
        self.aspect = aspect
        self.near = near
        self.far = far
        self.skybox_color = skybox_color

        self.shaders = {}
        self.active_shader_name = None

        self._init_opengl(skybox_color)

        # Initilize shader
        self.shaders["default"] = Shader("default", SHADERS_PATH + DEFAULT_VERT_SHADER, SHADERS_PATH + DEFAULT_FRAG_SHADER, fovy, aspect, near, far)
        # Use default shader
        self._use_shader("default")

        # Pre-draw instructions
        self.pending_skybox_color = None
        self.new_skybox_color = None

        self.pending_shader_creations = []
        self.new_shader_creations = []

        # Draw lists
        self.pending_draw_instructions = []
        self.active_draw_instructions = []

        self.lock = Lock()
    
    # Init helper functions #
    @staticmethod
    def _init_opengl(skybox_color: ColorRGBA):
        # Initilize OpenGL
        gl.glClearColor(*skybox_color)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    # Public functions #
    def create_shader(self, name: str, vertex_path: str, fragment_path: str, fovy: float = None, aspect: float = None, near: float = None, far: float = None, compile_time_config: dict = {}):
        if fovy is None:
            fovy = self.fovy
        if aspect is None:
            aspect = self.aspect
        if near is None:
            near = self.near
        if far is None:
            far = self.far

       # Validate parameters
        validate_types([('fovy', fovy, Real),
                        ('aspect', aspect, Real),
                        ('near', near, Real),
                        ('far', far, Real)])
        
        if not far > near:
            raise ValueError(f"The value for far '{far}' is not greater than near '{near}'.")
        
        if name in self.shaders:
            log.warn(f"Create shader called for '{name}', but the shader '{name}' already exists. The old shader will be overridden.")
        
        log.info(f"Shader creation for '{name}' pass variable validation.")

        self.pending_shader_creations.append((name, vertex_path, fragment_path, fovy, aspect, near, far, compile_time_config))

    def clear(self):
        """Clears pending draw list"""
        self.pending_draw_instructions = []

    def use_shader(self, shader_name: str):
        validate_type("shader_name", shader_name, str)

        instruction_args = (shader_name)
        self._add_draw_instruction(self._use_shader, instruction_args)

    def set_view(self, pos: Coordinate = None, rotation: ToDecide = None):
        """Set camera position and rotation relative to world"""
        validate_type("pos", pos, Coordinate)
        #validate_type("rotation", rotation, ToDecide) TODO

        instruction_args = (pos, rotation)
        self._add_draw_instruction(self._set_view, instruction_args)

    def set_skybox_color(self, skybox_color: ColorRGBA):
        """Set the clear color in normalised RGBA"""
        validate_type("skybox_color", skybox_color, ColorRGBA)

        self.pending_skybox_color = skybox_color
    
    def update(self):
        with self.lock:
            self.new_shader_creations = self.pending_shader_creations
            self.new_skybox_color = self.pending_skybox_color

            self.active_draw_instructions = self.pending_draw_instructions

    # Private functions #
    def _create_shaders(self, new_shader_creations: list[tuple[str, str, str, float, float, float, float, dict]]):
        for new_shader in new_shader_creations:
            name, vertex_path, fragment_path, fovy, aspect, near, far, compile_time_config = new_shader
            self.shaders[name] = Shader(name, vertex_path, fragment_path, fovy, aspect, near, far, compile_time_config)

    def _use_shader(self, shader_name: str):
        if shader_name not in self.shaders:
            raise Exception("Shader '{shader_name}' does not exist in shaders.")

        self.shaders[shader_name].use()
        self.active_shader_name = shader_name

    def _add_draw_instruction(self, draw_function: Callable, args: tuple):
        self.pending_draw_instructions.append((draw_function, args))

    def _set_view(self, pos: Coordinate, rotation: ToDecide):
        view_transform = pyrr.matrix44.create_look_at(
            eye = np.zeros(3, dtype = np.float32),
            target = np.array([1, 0, 0], dtype = np.float32),
            up = np.array([0, 1, 0], dtype = np.float32),
            dtype = np.float32
        )

        self.shaders[self.active_shader_name].set_view(view_transform)

    @staticmethod
    def _update_skybox_color(skybox_color: ColorRGBA):
        """Set the clear color"""
        if skybox_color is not None:
            log.info(f"Updating skybox color to {skybox_color}")
            gl.glClearColor(*skybox_color)

    @staticmethod
    def _clear_screen():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def _complete_draw_instructions(self, draw_instructions):
        for draw_instruction in draw_instructions:
            draw_function, args = draw_instruction

            draw_function(*args)
    
    def _render(self):
        with self.lock:
            # Get render instructions
            new_shader_creations = copy.deepcopy(self.new_shader_creations)
            new_skybox_color = copy.deepcopy(self.new_skybox_color)
            active_draw_instructions = copy.deepcopy(self.active_draw_instructions)

            # Reset one-time functions
            self.new_shader_creations = []
            self.new_skybox_color = None

        self._create_shaders(new_shader_creations)
        self._update_skybox_color(new_skybox_color)

        self._clear_screen()
        self._use_shader("default")
        self._complete_draw_instructions(active_draw_instructions)