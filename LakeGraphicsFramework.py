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
    
    def main(self) -> None:
        while self.active:
            gl.glFlush() # Wait for pipeline

            self._gl_check_error()

            self.graphics_engine._render()
            glfw.swap_buffers(self.window)

            self._tick()

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

    def _init_window(self, monitor: glfw._GLFWmonitor, size: Size, caption: str, windowed: bool, vsync: bool, max_fps: int) -> glfw._GLFWwindow:
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
        if windowed:
            self._center_window(window, monitor, size)

        # Set vsync if enabled
        fps_value = VSYNC_VALUE if vsync else max_fps
        glfw.swap_interval(fps_value)

        return window
    
    @staticmethod
    def _center_window(window: glfw._GLFWwindow, monitor: glfw._GLFWmonitor, size: Size) -> None:
        """
        [Private]
        Centers the GLFW window within the work area of the specified monitor.

        Parameters:
            window (glfw._GLFWwindow): The GLFW window to be centered.
            monitor (glfw._GLFWmonitor): The monitor whose work area is used for positioning.
            size (Size): The dimensions of the window (width, height).
        """
        log.info("Centering window.")

        monitor_x, monitor_y, monitor_width, monitor_height = glfw.get_monitor_workarea(monitor)

        pos_x = monitor_x + (monitor_width - size[0]) // 2
        pos_y = monitor_y + (monitor_height - size[1]) // 2

        glfw.set_window_pos(window, pos_x, pos_y)
    
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
    def _gl_check_error(self) -> None:
        """
        [Private]
        Checks for OpenGL errors and logs a warning if an error is detected.
        """
        error = gl.glGetError()

        if error != gl.GL_NO_ERROR:
            log.warn(f"OpenGL error: {error}")

    def _tick(self) -> None:
        """
        [Private]
        Processes a single application tick by polling GLFW events and handling window-related events.
        """
        glfw.poll_events()

        self._handle_window_events()

    def _handle_window_events(self) -> None:
        """
        [Private]
        Handle GLFW events and closing the window.
        """
        if glfw.window_should_close(self.window) or glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            self.close()
            return
    
    def close(self) -> None:
        """
        [Private]
        Close the GLFW window and terminate glfw.
        """
        self.active = False
        self.graphics_engine.destroy()
        glfw.terminate()


class Shader:
    def __init__(self, name: str, vertex_path: str, fragment_path: str, fovy: float, aspect: float, near: float, far: float, model_name: str = DEFAULT_UNIFORM_NAMES["model"], view_name: str = DEFAULT_UNIFORM_NAMES["view"], projection_name: str = DEFAULT_UNIFORM_NAMES["projection"], texture_name: str = None, color_name: str = DEFAULT_UNIFORM_NAMES["color"], custom_uniform_names: dict = {}, compile_time_config: dict = {}) -> None:
        """
        [Private]
        Initializes the shader with the specified vertex and fragment shader paths, along with projection and model settings.
        Sets up uniform handles and initializes perspective projection.

        Parameters:
            name (str): The name of the shader.
            vertex_path (str): Path to the vertex shader file.
            fragment_path (str): Path to the fragment shader file.
            fovy (float): Field of view for the perspective projection.
            aspect (float): Aspect ratio for the perspective projection.
            near (float): Near clipping plane distance for the perspective projection.
            far (float): Far clipping plane distance for the perspective projection.
            model_name (str): The name for the model matrix uniform (default is "model").
            view_name (str): The name for the view matrix uniform (default is "view").
            projection_name (str): The name for the projection matrix uniform (default is "projection").
            texture_name (str): The name for the texture uniform (default is None).
            color_name (str): The name for the color uniform (default is "color").
            custom_uniform_names (dict): Optional custom uniform names to be handled (default is empty dictionary).
            compile_time_config (dict): Optional configuration settings for shader compilation (default is empty dictionary).
        """
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
        """
        [Private]
        Creates and compiles a shader program from the given vertex and fragment shader file paths.

        Parameters:
            vertex_path (str): Path to the vertex shader file.
            fragment_path (str): Path to the fragment shader file.
            compile_time_config (dict): Configuration settings to modify the shader source code during compilation.

        Returns:
            gls.ShaderProgram: The compiled shader program.
        """
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
        """
        [Private]
        Updates the shader source code by replacing placeholders with values from the compile-time configuration.
        E.G. Initializing constants in the shader.

        Parameters:
            file_src (str): The source code of the shader.
            compile_time_config (dict): A dictionary containing placeholders and their replacement values.

        Returns:
            str: The updated shader source code with placeholders replaced.
        """
        for placeholder, value, in compile_time_config:
            file_src = file_src.replace(placeholder, value)

        return file_src
    
    @staticmethod
    def _get_uniform_handles(shader: gls.ShaderProgram, model_name: str, view_name: str, projection_name: str, texture_name: str = None, color_name: str = None) -> dict:
        """
        [Private]
        Retrieves the locations of the specified uniforms in the provided shader program.

        Parameters:
            shader (gls.ShaderProgram): The compiled shader program to query for uniform locations.
            model_name (str): The name of the model matrix uniform.
            view_name (str): The name of the view matrix uniform.
            projection_name (str): The name of the projection matrix uniform.
            texture_name (str, optional): The name of the texture uniform (default is None).
            color_name (str, optional): The name of the color uniform (default is None).

        Returns:
            dict: A dictionary mapping uniform names to their respective locations in the shader.
        """
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
        """
        [Private]
        Retrieves the locations of custom uniforms specified in the provided dictionary for the given shader program.

        Parameters:
            shader (gls.ShaderProgram): The compiled shader program to query for uniform locations.
            uniforms (dict): A dictionary where the keys are uniform handle names and the values are the corresponding uniform names.

        Returns:
            dict: A dictionary mapping custom uniform handle names to their respective locations in the shader.
        """
        custom_uniform_handles = {uniform_handle_name: gl.glGetUniformLocation(shader, uniform_name) for uniform_handle_name, uniform_name in uniforms.items()}

        return custom_uniform_handles
    
    def _init_handles(self, handles) -> None:
        """
        [Private]
        Initializes shader handles, specifically setting the texture uniform to the default texture unit (0) if available.

        Parameters:
            handles (dict): A dictionary of uniform handles to initialize.
        """
        if "texture" in handles:
            gl.glUniform1i(handles["texture"], 0)
        else:
            log.info(f"No texture handle for the shader {self.name}.")

    @staticmethod
    def _init_perspective_projection(projection_handle, fovy: float, aspect: float, near: float, far: float) -> None:
        """
        [Private]
        Initializes the perspective projection matrix and sets it to the specified shader uniform.

        Parameters:
            projection_handle (int): The handle for the projection matrix uniform in the shader.
            fovy (float): The field of view angle (in radians) for the perspective projection.
            aspect (float): The aspect ratio of the viewport.
            near (float): The near clipping plane distance.
            far (float): The far clipping plane distance.
        """
        projection_transform = pyrr.matrix44.create_perspective_projection(fovy = fovy, aspect = aspect, near = near, far = far, dtype = np.float32)
        gl.glUniformMatrix4fv(projection_handle, 1, gl.GL_FALSE, projection_transform)
    
    def use(self) -> None:
        """
        Activates the shader program for use.
        """
        gl.glUseProgram(self.shader)

    def set_view(self, view_transform) -> None:
        """
        Sets the view matrix uniform in the shader.
        """
        gl.glUniformMatrix4fv(self.uniform_handles["view"], 1, gl.GL_FALSE, view_transform)

    def destroy(self) -> None:
        """
        Destroys the shader program and frees associated resources.
        """
        gl.glDeleteProgram(self.shader)


class GraphicsEngine:
    def __init__(self, fovy: float, aspect: float, near: float, far: float, skybox_color: ColorRGBA) -> None:
        """
        Initializes the Graphics Engine, setting up various parameters, shaders, and draw instructions.

        Parameters:
            fovy (float): The field of view angle (in radians) for the perspective projection.
            aspect (float): The aspect ratio of the viewport.
            near (float): The near clipping plane distance.
            far (float): The far clipping plane distance.
            skybox_color (ColorRGBA): The color to set for the skybox background.
        """
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
    def _init_opengl(skybox_color: ColorRGBA) -> None:
        """
        [Private]
        Initializes OpenGL settings, including background color, depth testing, and blending.

        Parameters:
            skybox_color (ColorRGBA): The color to set as the background for the skybox.
        """
        gl.glClearColor(*skybox_color)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    # Public functions #
    def create_shader(self, name: str, vertex_path: str, fragment_path: str, fovy: float = None, aspect: float = None, near: float = None, far: float = None, compile_time_config: dict = {}) -> None:
        """
        Creates a new shader and validates its parameters before adding it to the pending shader creation list.

        Parameters:
            name (str): The name of the shader.
            vertex_path (str): The file path to the vertex shader source.
            fragment_path (str): The file path to the fragment shader source.
            fovy (float, optional): The field of view angle for the perspective projection. Defaults to the instance's fovy if not provided.
            aspect (float, optional): The aspect ratio for the perspective projection. Defaults to the instance's aspect if not provided.
            near (float, optional): The near clipping plane distance. Defaults to the instance's near if not provided.
            far (float, optional): The far clipping plane distance. Defaults to the instance's far if not provided.
            compile_time_config (dict, optional): A dictionary of configuration values to apply during shader compilation. Defaults to an empty dictionary.
        """
        if fovy is None:
            fovy = self.fovy
        if aspect is None:
            aspect = self.aspect
        if near is None:
            near = self.near
        if far is None:
            far = self.far

       # Validate parameters
        validate_types([('name', name, str),
                        ('vertex_path', vertex_path, str),
                        ('fragment_path', fragment_path, str),
                        ('fovy', fovy, Real),
                        ('aspect', aspect, Real),
                        ('near', near, Real),
                        ('far', far, Real),
                        ('compile_time_config', compile_time_config, dict)])
        
        if not far > near:
            raise ValueError(f"The value for far '{far}' is not greater than near '{near}'.")
        
        shader_warn_count = 0        
        if name in self.shaders:
            log.warn(f"Create shader called for '{name}', but the shader '{name}' already exists. The old shader will be overridden.")
            shader_warn_count += 1

        if name in self.pending_shader_creations:
            log.warn(f"Create shader called for '{name}', but creation of the shader '{name}' is already pending. The old shader will be overridden.")
            shader_warn_count += 1
        
        log.info(f"Shader creation for '{name}' passed variable validation with {shader_warn_count} warn{"s" if shader_warn_count != 1 else ""}.")

        self.pending_shader_creations.append((name, vertex_path, fragment_path, fovy, aspect, near, far, compile_time_config))

    def clear(self) -> None:
        """
        Clears pending draw list
        """
        self.pending_draw_instructions = []

    def use_shader(self, shader_name: str) -> None:
        """
        Uses the specified shader for subsequent draw calls.

        Parameters:
            shader_name (str): The name of the shader to use.
        """
        validate_type("shader_name", shader_name, str)

        instruction_args = (shader_name)
        self._add_draw_instruction(self._use_shader, instruction_args)

    def set_view(self, pos: Coordinate = None, rotation: ToDecide = None) -> None:
        """
        Set camera position and rotation relative to world

        To finish
        """
        validate_type("pos", pos, Coordinate)
        #validate_type("rotation", rotation, ToDecide) TODO

        instruction_args = (pos, rotation)
        self._add_draw_instruction(self._set_view, instruction_args)

    def set_skybox_color(self, skybox_color: ColorRGBA) -> None:
        """
        Sets the skybox color to be used for clearing the screen, in normalized RGBA.

        Parameters:
            skybox_color (ColorRGBA): The color to set as the skybox background in normalized RGBA format.
        """
        validate_type("skybox_color", skybox_color, ColorRGBA)

        self.pending_skybox_color = skybox_color
    
    def update(self) -> None:
        """
        Updates the internal state by transferring pending shader creations, skybox color, and draw instructions.
        This method ensures that changes to shader creation, skybox color, and draw instructions are applied safely using a lock to avoid race conditions.
        """
        with self.lock:
            self.new_shader_creations = self.pending_shader_creations
            self.new_skybox_color = self.pending_skybox_color

            self.active_draw_instructions = self.pending_draw_instructions

    def destroy(self) -> None:
        """
        Destroys all shaders in the engine, releasing any associated resources.
        """
        for shader in self.shaders.values():
            shader.destroy()

    # Private functions #
    def _create_shaders(self, new_shader_creations: list[tuple[str, str, str, float, float, float, float, dict]]) -> None:
        """
        [Private]
        Creates and stores new shaders based on the provided shader creation data.
        Processes a list of shader creation instructions and adds each shader to the engine's shader collection.

        Parameters:
            new_shader_creations (list): A list of tuples containing the shader creation details.
                Each tuple contains the following:
                - name (str): The name of the shader.
                - vertex_path (str): The path to the vertex shader file.
                - fragment_path (str): The path to the fragment shader file.
                - fovy (float): The field of view for the shader.
                - aspect (float): The aspect ratio for the shader.
                - near (float): The near clipping plane for the shader.
                - far (float): The far clipping plane for the shader.
                - compile_time_config (dict): A dictionary containing compile-time configuration for the shader.
        """
        for new_shader in new_shader_creations:
            name, vertex_path, fragment_path, fovy, aspect, near, far, compile_time_config = new_shader
            self.shaders[name] = Shader(name, vertex_path, fragment_path, fovy, aspect, near, far, compile_time_config)

    def _use_shader(self, shader_name: str) -> None:
        """
        [Private]
        Activates the shader by name and sets it as the active shader.
        Raises an error if the shader does not exist.

        Parameters:
            shader_name (str): The name of the shader to be activated.
        """
        if shader_name not in self.shaders:
            raise Exception("Shader '{shader_name}' does not exist in shaders.")

        self.shaders[shader_name].use()
        self.active_shader_name = shader_name

    def _add_draw_instruction(self, draw_function: Callable, args: tuple) -> None:
        """
        [Private]
        Adds a drawing instruction to the pending instructions list.
        Stores the draw function and its arguments in the pending queue, will be processed after update.

        Parameters:
            draw_function (Callable): The function to be called for drawing.
            args (tuple): The arguments to be passed to the drawing function.
        """
        self.pending_draw_instructions.append((draw_function, args))

    def _set_view(self, pos: Coordinate, rotation: ToDecide) -> None:
        """
        [Private]
        Sets the view transformation matrix for the currently active shader.
        This method calculates a view transformation matrix using a camera setup with a fixed position and orientation, then updates the shader's view uniform.

        Parameters:
            pos (Coordinate): The position of the camera (currently unused).
            rotation (ToDecide): The rotation of the camera (currently unused).
        """
        view_transform = pyrr.matrix44.create_look_at(
            eye = np.zeros(3, dtype = np.float32),
            target = np.array([1, 0, 0], dtype = np.float32),
            up = np.array([0, 1, 0], dtype = np.float32),
            dtype = np.float32
        )

        self.shaders[self.active_shader_name].set_view(view_transform)

    def _set_global_view(self, pos: Coordinate, rotation: ToDecide) -> None:
        """
        [Private]
        TODO
        Sets the view transformation matrix for all existing shaders.

        Parameters:
            pos (Coordinate): The position of the camera (currently unused).
            rotation (ToDecide): The rotation of the camera (currently unused).
        """
        pass

    @staticmethod
    def _update_skybox_color(skybox_color: ColorRGBA) -> None:
        """
        [Private]
        Updates the OpenGL clear color for the skybox.

        Parameters:
            skybox_color (ColorRGBA): The new color to be used for the skybox in normalised RGBA format.
        """
        if skybox_color is not None:
            log.info(f"Updating skybox color to {skybox_color}")
            gl.glClearColor(*skybox_color)

    @staticmethod
    def _clear_screen() -> None:
        """
        [Private]
        Clears the screen by resetting the color and depth buffers.
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def _complete_draw_instructions(self, draw_instructions) -> None:
        """
        [Private]
        Executes all draw instructions with passed arguments.
        Processes and executes each draw instruction in order, calling the respective draw function with the provided arguments.

        Parameters:
            draw_instructions (list): A list of tuples, where each tuple contains a draw function and its associated arguments.
        """
        for draw_instruction in draw_instructions:
            draw_function, args = draw_instruction

            draw_function(*args)
    
    def _render(self) -> None:
        """
        [Private]
        Executes the render process by updating shaders, setting skybox color, clearing the screen, and processing draw instructions.

        This method handles the rendering process by:
        1. Creating new shaders if necessary.
        2. Updating the skybox color.
        3. Clearing the screen.
        4. Using the default shader.
        5. Completing draw instructions.
        """
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