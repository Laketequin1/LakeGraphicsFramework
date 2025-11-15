### Imports ###
import glfw
import numpy as np
import pyrr
import atexit
import ctypes
import uuid
import OpenGL.GL as gl
import OpenGL.GL.shaders as gls
from variable_type_validation import *
from MessageLogger import MessageLogger as log
from threading import Lock, Event, Thread
from typing import Callable, Literal
import copy
from safe_file_readlines import safe_file_readlines
import sys
from PIL import Image
from key_handler import KEY_NAMES, KEYS, get_key_name

### Type hints ###
ColorRGBA = Tuple[float, float, float, float]
ToDecide = Any # Variables with unknown types
GLUniformFunction = Literal[
    "glUniform1f",
    "glUniform2f",
    "glUniform3f",
    "glUniform4f",
    "glUniform1i",
    "glUniform2i",
    "glUniform3i",
    "glUniform4i",
    "glUniform1ui",
    "glUniform2ui",
    "glUniform3ui",
    "glUniform4ui",
    "glUniform1fv",
    "glUniform2fv",
    "glUniform3fv",
    "glUniform4fv",
    "glUniform1iv",
    "glUniform2iv",
    "glUniform3iv",
    "glUniform4iv",
    "glUniform1uiv",
    "glUniform2uiv",
    "glUniform3uiv",
    "glUniform4uiv",
    "glUniformMatrix2fv",
    "glUniformMatrix3fv",
    "glUniformMatrix4fv",
    "glUniformMatrix2x3fv",
    "glUniformMatrix3x2fv",
    "glUniformMatrix2x4fv",
    "glUniformMatrix4x2fv",
    "glUniformMatrix3x4fv",
    "glUniformMatrix4x3fv"
]


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


### Exit Handling ###
def exit_handler() -> None:
    """
    Runs before main threads terminates.
    """
    #events["exit"].set()
    log.info("Program terminating")
    glfw.terminate()

atexit.register(exit_handler)


### Classes ###
class Window:
    """
    Creates an application window with a rendering loop.

    Handles GLFW initialization, window creation, input setup, event processing, and delegates rendering tasks to a GraphicsEngine.
    """
    def __init__(self, size: Size = (0, 0), caption: str = "", fullscreen: bool = False, windowed: bool = True, vsync: bool = True, max_fps: int = 0, raw_mouse_input: bool = True, center_cursor_on_creation: bool = True, hide_cursor: bool = False, fovy: float = DEFAULT_FOVY, near: float = DEFAULT_NEAR, far: float = DEFAULT_FAR, skybox_color: ColorRGBA = DEFAULT_SKYBOX_COLOR) -> None:
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
            center_cursor_on_creation (bool): Whether the cursor is centered in the window.
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
                        ('center_cursor_on_creation', center_cursor_on_creation, bool),
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
        self.center_cursor_on_creation = center_cursor_on_creation
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
        self.window_requesting_close = False
        self.should_close_event = Event()
        self.lock = Lock()
        self.key_states = {}
        self.graphics_engine = GraphicsEngine(fovy, self.aspect_ratio, near, far, skybox_color)

    def start(self):
        glfw.make_context_current(None)
        
        self.render_thread = Thread(target=self._render_loop)
        self.render_thread.start()

        glfw.set_key_callback(self.window, self._key_callback)
    
    def _render_loop(self) -> None:
        glfw.make_context_current(self.window)

        while not self.should_close_event.is_set():
            gl.glFlush() # Wait for pipeline

            self._gl_check_error()

            self.graphics_engine._render()
            glfw.swap_buffers(self.window)
            #self._tick() BAD BAD BAD BAD BAD BAD
        
        self._close()

    def poll_events(self) -> None:
        """
        Processes a single application tick by polling GLFW events and handling window-related events.
        """
        for unique_key_name, unique_key_states in list(self.key_states.items()):
            if unique_key_states["released"]:
                del self.key_states[unique_key_name]

        glfw.poll_events()
        log.info(f"Keystates: {self.key_states}")
            
        self._handle_window_events()
    
    def get_key_states(self) -> dict:
        """
        [Public]
        Get all keystates presses recieved by the GLFW window after the last poll_events(). Only contains pressed keys.
        
        Note:
            poll_events() updates the keys.

        Returns:
            dict {"SOMEKEY": {"pressed": bool, "released": bool, "num_lock": bool, "caps_lock": bool, "super_key": bool, "alt": bool, "control": bool, "shift": bool}}: Key state.
        """
        return copy.deepcopy(self.key_states)
    
    def get_requesting_close(self) -> bool:
        """
        [Public]
        Get whether the GLFW window is requesting to close.

        Returns:
            bool: True if the window should close, False otherwise.
        """
        return copy.deepcopy(self.window_requesting_close)

    def close(self) -> None:
        """
        [Public]
        Requests to close the GLFW window.
        """
        log.info("Window close requested.")
        
        self.should_close_event.set()

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
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE) # Necessary for macOS
        
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

    def _key_callback(self, window_, key, scancode_, action, mods) -> None:
        """
        [Private]
        Key callback to updates the local saved key states.
        """
        log.info((get_key_name(key), "KEYDOWN" if action else "KEYUP", mods))

        if get_key_name(key) not in self.key_states:
            self.key_states[get_key_name(key)] = {}
        unique_key_states = self.key_states[get_key_name(key)]

        MOD_SHIFT = 0x0001
        MOD_CONTROL = 0x0002
        MOD_ALT = 0x0004
        MOD_SUPER = 0x0008
        MOD_CAPS_LOCK = 0x0010
        MOD_NUM_LOCK = 0x0020

        if action:
            unique_key_states["pressed"] = True
            unique_key_states["released"] = False

            unique_key_states["num_lock"] = (mods & MOD_NUM_LOCK) != 0
            unique_key_states["caps_lock"] = (mods & MOD_CAPS_LOCK) != 0
            unique_key_states["super_key"] = (mods & MOD_SUPER) != 0
            unique_key_states["alt"] = (mods & MOD_ALT) != 0
            unique_key_states["control"] = (mods & MOD_CONTROL) != 0
            unique_key_states["shift"] = (mods & MOD_SHIFT) != 0
        else:
            unique_key_states["released"] = True

    def _handle_window_events(self) -> bool:
        """
        [Private]
        Handle GLFW events to check if the window should close.
        """
        if glfw.window_should_close(self.window):
            self.window_requesting_close = True
            log.info("Window event requesting close")
    
    def _close(self) -> None:
        """
        [Private]
        Close the GLFW window and terminate glfw.
        """
        log.info("Closing window...")
        self.graphics_engine.destroy()
        glfw.destroy_window(self.window)
        glfw.terminate()
        log.info("Window closed.")
        sys.exit(0)


class Mesh:
    def __init__(self, path) -> None:
        log.info(f"Creating mesh for {path}")

        # x, y, z, s, t, nx, ny, nz
        self.vertices = self.load_mesh(path)
        
        # Each vertex consists of 3 components (x, y, z)
        self.vertex_count = len(self.vertices) // 3
        self.vertices = np.array(self.vertices, dtype=np.float32)
        
        # Vertex Array Object (vao) stores buffer attributes (defines how vertex data is laid out in memory, etc)
        self.vao = gl.glGenVertexArrays(1)
        # Activate Vertex Array Object
        gl.glBindVertexArray(self.vao)
        
        # Vertex Buffer Object (vbo) stores raw data (vertex positions, normals, colors, etc)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW) # Upload the vertex data to the GPU

        # Add attribute pointer for position location in buffer so gpu can find vertex data in memory
        # Location 1 - Postion
        gl.glEnableVertexAttribArray(0)
        # Location, number of floats, format (float), gl.GL_FALSE, stride (total length of vertex, 4 bytes times number of floats), ctypes of starting position in bytes (void pointer expected)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 32, ctypes.c_void_p(0))
        
        # Location 2 - ST
        gl.glEnableVertexAttribArray(1)
        # Location, number of floats, format (float), gl.GL_FALSE, stride (total length of vertex, 4 bytes times number of floats), ctypes of starting position in bytes (void pointer expected)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 32, ctypes.c_void_p(12))
        
        # Location 3 - Normal
        gl.glEnableVertexAttribArray(2)
        # Location, number of floats, format (float), gl.GL_FALSE, stride (total length of vertex, 4 bytes times number of floats), ctypes of starting position in bytes (void pointer expected)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 32, ctypes.c_void_p(20))

        log.info(f"Created mesh for {path}")
    
    @staticmethod
    def load_mesh(filepath):
        vertices = []
        flags = {"v": [], "vt": [], "vn": []}
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                line.replace("\n", "")
                
                first_space = line.find(" ")
                flag = line[0:first_space]
                
                if flag in flags.keys():
                    line = line.replace(flag + " ", "")
                    line = line.split(" ")
                    flags[flag].append([float(x) for x in line])
                elif flag == "f":
                    line = line.replace(flag + " ", "")
                    line = line.split(" ")
                    
                    face_vertices = []
                    face_textures = []
                    face_normals = []
                    
                    for vertex in line:
                        l = vertex.split("/")
                        face_vertices.append(flags["v"][int(l[0]) - 1])
                        face_textures.append(flags["vt"][int(l[1]) - 1])
                        face_normals.append(flags["vn"][int(l[2]) - 1])

                    triangles_in_face = len(line) - 2
                    vertex_order = []

                    for x in range(triangles_in_face):
                        vertex_order.extend((0, x + 1, x + 2))
                    for x in vertex_order:
                        vertices.extend((*face_vertices[x], *face_textures[x], *face_normals[x]))
        
        return vertices
    
    @staticmethod
    def get_fullscreen_quad_vertices() -> list[float]:
        """
        Generates vertex data for a quad that covers the entire screen/viewport in NDC.

        The quad is composed of two triangles. The vertex data includes positions (XYZ),
        texture coordinates (UV), and placeholder normals (XYZ), matching the format
        returned by `load_mesh`.

        Positions span from (-1, -1) to (1, 1) in X and Y. Z is 0.
        Texture coordinates span from (0, 0) to (1, 1).
        Normals are set to point along the positive Z-axis (0, 0, 1).

        Returns:
            A flat list of 48 floats (6 vertices * (3 pos + 2 tex + 3 norm) components).
            [v1_x, v1_y, v1_z, v1_u, v1_v, v1_nx, v1_ny, v1_nz, v2_x, ...]
        """
        # Vertices definitions: Pos (x, y, z), TexCoord (u, v), Normal (nx, ny, nz)
        # Note: Texture coords often have Y=0 at bottom, Y=1 at top.
        #       Normals are arbitrary for a 2D quad, setting to +Z.
        v1_tl = [-1.0,  1.0, 0.0,  0.0, 1.0,  0.0, 0.0, 1.0] # Top-left
        v2_bl = [-1.0, -1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 1.0] # Bottom-left
        v3_br = [ 1.0, -1.0, 0.0,  1.0, 0.0,  0.0, 0.0, 1.0] # Bottom-right
        v4_tr = [ 1.0,  1.0, 0.0,  1.0, 1.0,  0.0, 0.0, 1.0] # Top-right

        # Triangle 1: Top-left, Bottom-left, Bottom-right
        # Triangle 2: Top-left, Bottom-right, Top-right
        quad_vertices = []
        quad_vertices.extend(v1_tl)
        quad_vertices.extend(v2_bl)
        quad_vertices.extend(v3_br)

        quad_vertices.extend(v1_tl)
        quad_vertices.extend(v3_br)
        quad_vertices.extend(v4_tr)

        return quad_vertices
    
    def destroy(self):
        # Remove allocated memory
        gl.glDeleteVertexArrays(1, (self.vao, ))
        gl.glDeleteBuffers(1, (self.vbo, ))
        log.info("Destroyed mesh")


class Material:
    def __init__(self, filepath):
        log.info(f"Creating material for {filepath}")
        # Allocate space where texture will be stored
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        
        # S is horizontal of a texture, T is the vertical of a texture, GL_REPEAT means image will loop if S or T over/under 1. MIN_FILTER is downsizing. MAG_FILTER is enlarging.
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        # Load image, then get height, and the images data
        image = Image.open(filepath).convert("RGBA")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image_width, image_height = image.size
        image_data = image.tobytes("raw", "RGBA")
        
        # Get data for image, then generate the mipmap
        # Texture location, mipmap level, format image is stored as, width, height, border color, input image format, data format, image data
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image_width, image_height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image_data)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        log.info(f"Created material for {filepath}")
        
    def use(self, texture_layer: int = 0) -> None:
        """
        Bind the texture to the provided texture layer.
        """
        log.info(f"Binding texture {texture_layer}.")
        # Validate texture layer
        validate_type('texture_layer', texture_layer, int)
        
        gl_texture_n = getattr(gl, f"GL_TEXTURE{texture_layer}")

        gl.glActiveTexture(gl_texture_n)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        log.info(f"Binded texture {texture_layer}.")
        
    def destroy(self):
        # Remove allocated memory
        gl.glDeleteTextures(1, (self.texture, ))
        log.info("Destroyed material")


class Object:
    def __init__(self, mesh: Mesh, materials: Tuple[Material, ...], pos: Tuple[float, float, float], rotation: pyrr.Quaternion, scale: Tuple[float, float, float]):                
        self.pos = np.array(pos, dtype=np.float32)
        self.rotation = rotation
        self.scale = np.array(scale, dtype=np.float32)

        self.mesh = mesh
        self.materials = materials

        self.id = uuid.uuid4()

        log.info(f"Created object with id {self.id}")

    def render(self, model_matrix_handle):
        log.info(f"Rendering object {self.id}.")

        if self.materials:
            self.materials[0].use(0) # TODO Multiple materials - Normal maps etc
            log.dev("TODO Multiple materials - Normal maps etc")
        
        model_transform = pyrr.matrix44.create_identity(dtype = np.float32)

        # Scale
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_scale(self.scale, dtype = np.float32)
        )

        # Rotate around origin
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_quaternion(self.rotation, dtype = np.float32)
        )

        # Translate
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_translation(self.pos, dtype = np.float32)
        )
        
        # Complete transform
        gl.glUniformMatrix4fv(model_matrix_handle, 1, gl.GL_FALSE, model_transform)
        gl.glBindVertexArray(self.mesh.vao)
        
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.mesh.vertex_count)
        log.info(f"Rendered object {self.id}.")
    
    def destroy(self): # TO ADD CALL!!
        self.mesh.destroy()
        for material in self.materials:
            material.destroy()


class Shader:
    def __init__(self, vertex_path: str, fragment_path: str, fovy: float, aspect: float, near: float, far: float, model_name: str = DEFAULT_UNIFORM_NAMES["model"], view_name: str = DEFAULT_UNIFORM_NAMES["view"], projection_name: str = DEFAULT_UNIFORM_NAMES["projection"], texture_name: str = None, color_name: str = DEFAULT_UNIFORM_NAMES["color"], custom_uniform_names: dict = {}, compile_time_config: dict = {}) -> None:
        """
        [Private]
        Initializes the shader with the specified vertex and fragment shader paths, along with projection and model settings.
        Sets up uniform handles and initializes perspective projection.

        Parameters:
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
        self.id = uuid.uuid4()

        log.info(f"Creating a shader with id {self.id}")

        # Create shader
        self.shader = self._create_shader(vertex_path, fragment_path, compile_time_config)

        # Get shader handles
        self.uniform_handles = self._source_uniform_handles(self.shader, model_name, view_name, projection_name, texture_name, color_name)
        self.custom_uniform_handles = self._source_custom_uniform_handles(self.shader, custom_uniform_names)

        self.use()

        # Initilize shader handles
        self._init_handles(self.uniform_handles)
        if "projection" in self.uniform_handles:
            self._init_perspective_projection(self.uniform_handles["projection"], fovy, aspect, near, far)

        log.info(f"Shader {self.id} creation passed. Variables: {self.__dict__}")

    def _create_shader(self, vertex_path: str, fragment_path: str, compile_time_config: dict) -> gls.ShaderProgram:
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
        log.info(f"Compiling shaders for {self.id}. Vertex path: '{vertex_path}', Frag path: {fragment_path}")
        
        vertex_src_lines = safe_file_readlines(vertex_path)
        fragment_src_lines = safe_file_readlines(fragment_path)

        vertex_program = self._update_file_config(vertex_src_lines, compile_time_config, vertex_path)
        fragment_program = self._update_file_config(fragment_src_lines, compile_time_config, fragment_path)
        
        try:
            shader = gls.compileProgram(
                gls.compileShader(vertex_program, gl.GL_VERTEX_SHADER),
                gls.compileShader(fragment_program, gl.GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            log.error(f"Failed to compile shader {self.id}.")
            raise e
        
        return shader
    
    def _update_file_config(self, file_src_lines: list[str], compile_time_config: dict, shader_path: str) -> str:
        """
        [Private]
        Updates the shader source code by replacing placeholders with values from the compile-time configuration.
        E.G. Initializing constants in the shader.

        Parameters:
            file_src_lines (list[str]): The source code of the shader, with readlines.
            compile_time_config (dict): A dictionary containing placeholders and their replacement values.

        Returns:
            str: The updated shader source code with placeholders replaced.
        """
        for placeholder, value in compile_time_config.items():
            line_modifications_counter = 0

            for i in range(len(file_src_lines)):
                on_line_occurances = file_src_lines[i].count(placeholder)
                if on_line_occurances == 0:
                    continue
                file_src_lines[i] = file_src_lines[i].replace(placeholder, value)
                line_modifications_counter += on_line_occurances

            if line_modifications_counter == 0:
                log.warn(f"Placeholder text value '{placeholder}' not found in the shader file '{shader_path}' of {self.id}.")
            else:
                log.info(f"Placeholder text value '{placeholder}' occured {line_modifications_counter} times in the shader file '{shader_path}' of {self.id}.")

        return file_src_lines
    
    def _source_uniform_handles(self, shader: gls.ShaderProgram, model_name: str | None, view_name: str, projection_name: str, texture_name: str, color_name: str) -> dict:
        """
        [Private]
        Retrieves the locations of the specified uniforms in the provided shader program.
        Uniform names are optional, but will not be initilised if None.

        Parameters:
            shader (gls.ShaderProgram): The compiled shader program to query for uniform locations.
            model_name (str | None): The name of the model matrix uniform.
            view_name (str | None): The name of the view matrix uniform.
            projection_name (str | None): The name of the projection matrix uniform.
            texture_name (str | None): The name of the texture uniform.
            color_name (str | None): The name of the color uniform.

        Returns:
            dict: A dictionary mapping uniform names to their respective locations in the shader.
        """
        uniforms = {
            "model": model_name,
            "view": view_name,
            "projection": projection_name,
            "texture": texture_name,
            "color": color_name
        }

        uniform_handles = {uniform_handle_name: gl.glGetUniformLocation(shader, uniform_name) for uniform_handle_name, uniform_name in uniforms.items() if uniform_name is not None}

        log.info(f"Given uniform handles for {self.id}: {uniform_handles}")

        for key, value in uniform_handles.items():
            if value == -1:
                log.error(f"Uniform handle not found in created shader {self.id}: {key}")

        return uniform_handles
    
    def _source_custom_uniform_handles(self, shader: gls.ShaderProgram, uniforms: dict) -> dict:
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

        log.info(f"Given uniform handles for {self.id}: {custom_uniform_handles}")

        for key, value in custom_uniform_handles.items():
            if value == -1:
                log.error(f"Custom uniform handle not found in created shader {self.id}: {key}")

        return custom_uniform_handles
    
    def _init_handles(self, handles) -> None:
        """
        [Private]
        Initializes shader handles, specifically setting the texture uniform to the default texture unit (0) if available.

        Parameters:
            handles (dict): A dictionary of uniform handles to initialize.
        """
        if "texture" in handles:
            if handles["texture"] == -1:
                log.error(f"Texture handle promised, however was not found in shader {self.id}. Can't load texture.")

            gl.glUniform1i(handles["texture"], 0)
            log.info("Texture loaded for {self.id}")
        else:
            log.info(f"No texture handle for {self.id}")

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

    def get_uniform_handles(self) -> dict:
        """
        Returns a map of all uniforms to their handle locations for the given shader program.

        Returns:
            dict: A dictionary mapping uniform handle names to their respective locations in the shader.
        """
        uniform_handles = self.custom_uniform_handles
        uniform_handles.update(self.uniform_handles)

        return uniform_handles
    
    def get_uniform_handle(self, handle_name) -> dict:
        """
        Returns a map of a specified uniform handle name to its locations for the given shader program.

        Parameters:
            handle_name (dict): The name of the handle.

        Returns:
            dict: A dictionary mapping uniform handle names to their respective locations in the shader.
        """
        uniform_handles = self.get_uniform_handles()

        if handle_name not in uniform_handles:
            log.error(f"Handle name '{handle_name}' not in uniform_handles '{uniform_handles}' of {self.id}.")
            return None

        return uniform_handles[handle_name]
    
    def use(self) -> None:
        """
        Activates the shader program for use.
        """
        gl.glUseProgram(self.shader)
        #self.set_custom_handle()

    def set_view(self, view_transform) -> None:
        """
        Sets the view matrix uniform in the shader.
        """
        log.dev("TODO view validation")
        gl.glUniformMatrix4fv(self.uniform_handles["view"], 1, gl.GL_FALSE, view_transform)

    def set_model(self, model) -> None:
        """
        Sets the model uniform in the shader.
        """
        log.dev("TODO model validation and etc")
        gl.glUniformMatrix4fv(self.uniform_handles["view"], 1, gl.GL_FALSE, self._view_transform_TOFIX)

    def set_custom_handle(self, gl_uniform_func_name: GLUniformFunction, handle_name: str, args: tuple) -> None:
        """
        Sets a custom handle value.

        Parameters:
            gl_uniform_func_name (GLUniformFunction): The function which will be called with the shader handle. Options below.
            handle_name (str): Name of the custom handle.
            args (tuple): Tuple of all arguments to be passed after the uniform handle.
            
        GLUniformFunction = Literal[
            "glUniform1f",
            "glUniform2f",
            "glUniform3f",
            "glUniform4f",
            "glUniform1i",
            "glUniform2i",
            "glUniform3i",
            "glUniform4i",
            "glUniform1ui",
            "glUniform2ui",
            "glUniform3ui",
            "glUniform4ui",
            "glUniform1fv",
            "glUniform2fv",
            "glUniform3fv",
            "glUniform4fv",
            "glUniform1iv",
            "glUniform2iv",
            "glUniform3iv",
            "glUniform4iv",
            "glUniform1uiv",
            "glUniform2uiv",
            "glUniform3uiv",
            "glUniform4uiv",
            "glUniformMatrix2fv",
            "glUniformMatrix3fv",
            "glUniformMatrix4fv",
            "glUniformMatrix2x3fv",
            "glUniformMatrix3x2fv",
            "glUniformMatrix2x4fv",
            "glUniformMatrix4x2fv",
            "glUniformMatrix3x4fv",
            "glUniformMatrix4x3fv"
        ]
        """
        log.dev("Check this later, the type thing is weird, should it be a string or the funct??")
        if not hasattr(gl, gl_uniform_func_name):
            log.error(f"gl_uniform_func_name was not a literal GLUniformFunction type, '{gl_uniform_func_name}' is of type {type(gl_uniform_func_name)} in {self.id}")

        gl_uniform_func = getattr(gl, gl_uniform_func_name)

        gl_uniform_func(self.uniform_handles[handle_name], *args)
        #gl.glUniformMatrix4fv(self.uniform_handles["view"], 1, gl.GL_FALSE, view_transform) EXAMPLE OF ABOVE LINE IN USE

    def destroy(self) -> None:
        """
        Destroys the shader program and frees associated resources.
        """
        gl.glDeleteProgram(self.shader)
        log.info(f"Destroyed shader for {self.id}")


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
        self.active_shader_id = None

        self.objects = {}

        self._init_opengl(skybox_color)

        # Initilize shader
        self.default_shader_id = 0
        self.shaders[self.default_shader_id] = Shader(SHADERS_PATH + DEFAULT_VERT_SHADER, SHADERS_PATH + DEFAULT_FRAG_SHADER, fovy, aspect, near, far)
        # Use default shader
        self._use_shader(self.default_shader_id)

        # Pre-draw instructions
        self.pending_skybox_color = None
        self.new_skybox_color = None

        self.pending_shader_creations = []
        self.new_shader_creations = []

        self.pending_object_creations = []
        self.new_object_creations = []

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
    def create_shader(self, vertex_path: str, fragment_path: str, fovy: float = None, aspect: float = None, near: float = None, far: float = None, model_name: str = DEFAULT_UNIFORM_NAMES["model"], view_name: str = DEFAULT_UNIFORM_NAMES["view"], projection_name: str = DEFAULT_UNIFORM_NAMES["projection"], texture_name: str = None, color_name: str = DEFAULT_UNIFORM_NAMES["color"], custom_uniform_names: dict = {}, compile_time_config: dict = {}) -> int:
        """
        Creates a new shader and validates its parameters before adding it to the pending shader creation list.

        Parameters:
            vertex_path (str): The file path to the vertex shader source.
            fragment_path (str): The file path to the fragment shader source.
            fovy (float, optional): The field of view angle for the perspective projection. Defaults to the instance's fovy if not provided.
            aspect (float, optional): The aspect ratio for the perspective projection. Defaults to the instance's aspect if not provided.
            near (float, optional): The near clipping plane distance. Defaults to the instance's near if not provided.
            far (float, optional): The far clipping plane distance. Defaults to the instance's far if not provided.
            model_name (str): The name for the model matrix uniform (default is "model").
            view_name (str): The name for the view matrix uniform (default is "view").
            projection_name (str): The name for the projection matrix uniform (default is "projection").
            texture_name (str): The name for the texture uniform (default is None).
            color_name (str): The name for the color uniform (default is "color").
            custom_uniform_names (dict): Optional custom uniform names to be handled (default is empty dictionary).
            compile_time_config (dict): Optional configuration settings for shader compilation (default is empty dictionary).
            
        Returns:
            int: The id of the created shader object.        
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
        validate_types([('vertex_path', vertex_path, str),
                        ('fragment_path', fragment_path, str),
                        ('fovy', fovy, Real),
                        ('aspect', aspect, Real),
                        ('near', near, Real),
                        ('far', far, Real),
                        ('custom_uniform_names', custom_uniform_names, dict),
                        ('compile_time_config', compile_time_config, dict)])
        
        if model_name:
            validate_type(model_name, 'model_name', str)
        if view_name:
            validate_type(view_name, 'view_name', str)
        if projection_name:
            validate_type(projection_name, 'projection_name', str)
        if texture_name:
            validate_type(texture_name, 'texture_name', str)
        if color_name:
            validate_type(color_name, 'color_name', str)
                
        if not far > near:
            raise ValueError(f"The value for far '{far}' is not greater than near '{near}'.")
        
        log.info(f"Shader creation passed variable validation.")

        shader_id = len(self.shaders)

        self.pending_shader_creations.append((shader_id, vertex_path, fragment_path, fovy, aspect, near, far, model_name, view_name, projection_name, texture_name, color_name, custom_uniform_names, compile_time_config))
        return shader_id
    
    def create_object(self, mesh_path: str, material_paths: list[str, ], pos: Tuple[float, float, float] = np.zeros(3), rotation: pyrr.Quaternion = pyrr.quaternion.create(), scale: Tuple[float, float, float] = np.ones(3)):
        """
        Create and return a new object id.

        Parameters:
            mesh_path (str): The file path to the vertex shader source.
            material_paths (list): The file path to the fragment shader source.
            pos (Tuple[float, float, float]): Object world position.
            rotation (pyrr.Quaternion): World rotation in quaternions.
            scale (Tuple[float, float, float]): The scale/stretch for x y and z.

        Returns:
            int: The id of the created object.
        """
        validate_types([('mesh_path', mesh_path, str),
                        ('material_paths', material_paths, list)])
                        #('pos', pos, Tuple),
                        #('rotation', rotation, pyrr.Quaternion),
                        #('scale', scale, Tuple)

        log.dev("Todo type validation for pos, rotation, scale.")

        log.info(f"Object creation passed variable validation.")

        log.warn(f"Possible multithread issue here on this line")
        object_id = len(self.objects) + len(self.pending_object_creations)

        self.pending_object_creations.append((object_id, mesh_path, material_paths, pos, rotation, scale))
        return object_id

    def clear(self) -> None:
        """
        Clears pending draw lists.
        """
        self.pending_draw_instructions = []

    def fill(self, fill_color: ColorRGBA) -> None:
        """
        Clears the screen by filling everything with the specified color, in normalized RGBA.

        Parameters:
            fill_color (ColorRGBA): The color to fill the screen in normalized RGBA format.
        """
        validate_type("fill_color", fill_color, ColorRGBA)

        instruction_args = (fill_color,)
        self._add_draw_instruction(self._fill, instruction_args)

    def quad_fill(self) -> None:
        """
        Draws a single quad that fills the entire screen / viewport.

        Typically used as geometry for applying shader effects (post-processing / background generation) across the entire screen.
        """
        self._add_draw_instruction(self._quad_fill)

    def use_shader(self, shader_id: int) -> None:
        """
        Uses the specified shader for subsequent draw calls.

        Parameters:
            shader_id (int): The id of the shader to use.
        """
        validate_type("shader_id", shader_id, int)

        instruction_args = (shader_id,)
        self._add_draw_instruction(self._use_shader, instruction_args)

    def set_view(self, pos: Coordinate = None, rotation: pyrr.Quaternion = None) -> None:
        """
        Set camera position and rotation relative to world.

        TODO
        """
        #validate_type("pos", pos, Coordinate)
        #validate_type("rotation", rotation, pyrr.Quaternion)

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

    def modify_object_pos(self, object_id: int, new_pos: Tuple[float, float, float]) -> None:
        """
        Modifys the objects position.

        Parameters:
            object_id (int): The id of the object to be rendered.
            new_pos (Tuple[float, float, float]): The new object position in world space.
        """
        validate_type("object_id", object_id, int)

        instruction_args = (object_id, new_pos)
        self._add_draw_instruction(self._modify_object_pos, instruction_args)

    def render_object(self, object_id: int) -> None:
        """
        Renders the passed object id in the window.

        Parameters:
            object_id (int): The id of the object to be rendered.
        """
        validate_type("object_id", object_id, int)

        log.dev("TODO check object_id is valid and exists")

        instruction_args = (object_id, )
        self._add_draw_instruction(self._render_object, instruction_args)
    
    def update(self, preserve_draw_instructions: bool = False) -> None:
        """
        Updates the internal state by transferring pending shader creations, skybox color, and draw instructions. By default clears draw instruction.
        This method ensures that changes to shader creation, skybox color, and draw instructions are applied safely using a lock to avoid race conditions.

        Parameters:
            preserve_draw_instructions (bool): When true won't clear draw instructions. Future update calls will append to existing draw instructions.
        """
        with self.lock:
            if len(self.pending_shader_creations) > 0:
                self.new_shader_creations.extend(self.pending_shader_creations)
                self.pending_shader_creations = []

            if len(self.pending_object_creations) > 0:
                self.new_object_creations.extend(self.pending_object_creations)
                self.pending_object_creations = []
            
            if self.pending_skybox_color is not None:
                self.new_skybox_color = self.pending_skybox_color
                self.pending_skybox_color = None

            self.active_draw_instructions = self.pending_draw_instructions

            if not preserve_draw_instructions:
                self.clear()

    def destroy(self) -> None:
        """
        Destroys all shaders in the engine, releasing any associated resources.
        """
        log.info("Destroying shaders in GraphicsEngine.")
        for shader in self.shaders.values():
            shader.destroy()

    # Private functions #
    def _create_shaders(self, new_shader_creations: list[tuple[int, str, str, float, float, float, float, str, str, str, str, str, dict, dict], ]) -> None:
        """
        [Private]
        Creates and stores new shaders based on the provided shader creation data.
        Processes a list of shader creation instructions and adds each shader to the engine's shader collection.

        Parameters:
            new_shader_creations (list): A list of tuples containing the shader creation details:
                - shader_id (int): The id of the shader.
                - vertex_path (str): The path to the vertex shader file.
                - fragment_path (str): The path to the fragment shader file.
                - fovy (float): The field of view for the shader.
                - aspect (float): The aspect ratio for the shader.
                - near (float): The near clipping plane for the shader.
                - far (float): The far clipping plane for the shader.
                - model_name (str): The name for the model matrix uniform.
                - view_name (str): The name for the view matrix uniform.
                - projection_name (str): The name for the projection matrix uniform.
                - texture_name (str): The name for the texture uniform.
                - color_name (str): The name for the color uniform.
                - custom_uniform_names (dict): A dictionary containing extra uniform names for the shader.
                - compile_time_config (dict): A dictionary containing compile-time configuration for the shader.
        """
        for new_shader in new_shader_creations:
            shader_id, vertex_path, fragment_path, fovy, aspect, near, far, model_name, view_name, projection_name, texture_name, color_name, custom_uniform_names, compile_time_config = new_shader
            self.shaders[shader_id] = Shader(vertex_path, fragment_path, fovy, aspect, near, far, model_name, view_name, projection_name, texture_name, color_name, custom_uniform_names, compile_time_config)

    def _create_objects(self, new_object_creations: list[tuple[int, str, list, Tuple[float, float, float], pyrr.Quaternion, Tuple[float, float, float]], ]): #mesh_path: str, material_paths: list[str, ], pos: Tuple[float, float, float] = np.zeros(3), rotation: pyrr.Quaternion = pyrr.quaternion.create(), scale: Tuple[float, float, float] = np.ones(3)):
        """
        [Private]
        Create a new object.

        Parameters:
            new_object_creations (list): A list of tuples containing the object creation details:
                - object_id (int): The id of the object.
                - mesh_path (str): The file path to the vertex shader source.
                - material_paths (list): The file path to the fragment shader source.
                - pos (Tuple[float, float, float]): Object world position.
                - rotation (pyrr.Quaternion): World rotation in quaternions.
                - scale (Tuple[float, float, float]): The scale/stretch for x y and z.
        """      

        for new_object in new_object_creations:
            object_id, mesh_path, material_paths, pos, rotation, scale = new_object

            mesh = Mesh(mesh_path)
            materials = [Material(material_path) for material_path in material_paths]

            self.objects[object_id] = Object(mesh, materials, pos, rotation, scale)

    def _use_shader(self, shader_id: int) -> None:
        """
        [Private]
        Activates the shader by id and sets it as the active shader.
        Raises an error if the shader does not exist.

        Parameters:
            shader_id (int): The id of the shader to be activated.
        """
        if shader_id not in self.shaders:
            raise Exception(f"Shader '{shader_id}' does not exist in shaders.")

        self.shaders[shader_id].use()
        self.active_shader_id = shader_id

    def _add_draw_instruction(self, draw_function: Callable, args: tuple = tuple()) -> None:
        """
        [Private]
        Adds a drawing instruction to the pending instructions list.
        Stores the draw function and its arguments in the pending queue, will be processed after update.

        Parameters:
            draw_function (Callable): The function to be called for drawing.
            args (tuple): The arguments to be passed to the drawing function.
        """
        self.pending_draw_instructions.append((draw_function, tuple(args)))

    @staticmethod
    def _fill(fill_color: ColorRGBA) -> None:
        """
        [Private]
        Clears the screen by filling everything with the specified color, in normalized RGBA.

        Parameters:
            fill_color (ColorRGBA): The color to fill the screen in normalized RGBA format.
        """
        gl.glClearColor(*fill_color)

    @staticmethod
    def _quad_fill() -> None:
        """
        [Private]
        Draws a single quad that fills the entire screen / viewport.

        Typically used as geometry for applying shader effects (post-processing / background generation) across the entire screen.
        """
        pass

    def _set_view(self, pos: Coordinate, rotation: ToDecide) -> None: #TODO
        """
        [Private]
        Sets the view transformation matrix for the currently active shader.
        This method calculates a view transformation matrix using a camera setup with a fixed position and orientation, then updates the shader's view uniform.

        Parameters:
            pos (Coordinate): The position of the camera (currently unused).
            rotation (ToDecide): The rotation of the camera (currently unused).
        """
        log.dev("rotation variable type undecided in _set_view")

        view_transform = pyrr.matrix44.create_look_at(
            eye = np.zeros(3, dtype = np.float32),
            target = np.array([1, 0, 0], dtype = np.float32),
            up = np.array([0, 1, 0], dtype = np.float32),
            dtype = np.float32
        )

        self.shaders[self.active_shader_id].set_view(view_transform)

    def _set_global_view(self, pos: Coordinate, rotation: ToDecide) -> None:
        """
        [Private]
        TODO
        Sets the view transformation matrix for all existing shaders.

        Parameters:
            pos (Coordinate): The position of the camera (currently unused).
            rotation (ToDecide): The rotation of the camera (currently unused).
        """
        log.dev("rotation variable type undecided in _set_global_view")
        pass

    @staticmethod
    def _clear_screen() -> None:
        """
        [Private]
        Clears the screen by resetting the color and depth buffers.
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
    def _modify_object_pos(self, object_id: int, new_pos: Tuple[float, float, float]) -> None:
        """
        [Private]
        Modifys the objects position.

        Parameters:
            object_id (int): The id of the object to be rendered.
            new_pos (Tuple[float, float, float]): The new object position in world space.
        """
        self.objects[object_id].pos = new_pos

    def _render_object(self, object_id: int) -> None:
        """
        [Private]
        Renders the passed object id in the window.

        Parameters:
            object_id (int): The id of the object to be rendered.
        """
        self.objects[object_id].render(self.shaders[self.active_shader_id].get_uniform_handle("model"))

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
            new_object_creations = copy.deepcopy(self.new_object_creations)
            active_draw_instructions = copy.copy(self.active_draw_instructions)

            # Reset one-time functions
            self.new_shader_creations = []
            self.new_object_creations = []

        self._create_shaders(new_shader_creations)
        self._create_objects(new_object_creations)

        self._clear_screen()
        self._use_shader(self.default_shader_id)
        self._complete_draw_instructions(active_draw_instructions)