### Imports ###
import glfw
import OpenGL.GL as gl
from variable_type_validation import *
from MessageLogger import MessageLogger
from threading import Lock

### Constants ###
VSYNC_VALUE = 1

### Functions ###
def terminate_glfw():
    glfw.terminate()

### Classes ###
class Window:
    def __init__(self, size: Size = (0, 0), caption: str = "", fullscreen: bool = False, vsync: bool = True, max_fps: int = 0, raw_mouse_input: bool = True, center_cursor: bool = True, hide_cursor: bool = True) -> None:
        """
        Initializes a window and input system with the specified parameters.
        
        Parameters:
            size (Size): The initial size of the window (width, height).
            caption (str): The window's title.
            fullscreen (bool): Whether the window is fullscreen. Overrides size if True.
            vsync (bool): Whether vertical sync is enabled. Overrides max_fps if enabled.
            max_fps (int): The maximum frames per second. 0 means uncapped.
            raw_mouse_input (bool): Whether raw mouse input is used.
            center_cursor (bool): Whether the cursor is centered in the window.
            hide_cursor (bool): Whether the cursor is hidden.
        """
        # Validate parameters
        validate_types([('size', size, Size),
                        ('caption', caption, str),
                        ('fullscreen', fullscreen, bool),
                        ('vsync', vsync, bool),
                        ('max_fps', max_fps, int),
                        ('raw_mouse_input', raw_mouse_input, bool),
                        ('center_cursor', center_cursor, bool),
                        ('hide_cursor', hide_cursor, bool)])
        
        # Set parameters
        self.size = size
        self.caption = str(caption)
        self.fullscreen = fullscreen
        self.vsync = vsync
        self.max_fps = int(max_fps)
        self.raw_mouse_input = raw_mouse_input
        self.center_cursor = center_cursor
        self.hide_cursor = hide_cursor

        # Display init
        self.monitor, self.video_mode = self._init_display()
        self.display_size = self.video_mode.size

        # Update screen size if fullscreen
        if self.fullscreen:
            self.size = self.display_size
        
        if self.size[1] != 0:
            self.aspect_ratio = self.size[0] / self.size[1]
        else:
            self.aspect_ratio = 1

        # Window init
        self.window = self._init_window(self.monitor, self.size, self.caption, self.vsync, self.max_fps)

        # Input mode init
        self._init_input(self.window, self.raw_mouse_input, self.hide_cursor)

        # Set variables
        self.active = True
        self.lock = Lock()
        self.graphics_engine = GraphicsEngine()
    
    def main(self):
        while self.active:
            gl.glFlush() # Wait for pipeline

            with self.lock:
                pass

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
    def _init_window(monitor: glfw._GLFWmonitor, size: Size, caption: str, vsync: bool, max_fps: int) -> glfw._GLFWwindow:
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
        window = glfw.create_window(*size, caption, monitor, None)
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
                if MessageLogger.check_init_completed():
                    MessageLogger.warn("Raw mouse motion unsupported.")
    
    ### Private ###
    def _gl_check_error(self):
        


class GraphicsEngine:
    pass