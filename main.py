### Imports ###
import glfw
import sys
import atexit
from variable_type_validation import *
from MessageLogger import MessageLogger

### Constants ###
VSYNC_VALUE = 1

### Exit Handling ###
def exit_handler() -> None:
    """
    Runs before main threads terminates.
    """
    #events["exit"].set()
    MessageLogger.info("Program terminating")
    glfw.terminate()

atexit.register(exit_handler)

### Classes ###
class Window:
    def __init__(self, size: Size = (0, 0), caption: str = "", fullscreen: bool = False, vsync: bool = True, max_fps: int = 0, raw_mouse_input: bool = True, center_cursor: bool = True, print_gl_errors: bool = True):
        """
        fullscreen: [Overrides size]
        max_fps: [0 means uncapped]
        """
        # Validate parameters
        validate_types([('size', size, Size),
                        ('caption', caption, str),
                        ('fullscreen', fullscreen, bool),
                        ('vsync', vsync, bool),
                        ('max_fps', max_fps, int),
                        ('raw_mouse_input', raw_mouse_input, bool),
                        ('center_cursor', center_cursor, bool),
                        ('print_gl_errors', print_gl_errors, bool)])
        
        # Set parameters
        self.size = size
        self.caption = str(caption)
        self.fullscreen = fullscreen
        self.vsync = vsync
        self.max_fps = int(max_fps)
        self.raw_mouse_input = raw_mouse_input
        self.center_cursor = center_cursor
        self.print_gl_errors = print_gl_errors

        self.aspect_ratio = size[0] / size[1]

        # Display init
        self.monitor, self.video_mode = self._init_display(self.size, self.caption, self.fullscreen, self.vsync, self.max_fps)
        self.display_size = self.video_mode.size

        # Update screen size if fullscreen
        if self.fullscreen:
            self.size = self.display_size
            self.aspect_ratio = self.display_size[0] / self.display_size[1]

        # Window init
        self.window = self._init_window(self.monitor, self.size, self.caption, self.vsync, self.max_fps)

        # Input mode init


        # Set variables
        self.active = True
        self.graphics_engine = GraphicsEngine()

        self.gl_error_logs = []
    
    @staticmethod
    def _init_display() -> tuple[glfw._GLFWmonitor, glfw._GLFWvidmode]:
        """
        Initilize GLFW display
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
        Initilize GLFW
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
    def _init_window(monitor: glfw._GLFWmonitor, size: Size, caption: str, vsync: bool, max_fps: int) -> glfw._GLFWwindow:
        """
        Initilize GLFW window
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
    def _init_input(window: glfw._GLFWwindow) -> glfw._GLFWwindow:
        """
        Initilize window input
        """
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

        if glfw.raw_mouse_motion_supported():
            glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)


class GraphicsEngine:
    pass


def main():
    # Init MessageLogger (Pass from JSON)
    MessageLogger.init("LOG_ONLY")

    MessageLogger.info("Started main setup")
    MessageLogger.set_verbose_type("INFO")
    MessageLogger.warn("oooh!")
    MessageLogger.crucial("TESTTT")
    MessageLogger.error("Heheh!~")
    #window = Window((2, 5))

    sys.exit(0)

if __name__ == "__main__":
    main()