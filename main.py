### Imports ###
import glfw
import sys
import atexit
import traceback
import json
from variable_type_validation import *
from MessageLogger import MessageLogger
from get_json_data import *

### Constants ###
JSON_SETTINGS_FILEPATH = "settings/settings.json"
VSYNC_VALUE = 1
CAPTION = "Into Havoc"

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
    def __init__(self, size: Size = (0, 0), caption: str = "", fullscreen: bool = False, vsync: bool = True, max_fps: int = 0, raw_mouse_input: bool = True, center_cursor: bool = True, hide_cursor: bool = True):
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
        self.graphics_engine = GraphicsEngine()
    
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
    def _init_input(window: glfw._GLFWwindow, raw_mouse_input: bool, hide_cursor: bool) -> glfw._GLFWwindow:
        """
        Initilize window input
        """
        if hide_cursor:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

        if raw_mouse_input:
            if glfw.raw_mouse_motion_supported():
                glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)
            else:
                MessageLogger.warn("Raw mouse motion unsupported.")


class GraphicsEngine:
    pass


def catch_main():
    MessageLogger.init("LOG_ONLY")

    try:
        main()
    except Exception as e:
        error_message = f"Fatal termination error:\n\n{traceback.format_exc()}"
        print("\n\n\n-------------")
        print(type(e))
        print("-------------\n\n\n")
        MessageLogger.error(error_message, e)


def main():
    # Import settings
    settings = read_json_data(JSON_SETTINGS_FILEPATH)
    MessageLogger.set_verbose_type(settings["verbose_type"])
    MessageLogger.info(f"Imported settings: {settings}")
    
    window = Window(settings["window_resolution"], CAPTION, settings["fullscreen"], settings["vsync"], settings["max_fps"], settings["raw_mouse_input"], True)

    sys.exit(0)

if __name__ == "__main__":
    catch_main()