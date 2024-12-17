### Imports ###
import glfw
from variable_type_validation import *

### Constants ###
VSYNC = 1

### Classes ###
class Window:
    def __init__(self, size: Size = (0, 0), caption: str = "", fullscreen: bool = False, vsync: bool = True, max_fps: int = 0, print_gl_errors: bool = True):
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
                                 ('print_gl_errors', print_gl_errors, bool)])
        
        # Set parameters
        self.size = size
        self.caption = str(caption)
        self.fullscreen = fullscreen
        self.vsync = vsync
        self.max_fps = int(max_fps)
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

        # Max FPS (Disable VSYNC)
        if vsync:
            fps_value = 1 # 
        glfw.swap_interval(max_fps)

        return window


class GraphicsEngine:
    pass


def main():
    window = Window((2, 5))

if __name__ == "__main__":
    main()