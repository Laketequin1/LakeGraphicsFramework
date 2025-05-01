#import atexit
import traceback
import sys
import time
import threading
from MessageLogger import MessageLogger as log
from get_json_data import *

import LakeGraphicsFramework as lgf

### Constants ###
JSON_SETTINGS_FILEPATH = "settings/settings.json"
CAPTION = "Into Havoc"

"""
### Exit Handling ###
def exit_handler() -> None:
    
    #Runs before main threads terminates.
    
    #events["exit"].set()
    log.info("Program terminating")
    lgf.terminate_glfw()

atexit.register(exit_handler)
"""

### Example code ###
def catch_main():
    log.init("LOG_ONLY")
    
    try:
        setup()
    except Exception as e:
        error_message = f"Fatal termination error:\n\n{traceback.format_exc()}"
        log.error(error_message, e)

def setup():
    # Import settings
    settings = read_json_data(JSON_SETTINGS_FILEPATH)
    log.set_verbose_type(settings["verbose_type"])
    log.info(f"Imported settings: {settings}")
    
    # Create window
    window = lgf.Window(settings["window_resolution"], CAPTION, settings["fullscreen"], settings["windowed"], settings["vsync"], settings["max_fps"], settings["raw_mouse_input"], settings["center_cursor_on_creation"], settings["hide_cursor"])

    window.start()

    main(window)

    #main_thread = threading.Thread(target=main, args=(window, ))
    #main_thread.start()
    
    #window.main()

    #main_thread.join()
    #sys.exit(0)

def main(window: lgf.Window):
    graphics = window.graphics_engine
    shader_id = graphics.create_shader("shaders/" + "example.vert", "shaders/" + "example.frag", 100, None, 0.1, 200, compile_time_config={"SOMECOLOUR": "1, 0, 1, 1"})

    for x in range(254):
        if window.poll_events():
            window.close()
            break
        graphics.use_shader(shader_id)
        #graphics.set_()
        graphics.fill((0, 0, (255-x)/255, 1))
        #graphics.set_skybox_color((x/255, x/1000, x/1000, 1))
        graphics.update()
        time.sleep(1)

    log.crucial("Exit main thread")
    sys.exit(0)

if __name__ == "__main__":
    catch_main()