import atexit
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

### Exit Handling ###
def exit_handler() -> None:
    """
    Runs before main threads terminates.
    """
    #events["exit"].set()
    log.info("Program terminating")
    lgf.terminate_glfw()

atexit.register(exit_handler)

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
    window = lgf.Window(settings["window_resolution"], CAPTION, settings["fullscreen"], settings["windowed"], settings["vsync"], settings["max_fps"], settings["raw_mouse_input"], True)

    main_thread = threading.Thread(target=main, args=(window, ))
    main_thread.start()
    
    window.main()

    main_thread.join()
    sys.exit(0)

def main(window: lgf.Window):
    for x in range(254):
        window.graphics_engine.clear()
        window.graphics_engine.set_skybox_color((x/255, 0, 0, 1))
        window.graphics_engine.update()
        time.sleep(0.1)
    window.active = False
    log.crucial("Exit main thread")
    sys.exit(0)

if __name__ == "__main__":
    catch_main()