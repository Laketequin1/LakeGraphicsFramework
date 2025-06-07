### Imports ###
import os
from datetime import datetime
from typing import Literal, NoReturn

### Functions ###
class MessageLogger:
    """
    Used for debugging purposes during production.

    When initialized, a verbose level is set. Supported levels include:
    - NONE:     0 - Completely ignores all logs.
    - LOG_ONLY: 1 - Saves all logs to a file without printing to terminal.
    - ERROR:    2 - Prints only error messages.                 (logs all to file)
    - WARNING:  3 - Prints warnings and errors.                 (logs all to file)
    - CRUCIAL:  4 - Prints warnings, errors, and crucial info.  (logs all to file)
    - INFO:     5 - Prints all message levels.                  (logs all to file)

    Logs messages to a timestamped log file under the "logs/" folder.
    """
    # Constants
    LOG_FOLDER = "logs/"
    LOG_FILENAME = datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + ".log"
    LOG_FILEPATH = LOG_FOLDER + LOG_FILENAME

    VERBOSE_MAP = {
                    "NONE":     0,  # Everything completely ignored
                    "LOG_ONLY": 1,  # Only log to file
                    "ERROR":    2,  # Only print errors
                    "WARNING":  3,  # Only print warnings and errors
                    "CRUCIAL":  4,  # Only print warnings, errors, and crucial info
                    "INFO":     5   # Print all
                  }
    VERBOSE_KEYS = tuple(VERBOSE_MAP.keys())
    VerboseLiteral = Literal["NONE", "LOG_ONLY", "ERROR", "WARNING", "CRUCIAL", "INFO"]
    
    TEXT_STYLES = {
                    "ERROR":   "\033[31m",  # Red
                    "WARNING": "\033[33m",  # Yellow
                    "CRUCIAL": "\033[97m",  # White
                    "INFO":    "\033[90m",  # Light Grey
                    "CLEAR":   "\033[0m"    # Reset
                  }
    
    SETUP_MESSAGE_FORMAT = "Logfile created with verbose level {verbose_level} on {date_time}. Have fun <3\n"
    PREFIX_SPACE_PADDING = 7

    THROW_ERROR_IF_NOT_INITIATED = False

    # Verbose unset
    verbose_level = None
    
    @classmethod
    def init(cls, verbose_type: VerboseLiteral) -> None:
        """
        Initializes a MessageLogger instance with a specified verbose level.
        Will throw a ValueError if verbose level is invalid.
        
        Parameters:
            verbose_type (str): The verbosity level for logging. Must be one of the supported levels:
                                - NONE:     0 - Completely ignores all logs.
                                - LOG_ONLY: 1 - Saves all logs to a file without printing to terminal.
                                - ERROR:    2 - Prints only error messages.                 (logs all to file)
                                - WARNING:  3 - Prints warnings and errors.                 (logs all to file)
                                - CRUCIAL:  4 - Prints warnings, errors, and crucial info.  (logs all to file)
                                - INFO:     5 - Prints all message levels.                  (logs all to file)
        """
        cls.set_verbose_type(verbose_type)

        cls.logs = []

        date_time = datetime.now().strftime("%d/%m/%Y at %H:%M:%S")
        setup_message = cls.SETUP_MESSAGE_FORMAT.format(verbose_level = cls.VERBOSE_KEYS[cls.verbose_level], date_time = date_time)
        cls._setup_log_file(setup_message)

    @classmethod
    def _setup_log_file(cls, setup_message) -> None:
        """
        [Private]
        Generates the log folder if it doesn't exist. Creates a new log file and writes an initial setup message.

        Parameters:
            setup_message (str): The message to write at the beginning of the log file.
        """
        os.makedirs(cls.LOG_FOLDER, exist_ok=True)

        with open(cls.LOG_FILEPATH, "w") as log_file:
            log_file.write(setup_message + "\n")

    @classmethod
    def check_init_completed(cls) -> bool:
        """
        Returns if the class has been initiated.

        Returns:
            bool: If ClassLogger initiation has been completed.
        """
        if cls.verbose_level is None:
            return False
        return True

    @classmethod
    def error(cls, message: str, raise_exception: Exception = None) -> None:
        """
        Logs an error message to the file and prints it to the terminal based on the verbose level.
        Raises an exception.

        Parameters:
            message (str): The error message to be logged and/or printed.
        """
        if cls.THROW_ERROR_IF_NOT_INITIATED:
            cls._check_init_completed()
        elif not cls.check_init_completed():
            return
            
        cls._log("ERROR", str(message))

        if raise_exception is not None:
            raise raise_exception

    @classmethod
    def warn(cls, message: str) -> None:
        """
        Logs a warning message to the file and prints it to the terminal based on the verbose level.

        Parameters:
            message (str): The warning message to be logged and/or printed.
        """
        if cls.THROW_ERROR_IF_NOT_INITIATED:
            cls._check_init_completed()
        elif not cls.check_init_completed():
            return
        
        cls._log("WARNING", str(message))

    @classmethod
    def crucial(cls, message: str) -> None:
        """
        Logs a crucial message to the file and prints it to the terminal based on the verbose level.

        Parameters:
            message (str): The warning message to be logged and/or printed.
        """
        if cls.THROW_ERROR_IF_NOT_INITIATED:
            cls._check_init_completed()
        elif not cls.check_init_completed():
            return
        
        cls._log("CRUCIAL", str(message))

    @classmethod
    def info(cls, message: str) -> None:
        """
        Logs a info message to the file and prints it to the terminal based on the verbose level.

        Parameters:
            message (str): The warning message to be logged and/or printed.
        """
        if cls.THROW_ERROR_IF_NOT_INITIATED:
            cls._check_init_completed()
        elif not cls.check_init_completed():
            return
        
        cls._log("INFO", str(message))

    @classmethod
    def set_verbose_type(cls, verbose_type: VerboseLiteral) -> None:
        """
        Updates the verbose level.
        Will throw a ValueError if verbose level is invalid.
        
        Parameters:
            verbose_type (str): The verbosity level for logging. Must be one of the supported levels:
                                - NONE:     0 - Completely ignores all logs.
                                - LOG_ONLY: 1 - Saves all logs to a file without printing to terminal.
                                - ERROR:    2 - Prints only error messages.                 (logs all to file)
                                - WARNING:  3 - Prints warnings and errors.                 (logs all to file)
                                - CRUCIAL:  4 - Prints warnings, errors, and crucial info.  (logs all to file)
                                - INFO:     5 - Prints all message levels.                  (logs all to file)
        """
        cls.verbose_type = str(verbose_type).replace("\n", "").replace("\r", "").strip().upper()

        if cls.verbose_type not in cls.VERBOSE_MAP:
            raise ValueError(f"Verbose level '{cls.verbose_type}' is not an option. Available options are: {', '.join(cls.VERBOSE_MAP.keys())}.")
        
        previous_verbose_level = cls.verbose_level
        if previous_verbose_level is not None: # Check this isn't the init set
            previous_verbose_type = cls.VERBOSE_KEYS[previous_verbose_level]

        cls.verbose_level = cls.VERBOSE_MAP[cls.verbose_type]

        if previous_verbose_level is not None: # Check this isn't the init set
            cls.info(f"MessageLogger verbose type updated from {previous_verbose_type} to {cls.verbose_type}")
    
    @classmethod
    def _log(cls, verbose_type: str, message: str) -> None:
        """
        [Private]
        Handles the actual logging of messages, including formatting, saving to the file, and printing to the terminal - based on verbose settings.

        Parameters:
            verbose_type (str): The type of message being logged (e.g., "ERROR", "WARNING", "INFO").
            message (str): The content of the message to be logged and/or printed.
        """
        if cls.verbose_level < cls.VERBOSE_MAP["LOG_ONLY"]:
            return
        
        # Log message
        formatted_message = cls._format_message(verbose_type, message)
        cls.logs.append(message)
        cls._append_to_log_file(formatted_message)

        if cls.verbose_level < cls.VERBOSE_MAP[verbose_type]:
            return
        
        # Print message
        styled_message = cls._style_message(verbose_type, formatted_message)
        print(styled_message)

    @classmethod
    def _style_message(cls, style: str, message: str) -> str:
        """
        [Private]
        Applies ANSI styling to a message for formatted terminal output.

        Parameters:
            style (str): The style to be applied from TEXT_STYLES (e.g., "ERROR", "WARNING", "INFO").
            message (str): The message to be styled.

        Returns:
            str: The styled message.
        """
        styled_message = cls.TEXT_STYLES[style] + str(message) + cls.TEXT_STYLES["CLEAR"]
        return styled_message

    @classmethod
    def _format_message(cls, prefix: str, message: str) -> str:
        """
        [Private]
        Formats a message by adding a timestamp and prefix (e.g., "ERROR", "INFO").

        Parameters:
            prefix (str): The message prefix indicating its type (e.g., "ERROR", "INFO").
            message (str): The message to be formatted.

        Returns:
            str: The formatted message with a timestamp and prefix.
        """
        message_time = datetime.now().strftime("%H:%M:%S")
        padded_prefix = prefix.ljust(cls.PREFIX_SPACE_PADDING)
        formatted_message = f"[{message_time}] {padded_prefix} : {str(message)}"        
        return formatted_message

    @classmethod
    def _append_to_log_file(cls, message) -> None:
        """
        [Private]
        Appends a formatted message to the log file.

        Parameters:
            message (str): The message to be written to the log file.
        """
        with open(cls.LOG_FILEPATH, "a") as log_file:
            log_file.write(str(message) + "\n")
    
    @classmethod
    def _check_init_completed(cls) -> None | NoReturn:
        """
        Private implementation to check if ClassLogger has been initiated. Raises error if not initiated, otherwise continues.
        """
        if not cls.check_init_completed():
            raise RuntimeError("MessageLogger has not been initialized. Make sure to call MessageLogger.init(verbose_type)")
        

### Example code ###
def main():
    # Initialize the logger with a specific verbosity level
    verbose_level = "INFO"  # Can be "NONE", "LOG_ONLY", "ERROR", "WARNING", "CRUCIAL", or "INFO"
    
    # Initialize the MessageLogger with the specified verbose level
    MessageLogger.init(verbose_level)

    # Log different types of messages to demonstrate functionality
    MessageLogger.error("This is an error message.")
    MessageLogger.warn("This is a warning message.")
    MessageLogger.crucial("This is a crucial message.")
    MessageLogger.info("This is an info message.")
    
    # If you want to test other verbose levels, you can change the level and log more messages
    MessageLogger.set_verbose_type("WARNING")
    MessageLogger.info("This info message should not be printed, but will appear in the log file.")
    MessageLogger.error("This error message should still appear in the terminal.")

    # Nothing gets logged, nothing written to file.
    MessageLogger.set_verbose_type("NONE")
    MessageLogger.error("This error won't appear anywhere :)")

# Entry point
if __name__ == "__main__":
    main()