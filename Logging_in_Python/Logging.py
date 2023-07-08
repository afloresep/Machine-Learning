import logging 
import sys 

logger = logging.getLogger(__name__) 
"""
Obtain out logger object using getLogger(name). This method takes in an optional, single inputo parameter called 'name' that represents the name of the logger. 
Calling getLogger() with the same 'name' value returns the same logger object. If we give no 'name', the root logger is returned. 

Using Python variable __name__ is recommendedn since it will return the current module's name. 
"""

stream_handler = logging.StreamHandler(sys.stdout)
""" 
We need to inform the logger where we want our logged messages to go. To do this we use a 'handler'. 
For python we will use the logging module's StreamHandler. This class takes in an optional, single input called stream. 
We should supply sys.stdout as the stream value. Note that we must import the sys library to reference stdout as the stream value. 
"""

logger.addHandler(stream_handler)
"""
Now add our stream handler object to the logger. The logging module provides a method called addHandler(hldr) that adds a specific handler to the logger object.
 The hdlr input represents the handler object t oadd, which in our example is the StreamHandler object. 
"""


logging.basicConfig(filename='calculator.log', level=logging.DEBUG, format='[%(asctime)s] %(levelname)s - %(message)s')
"""The `basicConfig()` method allows for the basic configuration of the logger object by configuring the log level, any handlers,
log message formatting options, and more in one line of code."""


file_handler = logging.FileHandler("my_program.log")  
logger.addHandler(file_handler)
  
formatter1 = logging.Formatter("[%(asctime)s] {%(levelname)s} %(name)s: #%(lineno)d - %(message)s") 
formatter2 = logging.Formatter("[%(asctime)s] {%(levelname)s} - %(message)s")
  

file_handler.setFormatter(formatter1)
stream_handler.setFormatter(formatter2)
"""
Output for formatted.log: 
[2021-12-20 05:19:33,438] {DEBUG} __main__: #19 - Starting Division!

Output for stream: 
[2023-06-14 09:22:08,786] {DEBUG} - Starting Division!
"""

