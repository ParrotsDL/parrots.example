import time
import inspect
import ctypes
import threading
from test_convert import test_convert


# See https://stackoverflow.com/questions/323972
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid,
                                                     ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("Cancel failed, because convert test has finished.")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
    else:
        print('Close Successfully!')


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


if __name__ == '__main__':
    thread = threading.Thread(target=test_convert)
    thread.start()
    time.sleep(1)
    stop_thread(thread)
