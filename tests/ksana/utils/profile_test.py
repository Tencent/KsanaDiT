import time
import unittest

from ksana.utils.logger import log
from ksana.utils.profile import time_range

SLEEP_TIME = 0.01


@time_range
def func_with_no_name():
    time.sleep(SLEEP_TIME)


@time_range("my_func_with_name")
def func_with_name():
    time.sleep(SLEEP_TIME)


@time_range("my_func_with_name_and_print_func", log.info)
def func_with_name_and_log():
    time.sleep(SLEEP_TIME)


@time_range("my_func_with_name_and_print_func", print)
def func_with_name_and_print():
    time.sleep(SLEEP_TIME)


class TestTimeRange(unittest.TestCase):
    def test_with_no_name(self):
        with time_range():
            time.sleep(SLEEP_TIME)

    def test_with_name(self):
        with time_range("test_time_range"):
            time.sleep(SLEEP_TIME)

    def test_with_name_and_print_func(self):
        with time_range("test_time_range", log.info):
            time.sleep(SLEEP_TIME)

        with time_range("test_time_range", print):
            time.sleep(SLEEP_TIME)

    def test_func_with_no_name(self):
        func_with_no_name()

    def test_func_with_name(self):
        func_with_name()

    def test_func_with_name_and_print(self):
        func_with_name_and_print()

    def test_func_with_name_and_log(self):
        func_with_name_and_log()


if __name__ == "__main__":
    unittest.main()
