import unittest


def fib(n):
    return 1 if n <= 2 else fib(n - 1) + fib(n - 2)


class TestFib(unittest.TestCase):
    def setUp(self):
        self.n = 10

    def tearDown(self):
        del self.n

    def test_fib_assert_equal(self):
        self.assertEqual(fib(self.n), 55)

    def test_fib_assert_true(self):
        self.assertTrue(fib(self.n) == 55)