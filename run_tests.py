# run_tests.py
import unittest

if __name__ == '__main__':
    # This will run all test modules starting with 'test_' in the 'tests' directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir='tests', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)
