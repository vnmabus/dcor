
import doctest

collect_ignore = ['setup.py']

pytest_plugins = ("pytest_cov", "subtests")


doctest_normal_output_checker = doctest.OutputChecker

class Numpy1Numpy2AgnosticOutputChecker(doctest.OutputChecker):


    def numpy_doctest_output_fix(self, want, got, optionflags):
        if want == got:
            return True
        if want == f'np.float64({got})':
            return True



        return False



def pytest_configure(config):
    import sys
    sys._called_from_test = True
    doctest.OutputChecker = Numpy1Numpy2AgnosticOutputChecker



def pytest_unconfigure(config):
    import sys
    del sys._called_from_test
    doctest.OutputChecker = doctest_normal_output_checker



