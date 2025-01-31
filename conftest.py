# type: ignore
import re
import doctest

collect_ignore = ['setup.py']

pytest_plugins = ("pytest_cov", "subtests")


OutputChecker = doctest.OutputChecker



class Numpy1Numpy2AgnosticOutputChecker(doctest.OutputChecker):

    numpy2_float64_pattern = r'np\.float64\((?P<num>inf|[+-]?\d*\.\d+)\)'
    
    def check_output(self, want, got, optionflags):

        numpy_1_style_got = re.sub(
                self.numpy2_float64_pattern,
                lambda m: m['num'],
                got,
                )

        return super().check_output(want, numpy_1_style_got, optionflags)




def pytest_configure(config):
    import sys
    sys._called_from_test = True
    doctest.OutputChecker = Numpy1Numpy2AgnosticOutputChecker



def pytest_unconfigure(config):
    import sys
    del sys._called_from_test
    doctest.OutputChecker = OutputChecker



