# type: ignore
import re
import doctest

collect_ignore = ['setup.py']

pytest_plugins = ("pytest_cov", "subtests")


OutputChecker = doctest.OutputChecker


numpy2_float64_pattern = r'np\.float64\((?P<num>inf|[+-]?\d*\.\d+)\)'

def ensure_numpy_1_style_repr(repr_: str)-> str:
    return re.sub(
                numpy2_float64_pattern,
                lambda m: m['num'],
                repr_,
                )


class Numpy1Numpy2AgnosticOutputChecker(doctest.OutputChecker):

    def check_output(self, want, got, optionflags):

        numpy_1_style_want = ensure_numpy_1_style_repr(want)
        numpy_1_style_got = ensure_numpy_1_style_repr(got)

        return super().check_output(numpy_1_style_want, numpy_1_style_got, optionflags)




def pytest_configure(config):
    import sys
    sys._called_from_test = True
    doctest.OutputChecker = Numpy1Numpy2AgnosticOutputChecker



def pytest_unconfigure(config):
    import sys
    del sys._called_from_test
    doctest.OutputChecker = OutputChecker



