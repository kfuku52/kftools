import inspect
from typing import Any

import pytest

from kftools import kfexpression, kfog, kfphylo, kfplot, kfseq, kfspecies, kfstat, kfutil

PUBLIC_MODULES = (kfexpression, kfog, kfphylo, kfplot, kfseq, kfspecies, kfstat, kfutil)


def _defined_public_callables(module):
    for name, member in inspect.getmembers(module):
        is_export = name in getattr(module, "__all__", ())
        if name.startswith("_") or (getattr(member, "__module__", None) != module.__name__ and not is_export):
            continue
        if inspect.isfunction(member):
            yield f"{module.__name__}.{name}", member
        elif inspect.isclass(member):
            for method_name, method in inspect.getmembers(member):
                if method_name.startswith("_"):
                    continue
                if isinstance(inspect.getattr_static(member, method_name), property):
                    method = inspect.getattr_static(member, method_name).fget
                if inspect.isfunction(method):
                    yield f"{module.__name__}.{name}.{method_name}", method


@pytest.mark.parametrize(
    ("qualified_name", "callable_object"),
    [item for module in PUBLIC_MODULES for item in _defined_public_callables(module)],
)
def test_public_callables_are_documented_and_typed(qualified_name, callable_object):
    assert inspect.getdoc(callable_object), f"{qualified_name} is missing a docstring"
    signature = inspect.signature(callable_object)
    untyped_parameters = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name not in {"self", "cls"} and parameter.annotation is inspect.Parameter.empty
    ]
    assert not untyped_parameters, f"{qualified_name} has untyped parameters: {untyped_parameters}"
    assert signature.return_annotation is not inspect.Signature.empty, f"{qualified_name} has no return annotation"
    assert all(parameter.annotation not in (Any, "Any") for parameter in signature.parameters.values()), (
        f"{qualified_name} has a bare Any parameter; use an input type or object for genuinely opaque data"
    )
    assert signature.return_annotation not in (Any, "Any"), f"{qualified_name} has a bare Any return type"


@pytest.mark.parametrize(
    ("qualified_name", "public_class"),
    [
        (f"{module.__name__}.{name}", member)
        for module in PUBLIC_MODULES
        for name, member in inspect.getmembers(module, inspect.isclass)
        if not name.startswith("_") and (member.__module__ == module.__name__ or name in getattr(module, "__all__", ()))
    ],
)
def test_public_classes_are_documented(qualified_name, public_class):
    assert inspect.getdoc(public_class), f"{qualified_name} is missing a docstring"
