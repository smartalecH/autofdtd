from importlib import import_module


def test_namespace_imports() -> None:
    module_names = (
        "autofdtd",
        "autofdtd.api",
        "autofdtd.boundaries",
        "autofdtd.compiler",
        "autofdtd.core",
        "autofdtd.diagnostics",
        "autofdtd.examples",
        "autofdtd.geometry",
        "autofdtd.grid",
        "autofdtd.ir",
        "autofdtd.kernels",
        "autofdtd.materials",
        "autofdtd.modes",
        "autofdtd.monitors",
        "autofdtd.runtime",
        "autofdtd.sources",
    )
    for module_name in module_names:
        module = import_module(module_name)
        assert module.__name__ == module_name
