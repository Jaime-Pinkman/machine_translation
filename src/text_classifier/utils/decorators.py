import functools
from typing import Any, Callable, List, TypeVar


# The return type of the decorated function
RT = TypeVar("RT")


def check_is_fitted(
    attributes: List[str],
) -> Callable[[Callable[..., RT]], Callable[..., RT]]:
    """
    A decorator to check if certain attributes
    are not None before calling a method.
    """

    def decorator(method: Callable[..., RT]) -> Callable[..., RT]:
        @functools.wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> RT:
            for attribute in attributes:
                attribute_hierarchy = attribute.split(".")
                obj = self
                for attr in attribute_hierarchy:
                    obj = getattr(obj, attr)
                if obj is None:
                    raise ValueError(
                        f"'{attribute}' of {str(self).split('.')[-1].split(' ')[0]} "
                        "has not been initialized."
                    )
            return method(self, *args, **kwargs)

        return wrapper

    return decorator
