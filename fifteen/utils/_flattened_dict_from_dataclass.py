import dataclasses
from typing import Any, Dict

from typing_extensions import get_type_hints


def flattened_dict_from_dataclass(dataclass: Any) -> Dict[str, Any]:
    assert dataclasses.is_dataclass(dataclass)
    cls = type(dataclass)
    hints = get_type_hints(cls)

    output = {}
    for field in dataclasses.fields(dataclass):
        field_type = hints[field.name]
        value = getattr(dataclass, field.name)
        if dataclasses.is_dataclass(field_type):
            inner = flattened_dict_from_dataclass(value)
            inner = {".".join([field.name, k]): v for k, v in inner.items()}
            output.update(inner)
        else:
            output[field.name] = value
    return output
