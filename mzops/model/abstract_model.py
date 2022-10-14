

from typing import Any


class ModelMeta(type):
    """A Model metaclass that will be used for model creation
    """

    def __instancecheck__(cls, __instance: Any) -> bool:
        return cls.__subclasscheck__(type(__instance))

    def __subclasscheck__(cls, __subclass: type) -> bool:
        return (hasattr(__subclass), 'fit') and callable(__subclass.fit) and (hasattr(__subclass), 'evaluate') and callable(__subclass.evaluate)


class Model(metaclass = ModelMeta):
     """This interface is used for concrete classes to inherit from.
    There is no need to define the ModelMeta methods as any class
    as they are implicitly made available via .__subclasscheck__().
    """
    pass