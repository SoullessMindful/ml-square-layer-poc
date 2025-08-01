__all__ = ["Model", "BaseModel", "SquareModel"]

from .base_model import BaseModel
from .square_model import SquareModel

type Model = BaseModel | SquareModel