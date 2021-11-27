from pydantic import BaseModel, Extra
from pydantic.types import ConstrainedFloat, ConstrainedInt


class NonNegativeInt(ConstrainedInt):
    ge = 0


class NonNegativeFloat(ConstrainedFloat):
    ge = 0


class Fraction(ConstrainedFloat):
    ge = 0
    le = 1


class ProperFraction(ConstrainedFloat):
    gt = 0
    lt = 1


class NonZeroFraction(ConstrainedFloat):
    gt = 0
    le = 1


class NonOneFraction(ConstrainedFloat):
    ge = 0
    lt = 1


class ImmutableArgs(BaseModel):
    class Config:
        allow_mutation = True
        extra = Extra.forbid
