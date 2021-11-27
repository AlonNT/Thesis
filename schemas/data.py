from schemas.utils import ImmutableArgs


class DataArgs(ImmutableArgs):

    #: Whether to normalize of the values to a unit gaussian.
    normalization_to_unit_gaussian: bool = True

    #: Whether to use random cropping which is padding of 4 followed by random crop.
    random_crop: bool = True

    #: Whether to use random horizontal flipping (with probability 0.5).
    random_horizontal_flip: bool = True

    #: Whether to normalize of the values to the interval [-1,+1].
    normalization_to_plus_minus_one: bool = False
