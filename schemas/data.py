from schemas.utils import ImmutableArgs


class DataArgs(ImmutableArgs):

    #: Whether to use random cropping which is padding of 4 followed by random crop.
    random_crop: bool = True

    #: Whether to use random horizontal flipping (with probability 0.5).
    random_horizontal_flip: bool = True

    #: Whether to normalize of the values to a unit gaussian.
    normalization_to_unit_gaussian: bool = True

    #: Whether to normalize of the values to the interval [-1,+1].
    normalization_to_plus_minus_one: bool = False

    #: Indicator to shuffle the input images pixels before feeding to the neural-network.
    shuffle_images: bool = False

    #: Indicator to keep the RGB triplet intact when shuffling the image:
    #: sample a permutation from (32*32)! and not from (3*32*32)!
    keep_rgb_triplets_intact: bool = False
