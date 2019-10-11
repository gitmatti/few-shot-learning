import PIL.Image
import numpy


class ToSize(object):
    """Return height and width for  a ``PIL Image`` or ``numpy.ndarray``

    Converts a PIL Image or numpy.ndarray (H x W x C) to a tuple (H, W).
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image for which size is to be
            returned.

        Returns:
            Tuple: Image size.
        """
        if isinstance(pic, PIL.Image.Image):
            return pic.size
        elif isinstance(pic, numpy.ndarray):
            return pic.shape[:2]
        else:
            raise TypeError

    def __repr__(self):
        return self.__class__.__name__ + '()'
