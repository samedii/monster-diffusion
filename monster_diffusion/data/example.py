from PIL import Image
import numpy as np
import lantern


class Example(lantern.FunctionalBase):
    key: str
    image: lantern.Numpy.dims("HWC").dtype(np.float32)
    nonleaky_augmentations: lantern.Numpy.shape(9).dtype(np.float32)

    def hash(self):
        return hash(self.key)

    @staticmethod
    def from_row(row):
        return Example(
            key=row.key,
            image=Image.open(row.image_path).convert("RGBA"),
            nonleaky_augmentations=np.zeros((9,), dtype=np.float32),
        )

    def representation(self):
        return Image.fromarray(np.clip(self.image, 0, 255).astype(np.uint8))

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_

    def augment(self, augmenter):
        image = augmenter(image=self.image)
        return self.replace(image=image)

    def nonleaky_augment(self, augmenter):
        if (self.nonleaky_augmentations != 0).any():
            raise ValueError("Example has already been nonleaky augmented")
        else:
            image, nonleaky_augmentations = augmenter(image=self.image)
            return self.replace(
                image=image,
                nonleaky_augmentations=nonleaky_augmentations,
            )
