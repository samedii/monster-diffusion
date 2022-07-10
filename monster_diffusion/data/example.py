from uuid import UUID
from datetime import datetime
from PIL import Image
import lantern
from typing import List
from pydantic import Extra


class Example(lantern.FunctionalBase):
    key: str
    image: lantern.Numpy

    class Config:
        extra = Extra.ignore

    @staticmethod
    def from_row(row):
        return Example(
            image=Image.open(row.image_path).convert("RGBA"),
            **row.to_dict(),
        )

    def representation(self):
        return Image.fromarray(self.image)

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_

    def augment(self, augmenter):
        image = augmenter(image=self.image)
        return self.replace(image=image)

    def hash(self):
        return hash(self.key)
