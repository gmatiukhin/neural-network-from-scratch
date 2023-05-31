from manim import *
from manim_slides.slide import Slide


class NeuralNetworkPresentation(Slide):
    def construct(self):
        name = Text("Нейронная сеть своими руками!").scale(3)
        presenter = Text("Матюхин Григорий")
        self.add(name, presenter)
