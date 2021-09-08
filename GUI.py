from math import sin
from kivy.garden.graph import Graph, MeshLinePlot
from kivy.uix.widget import Widget
from kivy.properties import (
    NumericProperty, ReferenceListProperty, ObjectProperty
)
from kivy.app import App

class SetGraph(Widget):
    graph_test = ObjectProperty(None)

    def update_graph(self):
         plot = MeshLinePlot(color=[1, 0, 0, 1])
         plot.points = [(x, sin(x/10)) for x in range(0, 101)]
         self.graph_test.add_plot(plot)

class GUIApp(App):
    def build(self):
        disp = SetGraph()
        disp.update_graph()
        return disp


if __name__ == '__main__':
    GUIApp().run()
