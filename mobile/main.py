from kivy.core.window import Window
from kivy.lang.builder import Builder
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.list import IRightBodyTouch

from src.screens import *

Window.size = (336, 600)


class HgOApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"
        self.theme_cls.primary_hue = "700"
        return Builder.load_file('src/template.kv')


HgOApp().run()
