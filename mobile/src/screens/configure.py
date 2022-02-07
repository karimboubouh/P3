from kivy.clock import Clock
from kivy.properties import StringProperty, BooleanProperty
from kivy.uix.screenmanager import Screen
from kivymd.toast import toast
from kivymd.uix.dialog import MDDialog

from src import protocol
from src.conf import BRIDGE_HOST, BRIDGE_PORT


class ConfScreen(Screen):
    connect_logs = StringProperty()
    connect_btn = StringProperty()
    join_disabled = BooleanProperty()

    def __init__(self, **kwargs):
        super(ConfScreen, self).__init__(**kwargs)
        self.dialog = None
        self.connect_logs = "Connect to get the list of your neighbors within the network"
        self.bridge_host = BRIDGE_HOST
        self.bridge_port = BRIDGE_PORT
        self.request_data = True
        self.connect_btn = "Connect"
        self.join_disabled = True
        self.share_logs = True
        Clock.schedule_once(self.init, 1)

    def init(self, *args):
        self.ids.bridge_host.text = self.bridge_host
        self.ids.bridge_host.focus = True
        self.ids.bridge_host.focus = False
        self.ids.bridge_port.text = str(self.bridge_port)
        self.ids.bridge_port.focus = True
        self.ids.bridge_port.focus = False
        self.ids.request_data.active = self.request_data
        self.ids.share_logs.active = self.share_logs

    def connect(self):
        self.connect_logs = f"HOST:{self.manager.node.host} / PORT:{self.manager.node.port}\n"
        self.dialog = MDDialog(title="Connection")  # , auto_dismiss=False
        try:
            if self.manager.node.connect_bridge(self.bridge_host, self.bridge_port):
                self.ids.connect_btn.disabled = True
                self.connect_btn = "Connected"
                toast("Connected successfully")
                self.manager.node.bridge.send_pref(self.request_data, self.share_logs)
                self.connect_logs = "Connecting to a set of peers ..."
            else:
                self.dialog.text = f"Could not connect to bridge."
                self.dialog.open()
                return False
        except Exception as e:
            self.dialog.text = f"Error while connecting to bridge: {str(e)}"
            self.dialog.open()
            return False

    def toggle_join(self, *args):
        self.ids.join_btn.disabled = not self.ids.join_btn.disabled
        self.manager.node.bridge.send(protocol.return_method("populate", {'s': True}))

    def log_pref(self):
        pref = ""
        pref = "\n".join([pref, f"[b]Id[/b] {self.manager.node.id}"])
        pref = "\n".join([pref, f"[b]Neighbors[/b] {self.manager.node.neighbors_ids}"])
        pref = "\n".join([pref, f"[b]Dataset[/b] {len(self.manager.node.train)} train samples"])
        pref = " | ".join([pref, f"{len(self.manager.node.val)} validation samples"])
        pref = " | ".join([pref, f"{len(self.manager.node.inference)} test samples"])
        pref = "\n".join([pref, f"[b]Epochs[/b] {self.manager.node.params.epochs} epochs."])
        pref = "\n".join([pref, f"[b]Batch size[/b] {self.manager.node.params.batch_size} samples."])
        self.connect_logs = pref
        Clock.schedule_once(self.toggle_join, 0)

    def join_train(self):
        self.manager.current = 'train'
