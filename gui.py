from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
import time
import threading
from kivy.clock import Clock, mainthread


"""class Gui(Widget):
    def __init__(self):
        self.logs = TextInput(text='fups', size_hint=(None, None), size=(100, 50))
"""

class Nnet(FloatLayout):
    stop = threading.Event()
    def __init__(self, **kwargs):
        super(Nnet, self).__init__(**kwargs)
        self.loggerQueue = []
        self.logger = TextInput(pos_hint={'center_y': .5}, size_hint=(None, 0.95), x=5)
        self.logger.width = 300
        self.add_widget(self.logger)
        Clock.schedule_interval(self.logger_updater_thread, timeout = 0)  
#        threading.Thread(target=self.logger_updater_thread).start()
        threading.Thread(target=self.nnet_thread).start()
  


#    @mainthread
    def logger_updater_thread(self, *args):
        if self.loggerQueue:
            self.logger.text += self.loggerQueue.pop(0)

    def nnet_thread(self):
        iteration = 0
        while not self.stop.is_set():
            iteration +=1
            self.loggerQueue.append(str(iteration))
            print('iteration %s' % iteration)
            time.sleep(1)

class GuiApp(App):
    def __init__(self, **kwargs):
        super(GuiApp, self).__init__(**kwargs)
        self.real_app = Nnet()

    def on_stop(self):
        self.real_app.stop.set()

    def build(self):
        return self.real_app

if __name__ == '__main__':
    GuiApp().run()

