import sys, time
import threading



from wnd1 import Ui_MainWindow

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication



class MainWnd(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWnd, self).__init__(parent)
        self.setupUi(self)



def fb():
    splash = QtWidgets.QSplashScreen(QtGui.QPixmap("img.jpg"))
    splash.showMessage("软件初始化中,请稍后.....     designed by alinn ", QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
                       QtCore.Qt.black)
    splash.show()  # 显示启动界面
    time.sleep(3)
    splash.finish()  # 隐藏启动界面

if __name__ == '__main__':
    import os
    app = QApplication(sys.argv)
    t1 = threading.Thread(target=fb, args=())
    t1.start()
    win = MainWnd()
    time.sleep(3)
    win.show()
    print(os.getcwd())
    sys.exit(app.exec_())