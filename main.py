# -*- coding: UTF-8 -*-
import sys
import os
import cv2
import time
from tkinter import *
import tkinter.ttk as ttk
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5 import QtWidgets, QtGui, QtCore, Qt
from PyQt5.QtGui import QIcon, QBrush, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import *
from set_video import Ui_setVideoPath
from PyQt5.QtCore import QObject,pyqtSignal, QDir
from wnd1 import Ui_MainWindow


class MainWnd(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWnd, self).__init__(parent)
        self.setupUi(self)
        self.initUI()

    # 初始化状态函数
    def initUI(self):
        self.setWindowIcon(QIcon('./res/face_detect.ico'))
        self.setWindowTitle("深度学习快速检测软件")
        self.directory = None   # 打开文件目录
        self.img_index = 0  # 显示图片的索引
        self.path_dir = []  # 整个目录的文件索引集
        self.path_files = []  # 整个目录的文件名集合
        self.icon = None  # 原始icon
        self.new_icon = QIcon("./res/process.ico")
        self.itemlist = []   # 所有item的集合


        self.new_pro.triggered.connect(self.setuppro)
        self.open_pro.triggered.connect(self.openpro)
        self.save_pro.triggered.connect(self.savepro)
        self.saveas_pro.triggered.connect(self.saveaspro)

        self.set_pic_path.triggered.connect(self.setp_path)
        self.set_video_path.triggered.connect(self.setv_path)
        self.save_results_path.triggered.connect(self.saver_path)
        self.read_module.triggered.connect(self.readmodule)

        self.open_results_path.triggered.connect(self.open_results)
        self.start_detect.triggered.connect(self.detect)

    def setuppro(self):
        print("here new pro")

    def openpro(self):
        print("here new pro")

    def savepro(self):
        print("here new pro")

    def saveaspro(self):
        print("here new pro")

    # 设置检测图片的路径
    def setp_path(self):
        self.directory = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        if self.directory:
            self.path_dir = []
            self.le_Pic_path.setText(self.directory)
            self.path_files = os.listdir(self.directory)

            fileInfo = Qt.QFileInfo(self.directory)
            fileIcon = Qt.QFileIconProvider()
            icon = QtGui.QIcon(fileIcon.icon(fileInfo))
            self.icon = icon

            self.tree.setRootIsDecorated(False)
            self.tree.setAlternatingRowColors(True)

            model = QFileSystemModel()
            model.setRootPath(self.directory)
            model.setFilter(QDir.Dirs)
            self.tree.setModel(model)
            self.file_num_max = len(self.path_files)
            for paths in self.path_files:
                self.path_dir.append(self.directory + '/' + paths)
            self.tree.setModel(model)

            self.loadImg(self.path_dir[0])






    # 设置检测视频源路径
    def setv_path(self):
        set_video_wnd.show()

    # 加载图像文件显示
    def loadImg(self, filePath=None):
        if filePath is not None:
            jpg = QtGui.QPixmap(filePath).scaled(self.pic_show.width(), self.pic_show.height())
            if jpg:   # 这里判断有问题
                self.pic_show.setPixmap(jpg)
            else:
                QMessageBox.warning(self, "警告!", "请检查图片路径，并重新输入！", QMessageBox.Yes)


    # 显示视频源路径
    def show_video_text(self, str):
        self.le_Video_path.setText(str)

    # 设置检测结果保存路径
    def saver_path(self):
        print("here new pro")

    # 读取模型文件h5
    def readmodule(self):
        print("here new pro")

    # 打开结果保存文件夹
    def open_results(self):
        print("here new pro")

    # 检测图片或视频
    def detect(self):
        pass




class VideoWnd(QtWidgets.QMainWindow, Ui_setVideoPath):
    send_video_path = pyqtSignal(str)

    def __init__(self, parent=None):
        super(VideoWnd, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.path_real = ''
        self.bt_sure.clicked.connect(self.sure)
        self.bt_test.clicked.connect(self.video_test)

    def sure(self):
        if self.le_vpath.text():
            self.path_real = self.le_vpath.text()
            self.send_video_path.emit(self.path_real)
            QMessageBox.information(self, "提示", "已输入视频源地址！", QMessageBox.Yes)
        else:
            QMessageBox.information(self, "提示", "请使用默认本机摄像头或重新输入地址！", QMessageBox.Yes)
            self.path_real = ''
            self.send_video_path.emit('0')

    def video_test(self, path=0):
        if self.path_real is not '':
            path = self.path_real
            cap = cv2.VideoCapture(str(path))
        else:
            cap = cv2.VideoCapture(0)

        while (1):
            # get a frame
            ret, frame = cap.read()
            # show a frame
            if ret:
                cv2.imshow("video test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                QMessageBox.warning(self, "警告!", "请检查视频源地址，并重新输入！", QMessageBox.Yes)
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWnd()
    win.show()
    set_video_wnd = VideoWnd()
    set_video_wnd.send_video_path.connect(win.show_video_text)
    sys.exit(app.exec_())