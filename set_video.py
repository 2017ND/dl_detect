# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'set_video.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_setVideoPath(object):
    def setupUi(self, setVideoPath):
        setVideoPath.setObjectName("setVideoPath")
        setVideoPath.resize(447, 176)
        self.groupBox = QtWidgets.QGroupBox(setVideoPath)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 431, 171))
        self.groupBox.setObjectName("groupBox")
        self.layoutWidget = QtWidgets.QWidget(self.groupBox)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 40, 401, 27))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.le_vpath = QtWidgets.QLineEdit(self.layoutWidget)
        self.le_vpath.setObjectName("le_vpath")
        self.horizontalLayout.addWidget(self.le_vpath)
        self.layoutWidget1 = QtWidgets.QWidget(self.groupBox)
        self.layoutWidget1.setGeometry(QtCore.QRect(130, 90, 171, 70))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.bt_sure = QtWidgets.QPushButton(self.layoutWidget1)
        self.bt_sure.setObjectName("bt_sure")
        self.gridLayout.addWidget(self.bt_sure, 1, 0, 1, 1)
        self.bt_test = QtWidgets.QPushButton(self.layoutWidget1)
        self.bt_test.setObjectName("bt_test")
        self.gridLayout.addWidget(self.bt_test, 2, 0, 1, 1)

        self.retranslateUi(setVideoPath)
        QtCore.QMetaObject.connectSlotsByName(setVideoPath)

    def retranslateUi(self, setVideoPath):
        _translate = QtCore.QCoreApplication.translate
        setVideoPath.setWindowTitle(_translate("setVideoPath", "Form"))
        self.groupBox.setTitle(_translate("setVideoPath", "设置检测视频源：(初始化地址为本机摄像头)"))
        self.label.setText(_translate("setVideoPath", "网络地址:"))
        self.bt_sure.setText(_translate("setVideoPath", "确定"))
        self.bt_test.setText(_translate("setVideoPath", "测试视频源"))

