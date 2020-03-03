# -*- coding: UTF-8 -*-
'''pyqt5环境'''
import sys
import os
import cv2
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
import threading
from PyQt5 import QtWidgets, QtGui, QtCore, Qt
from PyQt5.QtGui import QIcon, QBrush, QImage, QPixmap
from PyQt5.QtWidgets import *
from wnd import Ui_MainWindow
from set_video import Ui_setVideoPath
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
'''pyqt5环境'''

'''keras-yolov3环境'''
import time
import colorsys
from timeit import  default_timer as timer
import tensorflow as tf
import numpy as np
from keras import backend as K
from  keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

# 解决GPU分配问题
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
'''keras-yolov3环境'''


file = None  # 保存结果的文件



class MainWnd(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWnd, self).__init__(parent)
        self.setupUi(self)
        self.initUI()

    # 初始化状态函数
    def initUI(self):
        self.setWindowIcon(QIcon('./res/face_detect.ico'))
        self.setWindowTitle("深度学习快速检测软件")
        self.cBox.addItem('Image')
        self.cBox.addItem('Video')
        self.pic_show.setScaledContents(True)
        self.saver_path()
        self.readmodule()

        self.directory = None  # 打开文件目录
        self.results_dir = None  # 保存结果文件目录
        self.img_index = 0  # 显示图片的索引
        self.path_dir = []  # 整个目录的文件索引集
        self.path_files = []  # 整个目录的文件名集合
        self.new_icon = QIcon("./res/process.ico")
        self.root = None  # 所有item的根节点
        self.detect_results_path = []  # 所有检测图片的路径

        self.treeWidget.setColumnCount(1)
        self.treeWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.treeWidget.itemDoubleClicked.connect(self.show_pic)

        self.set_pic_path.triggered.connect(self.setp_path)
        self.set_video_path.triggered.connect(self.setv_path)

        self.open_results.triggered.connect(self.view_results)
        self.start_detect.triggered.connect(self.detect)

        self.cap = None     # 视频源
        # 计算时间
        self.accum_time = None
        self.prev_time = None


    # 设置检测图片的路径
    def setp_path(self):
        self.directory = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        if self.directory:
            self.treeWidget.clear()
            self.path_dir = []
            self.le_Pic_path.setText(self.directory)
            self.path_files = os.listdir(self.directory)

            fileInfo = Qt.QFileInfo(self.directory)
            fileIcon = Qt.QFileIconProvider()
            icon = QtGui.QIcon(fileIcon.icon(fileInfo))
            root = QTreeWidgetItem(self.treeWidget)
            self.root = root
            root.setIcon(0, QtGui.QIcon(icon))
            root.setText(0, self.directory)
            brush_green = QBrush(QtCore.Qt.green)
            root.setBackground(0, brush_green)
            brush_blue = QBrush(QtCore.Qt.blue)
            root.setBackground(1, brush_blue)

            self.file_num_max = len(self.path_files)
            for paths in self.path_files:
                self.path_dir.append(self.directory + '/' + paths)
            self.CreateTree(root)
            self.treeWidget.addTopLevelItem(root)
            self.treeWidget.expandAll()
            self.loadImg(self.path_dir[0])

    def CreateTree(self, root):
        for file in self.path_dir:
            if file:
                fileInfo = Qt.QFileInfo(file)
                fileIcon = Qt.QFileIconProvider()
                icon = QtGui.QIcon(fileIcon.icon(fileInfo))
                child = QTreeWidgetItem()
                child.setText(0, file.split('/')[-1])
                child.setIcon(0, QtGui.QIcon(icon))
                root.addChild(child)

    def show_pic(self, item):
        path = self.directory + '/' + item.text(0)
        self.loadImg(path)

    # 设置检测视频源路径
    def setv_path(self):
        set_video_wnd.show()

    # 加载图像文件显示
    def loadImg(self, filePath=None):
        if filePath is not None:
            jpg = QtGui.QPixmap(filePath).scaled(self.pic_show.width(), self.pic_show.height())
            if jpg:  # 这里判断有问题
                self.pic_show.setPixmap(jpg)
                self.pic_show.repaint()
            else:
                QMessageBox.warning(self, "警告!", "请检查图片路径，并重新输入！", QMessageBox.Yes)

    # 显示视频源路径
    def show_video_text(self, str):
        self.le_Video_path.setText(str)

    # 设置检测结果保存路径
    def saver_path(self):
        result_path = './detect_results'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.le_Save_path.setText(result_path)
        # result如果之前存放的有文件，全部清除,无法删除文件夹
        for i in os.listdir(result_path):
            path_file = os.path.join(result_path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
        # 创建一个记录检测结果的文件
        txt_path = result_path + '/detect_results.txt'
        global file
        file = open(txt_path, 'w')
    # 设置模型文件路径
    def readmodule(self):
        self.module_path = './model_data'
        self.le_module_path.setText(self.module_path)

    # 浏览结果
    def view_results(self):
        if self.cBox.currentText() == "Image":
            for index, path in enumerate(self.detect_results_path):
                if path:
                    print(path)
                    self.loadImg(path)
        if self.cBox.currentText() == "Video":
            pass

    # 检测图片或视频
    def detect(self):
        # 检测图片
        if self.cBox.currentText() == "Image":
            t1 = time.time()
            for filepath in self.path_dir:
                file.write(filepath.split('/')[-1] + ' detect_result：\n')
                print(filepath)
                image = Image.open(filepath)
                r_image = yolo.detect_image(image)
                file.write('\n')
                image_save_path = os.getcwd() + '/detect_results/result_' + filepath.split('/')[-1]
                self.detect_results_path.append(image_save_path)
                r_image.save(image_save_path)
                self.loadImg(image_save_path)
            time_sum = time.time() - t1
            file.write('time sum: ' + str(time_sum) + 's')
            file.close()
            yolo.close_session()
        # 检测视频
        if self.cBox.currentText() == "Video":
            self.cap = cv2.VideoCapture()  # 开启摄像头
            self.timer = QTimer()
            self.timer.timeout.connect(self.capPicture)
            if self.le_Video_path.text() == '0':
                flag =self.cap.open(0)
            else:
                flag = self.cap.open(self.le_Video_path.text())
            if flag:
                video_FourCC = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                video_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                output_path = "./detect.mp4"
                print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
                self.out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
                self.accum_time = 0
                self.curr_fps = 0
                fps = "FPS: ??"
                self.prev_time = timer()

                self.timer.start(30)

    def capPicture(self):
        # get a frame
        ret, img = self.cap.read()
        height, width, bytesPerComponent = img.shape
        bytesPerLine = bytesPerComponent * width
        # 检测帧图像
        image = Image.fromarray(img)
        image = yolo.detect_image(image)
        img_result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        self.accum_time = self.accum_time + exec_time
        self.curr_fps = self.curr_fps + 1
        if self.accum_time > 1:
            self.accum_time = self.accum_time - 1
            fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0
        # cv2.putText(img_result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.50, color=(255, 0, 0), thickness=2)
        self.out.write(img_result)
        # 变换彩色空间顺序
        cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB, img_result)
        # 转为QImage对象
        self.image = QImage(img_result.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pic_show.setPixmap(QPixmap.fromImage(self.image).scaled(self.pic_show.width(), self.pic_show.height()))
        # self.cap.release()   # 停止检测视频






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


class YOLO(object):
    _defaults = {
        "model_path":    'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }
    print(_defaults)
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()  # 开始计时

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)  # 打印图片的尺寸
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  # 提示用于找到几个bbox

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(2e-2 * image.size[1] + 0.2).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500

        # 保存框检测出的框的个数
        file.write('find  ' + str(len(out_boxes)) + ' target(s) \n')

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 写入检测位置
            file.write(
                predicted_class + '  score: ' + str(score) + ' \nlocation: top: ' + str(top) + '、 bottom: ' + str(
                    bottom) + '、 left: ' + str(left) + '、 right: ' + str(right) + '\n')

            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print('time consume:%.3f s ' % (end - start))
        return image

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWnd()
    set_video_wnd = VideoWnd()
    set_video_wnd.send_video_path.connect(win.show_video_text)
    yolo = YOLO()
    win.show()
    sys.exit(app.exec_())