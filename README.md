# dl_detect
基于PyQt5 编写的检测平台
基于keras-yolov3的模型，读取图片和视频进行检测，并将检测结果保存。
功能如下：
1.读取训练模型h5文件和类别txt
2.设置图片路径或者视频源（可采用摄像头或者网络地址rtsp/rtmp)
3.选择检测类型Image/Video即可，开始检测；保存结果到detect_results文件夹下。

注：所需环境见./doc/requirments.txt
