source /opt/intel/openvino/bin/setupvars.sh 
python3 main.py -f ../intel/face-detection-adas-binary-0001.xml -fl ../intel/landmarks-regression-retail-0009.xml -hp ../intel/head-pose-estimation-adas-0001.xml -g ../intel/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -pt 0.6 -d "CPU" -fg 3
