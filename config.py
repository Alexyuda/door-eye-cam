import argparse


# Parse command line arguments
parser = argparse.ArgumentParser(description="Video processing pipeline")
parser.add_argument("-i", "--input", default="0",
                help="path to input video file or camera identifier")
parser.add_argument("-d", "--display", default=True, help="display video result")
parser.add_argument("-p", "--progress", default=True, help="display progress")
parser.add_argument("--prototxt", default="./models/face_detector/deploy.prototxt.txt",
                help="path to Caffe 'deploy' prototxt file")
parser.add_argument("--model", default="./models/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
                help="path to Caffe pre-trained model")
parser.add_argument("--confidence", type=float, default=0.5,
                help="minimum probability to filter weak face detections")
parser.add_argument("--batch-size", type=int, default=1,
                help="face detection batch size")
