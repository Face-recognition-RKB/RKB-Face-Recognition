import cv2
import time
import argparse
import os
import sys
import numpy as np
from detection.FaceDetector import FaceDetector
from recognition.FaceRecognition import FaceRecognition
from classifier.FaceClassifier import FaceClassifier


def main(args):
    face_detector = FaceDetector()
    face_recognition = FaceRecognition(args.model)
    face_classfier = FaceClassifier(args.classifier_filename)
    video_capture = cv2.VideoCapture(args.video_input)
    output_file = './media/result/'+os.path.basename(args.video_input)+'_result.avi'

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 24.0, (int(video_capture.get(3)),int(video_capture.get(4))))

    print('Start Recognition!')
    prevTime = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        curTime = time.time()  # calc fps
        find_results = []

        frame = frame[:, :, 0:3]
        boxes, scores = face_detector.detect(frame)
        face_boxes = boxes[np.argwhere(scores>0.3).reshape(-1)]
        face_scores = scores[np.argwhere(scores>0.3).reshape(-1)]
        print('Detected_FaceNum: %d' % len(face_boxes))

        if len(face_boxes) > 0:
            for i in range(len(face_boxes)):
                box = face_boxes[i]
                cropped_face = frame[box[0]:box[2], box[1]:box[3], :]
                cropped_face = cv2.resize(cropped_face, (160, 160), interpolation=cv2.INTER_AREA)
                feature = face_recognition.recognize(cropped_face)
                name = face_classfier.classify(feature)

                cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

                # plot result idx under box
                text_x = box[1]
                text_y = box[2] + 20
                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 0, 255), thickness=1, lineType=2)
        else:
            print('Unable to align')

        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / (sec)
        str = 'FPS: %2.3f' % fps
        text_fps_x = len(frame[0]) - 150
        text_fps_y = 20
        cv2.putText(frame, str, (text_fps_x, text_fps_y),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)

        out.write(frame)

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('face_detection_method', type=str, choices=['SSD', 'MTCNN'],
      help='Choose face detection method, either Mobilenet-SSD or MTCNN', default='SSD')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('video_input', type=str,
        help='Path to the data directory containing video input.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
