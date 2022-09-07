import argparse
import subprocess
import cv2

import tensorflow as tf


_CLASS_COLORS = [
    (0, 255, 0), (199, 21, 133), (0, 100, 0), (255, 0, 0 ),
    (154, 205, 50), (123, 104, 238), (255, 160, 122),
    (32, 178, 170), (216, 191, 216), (255, 255, 0),
    (210, 105, 30), (175, 238, 238), (135, 206, 250),
    (220, 220, 220), (255, 248, 220), (100, 149, 237),
]
_FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.75
_FONT_WEIGHT = 1

_MODEL_FILE = '../models/ssd_mobilenet_v1/mobilenet.tflite'
_LABELS_FILE = '../models/ssd_mobilenet_v1/labelmap.txt'


class ObjectDetector:
    def __init__(self):
        self.tflite = tf.lite.Interpreter(model_path=_MODEL_FILE)
        self.tflite.allocate_tensors()

        # Get index of input tensor
        input_details = self.tflite.get_input_details()
        self.input_tensor = input_details[0]['index']

        # Get indexes of resulting tensors
        output_details = self.tflite.get_output_details()
        self.boxes_tensor = output_details[0]['index']
        self.classes_tensor = output_details[1]['index']
        self.scores_tensor = output_details[2]['index']
        self.num_det_tensor = output_details[3]['index']

        # Load labels
        self.labelmap = {}
        with open(_LABELS_FILE) as f:
            for i, name in enumerate(f):
                if i == 0: # background category
                    continue

                classe = i-1
                name = name.strip()
                text_size, baseline = cv2.getTextSize(name, _FONT_FACE, _FONT_SCALE, _FONT_WEIGHT)
                self.labelmap[classe] = {
                    'name': name,
                    'txt_w': text_size[0],
                    'txt_h': text_size[1],
                    'baseline': baseline,
                    'color': _CLASS_COLORS[classe % len(_CLASS_COLORS)]
                }

    def detect(self, img_orig):
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300), cv2.INTER_AREA)
        img = img.reshape([1, 300, 300, 3])

        self.tflite.set_tensor(self.input_tensor, img)
        self.tflite.invoke()

        boxes = self.tflite.get_tensor(self.boxes_tensor)[0]
        classes = self.tflite.get_tensor(self.classes_tensor)[0]
        scores = self.tflite.get_tensor(self.scores_tensor)[0]
        num_det = self.tflite.get_tensor(self.num_det_tensor)[0]

        # Draw detection boxes
        for i in range(int(num_det)):
            coords = boxes[i]
            classe = int(classes[i])
            score = scores[i]

            h, w = img_orig.shape[:2]
            ymin, xmin, ymax, xmax = coords
            clr = _CLASS_COLORS[classe % len(_CLASS_COLORS)]
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            cv2.rectangle(img_orig, (x1, y1), (x2, y2), clr, 2)

            label = self.labelmap[classe]
            cv2.rectangle(img_orig, (x1, y1 + label['baseline']), (x1 + label['txt_w'], y1 - label['txt_h']), clr, -1)
            cv2.putText(img_orig, label['name'], (x1, y1), _FONT_FACE, _FONT_SCALE, (0, 0, 0), _FONT_WEIGHT)


class RtspReaderIterator:
    def __init__(self, cap):
        self.cap = cap

    def __next__(self):
        if not self.cap.isOpened():
            raise StopIteration()
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration()
        return frame

    def __iter__(self):
        return self


class RtspReader:
    def __init__(self, rtsp_url):
        print('Init RtspReader')
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            raise Exception(f'Failed to open RTSP stream {rtsp_url}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __enter__(self):
        return self

    def __exit__(self, et, ev, t):
        print('Close OpenCV')
        self.cap.release()

    def __iter__(self):
        return RtspReaderIterator(self.cap)


class RtspStreamer:
    def __init__(self, rtsp_url, fps):
        print('Init RtspStreamer')
        self.rtsp_url = rtsp_url
        self.proc = None
        self.fps = fps

    def start_proc(self, frame):
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        if len(frame.shape) == 2:
            pix_fmt = 'gray'
        else:
            if frame.shape[2] != 3:
                raise Exception('Unsupported frame format')
            pix_fmt = 'bgr24'

        command = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', f'{frame_w}x{frame_h}',
            '-pix_fmt', pix_fmt,
            '-r', str(self.fps),
            '-i', '-',
            '-an',
            '-c:v', 'libx264',
            '-g', str(self.fps), # num frames between keyframes, set to FPS to get 1sec
            '-preset', 'superfast',
            '-tune', 'zerolatency',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            f'{self.rtsp_url}'
        ]
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)

    def write(self, frame):
        if not self.proc:
            self.start_proc(frame)
        self.proc.stdin.write(frame.tostring())

    def __enter__(self):
        return self

    def __exit__(self, et, ev, t):
        print('Close RTSP streamer')
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()


def _detect_img_file(img_in: str, img_out: str, target_w: int):
    img = cv2.imread(img_in)

    ObjectDetector().detect(img)

    if target_w:
        img = _resize_image(img, target_w)

    if img_out:
        cv2.imwrite(img_out, img)
        return

    # Show output in a window
    cv2.imshow(img_in, img)
    print('Press ESC key in the image window to exit...')
    while True:
        if cv2.waitKey(100) == 27:
            break
        if cv2.getWindowProperty(img_in, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def _resize_image(frame, target_w: int):
    if not target_w:
        return frame
    source_h, source_w, ch = frame.shape
    target_h = int(source_h * (target_w / source_w))
    if source_w == target_w:
        return frame
    return cv2.resize(frame, (target_w, target_h))


def _detect_rtsp__window(rtsp_url: str, target_w: int):
    tflite = ObjectDetector()
    with RtspReader(rtsp_url) as rtsp:
        for frame in rtsp:
            frame = _resize_image(frame, target_w)

            tflite.detect(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


def _detect_rtsp__restream(rtsp_in: str, rtsp_out: str, target_w: int):
    tflite = ObjectDetector()
    with RtspReader(rtsp_in) as rtsp:
         with RtspStreamer(rtsp_out, rtsp.fps) as streamer:
            for frame in rtsp:
                frame = _resize_image(frame, target_w)
                tflite.detect(frame)
                streamer.write(frame)


def _main():
    print('Sample box detector')

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image or rtsp stream')
    parser.add_argument('-o', '--output', help='output image sile or rtsp stream')
    parser.add_argument('-r', '--resize', help='resize video frame to this width', type=int)
    args = parser.parse_args()

    if not args.input:
        raise Exception('Input source is not specified (--input)')

    if args.input.startswith('rtsp://'):
        if args.output:
            print(f'Detecting RTSP: {args.input} -> {args.output} (target_w={args.resize})')
            _detect_rtsp__restream(args.input, args.output, args.resize)
            return

        print(f'Detecting RTSP: {args.input} -> window (target_w={args.resize})')
        _detect_rtsp__window(args.input, args.resize)
        return

    print(f'Detecting image: {args.input} -> {args.output or "window"}')
    _detect_img_file(args.input, args.output, args.resize)


if __name__ == '__main__':
    _main()
