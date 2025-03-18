import cv2
import torch
import numpy as np
import time
import math
import pyautogui
import ObjectPolynomialRegression
import linedrawer


class ObjectDetection:
    def __init__(self, capture_index, module_name):
        self.capture_index = capture_index
        self.model = self.load_model(module_name)
        self.classes = self.model.names

        self.tracking_objects = []
        self.center_points_cur_frame = []
        self.center_points_prev_frame = []
        self.await_new_reappearance = False
        self.await_new_object_id = None
        self.saved_disappeared_info = None
        self.count = 0
        self.size = (pyautogui.size())
        self.cur_id = 1

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')

    def get_video_capture(self):
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, module_name):
        if module_name:
            model = torch.hub.load('yolov5-master', 'custom', path=module_name, source='local', force_reload=True)
        else:
            model = torch.hub.load('yolov5-master', 'yolo5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def track(self, results, frame):
        self.count += 1
        tracking_objects_copy = self.tracking_objects.copy()

        self.center_points_cur_frame = []
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                cx = int((x2 + x1) / 2)
                cy = int((y2 + y1) / 2)
                self.center_points_cur_frame.append(np.array([cx, cy]))
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (125, 125, 125), 2)
                frame = cv2.putText(frame, f'{cx}, {cy}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        center_points_cur_frame_copy = self.center_points_cur_frame.copy()
        for i in range(len(center_points_cur_frame_copy)):
            center_points_cur_frame_copy[i] = list(center_points_cur_frame_copy[i])

        # print(self.tracking_objects)
        for i in self.tracking_objects:
            i.updated = False
        if self.count >= 2 and len(self.center_points_cur_frame) != 0 and len(self.tracking_objects) != 0:
            for detobject in self.tracking_objects:
                shortest_distance = 1e+9
                count = 0

                for pt in center_points_cur_frame_copy:
                    if self.count >= 4:
                        prediction = detobject.v
                    else:
                        prediction = detobject.center_point
                    distance = math.hypot(prediction[0] - pt[0], prediction[1] - pt[1])

                    if distance < shortest_distance:
                        shortest_distance = distance
                        object_to_be_changed = detobject
                        pt_to_be_changed = pt
                        index = count
                    count += 1

                if len(center_points_cur_frame_copy) != 0:
                    del center_points_cur_frame_copy[index]
                else:
                    break
                object_to_be_changed.last_center_point = object_to_be_changed.center_point.copy()
                object_to_be_changed.center_point = pt_to_be_changed
                object_to_be_changed.updated = True
                detobject.frames_gone = 0

                # This algorithm is beautiful
            for i in range(len(center_points_cur_frame_copy)):
                center_points_cur_frame_copy[i] = np.array(center_points_cur_frame_copy[i])
            self.center_points_cur_frame = center_points_cur_frame_copy
        for pt in self.center_points_cur_frame:
            self.tracking_objects.append(ObjectPolynomialRegression.DetObject(pt, self.cur_id))
            self.cur_id += 1

        for detobject in self.tracking_objects:
            detobject.update(self.count)
            print('----')
            if not detobject.updated:
                detobject.frames_gone += 1
            if detobject.frames_gone >= 50:
                self.tracking_objects.remove(detobject)

        tracking_objects_copy = self.tracking_objects.copy()
        for detobject in tracking_objects_copy:
            if not detobject.updated:
                tracking_objects_copy.remove(detobject)

        # print(self.tracking_objects)

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                cx = int((x2 - x1) / 2 + x1)
                cy = int((y2 - y1) / 2 + y1)
                for detobject in tracking_objects_copy:
                    if cx == detobject.center_point[0] and cy == detobject.center_point[1]:
                        try:
                            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), detobject.bgr, 2)
                        except IndexError:
                            pass

        self.center_points_prev_frame = self.center_points_cur_frame.copy()
        # if self.count > 5:
        frame = linedrawer.draw_lines(frame, self.tracking_objects)
        return frame

    def __call__(self):
        if type(self.capture_index) != list:
            cap = self.get_video_capture()
            assert cap.isOpened()
        else:
            frame_nr = 0
        go_tracker = False
        go_tracker_wait = False
        count = 0
        while True:

            if type(self.capture_index) != list:
                try:
                    ret, frame = cap.read()
                    assert ret
                except IndexError:
                    break
            else:
                frame = self.capture_index[frame_nr]
                frame_nr += 1
                try:
                    self.capture_index[frame_nr + 1]
                except IndexError:
                    break
            count += 1

            # height_multiplier = self.size[1] / 704
            # width = int(self.size[0] / height_multiplier)
            # multiple_of_32 = int(width / 32)
            # if width % 32 > 16:
                # multiple_of_32 += 1
            # width = multiple_of_32 * 32

            # frame = cv2.resize(frame, (704, 704))

            start_time = time.time()
            results = self.score_frame(frame)
            frame = self.track(results, frame)

            end_time = time.time()
            fps = 1/np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv5 Detection', frame)

            if cv2.waitKey(25) & 0xFF == 27:
                break
        if type(self.capture_index) != list:
            cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection(capture_index='output.mp4', module_name='thimbles.pt')
detector()
