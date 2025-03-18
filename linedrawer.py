import cv2 as cv


def draw_lines(frame, tracking_in):
    tracking = []
    for detobject in tracking_in:
        if detobject.updated:
            tracking.append(detobject)

    for detobject_id in range(len(tracking)):
        for i in range(0, 14):
            cv.line(frame, (int(tracking[detobject_id].predicted[i][0]), int(tracking[detobject_id].predicted[i][1])), (int(tracking[detobject_id].predicted[i + 1][0]), int(tracking[detobject_id].predicted[i + 1][1])), (0, 0, 255), 2)
            # print((int(tracking[detobject_id].predicted[i][0]), int(tracking[detobject_id].predicted[i][1])))
            # print((int(tracking[detobject_id].predicted[i + 1][0]), int(tracking[detobject_id].predicted[i + 1][1])))
            # print('---')
    return frame
