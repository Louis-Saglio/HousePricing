# Objectif : apprendre a manipuler les videos
import cv2
import matplotlib.pyplot as plt
import time

# TODO : ouvrir la video avec opencv VideoCapture()
video = cv2.VideoCapture('../data/data_types/data/video_drone.mp4')

# TODO : afficher cette video en temps reel (boucle avec cv2.imshow)
frame_number = 0
start = time.time()
ret = True
while ret:
    ret, frame = video.read()
    if frame is not None:
        fps = frame_number / (time.time() - start)
        cv2.putText(
            frame,
            str(fps),
            org=(0, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 0),
            thickness=2,
        )
        frame_number += 1
        cv2.imshow('', frame)
    if cv2.waitKey(1) & ord('q') == 0xFF:
        break

# TODO : ajouter le fps reel a laquelle vous affichez la video lors de l'affichage (cv2.putText)
# TODO : comment pouvez vous accelerer le fps ?
