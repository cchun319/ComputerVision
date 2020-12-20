import cv2
import os

image_folder = '/home/cchun3/Desktop/yolo_output'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 35, (480,480))

for image in images:
    resized = cv2.resize(cv2.imread(os.path.join(image_folder, image)), (480, 480), interpolation = cv2.INTER_AREA) 
    video.write(resized)

cv2.destroyAllWindows()
video.release()