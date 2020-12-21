import cv2
import time

width = 416
height = 416


cam = cv2.VideoCapture(0)
cam.set(3, width)
cam.set(4, height)

frames = 0
start = time.time()
while True:
    # Get webcam input
    ret_val, img = cam.read()

#     # Mirror 
    img = cv2.flip(img, 1)

#     # Free-up unneeded cuda memory
#     torch.cuda.empty_cache()

    # Show FPS
    frames += 1
    intv = time.time() - start
    if intv > 1:
        print("FPS of the video is {:5.2f}".format( frames / intv ))
        print(type(img), img.shape)
        start = time.time()
        frames = 0
    
    # Show webcam
    cv2.imshow('Demo webcam', img)
    if cv2.waitKey(1) == 27: 
        break  # Press Esc key to quit
cam.release()
cv2.destroyAllWindows()