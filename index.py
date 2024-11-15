import cv2 as cv

video_src = "video1.avi"

car_video = cv.VideoCapture(video_src)

car_model = cv.CascadeClassifier("cars.xml")

while True:
    ret, frame = car_video.read()

    if not ret:
        break

    car_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detect_car = car_model.detectMultiScale(car_gray)


    for (x, y, w, h) in detect_car:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 3)

    cv.imshow("video", frame)
    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()
