import cv2 as cv


# Arg could also be a path to a file
cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


if __name__ == '__main__':
    while True:
        ret, img = cap.read()
        img_cpy = img.copy()
        assert ret
        face_rects = face_cascade.detectMultiScale(img, 1.3, 3, 0, (200, 200))
        for rect in face_rects:
            cv.rectangle(img_cpy, rect, (255, 0, 0))
        print(face_rects)
        cv.imshow('image', img)
        cv.waitKey(2)
