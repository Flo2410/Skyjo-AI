# Import OpenCV libraries and tools
import cv2
import time


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Take that!")
        return

    img_num = 0
    lable = input("Lable: ")

    while True:
        if img_num >= 10:
            return

        ret, frame = cap.read()
        cv2.imwrite(f"{lable}/image_{img_num}.jpg", frame)
        img_num += 1
        time.sleep(0.5)

if __name__ == "__main__":
    main()