# Import OpenCV libraries and tools
import cv2
import time


def main():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Take that!")
        return

    img_num = 0
    threshold_min_area = 80_000
    threshold_max_area = 500_000

    while True:
        # image = cv2.imread("image_0.jpg")
        # cv2.imshow("Input", image)

        ret, image = cap.read()
        cv2.imshow("image", image)
        cv2.waitKey(1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            area = cv2.contourArea(c)
            x_min = min(c[:, :, 0])[0] - 20
            x_max = max(c[:, :, 0])[0] + 20
            y_min = min(c[:, :, 1])[0] - 20
            y_max = max(c[:, :, 1])[0] + 20

            if x_min < 0:
                x_min = 0
            if x_max < 0:
                x_max = 0
            if y_min < 0:
                y_min = 0
            if y_max < 0:
                y_max = 0

            if area > threshold_min_area and area < threshold_max_area and (x_max - x_min) < (y_max - y_min):
                # print(x_min, ": ", x_max)
                # print(y_min, ": ", y_max)

                crop_img = image[y_min:y_max, x_min:x_max].copy()

                # cv2.drawContours(image, [c], 0, (36, 255, 12), 3)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.imshow("image", image)
                cv2.moveWindow("image", 2000, -100)
                # cv2.imshow("crop", crop_img)
                cv2.waitKey(1)

                if img_num >= 200:
                    return

                cv2.imwrite(f"image_{img_num}.jpg", crop_img)
                img_num += 1

                print(img_num)
                # time.sleep(0.25)


if __name__ == "__main__":
    main()
