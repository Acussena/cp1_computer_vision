import cv2
import matplotlib.pyplot as plt


def load_image(path: str):

    image = cv2.imread(path)

    if image is None:
        raise ValueError("Erro ao carregar imagem")

    return image

def convert_to_gray(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray

def display_image(image, title="Imagem"):

    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_histogram(gray_image):

    hist = cv2.calcHist(
        [gray_image],
        [0],
        None,
        [256],
        [0,256]
    )

    plt.plot(hist)
    plt.title("Histograma de Intensidade")
    plt.xlabel("Intensidade")
    plt.ylabel("Pixels")
    plt.show()


def segment_image(gray):

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


def apply_morphology(binary):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5,5)
    )

    result = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    return result


def detect_contours(binary):

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def filter_and_draw_objects(image, contours, min_area=500):

    img_copy = image.copy()

    count = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        x,y,w,h = cv2.boundingRect(cnt)

        cv2.rectangle(
            img_copy,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        cv2.putText(
            img_copy,
            f"A:{int(area)}",
            (x,y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )

        count += 1

    return img_copy, count