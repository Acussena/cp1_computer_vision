from leitura_conversao import (
    load_image,
    convert_to_gray,
    display_image,
    plot_histogram,
    segment_image,
    apply_morphology,
    detect_contours,
    filter_and_draw_objects
)


if __name__ == "__main__":
    img = load_image("imagem.jpg")

    print("Dimensões da imagem:", img.shape)

    gray = convert_to_gray(img)

    display_image(gray, "Imagem em Cinza")

    plot_histogram(gray)

    binary = segment_image(gray)

    display_image(binary, "Segmentação")

    cleaned = apply_morphology(binary)

    display_image(cleaned, "Após Morfologia")

    contours = detect_contours(cleaned)

    result, total = filter_and_draw_objects(
        img,
        contours,
        min_area=500
    )

    display_image(result, "Objetos Detectados")

    print("Total de objetos detectados:", total)