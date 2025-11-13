import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_ducks_with_edges(image_path):
    img = cv2.imread(image_path)
    
    # Kropovanje slike (uzimanje centralnog dela)
    h, w, _ = img.shape
    crop_img = img[250:800, 200:800]  # uzimanje srednje trećine

    # Konvertovanje kropovane slike u grayscale
    grayscale_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Primena fiksnog threshold-a za binarizaciju slike
    _, binary_img = cv2.threshold(grayscale_img, 80, 255, cv2.THRESH_BINARY_INV)

    # Primenjujemo zatvaranje sa većim kernelom
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Primena detekcije ivica na zatvorenoj slici
    edges = cv2.Canny(closed_img, threshold1=50, threshold2=150)

    # Zatvaranje kontura nakon detekcije ivica
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Pronalazimo konture na slici sa detektovanim ivicama
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kopiramo sliku sa ivicama za popunjavanje kontura
    filled_img = edges_closed.copy()
    
    # Popunjavanje svake konture belom bojom
    for cnt in contours:
        cv2.drawContours(filled_img, [cnt], -1, 255, thickness=cv2.FILLED)

    # Ponovo pronalaženje kontura u popunjenoj slici
    contours, _ = cv2.findContours(filled_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtriranje kontura na osnovu površine za detekciju patkica
    duck_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Zadržavamo konture koje odgovaraju veličini patkica
        if 500 < area < 2000 or 4000 < area < 5000 or 8000 < area < 9000:
            duck_contours.append(cnt)

    # Broj prebrojanih patkica
    duck_count = len(duck_contours)

    # Kopija slike za prikaz kontura patkica
    contour_img = crop_img.copy()
    cv2.drawContours(contour_img, duck_contours, -1, (0, 255, 0), 2)  # Zeleno za konture patkica
    
    # Prikazivanje originalne slike, slike sa detektovanim ivicama, i slike sa konturama patkica
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    plt.title('Kropovana originalna slika')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(edges_closed, cmap='gray')
    plt.title('Slika sa detekcijom ivica i zatvorenim konturama')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Broj detektovanih patkica: {duck_count}')
    plt.axis('off')

    plt.show()

    # Ispis broja detektovanih patkica
    print(f"Broj detektovanih patkica u slici '{image_path}': {duck_count}")

# Prebrojavanje patkica za svaku sliku
for i in range(1, 11):
    image_path = f'data/picture_{i}.jpg'
    count_ducks_with_edges(image_path)
