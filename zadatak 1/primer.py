import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_ducks_with_filled_contours(image_path):
    img = cv2.imread(image_path)
    
    # Kropovanje slike (uzimanje centralnog dela)
    h, w, _ = img.shape
    crop_img = img[250:800, 200:800]  # uzimanje srednje trećine

    # Konvertovanje kropovane slike u grayscale
    grayscale_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Binarizacija sa fiksnim pragom
    _, binary_img = cv2.threshold(grayscale_img, 80, 255, cv2.THRESH_BINARY_INV)

    # Uklanjanje šuma sa zatvaranjem kontura
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Detekcija ivica sa Canny
    edges = cv2.Canny(closed_img, threshold1=50, threshold2=150)

    # Dilacija na ivicama radi spajanja kontura
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    dilated_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    
    # Zatvaranje da bi se povezali obližnji regioni
    edges_closed = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Popunjavanje kontura kako bi se formirali puni oblici
    filled_img = np.zeros_like(grayscale_img)  # Crna pozadina za popunjene konture
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_contour_areas = []  # Lista za skladištenje površina kontura
    contour_labels = []  # Lista za skladištenje parova (broj, kontura)

    for i, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        if 750 < area < 2000 or 2200 < area < 3000 or 9000 < area < 10000:
            # Popunjavamo konture belom bojom
            cv2.drawContours(filled_img, [cnt], -1, 255, thickness=cv2.FILLED)
            filled_contour_areas.append(area)
            contour_labels.append((i, cnt))  # Čuvamo broj konture i samu konturu

    # Ispis površina popunjenih kontura
    print("Površine svih popunjenih kontura koje zadovoljavaju uslov:")
    for i, area in enumerate(filled_contour_areas, 1):
        print(f"Kontura {i}: Površina = {area}")

    # Kopija slike za prikaz kontura patkica sa oznakama
    contour_img = crop_img.copy()
    for label, cnt in contour_labels:
        # Pronalazak centra konture za ispisivanje broja
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(contour_img, str(label), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Prikazivanje slike nakon popunjavanja kontura i konačnih detekcija
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    plt.title('Kropovana originalna slika')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(filled_img, cmap='gray')
    plt.title('Popunjene konture (samo u opsegu)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Broj detektovanih patkica: {len(filled_contour_areas)}')
    plt.axis('off')

    plt.show()

    # Ispis broja detektovanih patkica
    print(f"Broj detektovanih patkica u slici '{image_path}': {len(filled_contour_areas)}")

# Poziv funkcije za svaku sliku
for i in range(1, 11):
    image_path = f'data/picture_{i}.jpg'
    count_ducks_with_filled_contours(image_path)
