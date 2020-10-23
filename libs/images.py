# Objectif : apprendre a manipuler les images avec opencv
import cv2
import matplotlib.pyplot as plt
import numpy as np


# TODO : lire l'image avec opencv (imread) et l'afficher avec pyplot (imshow) pour verifier que tout va bien
def load_image() -> np.ndarray:
    return cv2.cvtColor(cv2.imread('../data/data_types/data/cat.jpg'), cv2.COLOR_BGR2RGB)


# TODO : faire une boucle, et changer tout les pixels noirs en blancs
image = load_image()
image[np.where((image[:, :, 0] < 45) & (image[:, :, 1] < 45) & (image[:, :, 2] < 45))] = (255, 255, 255)
plt.imshow(image)
plt.show()

# TODO : ecrire le code ci-dessus en plus efficace grace a cv2.floodfill()
# cv2.floodFill(image, )


image = load_image()
cv2.floodFill(
    image,
    None,
    seedPoint=(0, 0),
    newVal=(255, 255, 255),
    loDiff=(1, 2, 2, 2),
    upDiff=(2, 2, 2, 2),
)
plt.imshow(image)
plt.show()

# TODO: sauvegarder votre image dans ./out et verifier que celle ci est bien celle que vous voulez
plt.imsave("../out/cat.png", image)
