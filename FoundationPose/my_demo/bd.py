import cv2

aruco = cv2.aruco

dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_6X6_250
)

board = aruco.CharucoBoard(
    (7, 5),
    0.04,
    0.03,
    dictionary
)

img = board.generateImage((1400, 1000))

cv2.imwrite("charuco.png", img)