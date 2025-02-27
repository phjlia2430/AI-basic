import cv2

image = 'C:/python/lenna.png'

image = cv2.imread(image, cv2.IMREAD_COLOR)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()