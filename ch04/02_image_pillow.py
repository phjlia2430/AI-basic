# Pillow를 이용하여 이미지 파일 열기
from PIL import Image
# 이미지 불러오기
image = Image.open('./lenna.png')
# 이미지 정보 요약
print (image.format)
print (image.mode)
print (image.size)

# 픽셀 보여주기
print (list(image.getdata()))
# 이미지 실행
image.show()

# 이미지 잘라내기
xy = (120, 50, 400,400) # left, up, right, down
crop_image = image.crop(xy)
crop_image.show()
crop_image.save('./crop_lenna.jpg')


'''
########################################
# 그레이 스케일로 변환하기
gs_image = image.convert(mode='L')
# 이미지 저장하기
gs_image.save('C:/python/lenna_grayscale.png')
# 픽셀 보여주기
print (list(gs_image.getdata()))
# 이미지 실행
gs_image.show()


'''

