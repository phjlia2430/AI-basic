import cv2

video = 'C:/python/people.mp4'

# 비디오 파일 읽어오기
cap = cv2.VideoCapture(video)

if cap.isOpened(): # 비디오파일이 존재하여 읽어왔다면
    while True:
        ret, img = cap.read() # 각 프레임을 읽어오기
        if ret:
            cv2.imshow(video, img) # 프레임 재생
            if cv2.waitKey(33) == ord('q'): # 키보드 q를 누르면 종료
                break
else:
    print("cannot open video file")

cap.release() # 비디오 객체 해제
cv2.destroyAllWindows()