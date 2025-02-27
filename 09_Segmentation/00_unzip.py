import tarfile
import os

def extract_tar_gz(file_path, extract_path="."):
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            print(f"압축 해제 중: {file_path} -> {extract_path}")
            # 파일 압축 해제
            tar.extractall(path=extract_path)
            print("압축 해제 완료.")
    except Exception as e:
        print(f"에러 발생: {e}")


image_path = "./oxford-iiit-pet/images.tar.gz"
annotation_path = "./oxford-iiit-pet/annotations.tar.gz"
extract_path1 = "./oxford-iiit-pet/images"
extract_path2 = "./oxford-iiit-pet/annotations"


# 경로가 없다면 생성
os.makedirs(extract_path1, exist_ok=True)
os.makedirs(extract_path2, exist_ok=True)

# 함수 호출
extract_tar_gz(image_path, extract_path1)
extract_tar_gz(annotation_path, extract_path2)
