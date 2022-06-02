1. cv2 window 관련
- 코드 실행시 cv2 window 창 두번 켜지는 현상 수정
- window 창에서 종료할 수 있도록 수정 : yolov5/utils/dataloaders.py 수정

2. 음성 출력 모듈 추가 : warning_sound.py
- 해당 모듈이 동작하기 위해서

3. Multi-processor 동작 코드 추가
- 이제 AI 모델 구동 동작코드와 서버 전송 동작 코드를 병렬적으로 실행합니다.
- 전역변수 send_q Queue를 통해 코드 간 데이터를 전송합니다.

4. detect_crosswalk_test.py 관련
- 이제 datetime의 timestamp가 한국시간(Asia, seoul)으로 설정되었습니다.

5. 이미지가 저장되는 tmp_raw_images, tmp_result_images 폴더 추가
- 위반행위 검출시 이미지가 해당 폴더에 저장됩니다.
- 해당 폴더의 메모리는 "upload_to_sever.py" 에서 관리됩니다.

6. Upload_to_sever_script.py 관련
- Renamed "Upload_to_Sever_script.py" to "upload_to_sever.py"
- 서버에 이미지 전송 후 폴더 내 이미지를 삭제하는 코드 추가 : 더이상 메모리를 신경쓸 필요 없습니다

7. Renamed "Detect_bike_road.py" to "detect_bike_road.py"
