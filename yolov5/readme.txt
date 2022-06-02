ver 2022.06.03 05:22

1. cv2 window 관련
- 코드 실행시 cv2 window 창 두번 켜지는 현상 수정
- window 창에서 종료할 수 있도록 수정 : yolov5/utils/dataloaders.py 수정

2. 음성 출력 모듈 추가 : warning_sound.py
- 현재 간혈적 작동 불가로 수정 중 with 범준

3. Multi-processor 동작 삭제
- 실험결과 더 성능이 악화되거나 해당 프로젝트에 사용되기 어려움 -> 제거

4. detect_crosswalk_test.py 관련
- datetime package는 더이상 pandas에 존재하지 않아 경고가 반복적으로 발생하므로 수정.
- datetime의 timestamp가 한국시각(Asia, seoul)으로 설정되었습니다.
- 이제 위반 Class를 서버에 전송합니다.

5. 이미지 저장 관련
- raw 이미지가 서버에 전송되지 않던 오류 수정
- 이미지가 저장되는 tmp_raw_images, tmp_result_images 폴더가 추가되었습니다.
- 위반행위 검출시 이미지가 해당 폴더에 저장됩니다.
- 웹페이지 오작동 버그를 해결하기 위해 저장되는 이미지의 이름이 count 기반이 아니라 시간 기반으로 변경되었습니다.
- 해당 폴더의 메모리는 "upload_to_sever.py" 에서 관리됩니다.
- upload_to_sever.py 에 서버에 이미지 전송 후 폴더 내 이미지를 삭제하는 코드 추가
- 더이상 저장공간을 신경 쓸 필요 없습니다.

6. ransec 오류로 프로그램 에러발생시 자동으로 코드 재시작하도록 설정

7. 매우 간단한 GUI 제작
- cv2 윈도우 두개 붙여 만듦

8. Renamed
- Renamed "Detect_bike_road.py" to "detect_bike_road.py"
- Renamed "Upload_to_Sever_script.py" to "upload_to_sever.py"