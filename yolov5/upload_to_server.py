import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import os

# Firebase 인증서 위치
certi_route = "capstone-continue-firebase-adminsdk-ubpi1-68bc0133b4.json"

# Firebase initialize
cred = credentials.Certificate(certi_route)
firebase_admin.initialize_app(cred, {
        'projectId' : 'capstone-continue',
        'storageBucket': 'capstone-continue.appspot.com'
})
bucket = storage.bucket()
db = firestore.client()

# Upload('2020', "ai center", 1, 1, "1.jpg", "2.jpg")
# Upload(raw data : 0, 시간, 장소, case, class_num, raw_image, result_image)

def main(msg):
    # time, place, case, class_num, raw_image_route, result_image_route
    time = msg[0]
    place = msg[1]
    case = msg[2]
    class_num = msg[3]
    raw_image_route = msg[4]
    result_image_route = msg[5]

    filename = raw_image_route
    save_in = "Detected_raw_images/"
    blob_raw = bucket.blob(save_in + filename)
    blob_raw.upload_from_filename("tmp_raw_images/" + filename)
    blob_raw.make_public()

    filename = result_image_route
    save_in = "Detected_result_images/"
    blob_result = bucket.blob(save_in + filename)
    blob_result.upload_from_filename("tmp_result_images/" + filename)
    blob_result.make_public()

    doc_ref = db.collection(u'Detection').document(u'json')
    doc_ref.set({
        u'time': time,
        u'place': place,
        u'case': case,
        u'class': class_num,
        u'raw_imageURL': blob_raw.public_url,
        u'result_imageURL' : blob_result.public_url
    })
    print("[info] Upload Finished")

    # Delete images in temporary folder
    raw_file = "tmp_raw_images/" + raw_image_route
    result_file = "tmp_result_images/" + result_image_route
    if os.path.isfile(raw_file):
        os.remove(raw_file)
        os.remove(result_file)

    print("[info] Cleaned temporary image folder")
if __name__ == '__main__':
    main()

