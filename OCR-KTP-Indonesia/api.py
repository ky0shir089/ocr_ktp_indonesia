import cv2
import numpy as np
import ocr
import time
# Note: Uncomment for YOLO feature
import yolo_detect
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image

app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/ocr', methods = ['POST'])
@cross_origin(supports_credentials=True)
def upload_file():
    start_time = time.time()

    if 'image' not in request.files:
        finish_time = time.time() - start_time

        return jsonify({
            'error': True,
            'message': "Foto wajib ada"
        })
    
    else:
        imagefile = request.files['image'].read()
        npimg = np.frombuffer(imagefile, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Note: Uncomment for YOLO feature
        image = yolo_detect.main(image)
        (nik, nama, tempat_lahir, tgl_lahir, jenis_kelamin, gol_darah, agama,
                status_perkawinan, provinsi, kabupaten, alamat, rt_rw, 
                kel_desa, kecamatan, pekerjaan, kewarganegaraan, berlaku_hingga) = ocr.main(image)

        finish_time = time.time() - start_time

        return jsonify({
            'error': False,
            'message': 'Proses OCR Berhasil',
            'data': {
                'nik': str(nik),
                'nama': str(nama),
                'tempat_lahir': str(tempat_lahir),
                'tgl_lahir': str(tgl_lahir),
                'jenis_kelamin': str(jenis_kelamin),
                'gol_darah': str(gol_darah),
                'agama': str(agama),
                'status_perkawinan': str(status_perkawinan),
                'pekerjaan': str(pekerjaan),
                'kewarganegaraan': str(kewarganegaraan),
                'berlaku_hingga': str(berlaku_hingga),
                'alamat': {
                    'name': str(alamat),
                    'rt_rw': str(rt_rw),
                    'kel_desa': str(kel_desa),
                    'kecamatan': str(kecamatan),
                    'kabupaten': str(kabupaten),
                    'provinsi': str(provinsi)
                },
                'time_elapsed': str(round(finish_time, 3))
            }
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug = True, threaded=True)
