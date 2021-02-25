# Hệ thống tìm kiếm hình ảnh đơn giản

## Overview
- Hệ thống tìm kiếm hình ảnh đơn giản gồm có 2 file:
- `offline.py`: File này trích xuất deep-feature từ kho dữ liệu. Những feature là 4096D fc6 activation từ VGG16 model với ImageNet pre-trained weights.
- `server.py`:  script này chạy server. Ta có thể query hình ảnh bằng Flask web-interface. Máy chủ tìm những hình ảnh phù hợp với câu query nhất bằng simple linear scan
- Đã chạy thành công trên hệ điều hành windows 10 - 64bit, python 3.6-64bit

## Nội dung tham khảo
- [Demo](http://www.simple-image-search.xyz/)
- [Course at CVPR2020](https://matsui528.github.io/cvpr2020_tutorial_retrieval/) [[slides](https://speakerdeck.com/matsui_528/cvpr20-tutorial-live-coding-demo-to-implement-an-image-search-engine-from-scratch)] [[video](https://www.youtube.com/watch?v=M0Y9_vBmYXU)]
- [Project page](http://yusukematsui.me/project/sis/sis.html)
- [Tutorial](https://ourcodeworld.com/articles/read/981/how-to-implement-an-image-search-engine-using-keras-tensorflow-with-python-3-in-ubuntu-18-04) and [Video](https://www.youtube.com/watch?v=Htu7b8PUyRg) by [@sdkcarlos](https://github.com/sdkcarlos)

## Các bước thực hiện
```bash
git clone https://github.com:phongultra/image-retrieval-system.git
cd image-retrieval-system
pip install -r requirements.txt

# Đưa kho ảnh dữ liệu vào (*.jpg) on static/img
download kho ảnh Oxford buildings 5K tại đây: http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/

# Sau đó fc6 features sẽ được trích xuất và lưu vào static/feature
# Sẽ mất thời gian khá lâu trong lần đầu tiên, vì Keras phải tính trọng số VGG
python offline.py

# Chạy file và đăng nhập vào localhost:5000
python server.py
```
## localhost:5000
Đây là trang chủ
Thực hiện chức năng tìm kiếm hình ảnh, gồm có:
- module nhập hình ảnh query
- module trả kết quả tương ứng với hình ảnh query

## localhost:5000/evaluation
Đây là trang đánh giá hệ thống
Thực hiện chức năng tìm kiếm hình ảnh dựa trên bộ groundtruth, gồm có:
- Đánh giá trên label good
- Đánh giá trên label ok
- Đánh giá trên label junk

## Nội dung tham khảo
- [1] C. Manning, P. Raghavan, H. Schütze: Introduction to Information Retrieval, 2008
- [2] http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/  -  The Oxford Buildings Dataset
Truy cập 19/02/2021
- [3] http://yusukematsui.me/project/sis/sis.html - Writing an Image Search Engine from Scratch - Truy cập 19/02/2021
- [4] https://giaphiep.com/blog/gioi-thieu-ve-cac-pre-trained-models-trong-linh-vuc-computer-vision-7187 - Giới thiệu về các pre-trained models trong lĩnh vực Computer Vision - Truy cập 25/02/2021
- [5] https://scikit-learn.org/stable/modules/feature_extraction.html  - Feature extraction
Truy cập 21/02/2021
- [6] https://github.com/Cartucho/mAP/   - Tính Mean Average Precision
Truy cập 22/02/2021
- [7] Jaeyoon Kim, Sung-Eui Yoon. School of Computing, Korea Advanced Institute of Science and Technology (KAIST), Daejeon, Korea: Regional Attention Based Deep Feature for Image Retrieval
