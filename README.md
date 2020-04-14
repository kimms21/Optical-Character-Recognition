Optical Character Recognition Challenge

1. 노이즈가 추가된 글자 이미지 생성 (Pillow.Imagedraw)
2. 생성 데이터로 OCR 모델 학습 (Recurrent Nets with Attention Modeling for OCR in the Wild 참고)
3. 2에서 학습한 모델을 베이스로 삼아 실제 데이터로 transfer learning
4. 처음부터 실제 데이터로 학습시킨 모델과 result 비교