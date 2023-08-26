chmod 777 $(pwd)
docker build -t train:0.0 .
docker run -v $(pwd):/train train:0.0
# model.pt 이름 수정 필요.
aws s3 cp model.pt s3://bucket-plugissue/model
