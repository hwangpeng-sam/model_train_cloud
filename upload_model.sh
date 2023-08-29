aws s3 sync s3://bucket-plugissue/model_input_data /home/ec2-user/model_train_cloud/model_input_data
chmod 777 $(pwd)
docker build -t train:0.0 .
#docker run -v $(pwd):/train train:0.0
nohup docker run -v$(pwd):/train train:0.0 &
# model.pt 이름 수정 필요.
#aws s3 sync ./model_output s3://bucket-plugissue/model_output