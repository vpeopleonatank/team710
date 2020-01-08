Các bước đội mình chạy :

- sudo docker build -t team710 .

- sudo docker run --rm -it --gpus=all --network=host -v /home/vpoat/catkin_ws/src:/catkin_ws/src --name team710 team710:latest bash

- sau khi vào container: roslaunch team710 team710.launch

Ghi chú: - Port websocket server trong file launch là 9090 
	 - cấp quyền cho file script: chmod +x src/Main.py

# team710
