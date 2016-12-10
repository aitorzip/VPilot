import argparse
from PIL import Image
import socket, struct
import numpy as np
from array import array
import model

class Server:
	def __init__(self, port=8000, image_size=(200,66)):
		print('Started server')
		self.image_size = image_size
		self.buffer_size = image_size[0]*image_size[1]*3;
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.bind(('127.0.0.1', port))
		self.s.listen(1)

		self.conn, self.addr = self.s.accept()
		print('GTAV connected')

	def recvImage(self):
		data = b""
		while len(data) < self.buffer_size:
			packet = self.conn.recv(self.buffer_size - len(data))
			if not packet: return None
			data += packet

		print('Received image')
		return ((np.array(Image.frombytes('RGB', (self.image_size[1], self.image_size[0]), data)).astype('float32'))[None,:,:,:])

	def sendCommands(self,throttle, brake, steering):
		data = array('f', [throttle, brake, steering])
		self.conn.sendall(data.tobytes())
		print('Sent commands')

	def recvReward(self):
		data = b""
		while len(data) < 4:
			packet = self.conn.recv(self.buffer_size - len(data))
			if not packet: return None
			data += packet

		print('Received reward')
		return struct.unpack('f', data)[0]

	def close(self):
		self.s.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Remote Driving')
	parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
	parser.add_argument('port', type=int, help='Port to listen to')
	parser.add_argument('width', type=int, help='Width of the image to receive')
	parser.add_argument('height', type=int, help='Height of the image to receive')
	args = parser.parse_args()
	model = model.getModel(model_path=args.model)

	server = Server(port=args.port, image_size=(args.width, args.height))
	while 1:
		image = server.recvImage()
		if (image == None): break
		throttle = 1.0
		brake = 0.0
		steering = float(model.predict(image, batch_size=1))
		server.sendCommands(throttle, brake, steering)
		reward = server.recvReward()
		if (reward == None): break
		print(reward)

	server.close()


   
