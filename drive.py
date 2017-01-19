import argparse
import socket, struct
import numpy as np
from array import array
from model import nanoAitorNet

class Server:
	def __init__(self, port=8000, image_size=(200,66)):
		print('Started server')
		self.image_size = image_size
		self.buffer_size = image_size[0]*image_size[1]*3;
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.bind(('0.0.0.0', port))
		self.s.listen(1)

		self.conn, self.addr = self.s.accept()
		print('GTAV connected')

	def recvImage(self):
		data = b""
		while len(data) < self.buffer_size:
			packet = self.conn.recv(self.buffer_size - len(data))
			if not packet: return None
			data += packet

		return np.resize(np.fromstring(data, dtype='uint8'), (self.image_size[1], self.image_size[0], 3)).astype('float32')

	def sendCommands(self, throttle, steering):		
		data = array('f', [throttle, steering])
		self.conn.sendall(data.tobytes())
		print('Sent commands', data)

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
	parser.add_argument('weights', type=str, help='Path to model weights')
	parser.add_argument('port', type=int, help='Port to listen to')
	parser.add_argument('width', type=int, help='Width of the image to receive')
	parser.add_argument('height', type=int, help='Height of the image to receive')
	args = parser.parse_args()

	aitorNet = nanoAitorNet()
	model = aitorNet.getModel(weights_path=args.weights)
	x = np.zeros((50, args.height, args.width, 3), dtype='float32')

	server = Server(port=args.port, image_size=(args.width, args.height))
	while 1:
		img = server.recvImage()
		if (img == None): break
		x = np.roll(x,-1, axis=0)
		x[-1] = img

		commands = model.predict(x[None,:,:,:,:], batch_size=1)
		server.sendCommands(commands[0,0], commands[0,1])
		reward = server.recvReward()
		if (reward == None): break
		print(reward)

	server.close()


   
