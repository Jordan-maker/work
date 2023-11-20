# name, year and the propertyCode are attributes of the class Mobile, but
#receive_message and send_message are methods of the class.
# 'text' param of the send_message method is not a attribute of the class.

class Mobile:

	"""
	@ Data attributes - instance variables (unique for each object)
	name: name of the Mobile
	year: year of construction of the Mobile

	@ class attributes - class variables (shared for all objects)
	country: where was constructed the mobile
	"""

	country = 'USA' # The same for each created object. This can not be changed. Default value
	tactil = True

	def __init__(self, name, year):

		self.mobile_name = name
		self.year = year
		self.propertyCode = f'{name}_{year}'

	def receive_message(self):
		print(f"Receive message using {self.mobile_name} Mobile.")

	def send_message(self, text):
		print(f"Send message using {self.mobile_name} Mobile.")
		print(text)


def main():

	nokia = Mobile(name="Nokia", year=2015)
	nokia.receive_message()
	nokia.send_message(text="Hola a todos.")

	samsung = Mobile(name="Samsung", year=2019)

	nokia.version = '1100'  # Here, a new attribute was created only on nokia class and not on the samsung class.

	print(nokia.__doc__)   # Printing the documentation about the Mobile class.
	print(nokia.__dict__)  # Printing the attributes and their values of this object nokia (in a dictionary).

	print(nokia.country)
	print(samsung.country)

if __name__ == "__main__":
	main()
