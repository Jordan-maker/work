import math
import sys

class circle:
    def __init__(self, radius):
        self.radius = radius # only radius is the attibute of this class

    def circunference(self):
        return  2*math.pi*self.radius

    def area(self):
        return math.pi*(self.radius**2)

    def calculate_arch_lenght(self, angle): # angle is not an attribute
        if 0 < angle < 2*math.pi:
            return circle.circunference(self)*(angle/2*math.pi)
        else:
            sys.exit("The angle must be in [0, 2*pi]. Calculated in Radians.")


def main():
    myCircle = circle(radius=1)
    print(myCircle.circunference())
    print(myCircle.area())
    lenght = myCircle.calculate_arch_lenght(angle=4)

if __name__ == "__main__":
    main()

