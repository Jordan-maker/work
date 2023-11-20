#BaseClass and SubClass

class animal:  # BaseClass

    def __init__(self, name, feeding, environment):
        self.name = name
        self.feeding = feeding
        self.enviroment = environment

    def eating(self, name):
        print(f"The {self.name} is {self.feeding}.")

    def movement(self, name):
        print(f"The {self.name} can move in {self.enviroment}.")


class flyiers(animal):

    def __init__(self, name, feeding, environment, size, feathers:bool=True): # Here are located the attributes of the BaseCLass + SubClass
        super().__init__(name, feeding, environment) # Here are located only the attributes of the BaseClass
        self.feathers = feathers
        self.size = size

    def size(self):
        print(f"The {self.name} is {self.size}")
        animal.eating(self, name=self.name)      # 1 way
        super().eating(name=self.name)           # 2 way

# Se pueden crear varias sub-classes, como animales de tierra y de agua (todos ellos siendo animales)
# Si dentro de la SubClass quisieramos usar un método creado en la BaseClass, entonces lo podemos llamar de dos formas:
  #  1) BaseClass.method(self) ;  animal.eating(self)
  #  2) super().method()       ;  super().eating()
  # The using of super() is recommended because the name of the BaseClass can changed many times, while super() always will do the same action without causing problems.

parrot = flyiers(name="parrot", feeding="herbivorous", environment="sky", size="small", feathers=True)
parrot.size()
