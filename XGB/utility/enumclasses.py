import enum

class MLFunction(enum.Enum):
	Train = 1
	Validate = 2

class Startegy(enum.Enum):
	Continous = 1
	Mod10	  = 2