from pyboy import PyBoy
pyboy = PyBoy('rom/red.gb')
while not pyboy.tick():
    pass