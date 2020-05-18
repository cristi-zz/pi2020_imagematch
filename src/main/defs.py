class Keypoint:
    def __init__(self, position, color, direction):
        assert (isinstance(position, list))
        assert (isinstance(color, list))
        assert (len(color) == 3)
        for _color in color:
            assert (0 <= _color < 256)
        assert (isinstance(direction, list))
        assert (len(direction) == 3)
        self.position = position
        self.color = color
        self.direction = direction

