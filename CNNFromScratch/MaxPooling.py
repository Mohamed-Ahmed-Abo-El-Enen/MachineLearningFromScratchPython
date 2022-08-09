
class MaxPooling:
    def __init__(self, name, size=3, stride=2):
        self.name = name
        self.last_input = None
        self.stride = stride
        self.size = size

    def forward(self, image):
        self.last_input = image
        num_channels, h_prev, w_prev = image.shape
        h = int((h_prev - self.size) / self.stride) + 1
        w = int((w_prev - self.size) / self.stride) + 1
        downsampled = np.zeros((num_channels, h, w))

        for i in range(num_channels):
            curr_y = out_y = 0
            while curr_y + self.size <= h_prev:
                curr_x = out_x = 0
                while curr_x + self.size <= w_prev:
                    patch = image[i, curr_y:curr_y + self.size, curr_x:curr_x + self.size]
                    downsampled[i, out_y, out_x] = np.max(patch)
                    curr_x += self.stride
                    out_x += 1
                curr_y += self.stride
                out_y += 1

        return downsampled

    def backward(self, din, learning_rate):
        num_channels, orig_dim, *_ = self.last_input.shape
        dout = np.zeros(self.last_input.shape)

        for c in range(num_channels):
            tmp_y = out_y = 0
            while tmp_y + self.size <= orig_dim:
                tmp_x = out_x = 0
                while tmp_x + self.size <= orig_dim:
                    patch = self.last_input[c, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    (x, y) = np.unravel_index(np.nanargmax(patch), patch.shape)
                    dout[c, tmp_y + x, tmp_x + y] += din[c, out_y, out_x]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1

        return dout

    def get_weights(self):
        return 0

