import machine

class accel():
    def __init__(self, i2c, addr=0x68):
        self.iic = i2c
        self.addr = addr
        # Hapus start/stop manual, langsung tulis ke register
        # Mengaktifkan sensor (Power Management 1)
        self.iic.writeto(self.addr, bytearray([107, 0]))

    def get_raw_values(self):
        # Hapus start/stop manual
        # Membaca 14 byte data sekaligus mulai dari register 0x3B
        a = self.iic.readfrom_mem(self.addr, 0x3B, 14)
        return a

    def get_values(self):
        raw_ints = self.get_raw_values()
        vals = {}
        # Konversi byte ke integer
        vals["AcX"] = self.bytes_toint(raw_ints[0], raw_ints[1])
        vals["AcY"] = self.bytes_toint(raw_ints[2], raw_ints[3])
        vals["AcZ"] = self.bytes_toint(raw_ints[4], raw_ints[5])
        vals["Tmp"] = self.bytes_toint(raw_ints[6], raw_ints[7]) / 340.00 + 36.53
        vals["GyX"] = self.bytes_toint(raw_ints[8], raw_ints[9])
        vals["GyY"] = self.bytes_toint(raw_ints[10], raw_ints[11])
        vals["GyZ"] = self.bytes_toint(raw_ints[12], raw_ints[13])
        return vals

    def bytes_toint(self, firstbyte, secondbyte):
        if not firstbyte & 0x80:
            return firstbyte << 8 | secondbyte
        return - (((firstbyte ^ 255) << 8) | (secondbyte ^ 255) + 1)
