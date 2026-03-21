with open(r"anex80m.hdi", "rb") as f:
    data = f.read(524288)
with open(r"hdi_head.bin", "wb") as f:
    f.write(data)