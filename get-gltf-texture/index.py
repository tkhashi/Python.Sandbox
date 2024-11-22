from pygltflib import GLTF2
import base64

gltf = GLTF2().load("extract@0_edge.gltf")

for image in gltf.images:
    uri = image.uri
    if uri is None:
        continue

    if uri.startswith("data:image/"):
        header, encoded = uri.split(",", 1)
        data = base64.b64decode(encoded)
        with open("extracted_texture.png", "wb") as f:
            f.write(data)
    else:
        with open(uri, "rb") as f:
            data = f.read()
        with open(f"extracted_{uri}", "wb") as f:
            f.write(data)
