"""Fold the ladder Colab's download into the repository, without touching committed files."""
import hashlib, json, os, zipfile

SRC = r"C:\Users\COMPUTEL\Downloads\c4_ladder_end12_results.zip"
REPO = r"C:\Users\COMPUTEL\Desktop\projectos\fragility-paper"
z = zipfile.ZipFile(SRC)

d0 = json.loads(z.read("data/c4_end12_s3.json"))
print("log row keys:", sorted(d0["repair"]["log"][0]))
row = d0["repair"]["log"][0]
print("row:", {k: row[k] for k in sorted(row)})

for name in sorted(z.namelist()):
    data = z.read(name)
    dest = os.path.join(REPO, *name.split("/"))
    if name.startswith("data/c4_endgame_roots"):
        cur = open(dest, "rb").read()
        print("suite ", name,
              "identical to the committed file" if hashlib.sha256(cur).hexdigest()
              == hashlib.sha256(data).hexdigest() else "DIFFERENT -> repo copy left alone")
        continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and open(dest, "rb").read() == data:
        print("already", name)
        continue
    open(dest, "wb").write(data)
    print("wrote  ", name, len(data), "bytes")
