"""Bake a brain into play.html so it runs offline in the browser. Usage: python make_play.py brains/ttt_3x3_k3_h32.json"""
import json
import sys

brain = json.load(open(sys.argv[1]))
html = open("play_template.html", encoding="utf-8").read().replace("__BRAIN__", json.dumps(brain))
open("play.html", "w", encoding="utf-8").write(html)
print("play.html")
