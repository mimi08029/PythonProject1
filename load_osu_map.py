import json
import os
from collections import defaultdict

maps_dir = "maps"
j_path = os.path.join(maps_dir, "metadata.json")

class ExtraType:
    slider_slide=9
    slider_body=8
    slider_end=10

class slider_param:
    def __init__(self, curve_info):
        self.curve_type, self.curvePoints = self.parse(curve_info)

    @staticmethod
    def parse(string):
        return string[0], string[1:]

class TM:
    def __init__(self, time, beatLength, meter, sampleSet, sampleIndex, volume, uninherited, effects):
        self.time = time
        self.beatLength = beatLength
        self.meter = meter
        self.sampleSet = sampleSet
        self.sampleIndex = sampleIndex
        self.volume = volume
        self.uninherited = uninherited
        self.effects = effects

class HitObj:
    def __init__(self, x, y, t, type,  hitsound):
        self.x = min(max(0, int(x)), 1000)
        self.y = min(max(0, int(y)), 1000)
        self.t = max(0, int(t))
        self.type = int(type)
        self.hitsound = hitsound

class Circle(HitObj):
    def __init__(self, x, y, t, type, hitsound, *extra_params):
        super().__init__(x, y, t, type, hitsound)
        self.extra_params = extra_params


class Slider(HitObj):
    def __init__(self, x, y, t, type, hitsound, curvePoints, slides,length, edgeSounds=None, edgeSets=None, hitSample=None):
        super().__init__(x, y, t, type, hitsound)
        self.param = slider_param(curvePoints.split("|"))
        self.slides = int(slides)
        self.length = length
        self.edgeSounds = edgeSounds
        self.edgeSets = edgeSets
        self.hitSample = hitSample

def parse_one_osu_map(path):
    categories = ["General", "Editor", "Metadata", "Difficulty"]
    hit_categories = ["HitObjects"]
    tm_categories = ["TimingPoints"]
    res = defaultdict(dict)
    curr = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("["):
                curr = line[1:-1]
                continue

            if curr in categories:
                k, v = line.split(":", maxsplit=1)
                res[curr][k.strip()] = v.strip()

            elif curr in tm_categories:
                if "tm" not in res[curr]:
                    res[curr]["tm"] = []
                params = line.split(",")
                res[curr]["tm"].append(TM(*params))

            elif curr in hit_categories:
                if "nodes" not in res[curr]:
                    res[curr]["nodes"] = []

                params = line.split(",")
                if params[3] in ("0", "1", "3", "4", "5"):
                    res[curr]["nodes"].append(Circle(*params))
                elif params[3] in ("2", ):
                    res[curr]["nodes"].append(Slider(*params))
    return res

def load_osu_maps(j_path=j_path):
    dataset_obj = []
    with open(j_path, "r") as f:
        j = json.load(f)
        for dic in j:
            audio = dic["audio"]
            for map_id in dic["beatmaps"]:
                dataset_obj.append([audio, map_id])

    return dataset_obj
