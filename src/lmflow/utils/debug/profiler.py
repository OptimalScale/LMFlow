import time
import pprint


class Timer:
    def __init__(self, name):
        self.name = name
        self.runtimes = {}
        self.runtimes_readable = {}

    def start(self, tag):
        self.runtimes[tag] = {"start": time.time()}

    def end(self, tag):
        self.runtimes[tag]["end"] = time.time()
        self.runtimes[tag]["elapsed"] = self.runtimes[tag]["end"] - self.runtimes[tag]["start"]
        
    def get_runtime(self, tag):
        return self.runtimes[tag]["elapsed"]
    
    def show(self):
        self._to_readable()
        pprint.pprint(self.runtimes_readable)
        
    def _to_readable(self):
        for tag, runtime in self.runtimes.items():
            self.runtimes_readable[tag] = {"start": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(runtime["start"]))}
            self.runtimes_readable[tag]["end"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(runtime["end"]))
            self.runtimes_readable[tag]["elapsed"] = round(runtime["elapsed"], 5)
            

if __name__ == "__main__":
    timer = Timer("profiler")
    timer.start("main")
    time.sleep(1)
    timer.end("main")
    timer.show()