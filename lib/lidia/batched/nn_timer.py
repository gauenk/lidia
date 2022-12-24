
# -- separate class and logic --
from dev_basics.utils import clean_code
from dev_basics.utils.timer import ExpTimerList
__methods__ = []
register_method = clean_code.register_method(__methods__)

@register_method
def _update_times(self,timer):
    # print(timer.names)
    if not(self.use_timer): return
    for key in timer.names:
        if key in self.times.names:
            self.times[key].append(timer[key])
        else:
            self.times[key] = [timer[key]]

@register_method
def _reset_times(self):
    self.times = ExpTimerList(self.use_timer)
